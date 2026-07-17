#![feature(arbitrary_self_types)]
#![feature(arbitrary_self_types_pointers)]
#![allow(clippy::needless_return)] // tokio macro-generated code doesn't respect this

//! Concurrency and lifecycle stress for the GC modes.
//!
//! The functional suites drive GC from idle test hooks; these tests exercise the *production*
//! interleaving instead: `snapshot_and_persist` (operation drain -> inline GC pass -> tombstones
//! in the snapshot commit -> eviction) racing live change propagation on another thread. That
//! covers the surfaces the idle tests cannot: dead-list recording concurrent with a pass firing,
//! resurrection racing collection, eviction racing restoration, and the coordinator hand-off
//! under load. Run in a debug build these also sweep all the internal `debug_assert`s.
//!
//! The workload is deliberately nasty:
//! - per-lane churn subtrees keyed by generation (steady garbage every round)
//! - a 20-deep dependency chain per lane (internal aggregating nodes in the garbage — the
//!   follower-teardown path)
//! - shared leaves keyed by `generation % 3` (every third generation *resurrects* tasks the
//!   previous rounds orphaned, so stale dead-list entries and task-cache-hit reconnects happen
//!   constantly, sometimes mid-pass)

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use anyhow::Result;
use turbo_tasks::{
    ResolvedVc, State, TurboTasks, Vc, unmark_top_level_task_may_leak_eventually_consistent_state,
};
use turbo_tasks_backend::{
    BackendOptions, EvictionMode, GcMode, GitVersionInfo, TurboTasksBackend,
};

fn create_test_persistence_dir(name: &str) -> tempfile::TempDir {
    let parent = std::path::PathBuf::from(format!("{}/.cache", env!("CARGO_TARGET_TMPDIR")));
    std::fs::create_dir_all(&parent).unwrap();
    tempfile::Builder::new()
        .prefix(&format!("{name}-"))
        .tempdir_in(&parent)
        .unwrap()
}

fn open_tt_at(path: &std::path::Path, mode: GcMode) -> Arc<TurboTasks<TurboTasksBackend>> {
    TurboTasks::new(TurboTasksBackend::new(
        BackendOptions {
            num_workers: Some(4),
            small_preallocation: true,
            storage_mode: Some(turbo_tasks_backend::StorageMode::ReadWrite),
            eviction_mode: EvictionMode::Full,
            gc_mode: Some(mode),
            ..Default::default()
        },
        // Production-like persistence (ReadWrite, compaction on, not a short session): this suite
        // hammers frequent snapshot cycles, which the short-session/skip-compaction configuration
        // is not built for.
        turbo_tasks_backend::turbo_backing_storage(
            path,
            &GitVersionInfo {
                describe: "test-unversioned",
                dirty: false,
            },
            false,
            false,
            false,
        )
        .unwrap()
        .0,
    ))
}

const LANES: u32 = 4;
const LEAVES: u32 = 12;
const CHAIN: u32 = 20;
const SHARED: u32 = 6;

#[turbo_tasks::value(transparent)]
struct Generation(State<u32>);

#[turbo_tasks::function(operation, root)]
fn create_generation() -> Vc<Generation> {
    Generation(State::new(0)).cell()
}

#[turbo_tasks::function]
fn lane_leaf(lane: u32, generation: u32, index: u32) -> Vc<u32> {
    Vc::cell(
        lane.wrapping_mul(7919)
            .wrapping_add(generation.wrapping_mul(1_000_003))
            .wrapping_add(index),
    )
}

/// Shared across lanes AND across every third generation: `generation % 3` means these tasks are
/// orphaned by one bump and resurrected (task-cache hit) two bumps later — while GC may already
/// hold their ids on the dead list.
#[turbo_tasks::function]
fn shared_leaf(phase: u32, index: u32) -> Vc<u32> {
    Vc::cell(phase.wrapping_mul(31).wrapping_add(index))
}

#[turbo_tasks::function]
async fn chain_link(lane: u32, generation: u32, remaining: u32) -> Result<Vc<u32>> {
    if remaining == 0 {
        return Ok(Vc::cell(lane.wrapping_add(generation)));
    }
    Ok(Vc::cell(1u32.wrapping_add(
        *chain_link(lane, generation, remaining - 1).await?,
    )))
}

#[turbo_tasks::function]
async fn lane_mid(lane: u32, generation: u32) -> Result<Vc<u32>> {
    let mut sum = 0u32;
    for i in 0..LEAVES {
        sum = sum.wrapping_add(*lane_leaf(lane, generation, i).await?);
    }
    for i in 0..SHARED {
        sum = sum.wrapping_add(*shared_leaf(generation % 3, i).await?);
    }
    sum = sum.wrapping_add(*chain_link(lane, generation, CHAIN).await?);
    Ok(Vc::cell(sum))
}

#[turbo_tasks::function(operation, root)]
async fn stress_root(generation: ResolvedVc<Generation>) -> Result<Vc<u32>> {
    let generation = *generation.await?.get();
    let mut sum = 0u32;
    for lane in 0..LANES {
        sum = sum.wrapping_add(*lane_mid(lane, generation).await?);
    }
    Ok(Vc::cell(sum))
}

/// The value `stress_root` computes for a generation, computed independently.
fn expected(generation: u32) -> u32 {
    let mut sum = 0u32;
    for lane in 0..LANES {
        let mut mid = 0u32;
        for i in 0..LEAVES {
            mid = mid.wrapping_add(
                lane.wrapping_mul(7919)
                    .wrapping_add(generation.wrapping_mul(1_000_003))
                    .wrapping_add(i),
            );
        }
        for i in 0..SHARED {
            mid = mid.wrapping_add((generation % 3).wrapping_mul(31).wrapping_add(i));
        }
        mid = mid.wrapping_add(CHAIN.wrapping_add(lane.wrapping_add(generation)));
        sum = sum.wrapping_add(mid);
    }
    sum
}

async fn read_generation(tt: &Arc<TurboTasks<TurboTasksBackend>>, gen_value: u32) -> u32 {
    let tt_inner = tt.clone();
    turbo_tasks::run_once(tt.clone(), async move {
        unmark_top_level_task_may_leak_eventually_consistent_state();
        let generation_op = create_generation();
        let generation_vc = generation_op.resolve().strongly_consistent().await?;
        if gen_value > 0 {
            let generation = generation_op.read_strongly_consistent().await?;
            generation.set(gen_value);
        }
        let value = *stress_root(generation_vc)
            .read_strongly_consistent()
            .await?;
        let _ = &tt_inner;
        anyhow::Ok(value)
    })
    .await
    .unwrap()
}

/// Churn for many rounds while a background thread continuously runs the PRODUCTION
/// snapshot cycle (drain -> inline GC -> tombstones -> evict). Every round's result must be
/// correct despite passes constantly racing propagation, and after settling the resident set
/// must be flat (all churn garbage reclaimed) — in both modes.
fn concurrent_churn_in_mode(mode: GcMode) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async move {
        const ROUNDS: u32 = 24;
        let dir = create_test_persistence_dir(&format!("gc_concurrent_{mode:?}"));
        let tt = open_tt_at(dir.path(), mode);

        // Background snapshot+GC+evict hammer, like the production background job loop but much
        // more aggressive. Runs on the runtime's blocking pool: the snapshot path's parallel
        // scopes need a tokio runtime handle on the calling thread (production runs it from the
        // runtime-managed background loop).
        let stop = Arc::new(AtomicBool::new(false));
        let hammer = {
            let tt = tt.clone();
            let stop = stop.clone();
            tokio::task::spawn_blocking(move || {
                let mut cycles = 0u32;
                while !stop.load(Ordering::Relaxed) {
                    tt.backend().snapshot_and_evict_for_testing(&tt);
                    cycles += 1;
                    std::thread::sleep(std::time::Duration::from_millis(15));
                }
                cycles
            })
        };

        assert_eq!(read_generation(&tt, 0).await, expected(0));
        for round in 1..=ROUNDS {
            let value = read_generation(&tt, round).await;
            assert_eq!(
                value,
                expected(round),
                "wrong value at generation {round} under concurrent GC ({mode:?})"
            );
            if round % 6 == 0 {
                println!("[{mode:?}] round {round}/{ROUNDS} ok");
            }
        }

        stop.store(true, Ordering::Relaxed);
        let cycles = hammer.await.unwrap();
        println!("[{mode:?}] {ROUNDS} churn rounds raced {cycles} snapshot+GC+evict cycles");
        assert!(cycles > 0, "the hammer thread must have actually run");

        // Settle: with churn stopped, drive passes until the resident set stops shrinking.
        // Everything except the final generation's graph (and the bounded once-task residue)
        // must be reclaimed.
        let mut last = usize::MAX;
        for _ in 0..8 {
            tt.backend().snapshot_and_evict_for_testing(&tt);
            let now = tt.backend().resident_persistent_task_count_for_testing();
            if now == last {
                break;
            }
            last = now;
        }
        let resident = tt.backend().resident_persistent_task_count_for_testing();
        // Live set: per lane LEAVES + CHAIN+1 chain links + mid, plus SHARED*1 shared leaves for
        // the current phase, plus root/selector/const overhead. Give slack for one extra shared
        // phase and blocked residue, but a leak of even a few generations (~150 tasks each)
        // must fail.
        let live_estimate = (LANES * (LEAVES + CHAIN + 2) + 3 * SHARED + 16) as usize;
        println!("[{mode:?}] settled resident={resident} (live estimate {live_estimate})");
        assert!(
            resident <= live_estimate + 60,
            "[{mode:?}] resident set did not settle back to the live graph: {resident} > \
             {live_estimate} + slack — GC is leaking under concurrency"
        );

        // And the graph still computes.
        assert_eq!(read_generation(&tt, ROUNDS + 1).await, expected(ROUNDS + 1));
        tt.stop_and_wait().await;
    });
}

static SERIALIZE: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn concurrent_churn_incremental() {
    let _serialize = SERIALIZE.lock().unwrap();
    concurrent_churn_in_mode(GcMode::Incremental);
}

#[test]
fn concurrent_churn_scan() {
    let _serialize = SERIALIZE.lock().unwrap();
    concurrent_churn_in_mode(GcMode::Scan);
}

/// Crash-consistency shape: churn (creating garbage and dead-list entries), persist on shutdown,
/// reopen. The transient dead list is lost; the durable parent_counts are not. The reopened
/// session must compute correctly, and NEW garbage made after reopen must still be collected —
/// the lost entries only mean pre-shutdown garbage waits on disk (same blind spot the scan has).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reopen_after_churn_keeps_collecting() {
    let dir = create_test_persistence_dir("gc_reopen");

    // Session 1: churn several generations, run passes for some but not the last (so orphaned
    // garbage with dead-list entries is still pending at shutdown).
    {
        let tt = open_tt_at(dir.path(), GcMode::Incremental);
        assert_eq!(read_generation(&tt, 0).await, expected(0));
        for round in 1..=6 {
            assert_eq!(read_generation(&tt, round).await, expected(round));
            if round % 2 == 0 {
                tt.backend().snapshot_and_evict_for_testing(&tt);
            }
        }
        tt.stop_and_wait().await;
    }

    // Session 2: reopen. Dead list is empty; counts replayed/restored from disk.
    {
        let tt = open_tt_at(dir.path(), GcMode::Incremental);
        // The reopened graph computes (this restores the live tasks from disk).
        assert_eq!(read_generation(&tt, 7).await, expected(7));

        // New churn after reopen must be discovered and collected by the dead list as usual.
        for round in 8..=12 {
            assert_eq!(read_generation(&tt, round).await, expected(round));
        }
        let baseline = tt.backend().resident_persistent_task_count_for_testing();
        let mut last = baseline;
        for _ in 0..6 {
            tt.backend().snapshot_and_evict_for_testing(&tt);
            let now = tt.backend().resident_persistent_task_count_for_testing();
            if now == last {
                break;
            }
            last = now;
        }
        let settled = tt.backend().resident_persistent_task_count_for_testing();
        println!("reopen: baseline={baseline} settled={settled}");
        assert!(
            settled < baseline,
            "post-reopen churn garbage was not collected (baseline {baseline}, settled {settled})"
        );
        assert_eq!(read_generation(&tt, 13).await, expected(13));
        tt.stop_and_wait().await;
    }
}

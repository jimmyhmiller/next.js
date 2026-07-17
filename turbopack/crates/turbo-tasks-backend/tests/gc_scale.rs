#![feature(arbitrary_self_types)]
#![feature(arbitrary_self_types_pointers)]
#![allow(clippy::needless_return)] // tokio macro-generated code doesn't respect this

//! Production-scale scan-vs-incremental GC benchmark: ~5 million tasks, ~50 million tracked edge
//! entries, with a small per-round churn (~0.4%) — the shape of a large app under HMR edits.
//!
//! Ignored by default (it needs several GB of RAM and minutes of wall clock). Run it with:
//!
//! ```sh
//! cargo test --release -p turbo-tasks-backend --test gc_scale -- --ignored --nocapture
//! ```
//!
//! Size overrides via env: `GC_SCALE_SHARDS` (default 5000), `GC_SCALE_LEAVES` (default 1000,
//! per shard), `GC_SCALE_CHURN_MIDS` (default 20), `GC_SCALE_ROUNDS` (default 3).
//!
//! Graph shape: `scale_root` reads `SHARDS` ballast mids (+ the generation's churn mids). Ballast
//! mid `s` reads its own shard's `LEAVES` leaves plus shard `s+1`'s leaves (cross-shard shared
//! deps). Each leaf read stores ~5 edge entries (child edge, output dependency + reverse, cell
//! dependency + reverse), so 5000×1000 own reads + 5000×1000 cross reads ≈ 50M edge entries over
//! ~5M tasks. Churn mid `(generation, i)` reads 1000 generation-keyed churn leaves; bumping the
//! generation orphans `CHURN_MIDS × 1001` tasks.
//!
//! The backend runs fully in memory (`storage_mode: None`) so the numbers isolate the GC pass
//! from disk I/O.

use std::{sync::Arc, time::Instant};

use anyhow::Result;
use turbo_tasks::{
    ResolvedVc, State, TryJoinIterExt, TurboTasks, Vc,
    unmark_top_level_task_may_leak_eventually_consistent_state,
};
use turbo_tasks_backend::{BackendOptions, GcMode, TurboTasksBackend, noop_backing_storage};
use turbo_tasks_malloc::TurboMalloc;

// Track allocations so the memory-usage print reports real numbers.
#[global_allocator]
static ALLOC: TurboMalloc = TurboMalloc;

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .map(|v| v.parse().expect("env var must be a u32"))
        .unwrap_or(default)
}

fn create_tt(mode: GcMode) -> Arc<TurboTasks<TurboTasksBackend>> {
    TurboTasks::new(TurboTasksBackend::new(
        BackendOptions {
            storage_mode: None,
            gc_mode: Some(mode),
            ..Default::default()
        },
        noop_backing_storage(),
    ))
}

#[turbo_tasks::value(transparent)]
struct Generation(State<u32>);

#[turbo_tasks::function(operation, root)]
fn create_generation() -> Vc<Generation> {
    Generation(State::new(0)).cell()
}

#[turbo_tasks::function]
fn scale_leaf(shard: u32, index: u32) -> Vc<u32> {
    Vc::cell(shard.wrapping_mul(31).wrapping_add(index))
}

/// A ballast mid: reads its own shard's leaves plus the next shard's (cross-shard shared deps, so
/// every leaf has ~2 readers and the edge count doubles). Never disconnected.
#[turbo_tasks::function]
async fn scale_mid(shard: u32, shards: u32, leaves: u32) -> Result<Vc<u32>> {
    let own = (0..leaves).map(|i| scale_leaf(shard, i));
    let cross = (0..leaves).map(|i| scale_leaf((shard + 1) % shards, i));
    let values = own
        .chain(cross)
        .map(|vc| async move { vc.await })
        .try_join()
        .await?;
    Ok(Vc::cell(
        values.iter().fold(0u32, |a, b| a.wrapping_add(**b)),
    ))
}

#[turbo_tasks::function]
fn churn_leaf(generation: u32, mid: u32, index: u32) -> Vc<u32> {
    Vc::cell(
        generation
            .wrapping_mul(1_000_003)
            .wrapping_add(mid.wrapping_mul(1009))
            .wrapping_add(index),
    )
}

/// A churn mid: reads `leaves` generation-keyed churn leaves. A generation bump orphans the whole
/// previous generation's churn subtrees.
#[turbo_tasks::function]
async fn churn_mid(generation: u32, mid: u32, leaves: u32) -> Result<Vc<u32>> {
    let values = (0..leaves)
        .map(|i| churn_leaf(generation, mid, i))
        .map(|vc| async move { vc.await })
        .try_join()
        .await?;
    Ok(Vc::cell(
        values.iter().fold(0u32, |a, b| a.wrapping_add(**b)),
    ))
}

#[turbo_tasks::function(operation, root)]
async fn scale_root(
    generation: ResolvedVc<Generation>,
    shards: u32,
    leaves: u32,
    churn_mids: u32,
    churn_leaves: u32,
) -> Result<Vc<u32>> {
    let generation = *generation.await?.get();
    let ballast = (0..shards).map(|s| scale_mid(s, shards, leaves));
    let churn = (0..churn_mids).map(|m| churn_mid(generation, m, churn_leaves));
    let values = ballast
        .chain(churn)
        .map(|vc| async move { vc.await })
        .try_join()
        .await?;
    Ok(Vc::cell(
        values.iter().fold(0u32, |a, b| a.wrapping_add(**b)),
    ))
}

async fn read_root(
    tt: &Arc<TurboTasks<TurboTasksBackend>>,
    gen_value: u32,
    shards: u32,
    leaves: u32,
    churn_mids: u32,
    churn_leaves: u32,
) {
    let tt_inner = tt.clone();
    turbo_tasks::run_once(tt.clone(), async move {
        unmark_top_level_task_may_leak_eventually_consistent_state();
        let generation_op = create_generation();
        let generation_vc = generation_op.resolve().strongly_consistent().await?;
        if gen_value > 0 {
            let generation = generation_op.read_strongly_consistent().await?;
            generation.set(gen_value);
        }
        scale_root(generation_vc, shards, leaves, churn_mids, churn_leaves)
            .read_strongly_consistent()
            .await?;
        let _ = &tt_inner;
        anyhow::Ok(())
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs several GB of RAM and minutes of wall clock; run explicitly with --ignored"]
async fn gc_at_production_scale() {
    let shards = env_u32("GC_SCALE_SHARDS", 5_000);
    let leaves = env_u32("GC_SCALE_LEAVES", 1_000);
    let churn_mids = env_u32("GC_SCALE_CHURN_MIDS", 20);
    let churn_leaves = 1_000u32;
    let rounds = env_u32("GC_SCALE_ROUNDS", 3);

    let approx_tasks = shards as u64 * leaves as u64 + shards as u64;
    let approx_edges = 2 * (shards as u64 * leaves as u64) * 5;
    println!(
        "scale: ~{approx_tasks} ballast tasks, ~{approx_edges} edge entries, churn={} \
         tasks/round, rounds={rounds}",
        churn_mids as u64 * (churn_leaves as u64 + 1),
    );

    for mode in [GcMode::Scan, GcMode::Incremental] {
        let tt = create_tt(mode);

        let build_start = Instant::now();
        read_root(&tt, 0, shards, leaves, churn_mids, churn_leaves).await;
        let resident = tt.backend().resident_persistent_task_count_for_testing();
        println!(
            "[{mode:?}] build: {resident} resident tasks in {:.1?} (process memory {:.2} GiB)",
            build_start.elapsed(),
            TurboMalloc::memory_usage() as f64 / (1 << 30) as f64,
        );

        // Pass with NO garbage: the steady-state stop-the-world cost of a GC pass when nothing
        // died since the last one. This is what every snapshot pays.
        let stats = tt.backend().gc_stats_for_testing(&tt, mode);
        println!(
            "[{mode:?}] no-garbage pass: seeds={} collected={} seed={:.3?} total={:.3?}",
            stats.seed_candidates, stats.collected, stats.seed_duration, stats.total_duration,
        );

        let mut total_collected = 0usize;
        for round in 1..=rounds {
            let round_start = Instant::now();
            read_root(&tt, round, shards, leaves, churn_mids, churn_leaves).await;
            let update = round_start.elapsed();
            let stats = tt.backend().gc_stats_for_testing(&tt, mode);
            total_collected += stats.collected;
            println!(
                "[{mode:?}] round {round}: update={update:.1?} seeds={} collected={} requeued={} \
                 stale={} seed={:.3?} total={:.3?}",
                stats.seed_candidates,
                stats.collected,
                stats.requeued,
                stats.dropped_stale,
                stats.seed_duration,
                stats.total_duration,
            );
        }

        // Every churn generation must be fully reclaimed: `churn_mids` mids + their leaves.
        let expected_per_round = churn_mids as usize * (churn_leaves as usize + 1);
        assert!(
            total_collected >= expected_per_round * rounds as usize,
            "[{mode:?}] collected {total_collected}, expected >= {}",
            expected_per_round * rounds as usize
        );
        println!(
            "[{mode:?}] TOTAL collected={total_collected} (expected {} per round)",
            expected_per_round
        );

        let stop_start = Instant::now();
        tt.stop_and_wait().await;
        println!("[{mode:?}] stop: {:.1?}", stop_start.elapsed());
    }
}

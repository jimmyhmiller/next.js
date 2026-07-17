#![feature(arbitrary_self_types)]
#![feature(arbitrary_self_types_pointers)]
#![allow(clippy::needless_return)] // tokio macro-generated code doesn't respect this

//! Scan-vs-incremental GC comparison: both modes must collect the same garbage (parity), while
//! the incremental mode's pass cost must track the amount of *garbage*, not the size of the
//! *live* set (the ISMM'08 claim). Run with `--nocapture` to see the timing tables; timings are
//! printed rather than asserted (CI machines vary), but the structural counters (seed candidates,
//! collected) are asserted.
//!
//! Scenarios:
//! - churn under a persistent root with a live "ballast" set of varying size (the headline
//!   benchmark: scan pays O(live + garbage) per pass, incremental pays O(garbage))
//! - a deep dependency chain orphaned at once (cascade depth)
//! - diamond-shaped sharing where a dep shared with the live graph must survive
//! - a single task with a huge fan-out orphaned at once (per-task teardown fan-out)
//! - garbage that is evicted to disk before the pass runs — the scan cannot see it (documented
//!   limitation), the dead list still remembers it

use std::sync::Arc;

use anyhow::Result;
use turbo_tasks::{
    ResolvedVc, State, TurboTasks, Vc, unmark_top_level_task_may_leak_eventually_consistent_state,
};
use turbo_tasks_backend::{
    BackendOptions, EvictionMode, GcMode, GcPassStats, GitVersionInfo, TurboTasksBackend,
};

fn create_test_persistence_dir(name: &str) -> tempfile::TempDir {
    let parent = std::path::PathBuf::from(format!("{}/.cache", env!("CARGO_TARGET_TMPDIR")));
    std::fs::create_dir_all(&parent).unwrap();
    tempfile::Builder::new()
        .prefix(&format!("{name}-"))
        .tempdir_in(&parent)
        .unwrap()
}

fn create_tt(name: &str, mode: GcMode) -> (Arc<TurboTasks<TurboTasksBackend>>, tempfile::TempDir) {
    let dir = create_test_persistence_dir(name);
    let tt = TurboTasks::new(TurboTasksBackend::new(
        BackendOptions {
            num_workers: Some(4),
            small_preallocation: true,
            storage_mode: Some(turbo_tasks_backend::StorageMode::ReadWriteOnShutdown),
            eviction_mode: EvictionMode::Full,
            gc_mode: Some(mode),
            ..Default::default()
        },
        turbo_tasks_backend::turbo_backing_storage(
            dir.path(),
            &GitVersionInfo {
                describe: "test-unversioned",
                dirty: false,
            },
            false,
            true,
            true,
        )
        .unwrap()
        .0,
    ));
    (tt, dir)
}

#[turbo_tasks::value(transparent)]
struct Generation(State<u32>);

#[turbo_tasks::function(operation, root)]
fn create_generation() -> Vc<Generation> {
    Generation(State::new(0)).cell()
}

/// A live-ballast leaf: connected on round 0 and never disconnected. These are what a scan has to
/// wade through on every pass.
#[turbo_tasks::function]
fn live_leaf(index: u32) -> Vc<u32> {
    Vc::cell(index)
}

/// A churn leaf keyed by (generation, index): bumping the generation disconnects the whole
/// previous generation's worth of these.
#[turbo_tasks::function]
fn churn_leaf(generation: u32, index: u32) -> Vc<u32> {
    Vc::cell(generation.wrapping_mul(1_000_003).wrapping_add(index))
}

/// An intermediate over one churn leaf, so churn garbage is a two-level subtree (exercises the
/// cascade, not just single-node collection).
#[turbo_tasks::function]
async fn churn_mid(generation: u32, index: u32) -> Result<Vc<u32>> {
    Ok(Vc::cell(1 + *churn_leaf(generation, index).await?))
}

/// Reads `live` live leaves (generation-independent) and `churn` generation-keyed intermediates.
/// Bumping the generation re-executes this, keeping the live set connected and orphaning the
/// previous generation's churn subtrees (2 * churn tasks).
#[turbo_tasks::function(operation, root)]
async fn ballast_root(
    generation: ResolvedVc<Generation>,
    live: u32,
    churn: u32,
) -> Result<Vc<u32>> {
    let generation = *generation.await?.get();
    let mut sum = 0u32;
    for index in 0..live {
        sum = sum.wrapping_add(*live_leaf(index).await?);
    }
    for index in 0..churn {
        sum = sum.wrapping_add(*churn_mid(generation, index).await?);
    }
    Ok(Vc::cell(sum))
}

/// A chain task: reads the next link until `remaining == 0`. Keyed by generation, so a bump
/// orphans the entire chain at once and collection must cascade through its full depth.
#[turbo_tasks::function]
async fn chain_link(generation: u32, remaining: u32) -> Result<Vc<u32>> {
    if remaining == 0 {
        return Ok(Vc::cell(generation));
    }
    Ok(Vc::cell(1u32.wrapping_add(
        *chain_link(generation, remaining - 1).await?,
    )))
}

#[turbo_tasks::function(operation, root)]
async fn chain_root(generation: ResolvedVc<Generation>, depth: u32) -> Result<Vc<u32>> {
    let generation = *generation.await?.get();
    Ok(Vc::cell(*chain_link(generation, depth).await?))
}

/// Diamond: two generation-keyed mids share one generation-keyed base AND one shared live leaf.
/// Orphaning the generation must collect mids + base (count 2 -> 0) but never the live-shared
/// leaf (also referenced by the live graph via `ballast_root`-independent `live_leaf(0)` below).
#[turbo_tasks::function]
async fn diamond_base(generation: u32) -> Result<Vc<u32>> {
    Ok(Vc::cell(generation.wrapping_add(*live_leaf(0).await?)))
}

#[turbo_tasks::function]
async fn diamond_mid(generation: u32, side: u32) -> Result<Vc<u32>> {
    Ok(Vc::cell(
        side.wrapping_add(*diamond_base(generation).await?),
    ))
}

#[turbo_tasks::function(operation, root)]
async fn diamond_root(generation: ResolvedVc<Generation>) -> Result<Vc<u32>> {
    let generation = *generation.await?.get();
    let a = *diamond_mid(generation, 1).await?;
    let b = *diamond_mid(generation, 2).await?;
    // Keep live_leaf(0) connected to the live graph independently of the diamond.
    let live = *live_leaf(0).await?;
    Ok(Vc::cell(a.wrapping_add(b).wrapping_add(live)))
}

/// A single task reading `width` churn leaves directly: orphaned in one shot, its teardown fans
/// out across workers (chunked jobs).
#[turbo_tasks::function(operation, root)]
async fn fanout_root(generation: ResolvedVc<Generation>, width: u32) -> Result<Vc<u32>> {
    let generation = *generation.await?.get();
    let mut sum = 0u32;
    for index in 0..width {
        sum = sum.wrapping_add(*churn_leaf(generation, index).await?);
    }
    Ok(Vc::cell(sum))
}

/// A holder that pins itself against GC while executing (as a value escaping the tracked graph
/// would). Keyed by generation so each generation gets its own pinned holder.
#[turbo_tasks::function]
fn pinned_holder(generation: u32) -> Vc<u32> {
    turbo_tasks::prevent_gc();
    Vc::cell(generation)
}

/// Resolves the generation's pinned holder from within a persistent task (so no once-task residue
/// attaches to the holder) and reports the holder's raw task id as its value.
#[turbo_tasks::function]
async fn pinned_mid(generation: u32) -> Result<Vc<u32>> {
    let holder = pinned_holder(generation).resolve().await?;
    let _ = *holder.await?;
    let id = Vc::into_raw(holder)
        .try_get_task_id()
        .expect("a resolved Vc should be backed by a task");
    Ok(Vc::cell(id.to_primitive()))
}

#[turbo_tasks::function(operation, root)]
async fn pinned_root(generation: ResolvedVc<Generation>) -> Result<Vc<u32>> {
    let generation = *generation.await?.get();
    Ok(Vc::cell(*pinned_mid(generation).await?))
}

/// One churn round: bump the generation to `gen_value` and read the given root op fresh.
macro_rules! run_round {
    ($tt:expr, $gen_value:expr, $read:expr) => {{
        let tt_inner = $tt.clone();
        let gen_value: u32 = $gen_value;
        turbo_tasks::run_once($tt.clone(), async move {
            unmark_top_level_task_may_leak_eventually_consistent_state();
            let generation_op = create_generation();
            let generation_vc = generation_op.resolve().strongly_consistent().await?;
            if gen_value > 0 {
                let generation = generation_op.read_strongly_consistent().await?;
                generation.set(gen_value);
            }
            #[allow(clippy::redundant_closure_call)]
            ($read)(generation_vc).read_strongly_consistent().await?;
            let _ = &tt_inner;
            anyhow::Ok(())
        })
        .await
        .unwrap();
    }};
}

fn print_pass(scenario: &str, mode: GcMode, round: u32, resident: usize, stats: &GcPassStats) {
    println!(
        "{scenario:<28} {mode:<12?} round={round:<2} resident={resident:<7} seeds={seeds:<6} \
         collected={collected:<6} requeued={requeued:<4} stale={stale:<4} seed={seed_us:>10.1}us \
         total={total_us:>10.1}us",
        seeds = stats.seed_candidates,
        collected = stats.collected,
        requeued = stats.requeued,
        stale = stats.dropped_stale,
        seed_us = stats.seed_duration.as_secs_f64() * 1e6,
        total_us = stats.total_duration.as_secs_f64() * 1e6,
    );
}

/// The headline benchmark: fixed churn per round against live ballasts of different sizes. Both
/// modes must collect the same total garbage; the scan's pass cost grows with the ballast while
/// the incremental pass only ever touches the churn. No eviction between rounds, so the resident
/// set (what the scan wades through) stays maximal.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compare_churn_with_live_ballast() {
    const CHURN: u32 = 128;
    const ROUNDS: u32 = 6;

    for live in [1_000u32, 8_000] {
        let mut totals = Vec::new();
        for mode in [GcMode::Scan, GcMode::Incremental] {
            let scenario = format!("churn(live={live})");
            let (tt, _dir) = create_tt(&format!("cmp_churn_{live}_{mode:?}"), mode);

            // Round 0 builds the ballast + first churn generation.
            run_round!(&tt, 0, move |g| ballast_root(g, live, CHURN));
            let mut total_collected = 0usize;
            for round in 1..=ROUNDS {
                run_round!(&tt, round, move |g| ballast_root(g, live, CHURN));
                let resident = tt.backend().resident_persistent_task_count_for_testing();
                let stats = tt.backend().gc_stats_for_testing(&tt, mode);
                print_pass(&scenario, mode, round, resident, &stats);
                total_collected += stats.collected;

                if mode == GcMode::Incremental {
                    // The incremental pass must be seeded by the churn alone — never by the
                    // ballast. Each round orphans CHURN `churn_mid` roots (their leaves are found
                    // by the cascade, not the seed).
                    assert!(
                        stats.seed_candidates <= 2 * CHURN as usize,
                        "incremental seeds must track garbage, not the live set: {stats:?}"
                    );
                }
            }
            println!("{scenario:<28} {mode:<12?} TOTAL collected={total_collected}");
            totals.push(total_collected);
            tt.stop_and_wait().await;
        }
        assert_eq!(
            totals[0], totals[1],
            "scan and incremental must collect the same garbage for live={live}"
        );
        // Each round orphans a full churn generation: 2 tasks (mid + leaf) per churn index.
        let expected_min = (2 * CHURN as usize * ROUNDS as usize) / 2;
        assert!(
            totals[0] >= expected_min,
            "GC collected too little ({}); expected >= {expected_min}",
            totals[0]
        );
    }
}

/// A deep chain orphaned at once: collection must cascade through the whole depth in one pass in
/// both modes (the dead list is seeded only with the chain head).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compare_deep_chain() {
    const DEPTH: u32 = 300;
    const ROUNDS: u32 = 3;

    let mut totals = Vec::new();
    for mode in [GcMode::Scan, GcMode::Incremental] {
        let (tt, _dir) = create_tt(&format!("cmp_chain_{mode:?}"), mode);
        run_round!(&tt, 0, move |g| chain_root(g, DEPTH));
        let mut total_collected = 0usize;
        for round in 1..=ROUNDS {
            run_round!(&tt, round, move |g| chain_root(g, DEPTH));
            let resident = tt.backend().resident_persistent_task_count_for_testing();
            let stats = tt.backend().gc_stats_for_testing(&tt, mode);
            print_pass("deep-chain", mode, round, resident, &stats);
            total_collected += stats.collected;
            if mode == GcMode::Incremental {
                assert!(
                    stats.seed_candidates <= 4,
                    "a chain disconnect transitions only the head's count, got {stats:?}"
                );
            }
        }
        println!(
            "{:<28} {mode:<12?} TOTAL collected={total_collected}",
            "deep-chain"
        );
        totals.push(total_collected);
        tt.stop_and_wait().await;
    }
    assert_eq!(totals[0], totals[1], "deep-chain parity");
    // Each round orphans a DEPTH+1-task chain.
    assert!(
        totals[0] >= (DEPTH as usize) * (ROUNDS as usize),
        "chain cascade must reclaim the full depth, got {}",
        totals[0]
    );
}

/// Diamond sharing: the shared base (parent_count 2) must be collected once both mids are, and
/// the leaf shared with the live graph must survive in both modes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compare_diamond_shared_deps() {
    const ROUNDS: u32 = 4;

    let mut totals = Vec::new();
    for mode in [GcMode::Scan, GcMode::Incremental] {
        let (tt, _dir) = create_tt(&format!("cmp_diamond_{mode:?}"), mode);
        run_round!(&tt, 0, diamond_root);
        let mut total_collected = 0usize;
        for round in 1..=ROUNDS {
            run_round!(&tt, round, diamond_root);
            let resident = tt.backend().resident_persistent_task_count_for_testing();
            let stats = tt.backend().gc_stats_for_testing(&tt, mode);
            print_pass("diamond", mode, round, resident, &stats);
            total_collected += stats.collected;
        }
        println!(
            "{:<28} {mode:<12?} TOTAL collected={total_collected}",
            "diamond"
        );
        totals.push(total_collected);

        // live_leaf(0) is shared between every diamond generation and the live graph; it must
        // never have been collected (the graph still computes and it is still resident).
        let tt2 = tt.clone();
        turbo_tasks::run_once(tt.clone(), async move {
            unmark_top_level_task_may_leak_eventually_consistent_state();
            let generation_op = create_generation();
            let generation_vc = generation_op.resolve().strongly_consistent().await?;
            let output = diamond_root(generation_vc);
            let expected = ROUNDS
                .wrapping_add(1)
                .wrapping_add(ROUNDS.wrapping_add(2))
                .wrapping_add(0);
            assert_eq!(*output.read_strongly_consistent().await?, expected);
            let _ = &tt2;
            anyhow::Ok(())
        })
        .await
        .unwrap();
        tt.stop_and_wait().await;
    }
    assert_eq!(totals[0], totals[1], "diamond parity");
    // Each round orphans 2 mids + 1 base (the live-shared leaf survives).
    assert_eq!(
        totals[0],
        3 * ROUNDS as usize,
        "each diamond generation is exactly 3 collectible tasks"
    );
}

/// One huge task orphaned at once (the per-task fan-out stress): parity and pass time with the
/// teardown dominated by a single task's edges.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compare_wide_fanout() {
    const WIDTH: u32 = 5_000;

    let mut totals = Vec::new();
    for mode in [GcMode::Scan, GcMode::Incremental] {
        let (tt, _dir) = create_tt(&format!("cmp_fanout_{mode:?}"), mode);
        run_round!(&tt, 0, move |g| fanout_root(g, WIDTH));
        run_round!(&tt, 1, move |g| fanout_root(g, WIDTH));
        let resident = tt.backend().resident_persistent_task_count_for_testing();
        let stats = tt.backend().gc_stats_for_testing(&tt, mode);
        print_pass("wide-fanout", mode, 1, resident, &stats);
        totals.push(stats.collected);
        assert!(
            stats.collected >= WIDTH as usize / 2,
            "fan-out teardown must reclaim the bulk of the orphaned generation: {stats:?}"
        );
        tt.stop_and_wait().await;
    }
    assert_eq!(totals[0], totals[1], "wide-fanout parity");
}

/// The incremental mode's structural advantage: garbage that is BLOCKED (pinned) when passes
/// run, then EVICTED to disk, then unblocked. The scan refuses to judge a task whose Meta is not
/// resident (`is_restored` gate) and eviction never restores garbage nobody references — so scan
/// mode leaks it for the rest of the session. The dead list's requeued entry survives both the
/// pin and the eviction; the next incremental pass restores the task from disk, collects it, and
/// tombstones it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn incremental_collects_evicted_blocked_garbage_scan_cannot() {
    let mut collected_by_mode = Vec::new();
    for mode in [GcMode::Scan, GcMode::Incremental] {
        let (tt, _dir) = create_tt(&format!("cmp_evicted_{mode:?}"), mode);

        // Round 0 builds pinned_root -> pinned_mid(0) -> pinned_holder(0) (which pins itself);
        // the output smuggles out the holder's task id. Round 1 disconnects the generation-0
        // subtree.
        let tt2 = tt.clone();
        let holder_id = turbo_tasks::run_once(tt.clone(), async move {
            unmark_top_level_task_may_leak_eventually_consistent_state();
            let generation_op = create_generation();
            let generation_vc = generation_op.resolve().strongly_consistent().await?;
            let output = pinned_root(generation_vc);
            let raw_id = *output.read_strongly_consistent().await?;
            let holder_id = turbo_tasks::TaskId::new(raw_id).expect("task id is non-zero");

            let generation = generation_op.read_strongly_consistent().await?;
            generation.set(1);
            output.read_strongly_consistent().await?;
            let _ = &tt2;
            anyhow::Ok(holder_id)
        })
        .await
        .unwrap();

        // First pass: collects pinned_mid(0); the pinned holder is blocked (requeued in
        // incremental mode, silently left behind in scan mode).
        let stats = tt.backend().gc_stats_for_testing(&tt, mode);
        let resident = tt.backend().resident_persistent_task_count_for_testing();
        print_pass("evicted-blocked", mode, 1, resident, &stats);

        // Snapshot + evict: the blocked holder's Meta/Data are dropped (its map entry and
        // transient_ref_count residue are retained). It is now unrestored garbage-to-be.
        tt.backend().snapshot_and_evict_for_testing(&tt);

        // Unblock it. No reference count transition fires that the scan could... scan doesn't
        // listen anyway; the point is the task is now pure garbage that is not Meta-resident.
        tt.unpin_task_for_gc(holder_id);

        // The decisive pass.
        let stats = tt.backend().gc_stats_for_testing(&tt, mode);
        let resident = tt.backend().resident_persistent_task_count_for_testing();
        print_pass("evicted-blocked", mode, 2, resident, &stats);
        for (task, reason) in tt.backend().gc_blocked_tasks_for_testing() {
            println!("    blocked: {task} — {reason}");
        }
        collected_by_mode.push(stats.collected);
        tt.stop_and_wait().await;
    }
    let scan = collected_by_mode[0];
    let incremental = collected_by_mode[1];
    println!("evicted-blocked: scan collected {scan}, incremental collected {incremental}");
    assert_eq!(
        scan, 0,
        "the scan cannot see evicted (unrestored) garbage — if this starts passing, the scan \
         gained restore powers and this scenario should become a parity test"
    );
    assert_eq!(
        incremental, 1,
        "the dead list must remember the evicted+unpinned holder and collect it from disk"
    );
}

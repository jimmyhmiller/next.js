#![feature(arbitrary_self_types)]
#![feature(arbitrary_self_types_pointers)]
#![allow(clippy::needless_return)] // tokio macro-generated code doesn't respect this

//! Behavior specific to `GcMode::Incremental` (the ISMM'08-style dead list): garbage is discovered
//! by change propagation at disconnect time instead of by scanning the resident map, stale
//! dead-list entries for resurrected tasks are dropped, and parentless-but-blocked candidates are
//! requeued so no garbage is lost between passes.

use std::sync::Arc;

use anyhow::Result;
use turbo_tasks::{
    ResolvedVc, State, TaskId, TurboTasks, Vc, prevent_gc,
    unmark_top_level_task_may_leak_eventually_consistent_state,
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

/// A backend configured for incremental GC: candidate recording is on, and `gc_for_testing` seeds
/// from the dead list rather than scanning.
fn create_tt(name: &str) -> (Arc<TurboTasks<TurboTasksBackend>>, tempfile::TempDir) {
    let dir = create_test_persistence_dir(name);
    let tt = TurboTasks::new(TurboTasksBackend::new(
        BackendOptions {
            num_workers: Some(2),
            small_preallocation: true,
            storage_mode: Some(turbo_tasks_backend::StorageMode::ReadWriteOnShutdown),
            eviction_mode: EvictionMode::Full,
            gc_mode: Some(GcMode::Incremental),
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
struct Selector(State<bool>);

#[turbo_tasks::function(operation, root)]
fn create_selector(initial: bool) -> Vc<Selector> {
    Selector(State::new(initial)).cell()
}

#[turbo_tasks::function]
fn leaf(n: u32) -> Vc<u32> {
    Vc::cell(n)
}

#[turbo_tasks::function]
async fn branch_a() -> Result<Vc<u32>> {
    Ok(Vc::cell(1 + *leaf(10).await?))
}

#[turbo_tasks::function]
async fn branch_b() -> Result<Vc<u32>> {
    Ok(Vc::cell(2 + *leaf(20).await?))
}

/// A leaf that pins itself against GC while executing (as code handing a value across an untracked
/// boundary — e.g. a `spawn_detached` future sending a `Vc` over a channel — would).
#[turbo_tasks::function]
fn pinned_leaf() -> Vc<u32> {
    prevent_gc();
    Vc::cell(99)
}

/// Resolves `pinned_leaf` from within a persistent task (so no transient once-task edge or
/// aggregation residue attaches to it) and reports its raw task id as its value, letting the test
/// unpin it without ever touching it from a `run_once` context.
#[turbo_tasks::function]
async fn branch_pinned() -> Result<Vc<u32>> {
    let leaf = pinned_leaf().resolve().await?;
    let _ = *leaf.await?;
    let id = Vc::into_raw(leaf)
        .try_get_task_id()
        .expect("a resolved Vc should be backed by a task");
    Ok(Vc::cell(id.to_primitive()))
}

/// Like `select`, but reads `branch_pinned` instead of `branch_a` when the selector is false.
#[turbo_tasks::function(operation, root)]
async fn select_pinned(selector: ResolvedVc<Selector>) -> Result<Vc<u32>> {
    let use_b = *selector.await?.get();
    let value = if use_b {
        *branch_b().await?
    } else {
        *branch_pinned().await?
    };
    Ok(Vc::cell(value))
}

/// Reads exactly one branch depending on the selector; flipping it re-executes and disconnects the
/// previously-read branch (and its subtree).
#[turbo_tasks::function(operation, root)]
async fn select(selector: ResolvedVc<Selector>) -> Result<Vc<u32>> {
    let use_b = *selector.await?.get();
    let value = if use_b {
        *branch_b().await?
    } else {
        *branch_a().await?
    };
    Ok(Vc::cell(value))
}

/// Disconnecting a subtree must land its root on the dead list (recorded by the disconnect's
/// `AdjustParentCount`), and an incremental pass must collect the whole subtree through the
/// cascade — without ever scanning the live set. A follow-up pass must find nothing (the dead
/// list was drained; nothing re-fed it).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn incremental_collects_disconnected_subtree() {
    let (tt, _persistence_dir) = create_tt("incremental_collects_disconnected_subtree");
    let tt2 = tt.clone();

    let result = turbo_tasks::run_once(tt.clone(), async move {
        unmark_top_level_task_may_leak_eventually_consistent_state();

        let selector_op = create_selector(false);
        let selector_vc = selector_op.resolve().strongly_consistent().await?;
        let selector = selector_op.read_strongly_consistent().await?;

        // Build the connected graph: select -> branch_a -> leaf(10).
        let output = select(selector_vc);
        assert_eq!(*output.read_strongly_consistent().await?, 11);

        // Flip: select re-executes reading branch_b, disconnecting branch_a (whose subtree
        // includes leaf(10)).
        selector.set(true);
        assert_eq!(*output.read_strongly_consistent().await?, 22);

        anyhow::Ok(())
    })
    .await;
    result.unwrap();

    // Only branch_a's disconnect transitioned a reference count (leaf(10) is still listed as a
    // child by the garbage branch_a), so the dead list holds exactly the subtree root. The pass
    // collects branch_a and cascades into leaf(10): 2 tasks, from 1 seed.
    let stats = tt2
        .backend()
        .gc_stats_for_testing(&tt2, GcMode::Incremental);
    assert_eq!(
        stats.collected, 2,
        "expected branch_a + leaf(10) to be collected, got {stats:?}"
    );
    assert_eq!(
        stats.seed_candidates, 1,
        "only the subtree root should have been recorded on the dead list, got {stats:?}"
    );
    assert_eq!(stats.dropped_stale, 0, "nothing was resurrected: {stats:?}");

    // The dead list is drained; a second pass has nothing to do.
    let stats = tt2
        .backend()
        .gc_stats_for_testing(&tt2, GcMode::Incremental);
    assert_eq!(
        (stats.seed_candidates, stats.collected),
        (0, 0),
        "second pass must be empty, got {stats:?}"
    );

    tt.stop_and_wait().await;
}

/// A dead-list entry is a hint, not a verdict: a task that is reconnected (a task-cache hit — the
/// paper's "reuse") between being recorded and the pass must be dropped as stale, while the newly
/// disconnected branch is collected.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn incremental_drops_resurrected_candidate() {
    let (tt, _persistence_dir) = create_tt("incremental_drops_resurrected_candidate");
    let tt2 = tt.clone();

    let result = turbo_tasks::run_once(tt.clone(), async move {
        unmark_top_level_task_may_leak_eventually_consistent_state();

        let selector_op = create_selector(false);
        let selector_vc = selector_op.resolve().strongly_consistent().await?;
        let selector = selector_op.read_strongly_consistent().await?;

        let output = select(selector_vc);
        assert_eq!(*output.read_strongly_consistent().await?, 11);

        // Flip to b: branch_a is disconnected and recorded on the dead list.
        selector.set(true);
        assert_eq!(*output.read_strongly_consistent().await?, 22);

        // Flip back to a: branch_a is resurrected (reconnected via the task cache) BEFORE any GC
        // pass ran; branch_b is disconnected and recorded.
        selector.set(false);
        assert_eq!(*output.read_strongly_consistent().await?, 11);

        anyhow::Ok(())
    })
    .await;
    result.unwrap();

    // Dead list: {branch_a (stale — resurrected), branch_b}. The pass must collect branch_b +
    // leaf(20) and drop the stale branch_a entry without touching it.
    let stats = tt2
        .backend()
        .gc_stats_for_testing(&tt2, GcMode::Incremental);
    assert_eq!(
        stats.collected, 2,
        "expected branch_b + leaf(20) to be collected, got {stats:?}"
    );
    assert_eq!(
        stats.dropped_stale, 1,
        "the resurrected branch_a entry must be dropped as stale, got {stats:?}"
    );
    assert_eq!(stats.requeued, 0, "nothing is blocked: {stats:?}");

    // The graph still computes correctly (branch_a survived).
    let tt3 = tt.clone();
    let result = turbo_tasks::run_once(tt.clone(), async move {
        unmark_top_level_task_may_leak_eventually_consistent_state();
        let selector_op = create_selector(false);
        let selector_vc = selector_op.resolve().strongly_consistent().await?;
        let output = select(selector_vc);
        assert_eq!(*output.read_strongly_consistent().await?, 11);
        let _ = &tt3;
        anyhow::Ok(())
    })
    .await;
    result.unwrap();

    tt.stop_and_wait().await;
}

/// A parentless candidate that is *blocked* at pass time (here: pinned via `prevent_gc`, i.e.
/// `transient_ref_count > 0`) must be requeued rather than dropped — on both the seed path and the
/// cascade path. Nothing re-fires the dead-list hook when a pin is released (only reference-count
/// transitions on a fully-unreferenced task do), so without the requeue the task would leak
/// forever in a mode that never scans. A later pass (after unpin) must collect it from the
/// requeued entry alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn incremental_requeues_blocked_candidate() {
    let (tt, _persistence_dir) = create_tt("incremental_requeues_blocked_candidate");
    let tt2 = tt.clone();

    // Build select_pinned -> branch_pinned -> pinned_leaf (which pins itself while executing, as
    // code handing a value across an untracked boundary would), then disconnect the branch by
    // flipping the selector. The output value smuggles out pinned_leaf's task id so the test can
    // unpin it later — resolving it *here* instead would attach the run_once root's transient
    // edge + aggregation residue to it and block collection on that instead of on the pin.
    let pinned_leaf_id = turbo_tasks::run_once(tt.clone(), async move {
        unmark_top_level_task_may_leak_eventually_consistent_state();

        let selector_op = create_selector(false);
        let selector_vc = selector_op.resolve().strongly_consistent().await?;
        let selector = selector_op.read_strongly_consistent().await?;

        let output = select_pinned(selector_vc);
        let raw_id = *output.read_strongly_consistent().await?;
        let pinned_leaf_id = TaskId::new(raw_id).expect("task id is non-zero");

        // Flip: select_pinned re-executes reading branch_b, disconnecting branch_pinned (whose
        // subtree includes the pinned pinned_leaf).
        selector.set(true);
        assert_eq!(*output.read_strongly_consistent().await?, 22);

        anyhow::Ok(pinned_leaf_id)
    })
    .await
    .unwrap();

    // Pass 1: the dead list holds branch_pinned (the disconnect's count transition). It is
    // collected; the cascade drives pinned_leaf to parent_count 0 but finds it pinned — the
    // CASCADE requeue path must remember it.
    let stats = tt2
        .backend()
        .gc_stats_for_testing(&tt2, GcMode::Incremental);
    assert_eq!(
        (stats.collected, stats.requeued, stats.dropped_stale),
        (1, 1, 0),
        "branch_pinned collected, pinned_leaf cascade-requeued, got {stats:?}"
    );

    // Pass 2 (still pinned): the requeued entry seeds the pass and is still blocked — the SEED
    // requeue path must keep it.
    let stats = tt2
        .backend()
        .gc_stats_for_testing(&tt2, GcMode::Incremental);
    assert_eq!(
        (stats.seed_candidates, stats.collected, stats.requeued),
        (1, 0, 1),
        "pinned candidate must stay requeued, got {stats:?}"
    );

    // Release the pin. This does NOT re-record the task (no reference count transition fires on a
    // task that already has zero persistent parents and whose transient count just left a pin) —
    // wait: unpin IS a transient_ref_count 1 -> 0 transition, but it happens outside change
    // propagation (no aggregation job runs), so the requeued dead-list entry is what keeps the
    // task reachable by GC.
    tt.unpin_task_for_gc(pinned_leaf_id);

    // Pass 3: seeded purely by the requeued entry; collects pinned_leaf.
    let stats = tt2
        .backend()
        .gc_stats_for_testing(&tt2, GcMode::Incremental);
    assert_eq!(
        (stats.seed_candidates, stats.collected),
        (1, 1),
        "unpinned candidate must be collected from the requeued entry, got {stats:?}"
    );

    tt.stop_and_wait().await;
}

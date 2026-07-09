//! The simple truth: a task conditionally outputs different things based on a TRACKED "demanded"
//! input. Not demanded -> outputs empty, heavy work runs 0 times (even though the task ran and was
//! "emitted"). Flip demanded -> task recomputes and produces the real content. No new abstraction —
//! just a task branching on a State it reads.
#![cfg(test)]
#![feature(arbitrary_self_types)]
#![feature(arbitrary_self_types_pointers)]
#![allow(clippy::needless_return)]

use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use turbo_rcstr::RcStr;
use turbo_tasks::{State, TurboTasks, Vc};
use turbo_tasks_backend::{BackendOptions, TurboTasksBackend, noop_backing_storage};

// Counts when the HEAVY work actually runs.
static HEAVY_RUNS: AtomicUsize = AtomicUsize::new(0);

#[turbo_tasks::value]
struct Content {
    bytes: RcStr,
}

// The "demand" — a tracked, mutable input. Reading it registers a dependency; flipping it
// invalidates readers. This is the browser-requested-it signal.
#[turbo_tasks::value]
struct Demand {
    on: State<bool>,
}

#[turbo_tasks::value_impl]
impl Demand {
    #[turbo_tasks::function]
    fn new() -> Vc<Self> {
        Demand { on: State::new(false) }.cell()
    }

    #[turbo_tasks::function]
    fn is_on(&self) -> Vc<bool> {
        Vc::cell(*self.on.get()) // tracked read
    }

    #[turbo_tasks::function]
    fn turn_on(&self) -> Vc<()> {
        self.on.update_conditionally(|v| {
            let changed = !*v;
            *v = true;
            changed
        });
        Vc::cell(())
    }
}

// THE WHOLE IDEA: the content task branches on the tracked demand. Not demanded -> empty, no heavy
// work. Demanded -> do the expensive work. The task always "runs" (gets emitted); the COMPUTATION
// is behind the `if`.
#[turbo_tasks::function]
async fn content(demand: Vc<Demand>) -> Result<Vc<Content>> {
    if !*demand.is_on().await? {
        // Cheap branch: no heavy work. This is what runs during the eager emit.
        return Ok(Content { bytes: "".into() }.cell());
    }
    // Demanded branch: NOW do the expensive computation.
    HEAVY_RUNS.fetch_add(1, Ordering::SeqCst);
    Ok(Content { bytes: "REAL_HEAVY_CHUNK_BYTES".into() }.cell())
}

#[turbo_tasks::function(operation, root)]
async fn setup() -> Result<Vc<Demand>> {
    Ok(Demand::new())
}

#[turbo_tasks::function(operation, root)]
async fn emit(demand: turbo_tasks::ResolvedVc<Demand>) -> Result<Vc<RcStr>> {
    // Simulate the eager machinery: it reads content() (the "emit"/force). But content() branches on
    // demand, so if not demanded it returns empty WITHOUT heavy work.
    let c = content(*demand).await?;
    Ok(Vc::cell(c.bytes.clone()))
}

#[turbo_tasks::function(operation, root)]
async fn flip_on(demand: turbo_tasks::ResolvedVc<Demand>) -> Result<Vc<()>> {
    demand.turn_on().await?;
    Ok(Vc::cell(()))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn task_conditionally_outputs() -> Result<()> {
    let tt = TurboTasks::new(TurboTasksBackend::new(
        BackendOptions::default(),
        noop_backing_storage(),
    ));

    let demand: turbo_tasks::ResolvedVc<Demand> = tt
        .run_once(async move {
            let op = setup();
            let vc = op.resolve().strongly_consistent().await?;
            Ok(vc.to_resolved().await?)
        })
        .await?;

    // Cycle 1: EMIT while NOT demanded. content() runs (task executes, "emitted") but takes the
    // cheap branch -> heavy work runs 0 times, output is empty.
    let out1 = tt
        .run_once({
            let demand = demand;
            async move { Ok(emit(demand).read_strongly_consistent().await?.clone()) }
        })
        .await?;
    let heavy1 = HEAVY_RUNS.load(Ordering::SeqCst);
    eprintln!("[COND] cycle 1 (emit, NOT demanded): output={out1:?}, heavy ran {heavy1} time(s)");
    assert_eq!(&*out1, "", "not demanded -> empty output");
    assert_eq!(heavy1, 0, "not demanded -> heavy work did NOT run, even though the task ran & emitted");

    // Cycle 2: flip demand on (the 'browser requested it').
    tt.run_once({
        let demand = demand;
        async move {
            flip_on(demand).read_strongly_consistent().await?;
            Ok(())
        }
    })
    .await?;

    // Cycle 3: EMIT again. Now demanded -> task recomputes (incremental!) and does the heavy work.
    let out3 = tt
        .run_once({
            let demand = demand;
            async move { Ok(emit(demand).read_strongly_consistent().await?.clone()) }
        })
        .await?;
    let heavy3 = HEAVY_RUNS.load(Ordering::SeqCst);
    eprintln!("[COND] cycle 3 (emit, AFTER demand): output={out3:?}, heavy ran {heavy3} time(s)");
    assert_eq!(&*out3, "REAL_HEAVY_CHUNK_BYTES", "demanded -> real content, incrementally");
    assert_eq!(heavy3, 1, "demanded -> heavy work ran exactly once, now");

    eprintln!("[COND] === PROVEN: a task conditionally outputs empty-vs-real on a tracked demand; heavy work deferred until demanded, then recomputes incrementally ===");
    Ok(())
}

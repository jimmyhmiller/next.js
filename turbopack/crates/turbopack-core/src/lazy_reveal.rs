//! Lazy reveal registry for deferred dynamic-import subgraphs.
//!
//! A module reached through a `ChunkingType::Async` edge (a dynamic `import()`) is added to the
//! module graph but its own references are NOT expanded until its source path is *revealed* — see
//! [`is_module_deferred`] and the `[DEFERRED-SUBGRAPH]` sites in `module_graph`. Revealing flips a
//! tracked `State` (or, as a PoC poke, creates a `<file>.reveal` marker on the VFS), invalidating
//! the graph traversal, which re-runs and expands the subgraph as ordinary incremental recompute.
//!
//! # Persistence semantics
//!
//! The reveal registry is `serialization = "skip"` — **the reveal set does NOT persist**. On a
//! fresh session (dev-server restart, or a persistent-cache reload) nothing is revealed; every
//! deferred module starts deferred again. This is intentional: "revealed" means "demanded this
//! session".
//!
//! ## Intended semantics (NOT fully realized yet — see caveat)
//!
//! The *ideal* is:
//! - the **reveal state does not persist** (transient, session-scoped), but
//! - the **built bytes DO persist** — a module revealed and built in a prior session should not
//!   have to be re-parsed from scratch after restart, only re-*served*; and
//! - if the underlying **file changed**, the persisted build is invalidated and rebuilt (normal
//!   content-dependency tracking).
//!
//! ## Caveat / known soundness gap
//!
//! A task reading the registry is an ordinary `#[turbo_tasks::function]`, so its *result* is
//! cacheable and can be persisted by the turbo-persistence layer — but its dependency on the reveal
//! `State` is transient (`serialization = "skip"`). Across a persistent-cache reload this can leave
//! a cached result inconsistent with the (reset, empty) reveal set: a subgraph cached as *expanded*
//! in a prior session could be restored as expanded even though nothing has revealed it this
//! session (or vice versa). To be fully sound, one of:
//!   1. mark the gated task `session_dependent` so it re-executes on restore (re-reads the empty
//!      reveal set → deferred again; correct, but loses builtness across restart — conflicts with
//!      "bytes persist"); OR
//!   2. thread the reveal set through as a real task input (part of the cache key) rather than
//!      ambient `State`, so persistence handles it correctly and the "bytes persist, reveal
//!      doesn't" split becomes well-defined.
//!
//! This proves the *mechanism* (deferral, reveal→expand, edit-tracking); the persistence model
//! above is documented as intent, not yet implemented. `FxIndexSet` is not `Encode`, so a
//! persistent variant would use `State<Vec<RcStr>>` or route reveals as inputs.
//!
//! The `<file>.reveal` marker check in [`is_module_deferred`] is a tracked VFS read, so it does not
//! suffer the caveat — but it is a PoC poke, not the intended API.

use turbo_rcstr::RcStr;
use turbo_tasks::{FxIndexSet, State, Vc};

use crate::module::Module;

/// Process-global registry of revealed source paths. Runtime-only (never persisted).
#[turbo_tasks::value(serialization = "skip", eq = "manual", cell = "new")]
pub struct LazyRevealRegistry {
    #[turbo_tasks(debug_ignore, trace_ignore)]
    revealed: State<FxIndexSet<RcStr>>,
}

impl PartialEq for LazyRevealRegistry {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}
impl Eq for LazyRevealRegistry {}

#[turbo_tasks::value_impl]
impl LazyRevealRegistry {
    /// The singleton — a global cell so all callers share the same `State`.
    #[turbo_tasks::function]
    pub fn get() -> Vc<Self> {
        LazyRevealRegistry {
            revealed: State::new(FxIndexSet::default()),
        }
        .cell()
    }

    /// Whether `path` has been revealed. Tracked read.
    #[turbo_tasks::function]
    pub fn is_revealed(&self, path: RcStr) -> Vc<bool> {
        Vc::cell(self.revealed.get().contains(&path))
    }

    /// Reveal `path` — the "poke". Flips the tracked State, invalidating any source gated on it.
    #[turbo_tasks::function]
    pub fn reveal(&self, path: RcStr) -> Vc<()> {
        self.revealed.update_conditionally(|set| set.insert(path));
        Vc::cell(())
    }
}

/// Programmatic "poke": reveal a path via the global registry.
#[turbo_tasks::function]
pub fn reveal(path: RcStr) -> Vc<()> {
    LazyRevealRegistry::get().reveal(path)
}

/// Whether a module reached through a deferred (async) reference is still deferred — i.e. NOT
/// revealed. Keyed by the module's source path. Revealed if the in-process `State` registry says
/// so, OR (the PoC "poke") a `<file>.reveal` marker exists next to the source — a tracked VFS read,
/// so creating/editing it re-runs the module-graph traversal and expands the subgraph.
#[turbo_tasks::function]
pub async fn is_module_deferred(module: Vc<Box<dyn Module>>) -> anyhow::Result<Vc<bool>> {
    let ident = module.ident().await?;
    let path = &ident.path;
    let key = path.path.clone();

    let by_state = *LazyRevealRegistry::get().is_revealed(key).await?;

    let marker = path
        .parent()
        .join(&format!("{}.reveal", path.file_name()))?;
    let by_marker = !matches!(
        &*marker.read().await?,
        turbo_tasks_fs::FileContent::NotFound
    );

    Ok(Vc::cell(!(by_state || by_marker)))
}

//! Lazy source reveal registry.
//!
//! A `*.lazy.*` file's *source content* is empty until its path is *revealed* via [`reveal`]. Because
//! every build path (parse, SSR emit, raw-source fallback, chunking) ultimately reads the source
//! through `FileSource::content`, gating THERE defers all of them at once — the single upstream
//! door. Revealing flips a tracked `State`, invalidating the source's content task, after which the
//! file builds and behaves normally.
//!
//! # Persistence semantics
//!
//! The reveal registry is `serialization = "skip"` — **the reveal set does NOT persist**. On a fresh
//! session (dev-server restart, or a persistent-cache reload) nothing is revealed; every `*.lazy.*`
//! file starts deferred again. This is intentional: "revealed" means "demanded this session".
//!
//! ## Intended semantics (NOT fully realized yet — see caveat)
//!
//! The *ideal* is:
//! - the **reveal state does not persist** (transient, session-scoped), but
//! - the **built bytes DO persist** — a file revealed and built in a prior session should not have to
//!   be re-parsed from scratch after restart, only re-*served*; and
//! - if the underlying **file changed**, the persisted build is invalidated and rebuilt (normal
//!   content-dependency tracking).
//!
//! The content-change invalidation already holds: once revealed, `FileSource::content` reads the
//! file via the VFS, so edits invalidate and rebuild (proven in `turbopack-tests/tests/lazy_source`).
//!
//! ## Caveat / known soundness gap
//!
//! `FileSource::content` is an ordinary `#[turbo_tasks::function]`, so its *result* is cacheable and
//! can be persisted by the turbo-persistence layer — but its dependency on the reveal `State` is
//! transient (`serialization = "skip"`). Across a persistent-cache reload this can leave the cached
//! content result inconsistent with the (reset, empty) reveal set: a `.lazy.*` whose content was
//! cached as *real* in a prior session could be restored as real even though nothing has revealed it
//! this session (or vice versa). To be fully sound, one of:
//!   1. mark the gated content task `session_dependent` so it re-executes on restore (re-reads the
//!      empty reveal set → deferred again; correct, but loses builtness across restart — conflicts
//!      with "bytes persist"); OR
//!   2. thread the reveal set through as a real task input (part of the cache key) rather than
//!      ambient `State`, so persistence handles it correctly and the "bytes persist, reveal doesn't"
//!      split becomes well-defined.
//!
//! This build proves the *mechanism* (deferral, reveal→build, edit-tracking); the persistence model
//! above is documented as intent, not yet implemented. `FxIndexSet` is not `Encode`, so a persistent
//! variant would use `State<Vec<RcStr>>` or route reveals as inputs.

use turbo_rcstr::RcStr;
use turbo_tasks::{FxIndexSet, State, Vc};

use crate::module::Module;

/// Process-global registry of revealed `*.lazy.*` paths. Runtime-only (never persisted).
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

/// Is this path a lazy source that must be gated? (`*.lazy.*`)
pub fn is_lazy_path(path: &str) -> bool {
    path.contains(".lazy.")
}

/// Programmatic "poke": reveal a path via the global registry.
#[turbo_tasks::function]
pub fn reveal(path: RcStr) -> Vc<()> {
    LazyRevealRegistry::get().reveal(path)
}

/// Whether a module reached through a deferred (async) reference is still deferred — i.e. NOT
/// revealed. Keyed by the module's source path. Revealed if the in-process `State` registry says so,
/// OR (the PoC "poke") a `<file>.reveal` marker exists next to the source — a tracked VFS read, so
/// creating/editing it re-runs the module-graph traversal and expands the subgraph.
#[turbo_tasks::function]
pub async fn is_module_deferred(module: Vc<Box<dyn crate::module::Module>>) -> anyhow::Result<Vc<bool>> {
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

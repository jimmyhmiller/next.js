//! Proves the lazy-SOURCE gate end to end, across proper cycle boundaries:
//!   1. deferred: a `*.lazy.*` FileSource's content is EMPTY before reveal.
//!   2. reveal -> build: after reveal, the REAL file bytes flow through.
//!   3. edit-tracked: after reveal, editing the file rebuilds (content length changes) — i.e. once
//!      revealed it is a normal graph node with dependency tracking on its own bytes.
#![cfg(test)]
#![feature(arbitrary_self_types)]
#![feature(arbitrary_self_types_pointers)]
#![allow(clippy::needless_return)]

use anyhow::Result;
use turbo_rcstr::RcStr;
use turbo_tasks::{TurboTasks, Vc};
use turbo_tasks_backend::{BackendOptions, TurboTasksBackend, noop_backing_storage};
use turbo_tasks_fs::{DiskFileSystem, FileContent, FileSystem};
use turbopack_core::{
    asset::{Asset, AssetContent},
    file_source::FileSource,
    lazy_reveal::LazyRevealRegistry,
};

/// A SINGLE shared DiskFileSystem cell for `dir`, so all reads and the invalidate target the same
/// instance (each `DiskFileSystem::new` call is a distinct cell with its own tracked reads).
#[turbo_tasks::function]
fn test_fs(dir: RcStr) -> Vc<DiskFileSystem> {
    DiskFileSystem::new("test".into(), Vc::cell(dir))
}

/// Build the FileSource for `dir`/foo.lazy.js and return its content byte-length.
#[turbo_tasks::function(operation, root)]
async fn content_len(dir: RcStr) -> Result<Vc<usize>> {
    let fs: Vc<Box<dyn FileSystem>> = Vc::upcast(test_fs(dir.clone()));
    let path = fs.root().owned().await?.join("foo.lazy.js")?;
    let source = FileSource::new(path);
    let len = match &*source.content().await? {
        AssetContent::File(f) => match &*f.await? {
            FileContent::Content(file) => file.content().len(),
            FileContent::NotFound => 0,
        },
        _ => 0,
    };
    Ok(Vc::cell(len))
}

/// Reveal `dir`/foo.lazy.js.
#[turbo_tasks::function(operation, root)]
async fn do_reveal(dir: RcStr) -> Result<Vc<()>> {
    let fs: Vc<Box<dyn FileSystem>> = Vc::upcast(test_fs(dir.clone()));
    let path = fs.root().owned().await?.join("foo.lazy.js")?;
    LazyRevealRegistry::get().reveal(path.path.clone()).await?;
    Ok(Vc::cell(()))
}

/// Invalidate the shared DiskFileSystem so a subsequent read picks up an external file edit
/// (in a real dev server the file watcher does this automatically).
#[turbo_tasks::function(operation, root)]
async fn invalidate_fs(dir: RcStr) -> Result<Vc<()>> {
    test_fs(dir).await?.invalidate();
    Ok(Vc::cell(()))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lazy_source_reveal_and_edit_tracking() -> Result<()> {
    let uniq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("lazy_src_test_{uniq}"));
    std::fs::create_dir_all(&dir)?;
    let file = dir.join("foo.lazy.js");
    std::fs::write(&file, b"export const X = 'REAL_BYTES_HERE';\n")?; // 36 bytes
    let d: RcStr = dir.to_string_lossy().to_string().into();

    let tt = TurboTasks::new(TurboTasksBackend::new(
        BackendOptions::default(),
        noop_backing_storage(),
    ));

    // Cycle 1: NOT revealed -> empty (deferred).
    let before = {
        let d = d.clone();
        *tt.run_once(async move { Ok(content_len(d).read_strongly_consistent().await?) })
            .await?
    };
    eprintln!("[LAZY-SRC] cycle 1 (not revealed): len = {before}");

    // Cycle 2: reveal.
    {
        let d = d.clone();
        tt.run_once(async move {
            do_reveal(d).read_strongly_consistent().await?;
            Ok(())
        })
        .await?;
    }

    // Cycle 3: after reveal -> real bytes.
    let after = {
        let d = d.clone();
        *tt.run_once(async move { Ok(content_len(d).read_strongly_consistent().await?) })
            .await?
    };
    eprintln!("[LAZY-SRC] cycle 3 (after reveal): len = {after}");

    // ---- EDIT the file (append content) — a real change to the now-tracked bytes. ----
    std::fs::write(
        &file,
        b"export const X = 'REAL_BYTES_HERE';\nexport const Y = 'MORE_BYTES_ADDED_BY_EDIT';\n",
    )?;
    // Cycle 4: invalidate the fs (stands in for the dev-server watcher noticing the write).
    {
        let d = d.clone();
        tt.run_once(async move {
            invalidate_fs(d).read_strongly_consistent().await?;
            Ok(())
        })
        .await?;
    }

    // Cycle 5: read again -> the edit is reflected (it rebuilt). Proves edit tracking after reveal.
    let after_edit = {
        let d = d.clone();
        *tt.run_once(async move { Ok(content_len(d).read_strongly_consistent().await?) })
            .await?
    };
    eprintln!("[LAZY-SRC] cycle 5 (after edit + invalidate): len = {after_edit}");

    assert_eq!(before, 0, "before reveal: EMPTY (deferred)");
    assert!(after > 0, "after reveal: real bytes flow");
    assert!(
        after_edit > after,
        "after editing the file (post-reveal): content rebuilt with the new, longer bytes \
         (was {after}, now {after_edit})"
    );
    eprintln!(
        "[LAZY-SRC] === PROVEN: deferred(0) -> revealed({after}) -> edited & rebuilt({after_edit}). \
         Once revealed it's a normal, dependency-tracked module. ==="
    );
    Ok(())
}

use anyhow::{Result, bail};
use turbo_rcstr::RcStr;
use turbo_tasks::Vc;
use turbo_tasks_fs::{FileContent, FileSystemEntryType, FileSystemPath, LinkContent};

use crate::{
    asset::{Asset, AssetContent},
    ident::AssetIdent,
    source::Source,
};

/// The raw [Source]. It represents raw content from a path without any
/// references to other [Source]s.
#[turbo_tasks::value]
pub struct FileSource {
    path: FileSystemPath,
    query: RcStr,
    fragment: RcStr,
}

impl FileSource {
    pub fn new(path: FileSystemPath) -> Vc<Self> {
        FileSource::new_with_query_and_fragment(path, RcStr::default(), RcStr::default())
    }
    pub fn new_with_query(path: FileSystemPath, query: RcStr) -> Vc<Self> {
        FileSource::new_with_query_and_fragment(path, query, RcStr::default())
    }
}

#[turbo_tasks::value_impl]
impl FileSource {
    #[turbo_tasks::function]
    pub fn new_with_query_and_fragment(
        path: FileSystemPath,
        query: RcStr,
        fragment: RcStr,
    ) -> Vc<Self> {
        Self::cell(FileSource {
            path,
            query,
            fragment,
        })
    }
}

#[turbo_tasks::value_impl]
impl Source for FileSource {
    #[turbo_tasks::function]
    fn ident(&self) -> Vc<AssetIdent> {
        AssetIdent::from_path(self.path.clone())
            .with_query(self.query.clone())
            .with_fragment(self.fragment.clone())
            .into_vc()
    }

    #[turbo_tasks::function]
    fn description(&self) -> Vc<RcStr> {
        Vc::cell(format!("file content of {}", self.path).into())
    }
}

#[turbo_tasks::value_impl]
impl Asset for FileSource {
    #[turbo_tasks::function]
    async fn content(&self) -> Result<Vc<AssetContent>> {
        let file_type = &*self.path.get_type().await?;
        match file_type {
            FileSystemEntryType::Symlink => match &*self.path.read_link().await? {
                LinkContent::Link { target, link_type } => Ok(AssetContent::Redirect {
                    target: target.clone(),
                    link_type: *link_type,
                }
                .cell()),
                _ => bail!("Invalid symlink"),
            },
            FileSystemEntryType::File => {
                // [LAZY-SOURCE] Single-door gate. Every build path (parse, SSR, raw-source
                // fallback, chunking) reads the file through this one function. If this is a
                // `*.lazy.*` file that has NOT been revealed, hand back EMPTY content instead of the
                // real bytes — so nothing downstream has anything to build. Revealing the path (via
                // crate::lazy_reveal::reveal) flips a tracked State that invalidates this task,
                // after which the real file content flows through and the module builds normally.
                if crate::lazy_reveal::is_lazy_path(&self.path.path)
                    && !*crate::lazy_reveal::LazyRevealRegistry::get()
                        .is_revealed(self.path.path.clone())
                        .await?
                {
                    // Empty file content: valid, parses to nothing, no work downstream.
                    return Ok(AssetContent::File(
                        FileContent::Content(turbo_tasks_fs::File::from("")).resolved_cell(),
                    )
                    .cell());
                }
                Ok(AssetContent::File(self.path.read().to_resolved().await?).cell())
            }
            FileSystemEntryType::NotFound => {
                Ok(AssetContent::File(FileContent::NotFound.resolved_cell()).cell())
            }
            _ => bail!("Invalid file type {:?}", file_type),
        }
    }
}

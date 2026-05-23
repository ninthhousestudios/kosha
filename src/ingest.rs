use std::path::{Path, PathBuf};
use std::sync::mpsc;

use sqlx::PgPool;

use crate::chunk::{self, ChunkConfig};
use crate::decompose::Decomposer;
use crate::decompose::epub::EpubDecomposer;
use crate::decompose::html::HtmlDecomposer;
use crate::decompose::image::ImageDecomposer;
use crate::decompose::markdown::MarkdownDecomposer;
use crate::decompose::pdf::PdfDecomposer;
use crate::decompose::plain_text::PlainTextDecomposer;
use crate::embed::EmbedProvider;
use crate::store;

#[derive(Debug)]
pub struct IngestResult {
    pub content_hash: String,
    pub source_path: String,
    pub collection: String,
    pub tags: Vec<String>,
    pub segments: u32,
    pub chunks: u32,
    pub skipped: bool,
}

const SUPPORTED_EXTENSIONS: &[&str] = &[
    "pdf", "epub", "png", "jpg", "jpeg", "md", "markdown", "html", "htm", "txt",
];

fn decomposer_for(path: &Path) -> Box<dyn Decomposer> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("pdf") => Box::new(PdfDecomposer),
        Some("epub") => Box::new(EpubDecomposer),
        Some("png" | "jpg" | "jpeg") => Box::new(ImageDecomposer),
        Some("md" | "markdown") => Box::new(MarkdownDecomposer),
        Some("html" | "htm") => Box::new(HtmlDecomposer),
        _ => Box::new(PlainTextDecomposer),
    }
}

fn has_supported_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| SUPPORTED_EXTENSIONS.contains(&ext))
}

pub fn collect_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_files_rec(dir, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_files_rec(dir: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| !n.to_string_lossy().starts_with('.')) {
                collect_files_rec(&path, out)?;
            }
        } else if has_supported_extension(&path) {
            out.push(path);
        }
    }
    Ok(())
}

pub async fn ingest_file(
    pool: &PgPool,
    embedder: &dyn EmbedProvider,
    path: &Path,
    chunk_cfg: &ChunkConfig,
    collection: &str,
    tags: &[String],
) -> anyhow::Result<IngestResult> {
    let content = tokio::fs::read(path).await?;
    let source_path = path.to_string_lossy().to_string();

    let content_hash = blake3::hash(&content).to_hex().to_string();
    tracing::info!(%content_hash, %source_path, %collection, "checking leaf");

    match store::leaf_status(pool, &content_hash).await? {
        Some(ref s) if s == "ready" => {
            tracing::info!(%content_hash, "leaf already ingested, skipping");
            return Ok(IngestResult {
                content_hash,
                source_path,
                collection: collection.to_string(),
                tags: tags.to_vec(),
                segments: 0,
                chunks: 0,
                skipped: true,
            });
        }
        Some(ref status) => {
            tracing::info!(%content_hash, %status, "purging partial data for re-ingest");
            store::purge_leaf_children(pool, &content_hash).await?;
        }
        None => {}
    }

    let is_pdf = path.extension().and_then(|e| e.to_str()) == Some("pdf");

    if is_pdf {
        ingest_pdf_pipelined(pool, embedder, &content, &content_hash, &source_path, chunk_cfg, collection, tags).await
    } else {
        ingest_generic(pool, embedder, path, &content, &content_hash, &source_path, chunk_cfg, collection, tags).await
    }
}

async fn ingest_generic(
    pool: &PgPool,
    embedder: &dyn EmbedProvider,
    path: &Path,
    content: &[u8],
    content_hash: &str,
    source_path: &str,
    chunk_cfg: &ChunkConfig,
    collection: &str,
    tags: &[String],
) -> anyhow::Result<IngestResult> {
    let decomposer = decomposer_for(path);
    let format_name = decomposer.format_name();
    let segments = decomposer.decompose(content, source_path)?;
    let segment_count = segments.len() as i32;

    store::insert_leaf(pool, content_hash, source_path, format_name, None, collection, segment_count).await?;
    if !tags.is_empty() {
        store::set_leaf_tags(pool, content_hash, tags).await?;
    }

    match ingest_segments(pool, embedder, content_hash, &segments, chunk_cfg).await {
        Ok(total_chunks) => {
            store::mark_leaf_ready(pool, content_hash).await?;
            tracing::info!(%content_hash, segments = segment_count, chunks = total_chunks, "ingest complete");
            Ok(IngestResult {
                content_hash: content_hash.to_string(),
                source_path: source_path.to_string(),
                collection: collection.to_string(),
                tags: tags.to_vec(),
                segments: segment_count as u32,
                chunks: total_chunks,
                skipped: false,
            })
        }
        Err(e) => {
            let _ = store::mark_leaf_error(pool, content_hash, &e.to_string()).await;
            Err(e)
        }
    }
}

async fn ingest_pdf_pipelined(
    pool: &PgPool,
    embedder: &dyn EmbedProvider,
    content: &[u8],
    content_hash: &str,
    source_path: &str,
    chunk_cfg: &ChunkConfig,
    collection: &str,
    tags: &[String],
) -> anyhow::Result<IngestResult> {
    let (tx, rx) = mpsc::sync_channel::<crate::decompose::Segment>(8);

    let content_owned = content.to_vec();
    let decompose_handle = tokio::task::spawn_blocking(move || {
        let decomposer = PdfDecomposer;
        decomposer.decompose_streamed(&content_owned, tx)
    });

    // We don't know segment_count up front for the pipelined path.
    // Insert leaf with 0, update after we finish.
    store::insert_leaf(pool, content_hash, source_path, "pdf", None, collection, 0).await?;
    if !tags.is_empty() {
        store::set_leaf_tags(pool, content_hash, tags).await?;
    }

    let mut total_chunks = 0u32;
    let mut segment_count = 0i32;
    let mut ingest_error: Option<anyhow::Error> = None;

    for seg in rx {
        segment_count += 1;
        match ingest_one_segment(pool, embedder, content_hash, &seg, chunk_cfg).await {
            Ok(chunks) => total_chunks += chunks,
            Err(e) => {
                ingest_error = Some(e);
                break;
            }
        }
    }

    // Wait for the decompose thread to finish
    let decompose_result = decompose_handle.await?;

    if let Some(e) = ingest_error {
        let _ = store::mark_leaf_error(pool, content_hash, &e.to_string()).await;
        return Err(e);
    }

    if let Err(e) = decompose_result {
        let _ = store::mark_leaf_error(pool, content_hash, &e.to_string()).await;
        return Err(e);
    }

    store::update_leaf_segment_count(pool, content_hash, segment_count).await?;
    store::mark_leaf_ready(pool, content_hash).await?;
    tracing::info!(%content_hash, segments = segment_count, chunks = total_chunks, "ingest complete");

    Ok(IngestResult {
        content_hash: content_hash.to_string(),
        source_path: source_path.to_string(),
        collection: collection.to_string(),
        tags: tags.to_vec(),
        segments: segment_count as u32,
        chunks: total_chunks,
        skipped: false,
    })
}

async fn ingest_one_segment(
    pool: &PgPool,
    embedder: &dyn EmbedProvider,
    content_hash: &str,
    seg: &crate::decompose::Segment,
    chunk_cfg: &ChunkConfig,
) -> anyhow::Result<u32> {
    use crate::decompose::SegmentContent;

    let mut chunk_count = 0u32;

    match &seg.content {
        SegmentContent::Text(text) => {
            let segment_id = store::insert_segment(
                pool,
                content_hash,
                seg.index as i32,
                &seg.label,
                Some(text),
            )
            .await?;

            let chunks = chunk::chunk_segment(text, &seg.label, chunk_cfg);
            let chunk_texts: Vec<String> =
                chunks.iter().map(|c| c.content.clone()).collect();
            let embeddings = embedder.embed_batch(chunk_texts).await?;

            for (ch, emb) in chunks.iter().zip(embeddings.iter()) {
                store::insert_chunk(
                    pool,
                    segment_id,
                    ch.index as i32,
                    content_hash,
                    seg.index as i32,
                    &ch.label,
                    Some(&ch.content),
                    emb,
                    embedder.provider_name(),
                    embedder.model_name(),
                    embedder.dimension() as i32,
                )
                .await?;
                chunk_count += 1;
            }
        }
        SegmentContent::Image(png_data) => {
            let segment_id = store::insert_segment(
                pool,
                content_hash,
                seg.index as i32,
                &seg.label,
                None,
            )
            .await?;

            let embeddings = embedder
                .embed_image_bytes(vec![png_data.clone()])
                .await?;
            let emb = embeddings.into_iter().next().ok_or_else(|| {
                anyhow::anyhow!("empty image embedding result for {}", seg.label)
            })?;

            store::insert_chunk(
                pool,
                segment_id,
                0,
                content_hash,
                seg.index as i32,
                &seg.label,
                None,
                &emb,
                embedder.provider_name(),
                embedder.model_name(),
                embedder.dimension() as i32,
            )
            .await?;
            chunk_count += 1;
        }
    }

    Ok(chunk_count)
}

async fn ingest_segments(
    pool: &PgPool,
    embedder: &dyn EmbedProvider,
    content_hash: &str,
    segments: &[crate::decompose::Segment],
    chunk_cfg: &ChunkConfig,
) -> anyhow::Result<u32> {
    let mut total_chunks = 0u32;
    for seg in segments {
        total_chunks += ingest_one_segment(pool, embedder, content_hash, seg, chunk_cfg).await?;
    }
    Ok(total_chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn supported_extensions() {
        assert!(has_supported_extension(Path::new("doc.pdf")));
        assert!(has_supported_extension(Path::new("doc.epub")));
        assert!(has_supported_extension(Path::new("doc.md")));
        assert!(has_supported_extension(Path::new("doc.html")));
        assert!(has_supported_extension(Path::new("photo.png")));
        assert!(has_supported_extension(Path::new("photo.jpg")));
        assert!(has_supported_extension(Path::new("photo.jpeg")));
        assert!(has_supported_extension(Path::new("notes.txt")));
        assert!(!has_supported_extension(Path::new("binary.exe")));
        assert!(!has_supported_extension(Path::new("no_extension")));
    }

    #[test]
    fn collect_files_walks_recursively() {
        let tmp = tempdir("walk");
        fs::write(tmp.join("a.md"), "hello").unwrap();
        fs::write(tmp.join("b.txt"), "world").unwrap();
        fs::write(tmp.join("skip.rs"), "fn main() {}").unwrap();
        fs::create_dir(tmp.join("sub")).unwrap();
        fs::write(tmp.join("sub/c.pdf"), "fake pdf").unwrap();
        fs::create_dir(tmp.join(".hidden")).unwrap();
        fs::write(tmp.join(".hidden/d.md"), "secret").unwrap();

        let files = collect_files(&tmp).unwrap();
        let names: Vec<&str> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();

        assert!(names.contains(&"a.md"));
        assert!(names.contains(&"b.txt"));
        assert!(names.contains(&"c.pdf"));
        assert!(!names.contains(&"skip.rs"), "unsupported extension should be skipped");
        assert!(!names.contains(&"d.md"), "hidden directories should be skipped");
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn collect_files_sorted() {
        let tmp = tempdir("sort");
        fs::write(tmp.join("z.txt"), "").unwrap();
        fs::write(tmp.join("a.txt"), "").unwrap();

        let files = collect_files(&tmp).unwrap();
        assert!(files[0] < files[1]);
    }

    fn tempdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("kosha-test-{}-{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}

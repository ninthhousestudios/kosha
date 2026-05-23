use std::sync::mpsc;

use anyhow::Context;
use cairo::{Format, ImageSurface};
use poppler::Document;

use super::{Segment, SegmentContent};

const RENDER_DPI: f64 = 150.0;
const PDF_POINTS_PER_INCH: f64 = 72.0;
const PARALLEL_PAGES: usize = 4;

pub struct PdfDecomposer;

impl PdfDecomposer {
    pub fn decompose_streamed(
        &self,
        content: &[u8],
        tx: mpsc::SyncSender<Segment>,
    ) -> anyhow::Result<i32> {
        let doc = Document::from_data(content, None)
            .context("failed to parse PDF")?;
        let n_pages = doc.n_pages() as usize;

        for batch_start in (0..n_pages).step_by(PARALLEL_PAGES) {
            let batch_end = (batch_start + PARALLEL_PAGES).min(n_pages);
            let batch_size = batch_end - batch_start;

            let (batch_tx, batch_rx) =
                mpsc::sync_channel::<(usize, Result<Option<Segment>, anyhow::Error>)>(batch_size);

            std::thread::scope(|s| {
                for offset in 0..batch_size {
                    let page_idx = batch_start + offset;
                    let btx = batch_tx.clone();

                    s.spawn(move || {
                        let doc = match Document::from_data(content, None) {
                            Ok(d) => d,
                            Err(e) => {
                                let _ = btx.send((offset, Err(anyhow::anyhow!(
                                    "failed to open PDF for page {}: {e}", page_idx + 1
                                ))));
                                return;
                            }
                        };
                        let page = match doc.page(page_idx as i32) {
                            Some(p) => p,
                            None => {
                                let _ = btx.send((offset, Err(anyhow::anyhow!(
                                    "page {} not accessible", page_idx + 1
                                ))));
                                return;
                            }
                        };

                        let _ = btx.send((offset, Ok(decompose_page(&page, page_idx))));
                    });
                }
                drop(batch_tx);
            });

            let mut results: Vec<Result<Option<Segment>, anyhow::Error>> =
                (0..batch_size).map(|_| Ok(None)).collect();
            for (offset, result) in batch_rx {
                results[offset] = result;
            }

            for result in results {
                match result {
                    Err(e) => return Err(e),
                    Ok(Some(seg)) => {
                        if tx.send(seg).is_err() {
                            return Ok(n_pages as i32);
                        }
                    }
                    Ok(None) => {}
                }
            }
        }

        Ok(n_pages as i32)
    }
}

fn decompose_page(page: &poppler::Page, page_idx: usize) -> Option<Segment> {
    let text = page.text().unwrap_or_default();
    let text = text.trim().to_string();
    let label = format!("p. {}", page_idx + 1);

    if text.is_empty() {
        tracing::debug!(page = page_idx + 1, "page has no text layer, rendering to PNG");
        match render_page_to_png(page) {
            Ok(png_bytes) => Some(Segment {
                index: page_idx,
                label,
                content: SegmentContent::Image(png_bytes),
            }),
            Err(e) => {
                tracing::warn!(page = page_idx + 1, error = %e, "failed to render page, skipping");
                None
            }
        }
    } else {
        Some(Segment {
            index: page_idx,
            label,
            content: SegmentContent::Text(text),
        })
    }
}

fn render_page_to_png(page: &poppler::Page) -> anyhow::Result<Vec<u8>> {
    let (w_pts, h_pts) = page.size();
    let scale = RENDER_DPI / PDF_POINTS_PER_INCH;
    let w_px = (w_pts * scale).ceil() as i32;
    let h_px = (h_pts * scale).ceil() as i32;

    let surface = ImageSurface::create(Format::Rgb24, w_px, h_px)
        .map_err(|e| anyhow::anyhow!("cairo surface: {e}"))?;

    let cr = cairo::Context::new(&surface)
        .map_err(|e| anyhow::anyhow!("cairo context: {e}"))?;

    // White background
    cr.set_source_rgb(1.0, 1.0, 1.0);
    let _ = cr.paint();

    cr.scale(scale, scale);
    page.render(&cr);

    drop(cr);

    let mut buf = Vec::new();
    surface.write_to_png(&mut buf)
        .map_err(|e| anyhow::anyhow!("PNG write: {e}"))?;

    Ok(buf)
}

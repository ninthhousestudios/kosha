use std::sync::mpsc;

use anyhow::Context;
use cairo::{Format, ImageSurface};
use poppler::Document;

use super::{Decomposer, Segment, SegmentContent};

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

            let (batch_tx, batch_rx) = mpsc::sync_channel::<(usize, Option<Segment>)>(batch_size);

            std::thread::scope(|s| {
                for offset in 0..batch_size {
                    let page_idx = batch_start + offset;
                    let btx = batch_tx.clone();

                    s.spawn(move || {
                        let doc = match Document::from_data(content, None) {
                            Ok(d) => d,
                            Err(e) => {
                                tracing::warn!(page = page_idx + 1, error = %e, "failed to open PDF for page, skipping");
                                let _ = btx.send((offset, None));
                                return;
                            }
                        };
                        let page = match doc.page(page_idx as i32) {
                            Some(p) => p,
                            None => {
                                tracing::warn!(page = page_idx + 1, "page not accessible, skipping");
                                let _ = btx.send((offset, None));
                                return;
                            }
                        };

                        let _ = btx.send((offset, decompose_page(&page, page_idx)));
                    });
                }
                drop(batch_tx);
            });

            let mut results: Vec<Option<Segment>> = (0..batch_size).map(|_| None).collect();
            for (offset, seg) in batch_rx {
                results[offset] = seg;
            }

            for seg in results.into_iter().flatten() {
                if tx.send(seg).is_err() {
                    return Ok(n_pages as i32);
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

impl Decomposer for PdfDecomposer {
    fn format_name(&self) -> &'static str {
        "pdf"
    }

    fn decompose(&self, content: &[u8], _source_path: &str) -> anyhow::Result<Vec<Segment>> {
        let (tx, rx) = mpsc::sync_channel(PARALLEL_PAGES * 2);
        self.decompose_streamed(content, tx)?;
        Ok(rx.into_iter().collect())
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

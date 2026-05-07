use anyhow::Context;
use cairo::{Format, ImageSurface};
use poppler::Document;

use super::{Decomposer, Segment, SegmentContent};

const RENDER_DPI: f64 = 150.0;
const PDF_POINTS_PER_INCH: f64 = 72.0;

pub struct PdfDecomposer;

impl Decomposer for PdfDecomposer {
    fn format_name(&self) -> &'static str {
        "pdf"
    }

    fn decompose(&self, content: &[u8], _source_path: &str) -> anyhow::Result<Vec<Segment>> {
        let doc = Document::from_data(content, None)
            .context("failed to parse PDF")?;
        let n_pages = doc.n_pages();
        let mut segments = Vec::new();

        for i in 0..n_pages {
            let page = doc
                .page(i as i32)
                .with_context(|| format!("page {} not accessible", i + 1))?;
            let text = page.text().unwrap_or_default();
            let text = text.trim().to_string();
            let label = format!("p. {}", i + 1);

            if text.is_empty() {
                tracing::debug!(page = i + 1, "page has no text layer, rendering to PNG");
                match render_page_to_png(&page) {
                    Ok(png_bytes) => {
                        segments.push(Segment {
                            index: i as usize,
                            label,
                            content: SegmentContent::Image(png_bytes),
                        });
                    }
                    Err(e) => {
                        tracing::warn!(page = i + 1, error = %e, "failed to render page, skipping");
                    }
                }
            } else {
                segments.push(Segment {
                    index: i as usize,
                    label,
                    content: SegmentContent::Text(text),
                });
            }
        }

        Ok(segments)
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

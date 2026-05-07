use super::{Decomposer, Segment, SegmentContent};

pub struct ImageDecomposer;

impl Decomposer for ImageDecomposer {
    fn format_name(&self) -> &'static str {
        "image"
    }

    fn decompose(&self, content: &[u8], source_path: &str) -> anyhow::Result<Vec<Segment>> {
        let label = std::path::Path::new(source_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(source_path)
            .to_string();

        Ok(vec![Segment {
            index: 0,
            label,
            content: SegmentContent::Image(content.to_vec()),
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_produces_single_image_segment() {
        let png_bytes = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let segments = ImageDecomposer
            .decompose(&png_bytes, "/docs/chart.png")
            .unwrap();

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].index, 0);
        assert_eq!(segments[0].label, "chart.png");
        assert!(matches!(segments[0].content, SegmentContent::Image(ref b) if b == &png_bytes));
    }

    #[test]
    fn jpg_produces_single_image_segment() {
        let jpg_bytes = vec![0xFF, 0xD8, 0xFF, 0xE0];
        let segments = ImageDecomposer
            .decompose(&jpg_bytes, "/photos/sunset.jpg")
            .unwrap();

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].label, "sunset.jpg");
        assert!(matches!(segments[0].content, SegmentContent::Image(ref b) if b == &jpg_bytes));
    }

    #[test]
    fn label_falls_back_to_full_path() {
        let bytes = vec![0x89];
        let segments = ImageDecomposer.decompose(&bytes, "bare").unwrap();
        assert_eq!(segments[0].label, "bare");
    }
}

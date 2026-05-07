use super::{Decomposer, Segment, SegmentContent};

pub struct PlainTextDecomposer;

impl Decomposer for PlainTextDecomposer {
    fn format_name(&self) -> &'static str {
        "plain_text"
    }

    fn decompose(&self, content: &[u8], _source_path: &str) -> anyhow::Result<Vec<Segment>> {
        let text = std::str::from_utf8(content)?;
        Ok(text
            .split("\n\n")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .enumerate()
            .map(|(i, text)| Segment {
                index: i,
                label: format!("paragraph {}", i + 1),
                content: SegmentContent::Text(text.to_string()),
            })
            .collect())
    }
}

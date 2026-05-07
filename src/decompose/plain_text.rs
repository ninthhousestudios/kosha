use super::{Decomposer, Segment};

pub struct PlainTextDecomposer;

impl Decomposer for PlainTextDecomposer {
    fn format_name(&self) -> &'static str {
        "plain_text"
    }

    fn decompose(&self, content: &str, _source_path: &str) -> Vec<Segment> {
        content
            .split("\n\n")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .enumerate()
            .map(|(i, text)| Segment {
                index: i,
                label: format!("paragraph {}", i + 1),
                content: text.to_string(),
            })
            .collect()
    }
}

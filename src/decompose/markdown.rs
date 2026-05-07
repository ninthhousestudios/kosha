use super::{Decomposer, Segment, SegmentContent};

pub struct MarkdownDecomposer;

impl Decomposer for MarkdownDecomposer {
    fn format_name(&self) -> &'static str {
        "markdown"
    }

    fn decompose(&self, content: &[u8], _source_path: &str) -> anyhow::Result<Vec<Segment>> {
        let text = std::str::from_utf8(content)?;
        Ok(split_on_headings(text))
    }
}

fn heading_level(line: &str) -> Option<(usize, &str)> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }
    let hashes = trimmed.bytes().take_while(|&b| b == b'#').count();
    if hashes > 6 {
        return None;
    }
    let rest = &trimmed[hashes..];
    if !rest.starts_with(' ') && !rest.is_empty() {
        return None;
    }
    Some((hashes, rest.trim()))
}

fn split_on_headings(text: &str) -> Vec<Segment> {
    let mut segments: Vec<Segment> = Vec::new();
    let mut current_label: Option<String> = None;
    let mut current_body = String::new();

    for line in text.lines() {
        if let Some((_level, title)) = heading_level(line) {
            if !current_body.trim().is_empty() || current_label.is_some() {
                let label = current_label
                    .take()
                    .unwrap_or_else(|| "preamble".to_string());
                let body = current_body.trim().to_string();
                if !body.is_empty() {
                    segments.push(Segment {
                        index: segments.len(),
                        label,
                        content: SegmentContent::Text(body),
                    });
                }
                current_body.clear();
            }
            current_label = Some(title.to_string());
        } else {
            current_body.push_str(line);
            current_body.push('\n');
        }
    }

    let final_body = current_body.trim().to_string();
    if !final_body.is_empty() {
        let label = current_label
            .take()
            .unwrap_or_else(|| "preamble".to_string());
        segments.push(Segment {
            index: segments.len(),
            label,
            content: SegmentContent::Text(final_body),
        });
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_headings() {
        let md = "# Introduction\nHello world.\n\n# Methods\nWe did things.\n";
        let segs = MarkdownDecomposer
            .decompose(md.as_bytes(), "doc.md")
            .unwrap();

        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].label, "Introduction");
        assert!(matches!(&segs[0].content, SegmentContent::Text(t) if t.contains("Hello world")));
        assert_eq!(segs[1].label, "Methods");
        assert!(matches!(&segs[1].content, SegmentContent::Text(t) if t.contains("We did things")));
    }

    #[test]
    fn preamble_before_first_heading() {
        let md = "Some text before headings.\n\n# First\nContent here.\n";
        let segs = MarkdownDecomposer
            .decompose(md.as_bytes(), "doc.md")
            .unwrap();

        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].label, "preamble");
        assert_eq!(segs[1].label, "First");
    }

    #[test]
    fn no_headings_returns_single_segment() {
        let md = "Just a plain document.\nWith multiple lines.\n";
        let segs = MarkdownDecomposer
            .decompose(md.as_bytes(), "doc.md")
            .unwrap();

        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].label, "preamble");
    }

    #[test]
    fn nested_heading_levels() {
        let md = "# Top\nA\n## Sub\nB\n### Deep\nC\n";
        let segs = MarkdownDecomposer
            .decompose(md.as_bytes(), "doc.md")
            .unwrap();

        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0].label, "Top");
        assert_eq!(segs[1].label, "Sub");
        assert_eq!(segs[2].label, "Deep");
    }

    #[test]
    fn ignores_hash_without_space() {
        let md = "#notaheading\n# Real Heading\nContent.\n";
        let segs = MarkdownDecomposer
            .decompose(md.as_bytes(), "doc.md")
            .unwrap();

        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].label, "preamble");
        assert_eq!(segs[1].label, "Real Heading");
    }

    #[test]
    fn empty_section_skipped() {
        let md = "# First\n\n# Second\nContent.\n";
        let segs = MarkdownDecomposer
            .decompose(md.as_bytes(), "doc.md")
            .unwrap();

        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].label, "Second");
    }

    #[test]
    fn empty_input() {
        let segs = MarkdownDecomposer.decompose(b"", "doc.md").unwrap();
        assert!(segs.is_empty());
    }
}

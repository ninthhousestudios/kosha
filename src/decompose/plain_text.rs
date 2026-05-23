use super::{Decomposer, Segment, SegmentContent};

const SEGMENT_TARGET_CHARS: usize = 1500;
const STICKY_THRESHOLD: usize = 80;

pub struct PlainTextDecomposer;

impl Decomposer for PlainTextDecomposer {
    fn format_name(&self) -> &'static str {
        "plain_text"
    }

    fn decompose(&self, content: &[u8], _source_path: &str) -> anyhow::Result<Vec<Segment>> {
        let text = std::str::from_utf8(content)?;
        let paragraphs: Vec<&str> = text
            .split("\n\n")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if paragraphs.is_empty() {
            return Ok(vec![]);
        }

        let mut segments: Vec<Segment> = Vec::new();
        let mut current_parts: Vec<&str> = Vec::new();
        let mut current_len: usize = 0;

        for para in &paragraphs {
            let para_len = para.len();

            if current_parts.is_empty() {
                current_parts.push(para);
                current_len = para_len;
                continue;
            }

            let prev_is_sticky = current_parts.last().map_or(false, |p| p.len() <= STICKY_THRESHOLD);
            let over_budget = current_len + para_len > SEGMENT_TARGET_CHARS;
            let para_is_sticky = para_len <= STICKY_THRESHOLD;

            if !over_budget || (prev_is_sticky && para_is_sticky) {
                current_parts.push(para);
                current_len += para_len;
            } else {
                // Before emitting, peel off any trailing sticky parts so they
                // start the next segment instead of dangling at the end of this one.
                let mut carry: Vec<&str> = Vec::new();
                while current_parts.len() > 1
                    && current_parts.last().map_or(false, |p| p.len() <= STICKY_THRESHOLD)
                {
                    carry.push(current_parts.pop().unwrap());
                }
                carry.reverse();

                let body = current_parts.join("\n\n");
                segments.push(Segment {
                    index: segments.len(),
                    label: make_label(&current_parts),
                    content: SegmentContent::Text(body),
                });

                current_parts = carry;
                current_len = current_parts.iter().map(|p| p.len()).sum::<usize>();
                current_parts.push(para);
                current_len += para_len;
            }
        }

        if !current_parts.is_empty() {
            let body = current_parts.join("\n\n");
            segments.push(Segment {
                index: segments.len(),
                label: make_label(&current_parts),
                content: SegmentContent::Text(body),
            });
        }

        Ok(segments)
    }
}

fn make_label(parts: &[&str]) -> String {
    let first = parts[0];
    let trimmed = first.lines().next().unwrap_or(first);
    let trimmed = trimmed.trim_start_matches(|c: char| c == '-' || c == ' ' || c == '=' || c == '#');
    let trimmed = trimmed.trim();
    if trimmed.is_empty() || trimmed.len() > 80 {
        return "section".to_string();
    }
    trimmed.to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decompose(text: &str) -> Vec<Segment> {
        PlainTextDecomposer
            .decompose(text.as_bytes(), "test.txt")
            .unwrap()
    }

    fn seg_text(seg: &Segment) -> &str {
        match &seg.content {
            SegmentContent::Text(t) => t,
            _ => panic!("expected text segment"),
        }
    }

    #[test]
    fn empty_input() {
        assert!(decompose("").is_empty());
        assert!(decompose("   \n\n   ").is_empty());
    }

    #[test]
    fn single_paragraph() {
        let segs = decompose("Hello world.");
        assert_eq!(segs.len(), 1);
        assert_eq!(seg_text(&segs[0]), "Hello world.");
    }

    #[test]
    fn sticky_short_merges_with_next() {
        let text = "--- airavata | naga\n\nThe Burning Desire for Liberation\n\nBody text here.";
        let segs = decompose(text);
        assert_eq!(segs.len(), 1);
        assert!(seg_text(&segs[0]).contains("airavata"));
        assert!(seg_text(&segs[0]).contains("Burning Desire"));
        assert!(seg_text(&segs[0]).contains("Body text"));
    }

    #[test]
    fn budget_splits_large_content() {
        let big_para = "word ".repeat(400); // 2000 chars
        let text = format!("{}\n\n{}", big_para.trim(), big_para.trim());
        let segs = decompose(&text);
        assert!(segs.len() >= 2, "should split when over budget");
    }

    #[test]
    fn label_from_first_part() {
        let text = "--- airavata | naga\n\nSome body text that is long enough.";
        let segs = decompose(text);
        assert_eq!(segs[0].label, "airavata | naga");
    }

    #[test]
    fn label_from_heading() {
        let text = "=== Reflections\n\nSome reflective content here.";
        let segs = decompose(text);
        assert_eq!(segs[0].label, "reflections");
    }

    #[test]
    fn no_content_lost() {
        let text = "First.\n\nSecond.\n\n--- heading\n\nThird paragraph here.\n\nFourth paragraph.";
        let segs = decompose(text);
        let merged: String = segs.iter().map(|s| seg_text(s)).collect::<Vec<_>>().join("\n\n");
        let original_nonws: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        let merged_nonws: String = merged.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(original_nonws, merged_nonws);
    }

    #[test]
    fn multiple_sections_with_separators() {
        let section = |name: &str| {
            format!(
                "--- {name}\n\nTitle for {name}\n\n{}",
                "content ".repeat(200).trim()
            )
        };
        let text = format!("{}\n\n{}\n\n{}", section("alpha"), section("beta"), section("gamma"));
        let segs = decompose(&text);
        assert!(segs.len() >= 2, "separate sections should form distinct segments");
    }
}

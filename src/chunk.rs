const CHARS_PER_TOKEN: usize = 4;

const SCORE_HEADING_1: u32 = 100;
const SCORE_HEADING_2: u32 = 95;
const SCORE_HEADING_3: u32 = 90;
const SCORE_HEADING_OTHER: u32 = 85;
const SCORE_CODE_FENCE: u32 = 80;
const SCORE_BLANK_LINE: u32 = 20;
const SCORE_SENTENCE_END: u32 = 10;

const BREAK_THRESHOLD: f64 = 10.0;

#[derive(Debug, Clone)]
pub struct ChunkConfig {
    pub target_tokens: usize,
    pub tolerance_tokens: usize,
    pub overlap_tokens: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            target_tokens: 512,
            tolerance_tokens: 256,
            overlap_tokens: 0,
        }
    }
}

impl ChunkConfig {
    fn target_chars(&self) -> usize {
        self.target_tokens * CHARS_PER_TOKEN
    }
    fn min_chars(&self) -> usize {
        self.target_tokens.saturating_sub(self.tolerance_tokens) * CHARS_PER_TOKEN
    }
    fn ceiling_chars(&self) -> usize {
        (self.target_tokens + self.tolerance_tokens) * CHARS_PER_TOKEN
    }
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub index: usize,
    pub label: String,
    pub content: String,
}

// ── Boundary classification ──

#[derive(Debug, Clone, Copy, PartialEq)]
enum BoundaryKind {
    None,
    Heading(u32),
    CodeFenceOpen,
    CodeFenceClose,
    BlankLine,
    SentenceEnd,
}

impl BoundaryKind {
    fn score(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Heading(s) => s,
            Self::CodeFenceOpen | Self::CodeFenceClose => SCORE_CODE_FENCE,
            Self::BlankLine => SCORE_BLANK_LINE,
            Self::SentenceEnd => SCORE_SENTENCE_END,
        }
    }

    fn is_pre_break(self) -> bool {
        matches!(self, Self::Heading(_) | Self::CodeFenceOpen)
    }
}

fn classify_line(line: &str) -> BoundaryKind {
    let trimmed = line.trim();

    if trimmed.starts_with('#') {
        let level = trimmed.bytes().take_while(|&b| b == b'#').count();
        if level <= 6 && (trimmed.len() == level || trimmed.as_bytes()[level] == b' ') {
            return BoundaryKind::Heading(match level {
                1 => SCORE_HEADING_1,
                2 => SCORE_HEADING_2,
                3 => SCORE_HEADING_3,
                _ => SCORE_HEADING_OTHER,
            });
        }
    }

    if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
        return BoundaryKind::CodeFenceOpen;
    }

    if trimmed.is_empty() {
        return BoundaryKind::BlankLine;
    }

    if trimmed.ends_with('.') || trimmed.ends_with('?') || trimmed.ends_with('!') {
        return BoundaryKind::SentenceEnd;
    }

    BoundaryKind::None
}

fn resolve_fences(kinds: &mut [BoundaryKind]) {
    let mut in_fence = false;
    for kind in kinds.iter_mut() {
        match *kind {
            BoundaryKind::CodeFenceOpen if in_fence => {
                *kind = BoundaryKind::CodeFenceClose;
                in_fence = false;
            }
            BoundaryKind::CodeFenceOpen => {
                in_fence = true;
            }
            _ if in_fence => {
                *kind = BoundaryKind::None;
            }
            _ => {}
        }
    }
}

fn distance_decay(size: usize, target: usize) -> f64 {
    if target == 0 {
        return 1.0;
    }
    let dist = size.abs_diff(target);
    (1.0 - dist as f64 / target as f64).max(0.0)
}

// ── Chunker ──

pub fn chunk_segment(text: &str, segment_label: &str, cfg: &ChunkConfig) -> Vec<Chunk> {
    let target = cfg.target_chars();
    let min_size = cfg.min_chars();
    let ceiling = cfg.ceiling_chars();

    if text.chars().count() <= ceiling {
        return vec![Chunk {
            index: 0,
            label: segment_label.to_string(),
            content: text.to_string(),
        }];
    }

    let lines: Vec<&str> = text.split('\n').collect();
    let mut kinds: Vec<BoundaryKind> = lines.iter().map(|l| classify_line(l)).collect();
    resolve_fences(&mut kinds);

    let mut chunks: Vec<Chunk> = Vec::new();
    let mut start = 0;
    let mut size: usize = 0;

    for i in 0..lines.len() {
        let line_chars = lines[i].chars().count() + 1;
        size += line_chars;

        if size < min_size {
            if size >= ceiling && i > start {
                emit(&mut chunks, &lines[start..=i]);
                start = i + 1;
                size = 0;
            }
            continue;
        }

        let kind = kinds[i];

        if kind.is_pre_break() && i > start {
            let size_before = size - line_chars;
            if size_before >= min_size {
                let decay = distance_decay(size_before, target);
                let effective = kind.score() as f64 * decay;
                if effective >= BREAK_THRESHOLD || size_before >= target {
                    emit(&mut chunks, &lines[start..i]);
                    start = i;
                    size = line_chars;
                    if size < min_size {
                        continue;
                    }
                }
            }
        }

        if kind.score() > 0 && !kind.is_pre_break() {
            let decay = distance_decay(size, target);
            let effective = kind.score() as f64 * decay;
            if effective >= BREAK_THRESHOLD || size >= target {
                emit(&mut chunks, &lines[start..=i]);
                start = i + 1;
                size = 0;
                continue;
            }
        }

        if size >= ceiling {
            emit(&mut chunks, &lines[start..=i]);
            start = i + 1;
            size = 0;
        }
    }

    if start < lines.len() {
        emit(&mut chunks, &lines[start..]);
    }

    let total = chunks.len();
    for ch in &mut chunks {
        if total > 1 {
            ch.label = format!("{segment_label} [{}/{}]", ch.index + 1, total);
        } else {
            ch.label = segment_label.to_string();
        }
    }

    chunks
}

fn emit(chunks: &mut Vec<Chunk>, lines: &[&str]) {
    let text = lines.join("\n");
    let trimmed = text.trim_end();
    if !trimmed.is_empty() {
        chunks.push(Chunk {
            index: chunks.len(),
            label: String::new(),
            content: trimmed.to_string(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(target: usize, tolerance: usize) -> ChunkConfig {
        ChunkConfig {
            target_tokens: target,
            tolerance_tokens: tolerance,
            overlap_tokens: 0,
        }
    }

    #[test]
    fn small_text_single_chunk() {
        let chunks = chunk_segment("Hello world.", "seg", &cfg(20, 10));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "Hello world.");
        assert_eq!(chunks[0].label, "seg");
    }

    #[test]
    fn splits_at_blank_line() {
        // min=20, target=40, ceiling=60  (in chars)
        let text = format!(
            "{}\n\n{}",
            "word ".repeat(9).trim(),  // 44 chars
            "more ".repeat(9).trim()   // 44 chars
        );
        let chunks = chunk_segment(&text, "seg", &cfg(10, 5));
        assert!(
            chunks.len() >= 2,
            "expected split at blank line, got {} chunk(s): {:?}",
            chunks.len(),
            chunks.iter().map(|c| c.content.len()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn heading_starts_new_chunk() {
        // min=20, target=40, ceiling=60
        let text = format!(
            "{}\n# New Section\nMore content here and here.",
            "word ".repeat(9).trim() // 44 chars — above min, near target
        );
        let chunks = chunk_segment(&text, "seg", &cfg(10, 5));
        assert!(chunks.len() >= 2, "expected heading to trigger split");
        let heading_chunk = chunks.iter().find(|c| c.content.contains("# New Section"));
        assert!(heading_chunk.is_some(), "heading should appear in some chunk");
        assert!(
            heading_chunk.unwrap().content.starts_with("# New Section"),
            "heading should start its chunk, got: {:?}",
            heading_chunk.unwrap().content
        );
    }

    #[test]
    fn code_fence_stays_together() {
        // min=40, target=80, ceiling=120
        let code_block = "```\nfn main() {}\nlet x = 1;\nlet y = 2;\n```";
        let text = format!(
            "{}\n{}\n{}",
            "intro ".repeat(10).trim(), // 59 chars — above min
            code_block,
            "outro ".repeat(10).trim()
        );
        let chunks = chunk_segment(&text, "seg", &cfg(20, 10));
        // The opening and closing ``` should be in the same chunk
        for ch in &chunks {
            let fence_count = ch.content.matches("```").count();
            assert!(
                fence_count == 0 || fence_count == 2,
                "code fence split: chunk has {} fences: {:?}",
                fence_count,
                ch.content
            );
        }
    }

    #[test]
    fn hard_ceiling_forces_break() {
        // ceiling=60 chars. Lines with no structural boundaries.
        let lines: Vec<&str> = vec!["abcdefghij"; 20];
        let text = lines.join("\n"); // 20 lines, no headings/fences/blanks/sentence-ends
        let chunks = chunk_segment(&text, "seg", &cfg(10, 5));
        assert!(chunks.len() > 1, "hard ceiling should force splits");
    }

    #[test]
    fn decay_prevents_tiny_chunk_before_heading() {
        // min=40, target=80, ceiling=120
        // Only 5 chars before heading — way under min_size.
        let text = format!(
            "Hey.\n# Heading\n{}",
            "content ".repeat(25).trim() // 199 chars
        );
        let chunks = chunk_segment(&text, "seg", &cfg(20, 10));
        // "Hey." is under min_size, so heading should NOT create a tiny pre-chunk
        assert!(
            chunks[0].content.contains("Hey."),
            "tiny text should stay with heading"
        );
        assert!(
            chunks[0].content.contains("# Heading"),
            "heading should be in first chunk when preceded by tiny text"
        );
    }

    #[test]
    fn sentence_end_is_break_candidate() {
        // min=20, target=40, ceiling=60
        let text = format!(
            "{}.\n{}.",
            "word ".repeat(8).trim(), // 39 chars + '.'
            "more ".repeat(8).trim()
        );
        let chunks = chunk_segment(&text, "seg", &cfg(10, 5));
        assert!(
            chunks.len() >= 2,
            "expected split at sentence end, got {} chunk(s)",
            chunks.len()
        );
    }

    #[test]
    fn labels_with_total_count() {
        // ceiling=28 chars — forces many small chunks
        let text = format!("{}\n\n{}\n\n{}", "aaa ".repeat(8), "bbb ".repeat(8), "ccc ".repeat(8));
        let chunks = chunk_segment(&text, "page 1", &cfg(5, 2));
        if chunks.len() > 1 {
            for (i, ch) in chunks.iter().enumerate() {
                assert_eq!(ch.index, i);
                let expected = format!("page 1 [{}/{}]", i + 1, chunks.len());
                assert_eq!(ch.label, expected, "label mismatch at chunk {i}");
            }
        }
    }

    #[test]
    fn no_content_lost() {
        let text = format!(
            "# Title\n\n{}\n\n## Section\n\n{}\n\nEnd.",
            "paragraph one ".repeat(10).trim(),
            "paragraph two ".repeat(10).trim()
        );
        let chunks = chunk_segment(&text, "s", &cfg(15, 7));
        let original_non_ws: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        let chunked_non_ws: String = chunks
            .iter()
            .flat_map(|c| c.content.chars())
            .filter(|c| !c.is_whitespace())
            .collect();
        assert_eq!(
            original_non_ws, chunked_non_ws,
            "chunking lost or duplicated content"
        );
    }

    #[test]
    fn heading_hierarchy_scored() {
        assert!(
            classify_line("# H1").score() > classify_line("## H2").score(),
            "H1 should score higher than H2"
        );
        assert!(
            classify_line("## H2").score() > classify_line("### H3").score(),
            "H2 should score higher than H3"
        );
    }

    #[test]
    fn classify_various_lines() {
        assert_eq!(classify_line("# Heading").score(), SCORE_HEADING_1);
        assert_eq!(classify_line("## Sub").score(), SCORE_HEADING_2);
        assert_eq!(classify_line("```rust").score(), SCORE_CODE_FENCE);
        assert_eq!(classify_line("").score(), SCORE_BLANK_LINE);
        assert_eq!(classify_line("   ").score(), SCORE_BLANK_LINE);
        assert_eq!(classify_line("Hello world.").score(), SCORE_SENTENCE_END);
        assert_eq!(classify_line("What?").score(), SCORE_SENTENCE_END);
        assert_eq!(classify_line("Wow!").score(), SCORE_SENTENCE_END);
        assert_eq!(classify_line("no boundary here").score(), 0);
    }

    #[test]
    fn distance_decay_at_target() {
        assert!((distance_decay(100, 100) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn distance_decay_at_zero() {
        assert!((distance_decay(0, 100) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn distance_decay_symmetric() {
        let above = distance_decay(150, 100);
        let below = distance_decay(50, 100);
        assert!(
            (above - below).abs() < f64::EPSILON,
            "decay should be symmetric: above={above}, below={below}"
        );
    }
}

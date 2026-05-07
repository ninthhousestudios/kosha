const CHARS_PER_TOKEN: usize = 4;

#[derive(Debug, Clone)]
pub struct ChunkConfig {
    pub max_tokens: usize,
    pub overlap_tokens: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            overlap_tokens: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub index: usize,
    pub label: String,
    pub content: String,
}

impl ChunkConfig {
    fn max_chars(&self) -> usize {
        self.max_tokens * CHARS_PER_TOKEN
    }

    fn overlap_chars(&self) -> usize {
        self.overlap_tokens * CHARS_PER_TOKEN
    }
}

pub fn chunk_segment(text: &str, segment_label: &str, cfg: &ChunkConfig) -> Vec<Chunk> {
    let max_chars = cfg.max_chars();

    if text.len() <= max_chars {
        return vec![Chunk {
            index: 0,
            label: segment_label.to_string(),
            content: text.to_string(),
        }];
    }

    let overlap_chars = cfg.overlap_chars();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let mut end = (start + max_chars).min(text.len());

        while end < text.len() && !text.is_char_boundary(end) {
            end += 1;
        }

        if end < text.len()
            && let Some(ws_pos) = text[start..end].rfind(char::is_whitespace)
        {
            end = start + ws_pos + 1;
        }

        let chunk_text = text[start..end].trim();
        if !chunk_text.is_empty() {
            let idx = chunks.len();
            let label = format!("{segment_label} [{}/{}]", idx + 1, "?");
            chunks.push(Chunk {
                index: idx,
                label,
                content: chunk_text.to_string(),
            });
        }

        let window = end - start;
        let advance = if window > overlap_chars {
            window - overlap_chars
        } else {
            window
        };
        start += advance;
        while start < text.len() && !text.is_char_boundary(start) {
            start += 1;
        }
    }

    // Fix up labels now that we know total count.
    let total = chunks.len();
    if total > 1 {
        for ch in &mut chunks {
            ch.label = format!("{segment_label} [{}/{}]", ch.index + 1, total);
        }
    }

    chunks
}

use super::{Decomposer, Segment, SegmentContent};

pub struct HtmlDecomposer;

impl Decomposer for HtmlDecomposer {
    fn format_name(&self) -> &'static str {
        "html"
    }

    fn decompose(&self, content: &[u8], _source_path: &str) -> anyhow::Result<Vec<Segment>> {
        let text = std::str::from_utf8(content)?;
        Ok(split_html_on_headings(text))
    }
}

fn split_html_on_headings(html: &str) -> Vec<Segment> {
    let mut segments: Vec<Segment> = Vec::new();
    let mut current_label: Option<String> = None;
    let mut current_body = String::new();
    let mut pos = 0;
    let bytes = html.as_bytes();

    while pos < bytes.len() {
        if let Some(heading) = try_parse_heading(html, pos) {
            let body = strip_tags(&current_body);
            let body = body.trim().to_string();
            if !body.is_empty() {
                let label = current_label
                    .take()
                    .unwrap_or_else(|| "preamble".to_string());
                segments.push(Segment {
                    index: segments.len(),
                    label,
                    content: SegmentContent::Text(body),
                });
            } else {
                current_label.take();
            }
            current_body.clear();
            current_label = Some(heading.title.to_string());
            pos = heading.end;
        } else {
            current_body.push(bytes[pos] as char);
            pos += 1;
        }
    }

    let body = strip_tags(&current_body);
    let body = body.trim().to_string();
    if !body.is_empty() {
        let label = current_label
            .take()
            .unwrap_or_else(|| "preamble".to_string());
        segments.push(Segment {
            index: segments.len(),
            label,
            content: SegmentContent::Text(body),
        });
    }

    segments
}

struct HeadingMatch {
    title: String,
    end: usize,
}

fn try_parse_heading(html: &str, pos: usize) -> Option<HeadingMatch> {
    let rest = &html[pos..];
    if !rest.starts_with('<') {
        return None;
    }
    for level in 1..=6u8 {
        let open = format!("<h{}", level);
        let close = format!("</h{}>", level);
        let lower = rest.to_ascii_lowercase();
        if !lower.starts_with(&open) {
            continue;
        }
        let after_tag = &lower[open.len()..];
        if !after_tag.starts_with('>') && !after_tag.starts_with(' ') {
            continue;
        }
        let gt = lower[open.len()..].find('>')?;
        let content_start = open.len() + gt + 1;
        let close_pos = lower[content_start..].find(&close)?;
        let title_html = &rest[content_start..content_start + close_pos];
        let title = strip_tags(title_html).trim().to_string();
        let end = pos + content_start + close_pos + close.len();
        return Some(HeadingMatch { title, end });
    }
    None
}

pub(crate) fn strip_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(ch),
            _ => {}
        }
    }
    decode_entities(&result)
}

fn decode_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_html_headings() {
        let html = "<h1>Introduction</h1><p>Hello world.</p><h2>Methods</h2><p>We did things.</p>";
        let segs = HtmlDecomposer
            .decompose(html.as_bytes(), "doc.html")
            .unwrap();

        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].label, "Introduction");
        assert_eq!(segs[0].index, 0);
        assert!(matches!(&segs[0].content, SegmentContent::Text(t) if t.contains("Hello world")));
        assert_eq!(segs[1].label, "Methods");
    }

    #[test]
    fn strips_tags_from_content() {
        let html = "<h1>Title</h1><p>Some <strong>bold</strong> text.</p>";
        let segs = HtmlDecomposer
            .decompose(html.as_bytes(), "doc.html")
            .unwrap();

        assert_eq!(segs.len(), 1);
        assert!(matches!(&segs[0].content, SegmentContent::Text(t) if t == "Some bold text."));
    }

    #[test]
    fn decodes_html_entities() {
        let html = "<h1>Title</h1><p>A &amp; B &lt; C</p>";
        let segs = HtmlDecomposer
            .decompose(html.as_bytes(), "doc.html")
            .unwrap();

        assert!(matches!(&segs[0].content, SegmentContent::Text(t) if t == "A & B < C"));
    }

    #[test]
    fn preamble_before_first_heading() {
        let html = "<p>Preamble text.</p><h1>First</h1><p>Content.</p>";
        let segs = HtmlDecomposer
            .decompose(html.as_bytes(), "doc.html")
            .unwrap();

        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].label, "preamble");
        assert_eq!(segs[1].label, "First");
    }

    #[test]
    fn no_headings_returns_single_segment() {
        let html = "<p>Just some text.</p><p>More text.</p>";
        let segs = HtmlDecomposer
            .decompose(html.as_bytes(), "doc.html")
            .unwrap();

        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].label, "preamble");
    }

    #[test]
    fn case_insensitive_headings() {
        let html = "<H1>Title</H1><p>Content.</p>";
        let segs = HtmlDecomposer
            .decompose(html.as_bytes(), "doc.html")
            .unwrap();

        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].label, "Title");
    }

    #[test]
    fn heading_with_attributes() {
        let html = "<h1 class=\"title\">Title</h1><p>Content.</p>";
        let segs = HtmlDecomposer
            .decompose(html.as_bytes(), "doc.html")
            .unwrap();

        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].label, "Title");
    }

    #[test]
    fn empty_input() {
        let segs = HtmlDecomposer.decompose(b"", "doc.html").unwrap();
        assert!(segs.is_empty());
    }
}

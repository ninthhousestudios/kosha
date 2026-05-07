use std::collections::HashMap;
use std::io::{Cursor, Read};

use anyhow::Context;
use zip::ZipArchive;

use super::html::strip_tags;
use super::{Decomposer, Segment, SegmentContent};

pub struct EpubDecomposer;

impl Decomposer for EpubDecomposer {
    fn format_name(&self) -> &'static str {
        "epub"
    }

    fn decompose(&self, content: &[u8], _source_path: &str) -> anyhow::Result<Vec<Segment>> {
        let cursor = Cursor::new(content);
        let mut archive = ZipArchive::new(cursor).context("not a valid epub/zip")?;

        let opf_path = find_opf_path(&mut archive)?;
        let opf_dir = opf_path
            .rfind('/')
            .map(|i| &opf_path[..i + 1])
            .unwrap_or("")
            .to_string();

        let opf_xml = read_zip_entry(&mut archive, &opf_path)?;
        let manifest = parse_manifest(&opf_xml);
        let spine = parse_spine(&opf_xml);
        let toc = build_toc(&mut archive, &opf_xml, &manifest, &opf_dir);

        let mut segments = Vec::new();
        for idref in &spine {
            let Some(href) = manifest.get(idref.as_str()) else {
                continue;
            };
            let full_path = resolve_href(&opf_dir, href);
            let Ok(xhtml) = read_zip_entry(&mut archive, &full_path) else {
                continue;
            };
            let text = strip_tags(&xhtml).trim().to_string();
            if text.is_empty() {
                continue;
            }
            let label = toc
                .get(href.as_str())
                .or_else(|| toc.get(full_path.as_str()))
                .cloned()
                .unwrap_or_else(|| format!("chapter {}", segments.len() + 1));

            segments.push(Segment {
                index: segments.len(),
                label,
                content: SegmentContent::Text(text),
            });
        }

        Ok(segments)
    }
}

fn read_zip_entry(archive: &mut ZipArchive<Cursor<&[u8]>>, name: &str) -> anyhow::Result<String> {
    let mut file = archive
        .by_name(name)
        .with_context(|| format!("missing entry: {name}"))?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;
    Ok(buf)
}

fn find_opf_path(archive: &mut ZipArchive<Cursor<&[u8]>>) -> anyhow::Result<String> {
    let container = read_zip_entry(archive, "META-INF/container.xml")?;
    extract_attr(&container, "rootfile", "full-path")
        .context("no rootfile full-path in container.xml")
}

fn parse_manifest(opf: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let lower = opf.to_ascii_lowercase();
    let mut search_from = 0;
    while let Some(pos) = lower[search_from..].find("<item ") {
        let abs = search_from + pos;
        let tag_end = match lower[abs..].find("/>").or_else(|| lower[abs..].find('>')) {
            Some(e) => abs + e,
            None => break,
        };
        let tag = &opf[abs..tag_end + 1];
        if let (Some(id), Some(href)) = (get_attr(tag, "id"), get_attr(tag, "href")) {
            map.insert(id, href);
        }
        search_from = tag_end + 1;
    }
    map
}

fn parse_spine(opf: &str) -> Vec<String> {
    let mut refs = Vec::new();
    let lower = opf.to_ascii_lowercase();
    let mut search_from = 0;
    while let Some(pos) = lower[search_from..].find("<itemref ") {
        let abs = search_from + pos;
        let tag_end = match lower[abs..].find("/>").or_else(|| lower[abs..].find('>')) {
            Some(e) => abs + e,
            None => break,
        };
        let tag = &opf[abs..tag_end + 1];
        if let Some(idref) = get_attr(tag, "idref") {
            refs.push(idref);
        }
        search_from = tag_end + 1;
    }
    refs
}

fn build_toc(
    archive: &mut ZipArchive<Cursor<&[u8]>>,
    opf: &str,
    manifest: &HashMap<String, String>,
    opf_dir: &str,
) -> HashMap<String, String> {
    if let Some(toc) = try_ncx_toc(archive, opf, manifest, opf_dir) {
        return toc;
    }
    if let Some(toc) = try_nav_toc(archive, opf, manifest, opf_dir) {
        return toc;
    }
    HashMap::new()
}

fn try_ncx_toc(
    archive: &mut ZipArchive<Cursor<&[u8]>>,
    _opf: &str,
    manifest: &HashMap<String, String>,
    opf_dir: &str,
) -> Option<HashMap<String, String>> {
    let ncx_href = manifest
        .iter()
        .find(|(_, v)| v.ends_with(".ncx"))
        .map(|(_, v)| v.clone())?;
    let ncx_path = resolve_href(opf_dir, &ncx_href);
    let ncx = read_zip_entry(archive, &ncx_path).ok()?;
    Some(parse_ncx_nav_points(&ncx))
}

fn try_nav_toc(
    archive: &mut ZipArchive<Cursor<&[u8]>>,
    opf: &str,
    manifest: &HashMap<String, String>,
    opf_dir: &str,
) -> Option<HashMap<String, String>> {
    let lower = opf.to_ascii_lowercase();
    let nav_id = manifest.iter().find(|(id, _)| {
        let pattern = format!("id=\"{}\"", id);
        if let Some(pos) = lower.find(&pattern) {
            lower[..pos + pattern.len() + 100.min(lower.len() - pos - pattern.len())]
                .contains("properties=\"nav\"")
                || lower[pos.saturating_sub(200)..pos + pattern.len()].contains("properties=\"nav\"")
        } else {
            false
        }
    });
    let (_, href) = nav_id?;
    let path = resolve_href(opf_dir, href);
    let nav_html = read_zip_entry(archive, &path).ok()?;
    Some(parse_nav_links(&nav_html))
}

fn parse_ncx_nav_points(ncx: &str) -> HashMap<String, String> {
    let mut toc = HashMap::new();
    let lower = ncx.to_ascii_lowercase();
    let mut search_from = 0;

    while let Some(pos) = lower[search_from..].find("<navpoint") {
        let abs = search_from + pos;
        let block_end = lower[abs..]
            .find("</navpoint")
            .map(|e| abs + e)
            .unwrap_or(lower.len());

        let block = &ncx[abs..block_end];
        let block_lower = &lower[abs..block_end];

        let label = block_lower
            .find("<text>")
            .and_then(|s| {
                let start = s + 6;
                block_lower[start..].find("</text>").map(|e| {
                    strip_tags(&block[start..start + e])
                        .trim()
                        .to_string()
                })
            });

        let src = block_lower
            .find("<content")
            .and_then(|s| get_attr(&block[s..], "src"))
            .map(|s| s.split('#').next().unwrap_or(&s).to_string());

        if let (Some(label), Some(src)) = (label, src) {
            if !label.is_empty() {
                toc.entry(src).or_insert(label);
            }
        }
        search_from = abs + 1;
    }
    toc
}

fn parse_nav_links(html: &str) -> HashMap<String, String> {
    let mut toc = HashMap::new();
    let lower = html.to_ascii_lowercase();
    let mut search_from = 0;

    while let Some(pos) = lower[search_from..].find("<a ") {
        let abs = search_from + pos;
        let tag_end = lower[abs..].find("</a>").unwrap_or(lower.len() - abs);
        let tag = &html[abs..abs + tag_end + 4.min(html.len() - abs)];

        if let Some(href) = get_attr(tag, "href") {
            let href_base = href.split('#').next().unwrap_or(&href).to_string();
            let a_content_start = html[abs..].find('>').map(|p| abs + p + 1);
            let a_content_end = abs + tag_end;
            if let Some(start) = a_content_start {
                let label = strip_tags(&html[start..a_content_end]).trim().to_string();
                if !label.is_empty() && !href_base.is_empty() {
                    toc.entry(href_base).or_insert(label);
                }
            }
        }
        search_from = abs + tag_end.max(1);
    }
    toc
}

fn get_attr(tag: &str, name: &str) -> Option<String> {
    let lower = tag.to_ascii_lowercase();
    let needle = format!("{}=\"", name);
    if let Some(pos) = lower.find(&needle) {
        let start = pos + needle.len();
        let rest = &tag[start..];
        let end = rest.find('"')?;
        return Some(rest[..end].to_string());
    }
    let needle = format!("{}='", name);
    if let Some(pos) = lower.find(&needle) {
        let start = pos + needle.len();
        let rest = &tag[start..];
        let end = rest.find('\'')?;
        return Some(rest[..end].to_string());
    }
    None
}

fn extract_attr(xml: &str, element: &str, attr: &str) -> Option<String> {
    let lower = xml.to_ascii_lowercase();
    let needle = format!("<{}", element);
    let mut search_from = 0;
    while let Some(pos) = lower[search_from..].find(&needle) {
        let abs = search_from + pos;
        let after = lower.as_bytes().get(abs + needle.len()).copied();
        if matches!(after, Some(b' ') | Some(b'>') | Some(b'/') | None) {
            let tag_end = lower[abs..].find('>')? + abs;
            let tag = &xml[abs..tag_end + 1];
            return get_attr(tag, attr);
        }
        search_from = abs + 1;
    }
    None
}

fn resolve_href(base_dir: &str, href: &str) -> String {
    if href.starts_with('/') {
        href[1..].to_string()
    } else if base_dir.is_empty() {
        href.to_string()
    } else {
        format!("{}{}", base_dir, href)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_epub(files: &[(&str, &str)]) -> Vec<u8> {
        let buf = Vec::new();
        let cursor = Cursor::new(buf);
        let mut writer = zip::ZipWriter::new(cursor);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);

        for (name, content) in files {
            writer.start_file(*name, options).unwrap();
            std::io::Write::write_all(&mut writer, content.as_bytes()).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    fn minimal_epub(chapters: &[(&str, &str, &str)]) -> Vec<u8> {
        let mut manifest_items = String::new();
        let mut spine_refs = String::new();
        let mut nav_points = String::new();
        let mut files: Vec<(&str, String)> = Vec::new();

        for (i, (id, label, body)) in chapters.iter().enumerate() {
            let href = format!("{}.xhtml", id);
            manifest_items.push_str(&format!(
                "<item id=\"{}\" href=\"{}\" media-type=\"application/xhtml+xml\"/>\n",
                id, href
            ));
            spine_refs.push_str(&format!("<itemref idref=\"{}\"/>\n", id));
            nav_points.push_str(&format!(
                "<navPoint id=\"np{}\"><navLabel><text>{}</text></navLabel><content src=\"{}\"/></navPoint>\n",
                i, label, href
            ));
            let xhtml = format!(
                "<html><body><p>{}</p></body></html>",
                body
            );
            files.push((Box::leak(href.into_boxed_str()), xhtml));
        }

        let container = r#"<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"#;

        let opf = format!(
            r#"<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <manifest>
    {}
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
  </manifest>
  <spine toc="ncx">
    {}
  </spine>
</package>"#,
            manifest_items, spine_refs
        );

        let ncx = format!(
            r#"<?xml version="1.0"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/">
  <navMap>
    {}
  </navMap>
</ncx>"#,
            nav_points
        );

        let mut all_files: Vec<(&str, &str)> = vec![
            ("META-INF/container.xml", container),
        ];

        let opf_leaked: &str = Box::leak(opf.into_boxed_str());
        let ncx_leaked: &str = Box::leak(ncx.into_boxed_str());
        all_files.push(("content.opf", opf_leaked));
        all_files.push(("toc.ncx", ncx_leaked));

        for (name, content) in &files {
            all_files.push((name, Box::leak(content.clone().into_boxed_str())));
        }

        make_epub(&all_files)
    }

    #[test]
    fn epub_end_to_end() {
        let epub = minimal_epub(&[
            ("ch1", "Introduction", "Welcome to the book."),
            ("ch2", "Chapter One", "The story begins here."),
        ]);

        let segs = EpubDecomposer
            .decompose(&epub, "book.epub")
            .unwrap();

        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].label, "Introduction");
        assert!(matches!(&segs[0].content, SegmentContent::Text(t) if t.contains("Welcome")));
        assert_eq!(segs[1].label, "Chapter One");
        assert!(matches!(&segs[1].content, SegmentContent::Text(t) if t.contains("story begins")));
    }

    #[test]
    fn chapters_get_toc_labels() {
        let epub = minimal_epub(&[
            ("vibhuti", "ch. 3 — Vibhuti Pada", "Powers and attainments."),
        ]);

        let segs = EpubDecomposer
            .decompose(&epub, "yoga-sutras.epub")
            .unwrap();

        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].label, "ch. 3 — Vibhuti Pada");
    }

    #[test]
    fn html_tags_stripped_from_content() {
        let epub = minimal_epub(&[
            ("ch1", "Title", "Some <em>emphasized</em> and <strong>bold</strong> text."),
        ]);

        let segs = EpubDecomposer
            .decompose(&epub, "book.epub")
            .unwrap();

        assert!(matches!(
            &segs[0].content,
            SegmentContent::Text(t) if t.contains("Some emphasized and bold text.")
        ));
    }

    #[test]
    fn empty_chapters_skipped() {
        let epub = minimal_epub(&[
            ("cover", "Cover", ""),
            ("ch1", "Real Content", "Actual text here."),
        ]);

        let segs = EpubDecomposer
            .decompose(&epub, "book.epub")
            .unwrap();

        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].label, "Real Content");
        assert_eq!(segs[0].index, 0);
    }

    #[test]
    fn fallback_label_when_no_toc() {
        let container = r#"<?xml version="1.0"?>
<container><rootfiles><rootfile full-path="content.opf"/></rootfiles></container>"#;
        let opf = r#"<?xml version="1.0"?>
<package><manifest>
  <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
</manifest><spine><itemref idref="ch1"/></spine></package>"#;
        let ch1 = "<html><body><p>No TOC available.</p></body></html>";

        let epub = make_epub(&[
            ("META-INF/container.xml", container),
            ("content.opf", opf),
            ("ch1.xhtml", ch1),
        ]);

        let segs = EpubDecomposer
            .decompose(&epub, "book.epub")
            .unwrap();

        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].label, "chapter 1");
    }

    #[test]
    fn invalid_zip_returns_error() {
        let result = EpubDecomposer.decompose(b"not a zip", "bad.epub");
        assert!(result.is_err());
    }
}

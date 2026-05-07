pub mod html;
pub mod image;
pub mod markdown;
pub mod pdf;
pub mod plain_text;

#[derive(Debug, Clone)]
pub enum SegmentContent {
    Text(String),
    Image(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub index: usize,
    pub label: String,
    pub content: SegmentContent,
}

pub trait Decomposer {
    fn decompose(&self, content: &[u8], source_path: &str) -> anyhow::Result<Vec<Segment>>;
    fn format_name(&self) -> &'static str;
}

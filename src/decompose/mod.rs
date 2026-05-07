pub mod plain_text;

#[derive(Debug, Clone)]
pub struct Segment {
    pub index: usize,
    pub label: String,
    pub content: String,
}

pub trait Decomposer {
    fn decompose(&self, content: &str, source_path: &str) -> Vec<Segment>;
    fn format_name(&self) -> &'static str;
}

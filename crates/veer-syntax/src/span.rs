/// A byte range into a source file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Span { start, end }
    }
    pub fn to(self, other: Span) -> Span {
        Span { start: self.start.min(other.start), end: self.end.max(other.end) }
    }
}

/// A named source file with line lookup.
#[derive(Clone, Debug)]
pub struct Source {
    pub name: String,
    pub text: String,
    line_starts: Vec<usize>,
}

impl Source {
    pub fn new(name: impl Into<String>, text: impl Into<String>) -> Self {
        let text = text.into();
        let mut line_starts = vec![0];
        for (i, b) in text.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i + 1);
            }
        }
        Source { name: name.into(), text, line_starts }
    }

    /// 0-based (line, column-in-chars) of a byte offset.
    pub fn line_col(&self, offset: usize) -> (usize, usize) {
        let offset = offset.min(self.text.len());
        let line = match self.line_starts.binary_search(&offset) {
            Ok(l) => l,
            Err(l) => l - 1,
        };
        let start = self.line_starts[line];
        let col = self.text[start..offset].chars().count();
        (line, col)
    }

    pub fn line_text(&self, line: usize) -> &str {
        let start = self.line_starts[line];
        let end = self.line_starts.get(line + 1).copied().unwrap_or(self.text.len());
        self.text[start..end].trim_end_matches(['\n', '\r'])
    }

    pub fn slice(&self, span: Span) -> &str {
        &self.text[span.start.min(self.text.len())..span.end.min(self.text.len())]
    }
}

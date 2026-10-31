use crate::span::{Source, Span};
use std::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

/// A compiler diagnostic. `code` refers to an entry in [`crate::codes::CODES`].
#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub code: &'static str,
    pub severity: Severity,
    pub message: String,
    pub span: Span,
    pub labels: Vec<(Span, String)>,
    pub notes: Vec<String>,
}

impl Diagnostic {
    pub fn error(code: &'static str, span: Span, message: impl Into<String>) -> Self {
        debug_assert!(crate::codes::lookup(code).is_some(), "unregistered code {code}");
        Diagnostic { code, severity: Severity::Error, message: message.into(), span, labels: vec![], notes: vec![] }
    }
    pub fn warning(code: &'static str, span: Span, message: impl Into<String>) -> Self {
        let mut d = Self::error(code, span, message);
        d.severity = Severity::Warning;
        d
    }
    pub fn label(mut self, span: Span, msg: impl Into<String>) -> Self {
        self.labels.push((span, msg.into()));
        self
    }
    pub fn note(mut self, msg: impl Into<String>) -> Self {
        self.notes.push(msg.into());
        self
    }
    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }

    /// Render as plain text, rustc-style.
    pub fn render(&self, src: &Source) -> String {
        let mut out = String::new();
        let sev = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };
        let _ = writeln!(out, "{sev}[{}]: {}", self.code, self.message);
        let (line, col) = src.line_col(self.span.start);
        let _ = writeln!(out, "  --> {}:{}:{}", src.name, line + 1, col + 1);
        render_snippet(&mut out, src, self.span, "");
        for (span, msg) in &self.labels {
            let (l, c) = src.line_col(span.start);
            let _ = writeln!(out, "  ::: {}:{}:{}", src.name, l + 1, c + 1);
            render_snippet(&mut out, src, *span, msg);
        }
        for n in &self.notes {
            let _ = writeln!(out, "   = note: {n}");
        }
        let _ = writeln!(out, "   = help: run `veer explain {}` for details", self.code);
        out
    }
}

fn render_snippet(out: &mut String, src: &Source, span: Span, msg: &str) {
    let (line, col) = src.line_col(span.start);
    let (end_line, end_col) = src.line_col(span.end);
    let text = src.line_text(line);
    let width = if end_line == line { (end_col.saturating_sub(col)).max(1) } else { (text.chars().count().saturating_sub(col)).max(1) };
    let num = format!("{}", line + 1);
    let pad = " ".repeat(num.len());
    let _ = writeln!(out, "{pad} |");
    let _ = writeln!(out, "{num} | {text}");
    let _ = writeln!(out, "{pad} | {}{} {msg}", " ".repeat(col), "^".repeat(width));
}

pub fn render_all(diags: &[Diagnostic], src: &Source) -> String {
    diags.iter().map(|d| d.render(src)).collect::<Vec<_>>().join("\n")
}

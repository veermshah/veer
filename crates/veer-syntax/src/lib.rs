//! Lexer, parser, AST, and diagnostics for the Veer language.

pub mod ast;
pub mod codes;
pub mod diag;
pub mod lexer;
pub mod parser;
pub mod span;

pub use diag::{Diagnostic, Severity};
pub use parser::parse;
pub use span::{Source, Span};

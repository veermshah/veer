use crate::diag::Diagnostic;
use crate::span::Span;

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Ident(String),
    /// Integer with optional unit suffix, e.g. `2GiB`.
    Int(u64, Option<String>),
    Float(f64),
    Str(String),
    Kw(Kw),
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Lt,
    Gt,
    Le,
    Ge,
    EqEq,
    Ne,
    Eq,
    Comma,
    Colon,
    Semi,
    Dot,
    DotDot,
    Plus,
    PlusPlus,
    Minus,
    Star,
    Slash,
    At,
    Bang,
    AndAnd,
    OrOr,
    Pipe,
    Arrow,
    /// `#[`
    HashBracket,
    Eof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kw {
    Model,
    Prefix,
    Tool,
    Schema,
    Const,
    Fn,
    Let,
    Return,
    Fork,
    Gen,
    Max,
    As,
    Json,
    Choice,
    Stop,
    Drop,
    Splice,
    At,
    With,
    Speculate,
    Commit,
    Rollback,
    If,
    Else,
    For,
    In,
    Open,
    True,
    False,
}

impl Kw {
    fn from_str(s: &str) -> Option<Kw> {
        use Kw::*;
        Some(match s {
            "model" => Model,
            "prefix" => Prefix,
            "tool" => Tool,
            "schema" => Schema,
            "const" => Const,
            "fn" => Fn,
            "let" => Let,
            "return" => Return,
            "fork" => Fork,
            "gen" => Gen,
            "max" => Max,
            "as" => As,
            "json" => Json,
            "choice" => Choice,
            "stop" => Stop,
            "drop" => Drop,
            "splice" => Splice,
            "at" => At,
            "with" => With,
            "speculate" => Speculate,
            "commit" => Commit,
            "rollback" => Rollback,
            "if" => If,
            "else" => Else,
            "for" => For,
            "in" => In,
            "open" => Open,
            "true" => True,
            "false" => False,
            _ => return None,
        })
    }
    pub fn as_str(self) -> &'static str {
        use Kw::*;
        match self {
            Model => "model",
            Prefix => "prefix",
            Tool => "tool",
            Schema => "schema",
            Const => "const",
            Fn => "fn",
            Let => "let",
            Return => "return",
            Fork => "fork",
            Gen => "gen",
            Max => "max",
            As => "as",
            Json => "json",
            Choice => "choice",
            Stop => "stop",
            Drop => "drop",
            Splice => "splice",
            At => "at",
            With => "with",
            Speculate => "speculate",
            Commit => "commit",
            Rollback => "rollback",
            If => "if",
            Else => "else",
            For => "for",
            In => "in",
            Open => "open",
            True => "true",
            False => "false",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
    /// True if a newline separates this token from the previous one.
    pub nl_before: bool,
}

pub fn lex(src: &str) -> Result<Vec<Token>, Diagnostic> {
    let bytes = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    let mut nl = false;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'\n' {
            nl = true;
            i += 1;
            continue;
        }
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c == b'/' && bytes.get(i + 1) == Some(&b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        let tok = if c.is_ascii_alphabetic() || c == b'_' {
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let s = &src[start..i];
            match Kw::from_str(s) {
                Some(k) => Tok::Kw(k),
                None => Tok::Ident(s.to_string()),
            }
        } else if c.is_ascii_digit() {
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'_') {
                i += 1;
            }
            let mut is_float = false;
            if i + 1 < bytes.len() && bytes[i] == b'.' && bytes[i + 1].is_ascii_digit() {
                is_float = true;
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                let save = i;
                let mut j = i + 1;
                if j < bytes.len() && (bytes[j] == b'-' || bytes[j] == b'+') {
                    j += 1;
                }
                if j < bytes.len() && bytes[j].is_ascii_digit() {
                    while j < bytes.len() && bytes[j].is_ascii_digit() {
                        j += 1;
                    }
                    i = j;
                    is_float = true;
                } else {
                    i = save;
                }
            }
            let num = src[start..i].replace('_', "");
            if is_float {
                Tok::Float(num.parse().map_err(|_| Diagnostic::error("V0001", Span::new(start, i), "invalid float literal"))?)
            } else {
                let v: u64 = num.parse().map_err(|_| Diagnostic::error("V0001", Span::new(start, i), "integer literal too large"))?;
                let suf_start = i;
                while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                    i += 1;
                }
                let suffix = if i > suf_start { Some(src[suf_start..i].to_string()) } else { None };
                Tok::Int(v, suffix)
            }
        } else if c == b'"' {
            i += 1;
            let mut s = String::new();
            loop {
                if i >= bytes.len() {
                    return Err(Diagnostic::error("V0001", Span::new(start, i), "unterminated string literal"));
                }
                let ch = src[i..].chars().next().unwrap();
                if ch == '"' {
                    i += 1;
                    break;
                }
                if ch == '\\' {
                    let esc = src[i + 1..].chars().next();
                    let rep = match esc {
                        Some('n') => '\n',
                        Some('t') => '\t',
                        Some('"') => '"',
                        Some('\\') => '\\',
                        Some('r') => '\r',
                        _ => return Err(Diagnostic::error("V0001", Span::new(i, i + 2), "unknown escape sequence")),
                    };
                    s.push(rep);
                    i += 2;
                    continue;
                }
                s.push(ch);
                i += ch.len_utf8();
            }
            Tok::Str(s)
        } else {
            let two = if i + 1 < bytes.len() { &src[i..i + 2] } else { "" };
            let t2 = match two {
                "<=" => Some(Tok::Le),
                ">=" => Some(Tok::Ge),
                "==" => Some(Tok::EqEq),
                "!=" => Some(Tok::Ne),
                ".." => Some(Tok::DotDot),
                "++" => Some(Tok::PlusPlus),
                "&&" => Some(Tok::AndAnd),
                "||" => Some(Tok::OrOr),
                "|>" => Some(Tok::Pipe),
                "->" => Some(Tok::Arrow),
                "#[" => Some(Tok::HashBracket),
                _ => None,
            };
            if let Some(t) = t2 {
                i += 2;
                t
            } else {
                i += 1;
                match c {
                    b'(' => Tok::LParen,
                    b')' => Tok::RParen,
                    b'{' => Tok::LBrace,
                    b'}' => Tok::RBrace,
                    b'[' => Tok::LBracket,
                    b']' => Tok::RBracket,
                    b'<' => Tok::Lt,
                    b'>' => Tok::Gt,
                    b'=' => Tok::Eq,
                    b',' => Tok::Comma,
                    b':' => Tok::Colon,
                    b';' => Tok::Semi,
                    b'.' => Tok::Dot,
                    b'+' => Tok::Plus,
                    b'-' => Tok::Minus,
                    b'*' => Tok::Star,
                    b'/' => Tok::Slash,
                    b'@' => Tok::At,
                    b'!' => Tok::Bang,
                    _ => {
                        let ch = src[start..].chars().next().unwrap();
                        return Err(Diagnostic::error("V0001", Span::new(start, start + ch.len_utf8()), format!("unexpected character `{ch}`")));
                    }
                }
            }
        };
        out.push(Token { tok, span: Span::new(start, i), nl_before: nl });
        nl = false;
    }
    out.push(Token { tok: Tok::Eof, span: Span::new(src.len(), src.len()), nl_before: true });
    Ok(out)
}

pub fn describe(t: &Tok) -> String {
    match t {
        Tok::Ident(s) => format!("identifier `{s}`"),
        Tok::Int(v, _) => format!("integer `{v}`"),
        Tok::Float(v) => format!("float `{v}`"),
        Tok::Str(_) => "string literal".into(),
        Tok::Kw(k) => format!("keyword `{}`", k.as_str()),
        Tok::Eof => "end of file".into(),
        other => format!("`{}`", punct(other)),
    }
}

fn punct(t: &Tok) -> &'static str {
    match t {
        Tok::LParen => "(",
        Tok::RParen => ")",
        Tok::LBrace => "{",
        Tok::RBrace => "}",
        Tok::LBracket => "[",
        Tok::RBracket => "]",
        Tok::Lt => "<",
        Tok::Gt => ">",
        Tok::Le => "<=",
        Tok::Ge => ">=",
        Tok::EqEq => "==",
        Tok::Ne => "!=",
        Tok::Eq => "=",
        Tok::Comma => ",",
        Tok::Colon => ":",
        Tok::Semi => ";",
        Tok::Dot => ".",
        Tok::DotDot => "..",
        Tok::Plus => "+",
        Tok::PlusPlus => "++",
        Tok::Minus => "-",
        Tok::Star => "*",
        Tok::Slash => "/",
        Tok::At => "@",
        Tok::Bang => "!",
        Tok::AndAnd => "&&",
        Tok::OrOr => "||",
        Tok::Pipe => "|>",
        Tok::Arrow => "->",
        Tok::HashBracket => "#[",
        _ => "?",
    }
}

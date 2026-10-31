use crate::ast::*;
use crate::diag::Diagnostic;
use crate::lexer::{describe, lex, Kw, Tok, Token};
use crate::span::Span;

pub fn parse(src: &str) -> Result<Program, Diagnostic> {
    let toks = lex(src)?;
    let mut p = Parser { toks, pos: 0, next_id: 0 };
    p.program()
}

// Binding powers (higher binds tighter).
const P_PIPE: u8 = 1;
const P_OR: u8 = 2;
const P_AND: u8 = 3;
const P_CMP: u8 = 4;
const P_APPEND: u8 = 5;
const P_ADD: u8 = 6;
const P_MUL: u8 = 7;

struct Parser {
    toks: Vec<Token>,
    pos: usize,
    next_id: ExprId,
}

type PResult<T> = Result<T, Diagnostic>;

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }
    fn peek_at(&self, n: usize) -> &Tok {
        &self.toks[(self.pos + n).min(self.toks.len() - 1)].tok
    }
    fn tok(&self) -> &Token {
        &self.toks[self.pos]
    }
    fn span(&self) -> Span {
        self.toks[self.pos].span
    }
    fn prev_span(&self) -> Span {
        self.toks[self.pos.saturating_sub(1)].span
    }
    fn bump(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }
    fn at(&self, t: &Tok) -> bool {
        self.peek() == t
    }
    fn at_kw(&self, k: Kw) -> bool {
        matches!(self.peek(), Tok::Kw(x) if *x == k)
    }
    fn eat(&mut self, t: &Tok) -> bool {
        if self.at(t) {
            self.bump();
            true
        } else {
            false
        }
    }
    fn eat_kw(&mut self, k: Kw) -> bool {
        if self.at_kw(k) {
            self.bump();
            true
        } else {
            false
        }
    }
    fn err<T>(&self, expected: &str) -> PResult<T> {
        Err(Diagnostic::error("V0002", self.span(), format!("expected {expected}, found {}", describe(self.peek()))))
    }
    fn expect(&mut self, t: &Tok, what: &str) -> PResult<Span> {
        if self.at(t) {
            Ok(self.bump().span)
        } else {
            self.err(what)
        }
    }
    fn expect_kw(&mut self, k: Kw) -> PResult<Span> {
        if self.at_kw(k) {
            Ok(self.bump().span)
        } else {
            self.err(&format!("`{}`", k.as_str()))
        }
    }
    fn ident(&mut self) -> PResult<Ident> {
        match self.peek().clone() {
            Tok::Ident(name) => {
                let span = self.bump().span;
                Ok(Ident { name, span })
            }
            _ => self.err("an identifier"),
        }
    }
    fn uint(&mut self) -> PResult<u64> {
        match self.peek().clone() {
            Tok::Int(v, None) => {
                self.bump();
                Ok(v)
            }
            _ => self.err("an integer"),
        }
    }
    fn number(&mut self) -> PResult<f64> {
        let neg = self.eat(&Tok::Minus);
        let v = match self.peek().clone() {
            Tok::Int(v, None) => v as f64,
            Tok::Float(f) => f,
            _ => return self.err("a number"),
        };
        self.bump();
        Ok(if neg { -v } else { v })
    }
    fn mk(&mut self, kind: ExprKind, span: Span) -> Expr {
        let id = self.next_id;
        self.next_id += 1;
        Expr { id, kind, span }
    }

    // ---------------------------------------------------------------- items

    fn program(&mut self) -> PResult<Program> {
        let mut items = Vec::new();
        while !self.at(&Tok::Eof) {
            items.push(self.item()?);
            self.eat(&Tok::Semi);
        }
        Ok(Program { items })
    }

    fn item(&mut self) -> PResult<Item> {
        let start = self.span();
        let mut attrs = Vec::new();
        while self.at(&Tok::At) {
            attrs.push(self.attr()?);
        }
        if !attrs.is_empty() && !self.at_kw(Kw::Fn) {
            return self.err("`fn` after attribute");
        }
        match self.peek() {
            Tok::Kw(Kw::Model) => {
                self.bump();
                let name = self.ident()?;
                self.expect(&Tok::Eq, "`=`")?;
                let hf_id = match self.peek().clone() {
                    Tok::Str(s) => {
                        self.bump();
                        s
                    }
                    _ => return self.err("a model identifier string"),
                };
                self.expect(&Tok::LBrace, "`{`")?;
                let mut fields = Vec::new();
                while !self.at(&Tok::RBrace) {
                    let k = self.ident()?;
                    self.expect(&Tok::Colon, "`:`")?;
                    let v = match self.peek().clone() {
                        Tok::Int(n, None) => ModelVal::Int(n, self.bump().span),
                        Tok::Ident(_) => ModelVal::Ident(self.ident()?),
                        _ => return self.err("an integer or format name"),
                    };
                    fields.push((k, v));
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::RBrace, "`}`")?;
                Ok(Item::Model(ModelDecl { name, hf_id, fields, span: start.to(self.prev_span()) }))
            }
            Tok::Kw(Kw::Prefix) => {
                self.bump();
                let name = self.ident()?;
                self.expect(&Tok::Colon, "`:`")?;
                let model = self.ident()?;
                self.expect(&Tok::Eq, "`=`")?;
                let text = match self.peek().clone() {
                    Tok::Str(s) => {
                        self.bump();
                        s
                    }
                    _ => return self.err("a string literal"),
                };
                Ok(Item::Prefix(PrefixDecl { name, model, text, span: start.to(self.prev_span()) }))
            }
            Tok::Kw(Kw::Tool) => {
                self.bump();
                let name = self.ident()?;
                let params = self.params()?;
                self.expect(&Tok::Arrow, "`->`")?;
                let ret = self.ty()?;
                Ok(Item::Tool(ToolDecl { name, params, ret, span: start.to(self.prev_span()) }))
            }
            Tok::Kw(Kw::Schema) => {
                self.bump();
                let name = self.ident()?;
                self.expect(&Tok::LBrace, "`{`")?;
                let mut fields = Vec::new();
                while !self.at(&Tok::RBrace) {
                    let k = self.ident()?;
                    self.expect(&Tok::Colon, "`:`")?;
                    let t = self.ty()?;
                    fields.push((k, t));
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::RBrace, "`}`")?;
                Ok(Item::Schema(SchemaDecl { name, fields, span: start.to(self.prev_span()) }))
            }
            Tok::Kw(Kw::Const) => {
                self.bump();
                let name = self.ident()?;
                self.expect(&Tok::Eq, "`=`")?;
                let value = self.expr()?;
                Ok(Item::Const(ConstDecl { name, value, span: start.to(self.prev_span()) }))
            }
            Tok::Kw(Kw::Fn) => {
                self.bump();
                let name = self.ident()?;
                let params = self.params()?;
                let ret = if self.eat(&Tok::Arrow) { Some(self.ty()?) } else { None };
                let body = self.block()?;
                Ok(Item::Fn(FnDecl { name, attrs, params, ret, body, span: start.to(self.prev_span()) }))
            }
            _ => self.err("an item (`model`, `prefix`, `tool`, `schema`, `const`, `fn`)"),
        }
    }

    fn attr(&mut self) -> PResult<Attr> {
        let start = self.expect(&Tok::At, "`@`")?;
        let name = self.ident()?;
        let mut args = Vec::new();
        if self.eat(&Tok::LParen) {
            while !self.at(&Tok::RParen) {
                let key = self.ident()?;
                if !self.eat(&Tok::Eq) && !self.eat(&Tok::Le) {
                    return self.err("`=` or `<=`");
                }
                let vs = self.span();
                let value = match self.peek().clone() {
                    Tok::Int(v, suf) => {
                        self.bump();
                        AttrVal::Int(v, suf)
                    }
                    Tok::Float(f) => {
                        self.bump();
                        AttrVal::Float(f)
                    }
                    _ => return self.err("a number"),
                };
                args.push(AttrArg { span: key.span.to(vs), key, value });
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
            self.expect(&Tok::RParen, "`)`")?;
        }
        Ok(Attr { name, args, span: start.to(self.prev_span()) })
    }

    fn params(&mut self) -> PResult<Vec<Param>> {
        self.expect(&Tok::LParen, "`(`")?;
        let mut ps = Vec::new();
        while !self.at(&Tok::RParen) {
            let name = self.ident()?;
            self.expect(&Tok::Colon, "`:`")?;
            let ty = self.ty()?;
            ps.push(Param { name, ty });
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::RParen, "`)`")?;
        Ok(ps)
    }

    fn ty(&mut self) -> PResult<TypeExpr> {
        let start = self.span();
        let kind = match self.peek().clone() {
            Tok::LParen => {
                self.bump();
                let mut ts = Vec::new();
                while !self.at(&Tok::RParen) {
                    ts.push(self.ty()?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::RParen, "`)`")?;
                TypeExprKind::Tuple(ts)
            }
            Tok::LBracket => {
                self.bump();
                let elem = self.ty()?;
                self.expect(&Tok::Semi, "`;`")?;
                self.eat(&Tok::Le);
                let n = self.uint()?;
                self.expect(&Tok::RBracket, "`]`")?;
                TypeExprKind::List(Box::new(elem), n)
            }
            Tok::Ident(name) => {
                self.bump();
                match name.as_str() {
                    "str" => {
                        if self.eat(&Tok::Le) {
                            TypeExprKind::Str(Some(self.uint()?))
                        } else {
                            TypeExprKind::Str(None)
                        }
                    }
                    "int" => TypeExprKind::Int,
                    "float" => TypeExprKind::Float,
                    "bool" => TypeExprKind::Bool,
                    "Ctx" => {
                        self.expect(&Tok::Lt, "`<`")?;
                        let m = self.ident()?;
                        self.expect(&Tok::Gt, "`>`")?;
                        TypeExprKind::Ctx(m)
                    }
                    "Tensor" => {
                        self.expect(&Tok::Lt, "`<`")?;
                        self.expect(&Tok::LBracket, "`[`")?;
                        let mut shape = Vec::new();
                        while !self.at(&Tok::RBracket) {
                            shape.push(self.uint()?);
                            if !self.eat(&Tok::Comma) {
                                break;
                            }
                        }
                        self.expect(&Tok::RBracket, "`]`")?;
                        self.expect(&Tok::Comma, "`,`")?;
                        let fmt = self.ident()?;
                        let (mut abs, mut err) = (None, None);
                        while self.eat(&Tok::Comma) {
                            let k = self.ident()?;
                            self.expect(&Tok::Le, "`<=`")?;
                            let v = self.number()?;
                            match k.name.as_str() {
                                "abs" => abs = Some(v),
                                "err" => err = Some(v),
                                _ => return Err(Diagnostic::error("V0002", k.span, "expected `abs` or `err`")),
                            }
                        }
                        self.expect(&Tok::Gt, "`>`")?;
                        TypeExprKind::Tensor { shape, fmt, abs, err }
                    }
                    _ => TypeExprKind::Named(Ident { name, span: start }),
                }
            }
            _ => return self.err("a type"),
        };
        Ok(TypeExpr { kind, span: start.to(self.prev_span()) })
    }

    // ----------------------------------------------------------- statements

    fn block(&mut self) -> PResult<Block> {
        let start = self.expect(&Tok::LBrace, "`{`")?;
        let mut stmts = Vec::new();
        while !self.at(&Tok::RBrace) {
            if self.at(&Tok::Eof) {
                return self.err("`}`");
            }
            if self.eat(&Tok::Semi) {
                continue;
            }
            stmts.push(self.stmt()?);
        }
        let end = self.expect(&Tok::RBrace, "`}`")?;
        Ok(Block { stmts, span: start.to(end) })
    }

    fn stmt(&mut self) -> PResult<Stmt> {
        let start = self.span();
        let kind = match self.peek() {
            Tok::Kw(Kw::Let) => {
                self.bump();
                let pat = self.pattern()?;
                let ty = if self.eat(&Tok::Colon) { Some(self.ty()?) } else { None };
                self.expect(&Tok::Eq, "`=`")?;
                let value = self.expr()?;
                StmtKind::Let { pat, ty, value }
            }
            Tok::Kw(Kw::Return) => {
                self.bump();
                if self.at(&Tok::RBrace) || self.at(&Tok::Semi) || self.tok().nl_before {
                    StmtKind::Return(None)
                } else {
                    StmtKind::Return(Some(self.expr()?))
                }
            }
            Tok::Kw(Kw::Drop) => {
                self.bump();
                let mut es = vec![self.expr()?];
                while self.eat(&Tok::Comma) {
                    es.push(self.expr()?);
                }
                StmtKind::Drop(es)
            }
            Tok::Kw(Kw::Commit) => {
                self.bump();
                StmtKind::Commit(self.expr()?)
            }
            Tok::Kw(Kw::Rollback) => {
                self.bump();
                StmtKind::Rollback(self.expr()?)
            }
            Tok::Kw(Kw::If) => return self.if_stmt(),
            Tok::Kw(Kw::For) => {
                self.bump();
                let var = self.ident()?;
                self.expect_kw(Kw::In)?;
                let a = self.expr()?;
                let iter = if self.eat(&Tok::DotDot) { ForIter::Range(a, self.expr()?) } else { ForIter::Each(a) };
                let body = self.block()?;
                StmtKind::For { var, iter, body }
            }
            _ => {
                let e = self.expr()?;
                if self.eat(&Tok::Eq) {
                    let pat = expr_to_pattern(&e)?;
                    let value = self.expr()?;
                    StmtKind::Assign { pat, value }
                } else {
                    StmtKind::Expr(e)
                }
            }
        };
        let s = Stmt { kind, span: start.to(self.prev_span()) };
        self.eat(&Tok::Semi);
        Ok(s)
    }

    fn if_stmt(&mut self) -> PResult<Stmt> {
        let start = self.expect_kw(Kw::If)?;
        let cond = self.expr()?;
        let then = self.block()?;
        let els = if self.eat_kw(Kw::Else) {
            if self.at_kw(Kw::If) {
                let inner = self.if_stmt()?;
                Some(Block { span: inner.span, stmts: vec![inner] })
            } else {
                Some(self.block()?)
            }
        } else {
            None
        };
        Ok(Stmt { kind: StmtKind::If { cond, then, els }, span: start.to(self.prev_span()) })
    }

    fn pattern(&mut self) -> PResult<Pattern> {
        match self.peek().clone() {
            Tok::LParen => {
                let start = self.bump().span;
                let mut ps = Vec::new();
                while !self.at(&Tok::RParen) {
                    ps.push(self.pattern()?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                let end = self.expect(&Tok::RParen, "`)`")?;
                Ok(Pattern::Tuple(ps, start.to(end)))
            }
            Tok::Ident(n) if n == "_" => Ok(Pattern::Wildcard(self.bump().span)),
            Tok::Ident(_) => Ok(Pattern::Ident(self.ident()?)),
            _ => self.err("a pattern"),
        }
    }

    // ---------------------------------------------------------- expressions

    pub fn expr(&mut self) -> PResult<Expr> {
        self.binary(P_PIPE)
    }

    fn binop(&self) -> Option<(u8, Tok)> {
        let t = self.peek().clone();
        let p = match t {
            Tok::Pipe => P_PIPE,
            Tok::OrOr => P_OR,
            Tok::AndAnd => P_AND,
            Tok::EqEq | Tok::Ne | Tok::Lt | Tok::Le | Tok::Gt | Tok::Ge => P_CMP,
            Tok::PlusPlus => P_APPEND,
            Tok::Plus | Tok::Minus => P_ADD,
            Tok::Star | Tok::Slash => P_MUL,
            Tok::At if !self.tok().nl_before => P_MUL,
            _ => return None,
        };
        Some((p, t))
    }

    fn binary(&mut self, min: u8) -> PResult<Expr> {
        let mut lhs = self.unary()?;
        while let Some((p, t)) = self.binop() {
            if p < min {
                break;
            }
            self.bump();
            match t {
                Tok::Pipe => {
                    // `x |> f` or `x |> f(a, b)`  ==> f(x, a, b)
                    let f = self.ident()?;
                    let mut args = vec![lhs];
                    if self.at(&Tok::LParen) && !self.tok().nl_before {
                        args.extend(self.call_args()?);
                    }
                    let span = args[0].span.to(self.prev_span());
                    lhs = self.mk(ExprKind::Call(f, args), span);
                }
                Tok::PlusPlus => {
                    let label = if self.eat(&Tok::HashBracket) {
                        let l = self.ident()?;
                        self.expect(&Tok::RBracket, "`]`")?;
                        Some(l)
                    } else {
                        None
                    };
                    let rhs = self.binary(p + 1)?;
                    let span = lhs.span.to(rhs.span);
                    lhs = self.mk(ExprKind::Append { ctx: Box::new(lhs), label, text: Box::new(rhs) }, span);
                }
                _ => {
                    let op = match t {
                        Tok::OrOr => BinOp::Or,
                        Tok::AndAnd => BinOp::And,
                        Tok::EqEq => BinOp::Eq,
                        Tok::Ne => BinOp::Ne,
                        Tok::Lt => BinOp::Lt,
                        Tok::Le => BinOp::Le,
                        Tok::Gt => BinOp::Gt,
                        Tok::Ge => BinOp::Ge,
                        Tok::Plus => BinOp::Add,
                        Tok::Minus => BinOp::Sub,
                        Tok::Star => BinOp::Mul,
                        Tok::Slash => BinOp::Div,
                        Tok::At => BinOp::MatMul,
                        _ => unreachable!(),
                    };
                    let rhs = self.binary(p + 1)?;
                    let span = lhs.span.to(rhs.span);
                    lhs = self.mk(ExprKind::Binary(op, Box::new(lhs), Box::new(rhs)), span);
                }
            }
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> PResult<Expr> {
        let start = self.span();
        match self.peek() {
            Tok::Minus => {
                self.bump();
                let e = self.unary()?;
                let span = start.to(e.span);
                Ok(match e.kind {
                    ExprKind::Int(v) => self.mk(ExprKind::Int(-v), span),
                    ExprKind::Float(v) => self.mk(ExprKind::Float(-v), span),
                    _ => self.mk(ExprKind::Unary(UnOp::Neg, Box::new(e)), span),
                })
            }
            Tok::Bang => {
                self.bump();
                let e = self.unary()?;
                let span = start.to(e.span);
                Ok(self.mk(ExprKind::Unary(UnOp::Not, Box::new(e)), span))
            }
            Tok::Kw(Kw::Open) => {
                self.bump();
                let name = self.ident()?;
                Ok(self.mk(ExprKind::Open(name), start.to(self.prev_span())))
            }
            Tok::Kw(Kw::Fork) => {
                self.bump();
                let c = self.binary(P_APPEND)?;
                let span = start.to(c.span);
                Ok(self.mk(ExprKind::Fork(Box::new(c)), span))
            }
            Tok::Kw(Kw::Gen) => {
                self.bump();
                let ctx = self.binary(P_APPEND)?;
                self.expect_kw(Kw::Max)?;
                let max = self.binary(P_ADD)?;
                let mut constraint = None;
                if self.eat_kw(Kw::As) {
                    if self.eat_kw(Kw::Json) {
                        self.expect(&Tok::Lt, "`<`")?;
                        let s = self.ident()?;
                        self.expect(&Tok::Gt, "`>`")?;
                        constraint = Some(Constraint::Json(s));
                    } else if self.eat_kw(Kw::Choice) {
                        self.expect(&Tok::LBracket, "`[`")?;
                        let mut opts = Vec::new();
                        while let Tok::Str(s) = self.peek().clone() {
                            self.bump();
                            opts.push(s);
                            if !self.eat(&Tok::Comma) {
                                break;
                            }
                        }
                        self.expect(&Tok::RBracket, "`]`")?;
                        if opts.is_empty() {
                            return Err(Diagnostic::error("V0002", self.prev_span(), "choice[...] needs at least one option"));
                        }
                        constraint = Some(Constraint::Choice(opts));
                    } else {
                        return self.err("`json<Schema>` or `choice[...]`");
                    }
                }
                let stop = if self.eat_kw(Kw::Stop) { Some(Box::new(self.binary(P_ADD)?)) } else { None };
                Ok(self.mk(ExprKind::Gen { ctx: Box::new(ctx), max: Box::new(max), constraint, stop }, start.to(self.prev_span())))
            }
            Tok::Kw(Kw::Splice) => {
                self.bump();
                let ctx = self.binary(P_APPEND)?;
                self.expect_kw(Kw::At)?;
                let span_name = self.ident()?;
                self.expect_kw(Kw::With)?;
                let new_label = if self.eat(&Tok::HashBracket) {
                    let l = self.ident()?;
                    self.expect(&Tok::RBracket, "`]`")?;
                    Some(l)
                } else {
                    None
                };
                let text = self.binary(P_ADD)?;
                let span = start.to(text.span);
                Ok(self.mk(ExprKind::Splice { ctx: Box::new(ctx), span_name, new_label, text: Box::new(text) }, span))
            }
            Tok::Kw(Kw::Speculate) => {
                self.bump();
                let ctx = self.postfix()?;
                let body = self.block()?;
                let span = start.to(body.span);
                Ok(self.mk(ExprKind::Speculate { ctx: Box::new(ctx), body }, span))
            }
            _ => self.postfix(),
        }
    }

    fn call_args(&mut self) -> PResult<Vec<Expr>> {
        self.expect(&Tok::LParen, "`(`")?;
        let mut args = Vec::new();
        while !self.at(&Tok::RParen) {
            args.push(self.expr()?);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::RParen, "`)`")?;
        Ok(args)
    }

    fn postfix(&mut self) -> PResult<Expr> {
        let mut e = self.primary()?;
        loop {
            let nl = self.tok().nl_before;
            match self.peek() {
                Tok::LParen if !nl => {
                    let ExprKind::Var(name) = &e.kind else {
                        return Err(Diagnostic::error("V0002", self.span(), "only named functions can be called"));
                    };
                    let name = name.clone();
                    let args = self.call_args()?;
                    let span = e.span.to(self.prev_span());
                    e = self.mk(ExprKind::Call(name, args), span);
                }
                Tok::Dot => {
                    self.bump();
                    let f = self.ident()?;
                    let span = e.span.to(f.span);
                    e = self.mk(ExprKind::Field(Box::new(e), f), span);
                }
                Tok::LBracket if !nl => {
                    self.bump();
                    let idx = self.expr()?;
                    self.expect(&Tok::RBracket, "`]`")?;
                    let span = e.span.to(self.prev_span());
                    e = self.mk(ExprKind::Index(Box::new(e), Box::new(idx)), span);
                }
                _ => break,
            }
        }
        Ok(e)
    }

    fn primary(&mut self) -> PResult<Expr> {
        let start = self.span();
        match self.peek().clone() {
            Tok::Int(v, suf) => {
                if suf.is_some() {
                    return Err(Diagnostic::error("V0002", start, "unit suffixes are only allowed in attributes like @budget"));
                }
                self.bump();
                Ok(self.mk(ExprKind::Int(v as i64), start))
            }
            Tok::Float(f) => {
                self.bump();
                Ok(self.mk(ExprKind::Float(f), start))
            }
            Tok::Str(s) => {
                self.bump();
                Ok(self.mk(ExprKind::Str(s), start))
            }
            Tok::Kw(Kw::True) => {
                self.bump();
                Ok(self.mk(ExprKind::Bool(true), start))
            }
            Tok::Kw(Kw::False) => {
                self.bump();
                Ok(self.mk(ExprKind::Bool(false), start))
            }
            Tok::Ident(_) => {
                let id = self.ident()?;
                Ok(self.mk(ExprKind::Var(id), start))
            }
            Tok::LParen => {
                self.bump();
                if self.eat(&Tok::RParen) {
                    return Ok(self.mk(ExprKind::Tuple(vec![]), start.to(self.prev_span())));
                }
                let first = self.expr()?;
                if self.eat(&Tok::Comma) {
                    let mut es = vec![first];
                    while !self.at(&Tok::RParen) {
                        es.push(self.expr()?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.expect(&Tok::RParen, "`)`")?;
                    Ok(self.mk(ExprKind::Tuple(es), start.to(self.prev_span())))
                } else {
                    self.expect(&Tok::RParen, "`)`")?;
                    Ok(first)
                }
            }
            Tok::LBracket => {
                self.bump();
                let mut es = Vec::new();
                while !self.at(&Tok::RBracket) {
                    es.push(self.expr()?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::RBracket, "`]`")?;
                Ok(self.mk(ExprKind::List(es), start.to(self.prev_span())))
            }
            _ => {
                let _ = self.peek_at(0);
                self.err("an expression")
            }
        }
    }
}

fn expr_to_pattern(e: &Expr) -> PResult<Pattern> {
    match &e.kind {
        ExprKind::Var(id) if id.name == "_" => Ok(Pattern::Wildcard(id.span)),
        ExprKind::Var(id) => Ok(Pattern::Ident(id.clone())),
        ExprKind::Tuple(es) => Ok(Pattern::Tuple(es.iter().map(expr_to_pattern).collect::<PResult<_>>()?, e.span)),
        _ => Err(Diagnostic::error("V0002", e.span, "left side of `=` must be a variable or a tuple of variables")),
    }
}

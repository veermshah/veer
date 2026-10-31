use crate::ty::{compatible, fmt_num, Mismatch, Spans, Ty};
use std::collections::{HashMap, HashSet};
use veer_numerics::{bound, Bound, Format};
use veer_syntax::ast::*;
use veer_syntax::{Diagnostic, Span};

/// Static token measure for a literal. Byte length is a sound upper bound on
/// the token count of any byte-level BPE tokenizer (every token covers ≥ 1 byte).
pub fn tokens_of(s: &str) -> u64 {
    s.len() as u64
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub name: String,
    pub hf_id: String,
    pub layers: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub kv_fmt: Format,
    /// KV page size in tokens.
    pub block: u64,
    pub max_ctx: Option<u64>,
}

impl ModelInfo {
    /// Bytes of K+V cache per token across all layers.
    pub fn kv_bytes_per_token(&self) -> f64 {
        (self.layers * 2 * self.kv_heads * self.head_dim) as f64 * self.kv_fmt.bits_per_elem() / 8.0
    }
}

#[derive(Clone, Debug)]
pub struct PrefixInfo {
    pub name: String,
    pub model: String,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct Sig {
    pub params: Vec<(String, Ty)>,
    pub ret: Ty,
}

#[derive(Clone, Debug, Default)]
pub struct Budget {
    pub kv_bytes: Option<u64>,
    pub prefill: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub enum Accuracy {
    Abs(f64),
    Rel(f64),
}

#[derive(Clone, Debug)]
pub struct FnInfo {
    pub sig: Sig,
    pub budget: Budget,
    pub accuracy: Option<Accuracy>,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstVal {
    Int(i64),
    Float(f64),
}

impl ConstVal {
    pub fn as_f64(self) -> f64 {
        match self {
            ConstVal::Int(i) => i as f64,
            ConstVal::Float(f) => f,
        }
    }
}

/// Everything later stages need from a well-typed program.
#[derive(Debug, Default)]
pub struct Checked {
    pub models: HashMap<String, ModelInfo>,
    pub prefixes: HashMap<String, PrefixInfo>,
    pub tools: HashMap<String, Sig>,
    pub schemas: HashMap<String, Vec<(String, Ty)>>,
    pub consts: HashMap<String, ConstVal>,
    pub fns: HashMap<String, FnInfo>,
    /// Callees before callers.
    pub fn_order: Vec<String>,
    pub expr_types: HashMap<ExprId, Ty>,
}

impl Checked {
    pub fn ty(&self, e: &Expr) -> &Ty {
        self.expr_types.get(&e.id).unwrap_or(&Ty::Error)
    }

    /// Evaluate a constant integer/float expression (literals, consts, arithmetic).
    pub fn const_eval(&self, e: &Expr) -> Option<ConstVal> {
        const_eval(&self.consts, e)
    }

    /// Minimum serialized size (in tokens, under the byte bound) of a schema value.
    pub fn min_json_tokens(&self, ty: &Ty) -> u64 {
        match ty {
            Ty::Str(_) => 2,
            Ty::Int | Ty::Float => 1,
            Ty::Bool => 4,
            Ty::List(..) => 2,
            Ty::Record(name) => {
                let fields = &self.schemas[name];
                let inner: u64 = fields.iter().map(|(n, t)| n.len() as u64 + 3 + self.min_json_tokens(t)).sum();
                2 + inner + fields.len().saturating_sub(1) as u64
            }
            _ => 0,
        }
    }
}

pub fn const_eval(consts: &HashMap<String, ConstVal>, e: &Expr) -> Option<ConstVal> {
    match &e.kind {
        ExprKind::Int(i) => Some(ConstVal::Int(*i)),
        ExprKind::Float(f) => Some(ConstVal::Float(*f)),
        ExprKind::Var(id) => consts.get(&id.name).copied(),
        ExprKind::Unary(UnOp::Neg, x) => match const_eval(consts, x)? {
            ConstVal::Int(i) => Some(ConstVal::Int(-i)),
            ConstVal::Float(f) => Some(ConstVal::Float(-f)),
        },
        ExprKind::Binary(op, a, b) => {
            let (a, b) = (const_eval(consts, a)?, const_eval(consts, b)?);
            match (a, b) {
                (ConstVal::Int(x), ConstVal::Int(y)) => Some(ConstVal::Int(match op {
                    BinOp::Add => x.checked_add(y)?,
                    BinOp::Sub => x.checked_sub(y)?,
                    BinOp::Mul => x.checked_mul(y)?,
                    BinOp::Div => x.checked_div(y)?,
                    _ => return None,
                })),
                (x, y) => {
                    let (x, y) = (x.as_f64(), y.as_f64());
                    Some(ConstVal::Float(match op {
                        BinOp::Add => x + y,
                        BinOp::Sub => x - y,
                        BinOp::Mul => x * y,
                        BinOp::Div => x / y,
                        _ => return None,
                    }))
                }
            }
        }
        _ => None,
    }
}

pub const BUILTINS: &[&str] = &[
    "print", "len", "contains", "to_str", "trim", "rand_uniform", "zeros", "quant", "matmul", "add", "scale", "relu", "silu", "measured_err", "report",
];

// --------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum LinState {
    Live,
    Moved(Span),
}

#[derive(Clone, Debug)]
struct Var {
    ty: Ty,
    state: LinState,
}

#[derive(Clone, Debug, Default)]
struct Scope {
    vars: HashMap<String, Var>,
    /// Speculation barrier: outer contexts are frozen beyond this scope.
    barrier: bool,
}

struct SpecFrame {
    model: String,
    value_ty: Option<Ty>,
    commit_spans: Option<Spans>,
}

struct FnCtx {
    ret: Ty,
    /// Join of returned types (refines tensor bounds for the summary).
    returned: Option<Ty>,
}

pub struct Checker<'a> {
    pub out: Checked,
    pub diags: Vec<Diagnostic>,
    prog: &'a Program,
    scopes: Vec<Scope>,
    diverged: bool,
    specs: Vec<SpecFrame>,
    fnc: Option<FnCtx>,
}

pub fn check(prog: &Program) -> (Checked, Vec<Diagnostic>) {
    let mut c = Checker { out: Checked::default(), diags: vec![], prog, scopes: vec![], diverged: false, specs: vec![], fnc: None };
    c.run();
    (c.out, c.diags)
}

impl<'a> Checker<'a> {
    fn err(&mut self, code: &'static str, span: Span, msg: impl Into<String>) {
        self.diags.push(Diagnostic::error(code, span, msg));
    }

    fn run(&mut self) {
        // Duplicate item names.
        let mut seen: HashMap<String, Span> = HashMap::new();
        for it in &self.prog.items {
            let n = it.name();
            if let Some(prev) = seen.insert(n.name.clone(), n.span) {
                self.diags.push(Diagnostic::error("V0119", n.span, format!("`{}` is defined more than once", n.name)).label(prev, "first defined here"));
            }
            if BUILTINS.contains(&n.name.as_str()) || Format::parse(&n.name).is_some() {
                self.err("V0119", n.span, format!("`{}` is a builtin name", n.name));
            }
        }
        // Declarations, in dependency order.
        for it in &self.prog.items {
            if let Item::Model(m) = it {
                self.model_decl(m);
            }
        }
        for it in &self.prog.items {
            match it {
                Item::Const(c) => match const_eval(&self.out.consts, &c.value) {
                    Some(v) => {
                        self.out.consts.insert(c.name.name.clone(), v);
                    }
                    None => self.err("V0118", c.value.span, "constant must be a numeric expression of literals and other constants"),
                },
                Item::Schema(s) => {
                    // Schemas may reference earlier schemas.
                    let mut fields = Vec::new();
                    let mut names = HashSet::new();
                    for (n, t) in &s.fields {
                        if !names.insert(n.name.clone()) {
                            self.err("V0119", n.span, format!("duplicate field `{}`", n.name));
                        }
                        let ty = self.resolve_ty(t);
                        if ty.contains_ctx() || matches!(ty, Ty::Tensor { .. }) {
                            self.err("V0118", t.span, "schema fields must be strings, numbers, bools, lists or schemas");
                        }
                        fields.push((n.name.clone(), ty));
                    }
                    self.out.schemas.insert(s.name.name.clone(), fields);
                }
                _ => {}
            }
        }
        for it in &self.prog.items {
            match it {
                Item::Prefix(p) => {
                    if !self.out.models.contains_key(&p.model.name) {
                        self.err("V0100", p.model.span, format!("unknown model `{}`", p.model.name));
                    }
                    self.out.prefixes.insert(p.name.name.clone(), PrefixInfo { name: p.name.name.clone(), model: p.model.name.clone(), text: p.text.clone() });
                }
                Item::Tool(t) => {
                    let params = t.params.iter().map(|p| (p.name.name.clone(), self.resolve_ty(&p.ty))).collect::<Vec<_>>();
                    let ret = self.resolve_ty(&t.ret);
                    for (p, (_, ty)) in t.params.iter().zip(&params) {
                        if ty.contains_ctx() {
                            self.err("V0118", p.ty.span, "tools cannot take contexts");
                        }
                    }
                    if ret.contains_ctx() {
                        self.err("V0118", t.ret.span, "tools cannot return contexts");
                    }
                    if let Ty::Str(None) = ret {
                        self.diags.push(Diagnostic::error("V0118", t.ret.span, "tool results must be bounded (`str<=N`)").note("tool output is untrusted; the bound is enforced at runtime by truncation"));
                    }
                    self.out.tools.insert(t.name.name.clone(), Sig { params, ret });
                }
                _ => {}
            }
        }
        // Function signatures.
        let fns: Vec<&FnDecl> = self.prog.items.iter().filter_map(|i| if let Item::Fn(f) = i { Some(f) } else { None }).collect();
        for f in &fns {
            let mut names = HashSet::new();
            let mut params = Vec::new();
            for p in &f.params {
                if !names.insert(p.name.name.clone()) {
                    self.err("V0119", p.name.span, format!("duplicate parameter `{}`", p.name.name));
                }
                let ty = self.resolve_ty(&p.ty);
                params.push((p.name.name.clone(), ty));
            }
            let ret = f.ret.as_ref().map(|t| self.resolve_ty(t)).unwrap_or(Ty::Unit);
            let (budget, accuracy) = self.attrs(&f.attrs);
            self.out.fns.insert(f.name.name.clone(), FnInfo { sig: Sig { params, ret }, budget, accuracy, span: f.span });
        }
        // Call graph → order, recursion check.
        let mut graph: HashMap<String, Vec<(String, Span)>> = HashMap::new();
        for f in &fns {
            let mut calls = Vec::new();
            collect_calls_block(&f.body, &mut calls);
            calls.retain(|(n, _)| self.out.fns.contains_key(n));
            graph.insert(f.name.name.clone(), calls);
        }
        let mut order = Vec::new();
        let mut state: HashMap<String, u8> = HashMap::new();
        let mut cyclic = HashSet::new();
        for f in &fns {
            topo(&f.name.name, &graph, &mut state, &mut order, &mut cyclic, &mut Vec::new());
        }
        for f in &fns {
            if cyclic.contains(&f.name.name) {
                self.err("V0114", f.name.span, format!("function `{}` is recursive", f.name.name));
            }
        }
        self.out.fn_order = order.clone();
        for name in order {
            if cyclic.contains(&name) {
                continue;
            }
            let f = fns.iter().find(|f| f.name.name == name).unwrap();
            self.check_fn(f);
        }
    }

    fn model_decl(&mut self, m: &ModelDecl) {
        let mut info = ModelInfo {
            name: m.name.name.clone(),
            hf_id: m.hf_id.clone(),
            layers: 0,
            kv_heads: 0,
            head_dim: 0,
            kv_fmt: Format::Bf16,
            block: 16,
            max_ctx: None,
        };
        let mut have = HashSet::new();
        for (k, v) in &m.fields {
            have.insert(k.name.as_str());
            match (k.name.as_str(), v) {
                ("layers", ModelVal::Int(n, _)) => info.layers = *n,
                ("kv_heads", ModelVal::Int(n, _)) => info.kv_heads = *n,
                ("head_dim", ModelVal::Int(n, _)) => info.head_dim = *n,
                ("block", ModelVal::Int(n, _)) if *n > 0 => info.block = *n,
                ("max_ctx", ModelVal::Int(n, _)) => info.max_ctx = Some(*n),
                ("kv", ModelVal::Ident(id)) => match Format::parse(&id.name) {
                    Some(f) => info.kv_fmt = f,
                    None => self.err("V0303", id.span, format!("unknown KV format `{}`", id.name)),
                },
                _ => self.err("V0118", k.span, format!("invalid model field `{}`", k.name)),
            }
        }
        for req in ["layers", "kv_heads", "head_dim"] {
            if !have.contains(req) {
                self.err("V0118", m.name.span, format!("model `{}` is missing `{req}`", m.name.name));
            }
        }
        self.out.models.insert(m.name.name.clone(), info);
    }

    fn attrs(&mut self, attrs: &[Attr]) -> (Budget, Option<Accuracy>) {
        let mut b = Budget::default();
        let mut acc = None;
        for a in attrs {
            match a.name.name.as_str() {
                "budget" => {
                    for arg in &a.args {
                        let val = match &arg.value {
                            AttrVal::Int(v, suf) => match unit_mult(suf.as_deref()) {
                                Some(m) => v.checked_mul(m),
                                None => {
                                    self.err("V0118", arg.span, format!("unknown unit `{}`", suf.clone().unwrap_or_default()));
                                    None
                                }
                            },
                            AttrVal::Float(_) => {
                                self.err("V0118", arg.span, "budgets must be integers");
                                None
                            }
                        };
                        match arg.key.name.as_str() {
                            "kv" => b.kv_bytes = val,
                            "prefill" => b.prefill = val,
                            _ => self.err("V0118", arg.key.span, "budget keys are `kv` and `prefill`"),
                        }
                    }
                }
                "accuracy" => {
                    for arg in &a.args {
                        let v = match arg.value {
                            AttrVal::Int(v, None) => v as f64,
                            AttrVal::Float(f) => f,
                            _ => {
                                self.err("V0118", arg.span, "accuracy needs a number");
                                continue;
                            }
                        };
                        acc = match arg.key.name.as_str() {
                            "abs" => Some(Accuracy::Abs(v)),
                            "rel" => Some(Accuracy::Rel(v)),
                            _ => {
                                self.err("V0118", arg.key.span, "accuracy keys are `abs` and `rel`");
                                None
                            }
                        };
                    }
                }
                other => self.err("V0118", a.name.span, format!("unknown attribute `@{other}`")),
            }
        }
        (b, acc)
    }

    pub fn resolve_ty(&mut self, t: &TypeExpr) -> Ty {
        match &t.kind {
            TypeExprKind::Str(b) => Ty::Str(*b),
            TypeExprKind::Int => Ty::Int,
            TypeExprKind::Float => Ty::Float,
            TypeExprKind::Bool => Ty::Bool,
            TypeExprKind::Ctx(m) => {
                if !self.out.models.contains_key(&m.name) {
                    self.err("V0100", m.span, format!("unknown model `{}`", m.name));
                    return Ty::Error;
                }
                Ty::ctx(&m.name)
            }
            TypeExprKind::Tuple(ts) => {
                if ts.is_empty() {
                    Ty::Unit
                } else {
                    Ty::Tuple(ts.iter().map(|t| self.resolve_ty(t)).collect())
                }
            }
            TypeExprKind::Named(n) => {
                if self.out.schemas.contains_key(&n.name) {
                    Ty::Record(n.name.clone())
                } else {
                    self.err("V0100", n.span, format!("unknown type `{}`", n.name));
                    Ty::Error
                }
            }
            TypeExprKind::List(e, n) => Ty::List(Box::new(self.resolve_ty(e)), *n),
            TypeExprKind::Tensor { shape, fmt, abs, err } => {
                let Some(f) = Format::parse(&fmt.name) else {
                    self.err("V0303", fmt.span, format!("unknown number format `{}`", fmt.name));
                    return Ty::Error;
                };
                Ty::Tensor { shape: shape.clone(), fmt: f, bound: Bound { m: abs.unwrap_or(f64::INFINITY), e: err.unwrap_or(if abs.is_some() { 0.0 } else { f64::INFINITY }) } }
            }
        }
    }

    // ------------------------------------------------------------ scopes

    fn push(&mut self, barrier: bool) {
        self.scopes.push(Scope { vars: HashMap::new(), barrier });
    }

    /// Pop a scope, reporting leaked contexts.
    fn pop(&mut self, end: Span) {
        let s = self.scopes.pop().unwrap();
        if self.diverged {
            return;
        }
        let mut leaked: Vec<_> = s.vars.iter().filter(|(_, v)| v.ty.is_ctx() && matches!(v.state, LinState::Live)).map(|(n, _)| n.clone()).collect();
        leaked.sort();
        for n in leaked {
            self.diags.push(
                Diagnostic::error("V0102", end, format!("context `{n}` is never consumed"))
                    .note("drop it with `drop`, return it, or pass it on: leaked contexts would hold KV memory forever"),
            );
        }
    }

    fn lookup(&self, name: &str) -> Option<(usize, bool)> {
        let mut crossed = false;
        for (i, s) in self.scopes.iter().enumerate().rev() {
            if s.vars.contains_key(name) {
                return Some((i, crossed));
            }
            if s.barrier {
                crossed = true;
            }
        }
        None
    }

    fn bind(&mut self, name: &Ident, ty: Ty) {
        let scope = self.scopes.last_mut().unwrap();
        if let Some(prev) = scope.vars.get(&name.name) {
            if prev.ty.is_ctx() && matches!(prev.state, LinState::Live) && !self.diverged {
                self.diags.push(Diagnostic::error("V0102", name.span, format!("shadowing live context `{}` would leak it", name.name)));
            }
        }
        self.scopes.last_mut().unwrap().vars.insert(name.name.clone(), Var { ty, state: LinState::Live });
    }

    /// Linear state of every context visible right now, keyed by (scope index, name).
    fn lin_snapshot(&self) -> Vec<((usize, String), bool, Option<Spans>)> {
        let mut v = Vec::new();
        for (i, s) in self.scopes.iter().enumerate() {
            for (n, var) in &s.vars {
                if let Ty::Ctx { spans, .. } = &var.ty {
                    v.push(((i, n.clone()), matches!(var.state, LinState::Live), Some(spans.clone())));
                }
            }
        }
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    // ------------------------------------------------------------ functions

    fn check_fn(&mut self, f: &FnDecl) {
        let info = self.out.fns[&f.name.name].clone();
        self.scopes.clear();
        self.diverged = false;
        self.specs.clear();
        self.fnc = Some(FnCtx { ret: info.sig.ret.clone(), returned: None });
        self.push(false);
        for (p, (_, ty)) in f.params.iter().zip(&info.sig.params) {
            self.bind(&p.name, ty.clone());
        }
        self.block_stmts(&f.body);
        if !self.diverged && info.sig.ret != Ty::Unit && info.sig.ret != Ty::Error {
            self.err("V0110", Span::new(f.body.span.end.saturating_sub(1), f.body.span.end), format!("function `{}` must return a value of type {}", f.name.name, info.sig.ret));
        }
        self.pop(Span::new(f.body.span.end.saturating_sub(1), f.body.span.end));
        let fnc = self.fnc.take().unwrap();
        // Refine the summary: returned tensor bounds are tighter than declared ones.
        let mut sig = info.sig.clone();
        if let Some(r) = fnc.returned {
            sig.ret = refine(&sig.ret, &r);
        }
        if let (Some(acc), Ty::Tensor { bound: b, .. }) = (info.accuracy, &sig.ret) {
            let (ok, what) = match acc {
                Accuracy::Abs(x) => (b.e <= x, format!("abs error bound {} > {}", fmt_num(b.e), fmt_num(x))),
                Accuracy::Rel(x) => (b.rel() <= x, format!("relative error bound {} > {}", fmt_num(b.rel()), fmt_num(x))),
            };
            if !ok {
                self.diags.push(
                    Diagnostic::error("V0301", f.name.span, format!("`{}` violates its @accuracy contract: {what}", f.name.name))
                        .note(format!("inferred return type: {}", sig.ret))
                        .note("use a wider format (e.g. bf16 / mxfp4_oas / nvfp4), or loosen the contract"),
                );
            }
        }
        self.out.fns.get_mut(&f.name.name).unwrap().sig = sig;
    }

    fn block(&mut self, b: &Block) {
        self.push(false);
        self.block_stmts(b);
        self.pop(Span::new(b.span.end.saturating_sub(1), b.span.end));
    }

    fn block_stmts(&mut self, b: &Block) {
        for s in &b.stmts {
            if self.diverged {
                break;
            }
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        match &s.kind {
            StmtKind::Let { pat, ty, value } => {
                let vt = self.expr(value);
                let t = if let Some(te) = ty {
                    let decl = self.resolve_ty(te);
                    self.expect_compat(&decl, &vt, value.span);
                    match (&decl, &vt) {
                        (Ty::Ctx { .. }, _) | (Ty::Tensor { .. }, Ty::Tensor { .. }) => refine(&decl, &vt),
                        _ => decl,
                    }
                } else {
                    vt
                };
                self.bind_pattern(pat, t);
            }
            StmtKind::Assign { pat, value } => {
                let vt = self.expr(value);
                self.assign_pattern(pat, vt, value.span);
            }
            StmtKind::Expr(e) => {
                let t = self.expr(e);
                if t.contains_ctx() {
                    self.err("V0102", e.span, "the context produced here is discarded; bind it or `drop` it");
                }
            }
            StmtKind::Drop(es) => {
                for e in es {
                    let t = self.expr(e);
                    if !t.is_ctx() && t != Ty::Error {
                        self.err("V0110", e.span, format!("`drop` expects a context, found {t}"));
                    }
                }
            }
            StmtKind::Return(e) => {
                if !self.specs.is_empty() {
                    self.err("V0132", s.span, "`return` is not allowed inside `speculate`");
                    return;
                }
                let t = e.as_ref().map(|e| self.expr(e)).unwrap_or(Ty::Unit);
                let ret = self.fnc.as_ref().unwrap().ret.clone();
                self.expect_compat(&ret, &t, e.as_ref().map(|e| e.span).unwrap_or(s.span));
                let fnc = self.fnc.as_mut().unwrap();
                fnc.returned = Some(match fnc.returned.take() {
                    None => t,
                    Some(prev) => join_ty(&prev, &t),
                });
                // Every other live context must already be consumed.
                let mut leaked = Vec::new();
                for sc in &self.scopes {
                    for (n, v) in &sc.vars {
                        if v.ty.is_ctx() && matches!(v.state, LinState::Live) {
                            leaked.push(n.clone());
                        }
                    }
                }
                leaked.sort();
                for n in leaked {
                    self.err("V0102", s.span, format!("context `{n}` is still live at this return"));
                }
                self.diverged = true;
            }
            StmtKind::If { cond, then, els } => {
                let ct = self.expr(cond);
                self.expect_compat(&Ty::Bool, &ct, cond.span);
                let before = self.scopes.clone();
                self.block(then);
                let (then_scopes, then_div) = (std::mem::replace(&mut self.scopes, before), self.diverged);
                self.diverged = false;
                if let Some(e) = els {
                    self.block(e);
                }
                let else_div = self.diverged;
                match (then_div, else_div) {
                    (true, true) => self.diverged = true,
                    (true, false) => {}
                    (false, true) => {
                        self.scopes = then_scopes;
                        self.diverged = false;
                    }
                    (false, false) => {
                        let else_scopes = std::mem::replace(&mut self.scopes, then_scopes);
                        let a = self.lin_snapshot();
                        let saved = std::mem::replace(&mut self.scopes, else_scopes);
                        let b = self.lin_snapshot();
                        self.scopes = saved;
                        for (x, y) in a.iter().zip(&b) {
                            if x.1 != y.1 {
                                self.diags.push(
                                    Diagnostic::error("V0103", s.span, format!("context `{}` is consumed on one branch but not the other", x.0 .1))
                                        .note("consume (or keep) it on both paths"),
                                );
                            }
                            // Spans after the branch: those present on both paths.
                            if let (Some(sa), Some(sb)) = (&x.2, &y.2) {
                                let inter = sa.join(sb);
                                if let Some(v) = self.scopes[x.0 .0].vars.get_mut(&x.0 .1) {
                                    if let Ty::Ctx { spans, .. } = &mut v.ty {
                                        *spans = inter;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            StmtKind::For { var, iter, body } => {
                let elem_ty = match iter {
                    ForIter::Range(a, b) => {
                        for e in [a, b] {
                            self.expr(e);
                            if self.out.const_eval(e).is_none() {
                                self.diags.push(Diagnostic::error("V0113", e.span, "loop bounds must be constants").note("Veer needs a static trip count to bound KV memory and prefill cost"));
                            }
                        }
                        Ty::Int
                    }
                    ForIter::Each(e) => match self.expr(e) {
                        Ty::List(t, _) => *t,
                        Ty::Error => Ty::Error,
                        t => {
                            self.err("V0113", e.span, format!("can only iterate over a range or a bounded list, found {t}"));
                            Ty::Error
                        }
                    },
                };
                let before = self.lin_snapshot();
                self.push(false);
                self.bind(var, elem_ty);
                self.block_stmts(body);
                self.pop(Span::new(body.span.end.saturating_sub(1), body.span.end));
                if self.diverged {
                    // A loop body that always returns: treat the loop as possibly not running.
                    self.diverged = false;
                }
                let after = self.lin_snapshot();
                for (x, y) in before.iter().zip(&after) {
                    if x.1 != y.1 {
                        self.diags.push(
                            Diagnostic::error("V0104", s.span, format!("loop body changes the state of context `{}`", x.0 .1))
                                .note("to grow a context in a loop, consume and re-assign it: `c = c ++ \"...\"`"),
                        );
                    }
                    if let (Some(sa), Some(sb)) = (&x.2, &y.2) {
                        if x.1 && y.1 && sa != sb {
                            self.diags.push(
                                Diagnostic::error("V0104", s.span, format!("loop body changes the labelled spans of context `{}`", x.0 .1))
                                    .note("a label added in one iteration would be duplicated in the next; splice it away (or relabel) before the iteration ends"),
                            );
                        }
                    }
                }
            }
            StmtKind::Commit(e) => {
                if self.specs.is_empty() {
                    self.err("V0132", s.span, "`commit` is only allowed inside `speculate`");
                    return;
                }
                let t = self.expr(e);
                let frame_model = self.specs.last().unwrap().model.clone();
                match &t {
                    Ty::Tuple(ts) if ts.len() == 2 && ts[0].is_ctx() => {
                        if let Ty::Ctx { model, spans } = &ts[0] {
                            if *model != frame_model {
                                self.err("V0106", e.span, format!("committed context belongs to `{model}`, speculation is on `{frame_model}`"));
                            }
                            let fr = self.specs.last_mut().unwrap();
                            fr.commit_spans = Some(match fr.commit_spans.take() {
                                None => spans.clone(),
                                Some(p) => p.join(spans),
                            });
                        }
                        let vt = ts[1].clone();
                        self.spec_value(vt, e.span);
                    }
                    Ty::Error => {}
                    _ => self.err("V0110", e.span, format!("`commit` expects `(context, value)`, found {t}")),
                }
                self.auto_release();
                self.diverged = true;
            }
            StmtKind::Rollback(e) => {
                if self.specs.is_empty() {
                    self.err("V0132", s.span, "`rollback` is only allowed inside `speculate`");
                    return;
                }
                let t = self.expr(e);
                if t.contains_ctx() {
                    self.err("V0110", e.span, "`rollback` takes the fallback value, not a context");
                }
                self.spec_value(t, e.span);
                self.auto_release();
                self.diverged = true;
            }
        }
    }

    fn spec_value(&mut self, t: Ty, span: Span) {
        let fr = self.specs.last_mut().unwrap();
        match fr.value_ty.take() {
            None => fr.value_ty = Some(t),
            Some(prev) => {
                if compatible(&prev, &t).is_err() && compatible(&t, &prev).is_err() {
                    fr.value_ty = Some(prev.clone());
                    self.err("V0110", span, format!("speculation results disagree: {prev} vs {t}"));
                } else {
                    fr.value_ty = Some(join_ty(&prev, &t));
                }
            }
        }
    }

    /// commit/rollback release every context created inside the speculation.
    fn auto_release(&mut self) {
        for s in self.scopes.iter_mut().rev() {
            for v in s.vars.values_mut() {
                if v.ty.is_ctx() {
                    v.state = LinState::Moved(Span::default());
                }
            }
            if s.barrier {
                break;
            }
        }
    }

    fn bind_pattern(&mut self, pat: &Pattern, ty: Ty) {
        match pat {
            Pattern::Ident(id) => {
                if ty.contains_ctx() && !ty.is_ctx() {
                    self.err("V0105", id.span, format!("a value of type {ty} contains a context and must be destructured, e.g. `let (c, x) = ...`"));
                }
                self.bind(id, ty);
            }
            Pattern::Wildcard(sp) => {
                if ty.contains_ctx() {
                    self.err("V0102", *sp, "`_` would discard a context; bind it and `drop` it instead");
                }
            }
            Pattern::Tuple(ps, sp) => match ty {
                Ty::Tuple(ts) if ts.len() == ps.len() => {
                    for (p, t) in ps.iter().zip(ts) {
                        self.bind_pattern(p, t);
                    }
                }
                Ty::Error => {
                    for p in ps {
                        self.bind_pattern(p, Ty::Error);
                    }
                }
                t => {
                    self.err("V0110", *sp, format!("cannot destructure {t} into {} names", ps.len()));
                    for p in ps {
                        self.bind_pattern(p, Ty::Error);
                    }
                }
            },
        }
    }

    fn assign_pattern(&mut self, pat: &Pattern, ty: Ty, span: Span) {
        match pat {
            Pattern::Ident(id) => {
                let Some((si, crossed)) = self.lookup(&id.name) else {
                    self.err("V0100", id.span, format!("unknown variable `{}`", id.name));
                    return;
                };
                let var = self.scopes[si].vars[&id.name].clone();
                if crossed && var.ty.is_ctx() {
                    self.err("V0131", id.span, format!("outer context `{}` cannot be assigned inside `speculate`", id.name));
                    return;
                }
                self.expect_compat(&var.ty, &ty, span);
                if var.ty.is_ctx() && matches!(var.state, LinState::Live) {
                    self.err("V0102", id.span, format!("assigning to `{}` would leak the context it still holds", id.name));
                }
                let new_ty = if var.ty.is_ctx() || matches!(var.ty, Ty::Error) { ty } else { var.ty.clone() };
                self.scopes[si].vars.insert(id.name.clone(), Var { ty: new_ty, state: LinState::Live });
            }
            Pattern::Wildcard(sp) => {
                if ty.contains_ctx() {
                    self.err("V0102", *sp, "`_` would discard a context");
                }
            }
            Pattern::Tuple(ps, sp) => match ty {
                Ty::Tuple(ts) if ts.len() == ps.len() => {
                    for (p, t) in ps.iter().zip(ts) {
                        self.assign_pattern(p, t, span);
                    }
                }
                Ty::Error => {}
                t => self.err("V0110", *sp, format!("cannot destructure {t} into {} names", ps.len())),
            },
        }
    }

    fn expect_compat(&mut self, expected: &Ty, actual: &Ty, span: Span) {
        if let Err(m) = compatible(expected, actual) {
            let d = match m {
                Mismatch::Bound { expected, actual } => Diagnostic::error(
                    "V0112",
                    span,
                    match actual {
                        Some(a) => format!("string may be up to {a} tokens, but at most {expected} are allowed"),
                        None => format!("unbounded string where `str<={expected}` is required"),
                    },
                ),
                Mismatch::Model(a, b) => Diagnostic::error("V0106", span, format!("expected a context of model `{a}`, found `{b}`")),
                Mismatch::TensorBound { expected: e, actual: a } => Diagnostic::error(
                    "V0110",
                    span,
                    format!("tensor bound (abs<={}, err<={}) exceeds the declared (abs<={}, err<={})", fmt_num(a.m), fmt_num(a.e), fmt_num(e.m), fmt_num(e.e)),
                ),
                Mismatch::Plain => Diagnostic::error("V0110", span, format!("expected {expected}, found {actual}")),
            };
            self.diags.push(d);
        }
    }

    // ------------------------------------------------------------ expressions

    pub fn expr(&mut self, e: &Expr) -> Ty {
        let t = self.expr_inner(e);
        self.out.expr_types.insert(e.id, t.clone());
        t
    }

    fn expect_ctx(&mut self, e: &Expr) -> Option<(String, Spans)> {
        match self.expr(e) {
            Ty::Ctx { model, spans } => Some((model, spans)),
            Ty::Error => None,
            t => {
                self.err("V0110", e.span, format!("expected a context, found {t}"));
                None
            }
        }
    }

    fn bounded_text(&mut self, e: &Expr) -> Option<u64> {
        match self.expr(e) {
            Ty::Str(Some(n)) => Some(n),
            Ty::Str(None) => {
                self.diags.push(Diagnostic::error("V0117", e.span, "only bounded strings (`str<=N`) can be appended to a context"));
                None
            }
            Ty::Error => None,
            t => {
                self.err("V0110", e.span, format!("expected a string, found {t}"));
                None
            }
        }
    }

    fn expr_inner(&mut self, e: &Expr) -> Ty {
        match &e.kind {
            ExprKind::Int(_) => Ty::Int,
            ExprKind::Float(_) => Ty::Float,
            ExprKind::Str(s) => Ty::Str(Some(tokens_of(s))),
            ExprKind::Bool(_) => Ty::Bool,
            ExprKind::Var(id) => self.var_use(id),
            ExprKind::Binary(op, a, b) => self.binary(*op, a, b, e.span),
            ExprKind::Unary(op, x) => {
                let t = self.expr(x);
                match (op, &t) {
                    (UnOp::Not, Ty::Bool) => Ty::Bool,
                    (UnOp::Neg, Ty::Int | Ty::Float) => t,
                    (UnOp::Neg, Ty::Tensor { shape, fmt: _, bound: b }) => Ty::Tensor { shape: shape.clone(), fmt: Format::F32, bound: bound::scale(*b, -1.0) },
                    (_, Ty::Error) => Ty::Error,
                    _ => {
                        self.err("V0110", e.span, format!("invalid operand {t}"));
                        Ty::Error
                    }
                }
            }
            ExprKind::Call(name, args) => self.call(name, args, e.span),
            ExprKind::Field(x, f) => match self.expr(x) {
                Ty::Record(s) => match self.out.schemas[&s].iter().find(|(n, _)| *n == f.name) {
                    Some((_, t)) => t.clone(),
                    None => {
                        self.err("V0115", f.span, format!("schema `{s}` has no field `{}`", f.name));
                        Ty::Error
                    }
                },
                Ty::Error => Ty::Error,
                t => {
                    self.err("V0115", f.span, format!("{t} has no fields"));
                    Ty::Error
                }
            },
            ExprKind::Index(x, i) => {
                let xt = self.expr(x);
                let it = self.expr(i);
                self.expect_compat(&Ty::Int, &it, i.span);
                match xt {
                    Ty::List(t, _) => *t,
                    Ty::Error => Ty::Error,
                    t => {
                        self.err("V0110", x.span, format!("cannot index {t}"));
                        Ty::Error
                    }
                }
            }
            ExprKind::Tuple(es) => {
                if es.is_empty() {
                    Ty::Unit
                } else {
                    Ty::Tuple(es.iter().map(|x| self.expr(x)).collect())
                }
            }
            ExprKind::List(_) => {
                self.err("V0110", e.span, "list literals are only allowed as tensor shapes");
                Ty::Error
            }
            ExprKind::Open(name) => {
                if let Some(p) = self.out.prefixes.get(&name.name) {
                    Ty::ctx(&p.model.clone())
                } else if self.out.models.contains_key(&name.name) {
                    Ty::ctx(&name.name)
                } else {
                    self.err("V0100", name.span, format!("`open` needs a model or prefix, `{}` is neither", name.name));
                    Ty::Error
                }
            }
            ExprKind::Append { ctx, label, text } => {
                let c = self.expect_ctx(ctx);
                self.bounded_text(text);
                let Some((model, mut spans)) = c else { return Ty::Error };
                if let Some(l) = label {
                    if !spans.add(&l.name) {
                        self.err("V0121", l.span, format!("span `{}` is already present in this context", l.name));
                    }
                }
                Ty::Ctx { model, spans }
            }
            ExprKind::Fork(c) => match self.expect_ctx(c) {
                Some((model, spans)) => Ty::Tuple(vec![Ty::Ctx { model: model.clone(), spans: spans.clone() }, Ty::Ctx { model, spans }]),
                None => Ty::Error,
            },
            ExprKind::Gen { ctx, max, constraint, stop } => {
                let c = self.expect_ctx(ctx);
                self.expr(max);
                let maxv = match self.out.const_eval(max) {
                    Some(ConstVal::Int(n)) if n > 0 => n as u64,
                    _ => {
                        self.err("V0113", max.span, "`max` must be a positive integer constant");
                        1
                    }
                };
                if let Some(s) = stop {
                    let st = self.expr(s);
                    self.expect_compat(&Ty::Str(None), &st, s.span);
                }
                let out_ty = match constraint {
                    None => Ty::Str(Some(maxv)),
                    Some(Constraint::Choice(opts)) => {
                        let longest = opts.iter().map(|o| tokens_of(o)).max().unwrap_or(0);
                        if longest > maxv {
                            self.err("V0116", max.span, format!("the longest choice needs up to {longest} tokens but max is {maxv}"));
                        }
                        Ty::Str(Some(longest))
                    }
                    Some(Constraint::Json(s)) => {
                        if !self.out.schemas.contains_key(&s.name) {
                            self.err("V0100", s.span, format!("unknown schema `{}`", s.name));
                            Ty::Error
                        } else {
                            let t = Ty::Record(s.name.clone());
                            let need = self.out.min_json_tokens(&t);
                            if need > maxv {
                                self.diags.push(
                                    Diagnostic::error("V0116", max.span, format!("the smallest `{}` value needs {need} tokens but max is {maxv}", s.name))
                                        .note("token counts use Veer's conservative byte bound"),
                                );
                            }
                            t
                        }
                    }
                };
                match c {
                    Some((model, spans)) => Ty::Tuple(vec![Ty::Ctx { model, spans }, out_ty]),
                    None => Ty::Error,
                }
            }
            ExprKind::Splice { ctx, span_name, new_label, text } => {
                let c = self.expect_ctx(ctx);
                self.bounded_text(text);
                let Some((model, mut spans)) = c else { return Ty::Error };
                if !spans.has(&span_name.name) {
                    let known: Vec<_> = spans.must.iter().cloned().collect();
                    self.diags.push(
                        Diagnostic::error("V0120", span_name.span, format!("context has no span `{}`", span_name.name))
                            .note(if known.is_empty() { "this context has no labelled spans; add one with `c ++ #[name] text`".to_string() } else { format!("known spans: {}", known.join(", ")) }),
                    );
                }
                // The replacement text is unlabelled unless a new label is given.
                spans.remove(&span_name.name);
                if let Some(l) = new_label {
                    if !spans.add(&l.name) {
                        self.err("V0121", l.span, format!("span `{}` is already present in this context", l.name));
                    }
                }
                Ty::Ctx { model, spans }
            }
            ExprKind::Speculate { ctx, body } => {
                let ExprKind::Var(name) = &ctx.kind else {
                    self.err("V0110", ctx.span, "`speculate` expects a context variable");
                    return Ty::Error;
                };
                let Some((model, spans)) = self.expect_ctx(ctx) else { return Ty::Error };
                self.specs.push(SpecFrame { model: model.clone(), value_ty: None, commit_spans: None });
                let outer_div = self.diverged;
                self.push(true);
                self.bind(name, Ty::Ctx { model: model.clone(), spans: spans.clone() });
                self.block_stmts(body);
                if !self.diverged {
                    self.diags.push(
                        Diagnostic::error("V0130", Span::new(body.span.end.saturating_sub(1), body.span.end), "this path through `speculate` falls off the end")
                            .note("end every path with `commit (ctx, value)` or `rollback value`"),
                    );
                }
                self.diverged = true; // suppress leak reports: contexts are auto-released
                self.scopes.pop();
                self.diverged = outer_div;
                let fr = self.specs.pop().unwrap();
                let out_spans = match fr.commit_spans {
                    Some(cs) => cs.join(&spans),
                    None => spans,
                };
                Ty::Tuple(vec![Ty::Ctx { model, spans: out_spans }, fr.value_ty.unwrap_or(Ty::Error)])
            }
        }
    }

    fn var_use(&mut self, id: &Ident) -> Ty {
        if let Some((si, crossed)) = self.lookup(&id.name) {
            let var = self.scopes[si].vars[&id.name].clone();
            if var.ty.is_ctx() {
                if crossed {
                    self.err("V0131", id.span, format!("context `{}` is frozen inside this `speculate`", id.name));
                    return Ty::Error;
                }
                if let LinState::Moved(at) = var.state {
                    let mut d = Diagnostic::error("V0101", id.span, format!("use of moved context `{}`", id.name));
                    if at != Span::default() {
                        d = d.label(at, "moved here");
                    }
                    self.diags.push(d.note("every context operation consumes its input; rebind the result, e.g. `let c = c ++ \"...\"`"));
                    return Ty::Error;
                }
                self.scopes[si].vars.get_mut(&id.name).unwrap().state = LinState::Moved(id.span);
            }
            return var.ty;
        }
        if self.out.consts.contains_key(&id.name) {
            return match self.out.consts[&id.name] {
                ConstVal::Int(_) => Ty::Int,
                ConstVal::Float(_) => Ty::Float,
            };
        }
        if let Some(f) = Format::parse(&id.name) {
            return Ty::Format(f);
        }
        let hint = if self.out.prefixes.contains_key(&id.name) || self.out.models.contains_key(&id.name) { format!(" (did you mean `open {}`?)", id.name) } else { String::new() };
        self.err("V0100", id.span, format!("unknown name `{}`{hint}", id.name));
        Ty::Error
    }

    fn binary(&mut self, op: BinOp, a: &Expr, b: &Expr, span: Span) -> Ty {
        let ta = self.expr(a);
        let tb = self.expr(b);
        use BinOp::*;
        if ta == Ty::Error || tb == Ty::Error {
            return match op {
                Eq | Ne | Lt | Le | Gt | Ge | And | Or => Ty::Bool,
                _ => Ty::Error,
            };
        }
        match op {
            Add | Sub | Mul | Div => match (&ta, &tb) {
                (Ty::Int, Ty::Int) => Ty::Int,
                (Ty::Int | Ty::Float, Ty::Int | Ty::Float) => Ty::Float,
                (Ty::Str(x), Ty::Str(y)) if op == Add => Ty::Str(match (x, y) {
                    (Some(x), Some(y)) => Some(x + y),
                    _ => None,
                }),
                (Ty::Tensor { .. }, Ty::Tensor { .. }) if op == Add || op == Sub => self.tensor_add(&ta, &tb, span),
                (Ty::Tensor { shape, bound: bd, .. }, Ty::Int | Ty::Float) | (Ty::Int | Ty::Float, Ty::Tensor { shape, bound: bd, .. }) if op == Mul => {
                    let c = if matches!(ta, Ty::Tensor { .. }) { b } else { a };
                    match self.out.const_eval(c) {
                        Some(v) => Ty::Tensor { shape: shape.clone(), fmt: Format::F32, bound: bound::scale(*bd, v.as_f64()) },
                        None => {
                            self.err("V0110", c.span, "tensor scale factors must be constants (so the error bound is static)");
                            Ty::Error
                        }
                    }
                }
                _ => {
                    self.err("V0110", span, format!("cannot apply `{}` to {ta} and {tb}", op.as_str()));
                    Ty::Error
                }
            },
            MatMul => self.tensor_matmul(&ta, &tb, span),
            Eq | Ne => {
                let ok = matches!((&ta, &tb), (Ty::Str(_), Ty::Str(_)) | (Ty::Bool, Ty::Bool)) || (ta.is_numeric() && tb.is_numeric());
                if !ok {
                    self.err("V0110", span, format!("cannot compare {ta} and {tb}"));
                }
                Ty::Bool
            }
            Lt | Le | Gt | Ge => {
                if !(ta.is_numeric() && tb.is_numeric()) {
                    self.err("V0110", span, format!("cannot order {ta} and {tb}"));
                }
                Ty::Bool
            }
            And | Or => {
                self.expect_compat(&Ty::Bool, &ta, a.span);
                self.expect_compat(&Ty::Bool, &tb, b.span);
                Ty::Bool
            }
        }
    }

    fn tensor_add(&mut self, ta: &Ty, tb: &Ty, span: Span) -> Ty {
        let (Ty::Tensor { shape: sa, bound: ba, .. }, Ty::Tensor { shape: sb, bound: bb, .. }) = (ta, tb) else { return Ty::Error };
        let broadcast = sb.len() == 2 && sb[0] == 1 && sa.len() == 2 && sa[1] == sb[1];
        if sa != sb && !broadcast {
            self.err("V0300", span, format!("cannot add shapes {sa:?} and {sb:?} (the right side may also be a [1, n] row)"));
            return Ty::Error;
        }
        Ty::Tensor { shape: sa.clone(), fmt: Format::F32, bound: bound::add(*ba, *bb) }
    }

    fn tensor_matmul(&mut self, ta: &Ty, tb: &Ty, span: Span) -> Ty {
        let (Ty::Tensor { shape: sa, bound: ba, .. }, Ty::Tensor { shape: sb, bound: bb, .. }) = (ta, tb) else {
            self.err("V0110", span, format!("`@` needs two tensors, found {ta} and {tb}"));
            return Ty::Error;
        };
        if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[0] {
            self.err("V0300", span, format!("cannot multiply shapes {sa:?} @ {sb:?}"));
            return Ty::Error;
        }
        Ty::Tensor { shape: vec![sa[0], sb[1]], fmt: Format::F32, bound: bound::matmul(sa[1] as usize, *ba, *bb) }
    }

    fn shape_arg(&mut self, e: &Expr) -> Option<Vec<u64>> {
        let ExprKind::List(es) = &e.kind else {
            self.err("V0110", e.span, "expected a shape like `[4, 64]`");
            return None;
        };
        let mut dims = Vec::new();
        for d in es {
            match self.out.const_eval(d) {
                Some(ConstVal::Int(n)) if n > 0 => dims.push(n as u64),
                _ => {
                    self.err("V0110", d.span, "shape dimensions must be positive integer constants");
                    return None;
                }
            }
        }
        self.out.expr_types.insert(e.id, Ty::Unit);
        Some(dims)
    }

    fn call(&mut self, name: &Ident, args: &[Expr], span: Span) -> Ty {
        let n = name.name.as_str();
        let arity = |c: &mut Self, k: usize| -> bool {
            if args.len() != k {
                c.err("V0111", span, format!("`{n}` takes {k} argument(s), {} given", args.len()));
                for a in args {
                    if !matches!(a.kind, ExprKind::List(_)) {
                        c.expr(a);
                    }
                }
                false
            } else {
                true
            }
        };
        if let Some(info) = self.out.fns.get(n).cloned() {
            if !arity(self, info.sig.params.len()) {
                return Ty::Error;
            }
            for (a, (_, pt)) in args.iter().zip(&info.sig.params) {
                let at = self.expr(a);
                self.expect_compat(pt, &at, a.span);
            }
            return info.sig.ret.clone();
        }
        if let Some(sig) = self.out.tools.get(n).cloned() {
            if !arity(self, sig.params.len()) {
                return Ty::Error;
            }
            for (a, (_, pt)) in args.iter().zip(&sig.params) {
                let at = self.expr(a);
                self.expect_compat(pt, &at, a.span);
            }
            return sig.ret.clone();
        }
        match n {
            "print" => {
                for a in args {
                    let t = self.expr(a);
                    if t.contains_ctx() {
                        self.err("V0105", a.span, "contexts cannot be printed (that would duplicate them)");
                    }
                }
                Ty::Unit
            }
            "len" => {
                if !arity(self, 1) {
                    return Ty::Int;
                }
                let t = self.expr(&args[0]);
                if !matches!(t, Ty::Str(_) | Ty::List(..) | Ty::Error) {
                    self.err("V0110", args[0].span, format!("`len` expects a string or list, found {t}"));
                }
                Ty::Int
            }
            "contains" => {
                if !arity(self, 2) {
                    return Ty::Bool;
                }
                for a in args {
                    let t = self.expr(a);
                    self.expect_compat(&Ty::Str(None), &t, a.span);
                }
                Ty::Bool
            }
            "to_str" => {
                if !arity(self, 1) {
                    return Ty::Str(Some(24));
                }
                match self.expr(&args[0]) {
                    t @ Ty::Str(_) => t,
                    Ty::Int | Ty::Float | Ty::Bool | Ty::Error => Ty::Str(Some(24)),
                    t => {
                        self.err("V0110", args[0].span, format!("cannot convert {t} to a string"));
                        Ty::Error
                    }
                }
            }
            "trim" => {
                if !arity(self, 1) {
                    return Ty::Error;
                }
                let t = self.expr(&args[0]);
                self.expect_compat(&Ty::Str(None), &t, args[0].span);
                t
            }
            "rand_uniform" => {
                if !arity(self, 4) {
                    return Ty::Error;
                }
                let shape = self.shape_arg(&args[0]);
                let mut mags = Vec::new();
                for a in &args[1..3] {
                    self.expr(a);
                    match self.out.const_eval(a) {
                        Some(v) => mags.push(v.as_f64().abs()),
                        None => self.err("V0110", a.span, "range bounds must be constants (they set the static magnitude)"),
                    }
                }
                let st = self.expr(&args[3]);
                self.expect_compat(&Ty::Int, &st, args[3].span);
                match shape {
                    Some(shape) if mags.len() == 2 => Ty::Tensor { shape, fmt: Format::F32, bound: Bound::exact(mags[0].max(mags[1])) },
                    _ => Ty::Error,
                }
            }
            "zeros" => {
                if !arity(self, 1) {
                    return Ty::Error;
                }
                match self.shape_arg(&args[0]) {
                    Some(shape) => Ty::Tensor { shape, fmt: Format::F32, bound: Bound::exact(0.0) },
                    None => Ty::Error,
                }
            }
            "quant" => {
                if !arity(self, 2) {
                    return Ty::Error;
                }
                let t = self.expr(&args[0]);
                let f = match &args[1].kind {
                    ExprKind::Var(id) => match Format::parse(&id.name) {
                        Some(f) => Some(f),
                        None => {
                            self.err("V0303", id.span, format!("unknown number format `{}`", id.name));
                            None
                        }
                    },
                    _ => {
                        self.err("V0303", args[1].span, "expected a number format name");
                        None
                    }
                };
                if let Some(f) = f {
                    self.out.expr_types.insert(args[1].id, Ty::Format(f));
                }
                match (t, f) {
                    (Ty::Tensor { shape, bound: b, .. }, Some(f)) => match bound::quant(b, f) {
                        Ok(nb) => Ty::Tensor { shape, fmt: f, bound: nb },
                        Err(o) => {
                            self.diags.push(
                                Diagnostic::error("V0302", span, format!("values up to {} may overflow {} (max {})", fmt_num(o.magnitude), f.name(), fmt_num(o.max)))
                                    .note("scale the tensor down first, or use a block-scaled format (int8, mxfp4, nvfp4)"),
                            );
                            Ty::Error
                        }
                    },
                    (Ty::Error, _) | (_, None) => Ty::Error,
                    (t, _) => {
                        self.err("V0110", args[0].span, format!("`quant` expects a tensor, found {t}"));
                        Ty::Error
                    }
                }
            }
            "matmul" | "add" => {
                if !arity(self, 2) {
                    return Ty::Error;
                }
                let a = self.expr(&args[0]);
                let b = self.expr(&args[1]);
                if a == Ty::Error || b == Ty::Error {
                    return Ty::Error;
                }
                if n == "matmul" {
                    self.tensor_matmul(&a, &b, span)
                } else {
                    self.tensor_add(&a, &b, span)
                }
            }
            "scale" => {
                if !arity(self, 2) {
                    return Ty::Error;
                }
                let t = self.expr(&args[0]);
                self.expr(&args[1]);
                let c = self.out.const_eval(&args[1]);
                match (t, c) {
                    (Ty::Tensor { shape, bound: b, .. }, Some(c)) => Ty::Tensor { shape, fmt: Format::F32, bound: bound::scale(b, c.as_f64()) },
                    (Ty::Error, _) => Ty::Error,
                    (_, None) => {
                        self.err("V0110", args[1].span, "scale factor must be a constant");
                        Ty::Error
                    }
                    (t, _) => {
                        self.err("V0110", args[0].span, format!("`scale` expects a tensor, found {t}"));
                        Ty::Error
                    }
                }
            }
            "relu" | "silu" => {
                if !arity(self, 1) {
                    return Ty::Error;
                }
                match self.expr(&args[0]) {
                    Ty::Tensor { shape, bound: b, .. } => Ty::Tensor { shape, fmt: Format::F32, bound: if n == "relu" { bound::relu(b) } else { bound::silu(b) } },
                    Ty::Error => Ty::Error,
                    t => {
                        self.err("V0110", args[0].span, format!("`{n}` expects a tensor, found {t}"));
                        Ty::Error
                    }
                }
            }
            "measured_err" | "report" => {
                if !arity(self, 1) {
                    return Ty::Error;
                }
                match self.expr(&args[0]) {
                    Ty::Tensor { .. } | Ty::Error => {}
                    t => self.err("V0110", args[0].span, format!("`{n}` expects a tensor, found {t}")),
                }
                if n == "report" {
                    Ty::Unit
                } else {
                    Ty::Float
                }
            }
            _ => {
                for a in args {
                    if !matches!(a.kind, ExprKind::List(_)) {
                        self.expr(a);
                    }
                }
                self.err("V0100", name.span, format!("unknown function `{n}`"));
                Ty::Error
            }
        }
    }
}

/// Use the more precise of a declared type and an inferred one.
fn refine(declared: &Ty, inferred: &Ty) -> Ty {
    match (declared, inferred) {
        (Ty::Tensor { shape, fmt, .. }, Ty::Tensor { bound: bi, shape: si, fmt: fi }) if shape == si && fmt == fi => Ty::Tensor { shape: shape.clone(), fmt: *fmt, bound: *bi },
        (Ty::Ctx { .. }, Ty::Ctx { .. }) => inferred.clone(),
        (Ty::Tuple(a), Ty::Tuple(b)) if a.len() == b.len() => Ty::Tuple(a.iter().zip(b).map(|(x, y)| refine(x, y)).collect()),
        _ => declared.clone(),
    }
}

/// Least upper bound of two compatible types.
pub fn join_ty(a: &Ty, b: &Ty) -> Ty {
    match (a, b) {
        (Ty::Str(Some(x)), Ty::Str(Some(y))) => Ty::Str(Some(*x.max(y))),
        (Ty::Str(_), Ty::Str(_)) => Ty::Str(None),
        (Ty::Tensor { shape, fmt, bound: x }, Ty::Tensor { bound: y, .. }) => Ty::Tensor { shape: shape.clone(), fmt: *fmt, bound: Bound { m: x.m.max(y.m), e: x.e.max(y.e) } },
        (Ty::Tuple(xs), Ty::Tuple(ys)) if xs.len() == ys.len() => Ty::Tuple(xs.iter().zip(ys).map(|(x, y)| join_ty(x, y)).collect()),
        (Ty::Ctx { model, spans: s1 }, Ty::Ctx { spans: s2, .. }) => Ty::Ctx { model: model.clone(), spans: s1.join(s2) },
        (Ty::Error, x) | (x, Ty::Error) => x.clone(),
        _ => a.clone(),
    }
}

fn unit_mult(s: Option<&str>) -> Option<u64> {
    Some(match s {
        None | Some("B") => 1,
        Some("KB") => 1_000,
        Some("MB") => 1_000_000,
        Some("GB") => 1_000_000_000,
        Some("TB") => 1_000_000_000_000,
        Some("KiB") => 1 << 10,
        Some("MiB") => 1 << 20,
        Some("GiB") => 1 << 30,
        Some("TiB") => 1 << 40,
        Some("tokens") | Some("tok") => 1,
        _ => return None,
    })
}

fn topo(n: &str, g: &HashMap<String, Vec<(String, Span)>>, st: &mut HashMap<String, u8>, order: &mut Vec<String>, cyclic: &mut HashSet<String>, stack: &mut Vec<String>) {
    match st.get(n) {
        Some(2) => return,
        Some(1) => {
            // Mark the whole cycle.
            if let Some(pos) = stack.iter().position(|x| x == n) {
                for x in &stack[pos..] {
                    cyclic.insert(x.clone());
                }
            }
            return;
        }
        _ => {}
    }
    st.insert(n.to_string(), 1);
    stack.push(n.to_string());
    for (c, _) in g.get(n).map(|v| v.as_slice()).unwrap_or(&[]) {
        topo(c, g, st, order, cyclic, stack);
    }
    stack.pop();
    st.insert(n.to_string(), 2);
    order.push(n.to_string());
}

fn collect_calls_block(b: &Block, out: &mut Vec<(String, Span)>) {
    for s in &b.stmts {
        match &s.kind {
            StmtKind::Let { value, .. } | StmtKind::Assign { value, .. } | StmtKind::Expr(value) | StmtKind::Commit(value) | StmtKind::Rollback(value) => collect_calls(value, out),
            StmtKind::Drop(es) => es.iter().for_each(|e| collect_calls(e, out)),
            StmtKind::Return(e) => {
                if let Some(e) = e {
                    collect_calls(e, out)
                }
            }
            StmtKind::If { cond, then, els } => {
                collect_calls(cond, out);
                collect_calls_block(then, out);
                if let Some(e) = els {
                    collect_calls_block(e, out);
                }
            }
            StmtKind::For { iter, body, .. } => {
                match iter {
                    ForIter::Range(a, b) => {
                        collect_calls(a, out);
                        collect_calls(b, out);
                    }
                    ForIter::Each(e) => collect_calls(e, out),
                }
                collect_calls_block(body, out);
            }
        }
    }
}

pub fn collect_calls(e: &Expr, out: &mut Vec<(String, Span)>) {
    match &e.kind {
        ExprKind::Call(n, args) => {
            out.push((n.name.clone(), n.span));
            args.iter().for_each(|a| collect_calls(a, out));
        }
        ExprKind::Binary(_, a, b) => {
            collect_calls(a, out);
            collect_calls(b, out);
        }
        ExprKind::Unary(_, a) | ExprKind::Field(a, _) | ExprKind::Fork(a) => collect_calls(a, out),
        ExprKind::Index(a, b) => {
            collect_calls(a, out);
            collect_calls(b, out);
        }
        ExprKind::Tuple(es) | ExprKind::List(es) => es.iter().for_each(|x| collect_calls(x, out)),
        ExprKind::Append { ctx, text, .. } | ExprKind::Splice { ctx, text, .. } => {
            collect_calls(ctx, out);
            collect_calls(text, out);
        }
        ExprKind::Gen { ctx, max, stop, .. } => {
            collect_calls(ctx, out);
            collect_calls(max, out);
            if let Some(s) = stop {
                collect_calls(s, out);
            }
        }
        ExprKind::Speculate { ctx, body } => {
            collect_calls(ctx, out);
            collect_calls_block(body, out);
        }
        _ => {}
    }
}

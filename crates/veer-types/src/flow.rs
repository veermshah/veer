//! Static information-flow control for LLM programs.
//!
//! Every value carries a label from the four-point lattice
//! `{trusted, untrusted} x {public, secret}` (integrity x confidentiality).
//! The analysis is a flow-sensitive abstract interpretation over the
//! type-checked program, in the style of Volpano, Smith and Irvine (1996) and
//! Hunt and Sands (2006), specialised to Veer's linear KV contexts:
//!
//! * A context is a chain of segments. Each segment's label is *cumulative*:
//!   the KV entries of a token are computed while attending to every earlier
//!   token, so a segment is at least as tainted as everything before it.
//! * `gen` treats the model as an arbitrary (adversarial) function of its
//!   context: the output, and the KV it appends, get the label of the whole
//!   context (the rule CaMeL and Fides use dynamically).
//! * `splice` replaces a span, but the KV of every *later* segment was computed
//!   while attending to the old content, so later segments keep its taint. A
//!   splice only removes taint when the spliced span is the last segment.
//! * `speculate { ... rollback v }` restores the exact prior KV state, so the
//!   context leaves the block with its original label; only `v` (and the
//!   decision to roll back) can carry taint out. This is a static,
//!   type-checked quarantine (the dual-LLM pattern).
//! * Control flow raises the program-counter label `pc`: values assigned,
//!   contexts extended and tools called under an untrusted branch are tainted.
//!
//! Sinks are declared on tools: `@privileged` (all arguments must be trusted
//! and the call must not be controlled by untrusted data) and `@egress` (no
//! secret data, and not controlled by secret data). `endorse(x)` upgrades a
//! value with a finite, statically known set of possible values (a
//! `choice[...]`, a `bool`, or literals chosen between branches); the number
//! of bits admitted this way is counted and can be bounded with
//! `@budget(endorse = N bits)`.

use crate::check::{const_eval, Checked, ConstVal};
use crate::ty::Ty;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use veer_syntax::ast::*;
use veer_syntax::{Diagnostic, Span};

/// A point in the label lattice, plus (for diagnostics) where the taint came from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Label {
    pub untrusted: bool,
    pub secret: bool,
    /// First untrusted source that flowed here.
    pub u_src: Option<Span>,
    /// First secret source that flowed here.
    pub s_src: Option<Span>,
}

impl Label {
    pub const BOTTOM: Label = Label { untrusted: false, secret: false, u_src: None, s_src: None };

    pub fn join(self, o: Label) -> Label {
        Label {
            untrusted: self.untrusted || o.untrusted,
            secret: self.secret || o.secret,
            u_src: if self.untrusted { self.u_src.or(o.u_src) } else { o.u_src.or(self.u_src) },
            s_src: if self.secret { self.s_src.or(o.s_src) } else { o.s_src.or(self.s_src) },
        }
    }

    /// Just the lattice point (sources dropped).
    pub fn point(self) -> (bool, bool) {
        (self.untrusted, self.secret)
    }

    pub fn describe(self) -> &'static str {
        match (self.untrusted, self.secret) {
            (false, false) => "trusted, public",
            (true, false) => "untrusted, public",
            (false, true) => "trusted, secret",
            (true, true) => "untrusted, secret",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Seg {
    name: Option<String>,
    kv: Label,
}

#[derive(Clone, Debug, PartialEq)]
enum FVal {
    /// Any non-context value. `choices` is the finite set of strings it may hold, if known.
    Data { label: Label, choices: Option<BTreeSet<String>> },
    Ctx(Vec<Seg>),
    Tuple(Vec<FVal>),
}

impl FVal {
    fn data(label: Label) -> FVal {
        FVal::Data { label, choices: None }
    }

    fn label(&self) -> Label {
        match self {
            FVal::Data { label, .. } => *label,
            FVal::Ctx(segs) => segs.iter().fold(Label::BOTTOM, |a, s| a.join(s.kv)),
            FVal::Tuple(vs) => vs.iter().fold(Label::BOTTOM, |a, v| a.join(v.label())),
        }
    }

    /// Join `l` into every leaf (data and context segments).
    fn taint(mut self, l: Label) -> FVal {
        self.taint_mut(l, true);
        self
    }

    /// Join `l` into data leaves only (the assignment rule; contexts get `pc` from the operations that extend them).
    fn taint_data(mut self, l: Label) -> FVal {
        self.taint_mut(l, false);
        self
    }

    fn taint_mut(&mut self, l: Label, ctx: bool) {
        match self {
            FVal::Data { label, .. } => *label = label.join(l),
            FVal::Ctx(segs) => {
                if ctx {
                    for s in segs {
                        s.kv = s.kv.join(l);
                    }
                }
            }
            FVal::Tuple(vs) => vs.iter_mut().for_each(|v| v.taint_mut(l, ctx)),
        }
    }

    fn segs(&self) -> Vec<Seg> {
        match self {
            FVal::Ctx(s) => s.clone(),
            other => vec![Seg { name: None, kv: other.label() }],
        }
    }

    fn join(&self, o: &FVal) -> FVal {
        match (self, o) {
            (FVal::Data { label: a, choices: ca }, FVal::Data { label: b, choices: cb }) => FVal::Data {
                label: a.join(*b),
                choices: match (ca, cb) {
                    (Some(x), Some(y)) => Some(x.union(y).cloned().collect()),
                    _ => None,
                },
            },
            (FVal::Ctx(a), FVal::Ctx(b)) => {
                let same_shape = a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.name == y.name);
                if same_shape {
                    FVal::Ctx(a.iter().zip(b).map(|(x, y)| Seg { name: x.name.clone(), kv: x.kv.join(y.kv) }).collect())
                } else {
                    // Different shapes: keep the span names both sides have, every
                    // segment at the total join (sound, and stable under loops).
                    let total = self.label().join(o.label());
                    let names_b: BTreeSet<_> = b.iter().filter_map(|s| s.name.clone()).collect();
                    let mut segs = vec![Seg { name: None, kv: total }];
                    for s in a {
                        if let Some(n) = &s.name {
                            if names_b.contains(n) {
                                segs.push(Seg { name: Some(n.clone()), kv: total });
                            }
                        }
                    }
                    FVal::Ctx(segs)
                }
            }
            (FVal::Tuple(a), FVal::Tuple(b)) if a.len() == b.len() => FVal::Tuple(a.iter().zip(b).map(|(x, y)| x.join(y)).collect()),
            _ => self.clone().taint(o.label()),
        }
    }
}

fn last_kv(segs: &[Seg]) -> Label {
    segs.iter().fold(Label::BOTTOM, |a, s| a.join(s.kv))
}

// ------------------------------------------------------------------ report

/// A call to a `@privileged` or `@egress` tool, with the labels that reach it.
#[derive(Clone, Debug)]
pub struct SinkUse {
    pub tool: String,
    pub span: Span,
    pub privileged: bool,
    pub egress: bool,
    pub pc: Label,
    pub args: Vec<(String, Label)>,
    pub ok: bool,
}

/// A call to a tool whose results are untrusted or secret.
#[derive(Clone, Debug)]
pub struct SourceUse {
    pub tool: String,
    pub span: Span,
    pub label: Label,
}

/// An `endorse` or `declassify` site.
#[derive(Clone, Debug)]
pub struct CastUse {
    pub kind: &'static str,
    pub span: Span,
    /// Number of values the endorsed expression can take (endorse only).
    pub options: Option<usize>,
    /// log2(options) when the value was untrusted, else 0.
    pub bits: f64,
    /// Upper bound on how many times this site runs per call of the entry function.
    pub times: u64,
}

/// What one entry function can do with untrusted and secret data.
#[derive(Clone, Debug)]
pub struct FnAudit {
    pub name: String,
    /// Not called by any other function.
    pub root: bool,
    pub sinks: Vec<SinkUse>,
    pub sources: Vec<SourceUse>,
    pub casts: Vec<CastUse>,
    /// Total bits of untrusted influence admitted by `endorse` (sum of bits x times).
    pub endorsed_bits: f64,
    pub budget_bits: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct FlowReport {
    pub fns: Vec<FnAudit>,
}

// ------------------------------------------------------------------ analysis

#[derive(Clone, Copy, Debug, Default)]
struct Req {
    trusted: bool,
    public: bool,
}

#[derive(Clone, Debug)]
struct ToolFlow {
    privileged: bool,
    egress: bool,
    params: Vec<(String, Req)>,
    ret: Label,
}

fn has(t: &TypeExpr, n: &str) -> bool {
    t.labels.iter().any(|l| l.name == n)
}

#[derive(Clone, PartialEq, Debug)]
struct St {
    scopes: Vec<HashMap<String, FVal>>,
    pc: Label,
    /// Raised when the rest of the function only runs because some branch did not return/commit.
    sticky: Label,
    diverged: bool,
}

impl St {
    fn eff_pc(&self) -> Label {
        self.pc.join(self.sticky)
    }
}

/// Pointwise join of two states with the same scope structure; values that
/// differ between the paths are additionally tainted with `extra` (the label
/// of the branch decision).
fn merge(a: &St, b: &St, extra: Label) -> St {
    let mut out = a.clone();
    for (i, sa) in a.scopes.iter().enumerate() {
        let Some(sb) = b.scopes.get(i) else { continue };
        for (k, va) in sa {
            if let Some(vb) = sb.get(k) {
                if va != vb {
                    out.scopes[i].insert(k.clone(), va.join(vb).taint(extra));
                }
            }
        }
    }
    out.sticky = a.sticky.join(b.sticky);
    out
}

struct Frame {
    returns: Vec<FVal>,
    ret_ty: Option<TypeExpr>,
}

struct Exit {
    st: St,
    committed: Option<FVal>,
    value: FVal,
}

struct An<'a> {
    prog: &'a Program,
    ck: &'a Checked,
    tools: HashMap<String, ToolFlow>,
    diags: Vec<Diagnostic>,
    reported: HashSet<(&'static str, Span)>,
    frames: Vec<Frame>,
    specs: Vec<Vec<Exit>>,
    calls: Vec<Span>,
    mult: u64,
    sinks: BTreeMap<(Vec<Span>, Span), SinkUse>,
    sources: BTreeMap<(Vec<Span>, Span), SourceUse>,
    casts: BTreeMap<(Vec<Span>, Span), CastUse>,
}

/// Run the information-flow analysis on a type-checked program.
pub fn analyze(prog: &Program, ck: &Checked) -> (FlowReport, Vec<Diagnostic>) {
    let mut tools = HashMap::new();
    for it in &prog.items {
        if let Item::Tool(t) = it {
            let privileged = t.attrs.iter().any(|a| a.name.name == "privileged");
            let egress = t.attrs.iter().any(|a| a.name.name == "egress");
            let params = t
                .params
                .iter()
                .map(|p| {
                    let r = Req { trusted: has(&p.ty, "trusted") || (privileged && !has(&p.ty, "untrusted")), public: has(&p.ty, "public") || (egress && !has(&p.ty, "secret")) };
                    (p.name.name.clone(), r)
                })
                .collect();
            let ret = Label { untrusted: !has(&t.ret, "trusted"), secret: has(&t.ret, "secret"), u_src: None, s_src: None };
            tools.insert(t.name.name.clone(), ToolFlow { privileged, egress, params, ret });
        }
    }
    let mut an = An {
        prog,
        ck,
        tools,
        diags: vec![],
        reported: HashSet::new(),
        frames: vec![],
        specs: vec![],
        calls: vec![],
        mult: 1,
        sinks: BTreeMap::new(),
        sources: BTreeMap::new(),
        casts: BTreeMap::new(),
    };
    // Functions that some other function calls are not roots.
    let mut called = HashSet::new();
    for it in &prog.items {
        if let Item::Fn(f) = it {
            let mut cs = Vec::new();
            crate::check::collect_calls_block(&f.body, &mut cs);
            for (n, _) in cs {
                if n != f.name.name {
                    called.insert(n);
                }
            }
        }
    }
    let mut report = FlowReport::default();
    for name in &ck.fn_order {
        let Some(f) = an.fn_decl(name) else { continue };
        an.sinks.clear();
        an.sources.clear();
        an.casts.clear();
        an.mult = 1;
        let args: Vec<FVal> = f.params.iter().map(|p| initial(&p.ty)).collect();
        an.run_fn(f, args, None, Label::BOTTOM);
        let casts: Vec<CastUse> = an.casts.values().cloned().collect();
        let endorsed_bits: f64 = casts.iter().map(|c| c.bits * c.times as f64).sum();
        let budget_bits = ck.fns.get(name).and_then(|i| i.budget.endorse_bits);
        if let Some(b) = budget_bits {
            if endorsed_bits > b as f64 + 1e-9 {
                let mut d = Diagnostic::error("V0406", f.name.span, format!("`{name}` admits up to {} bits of untrusted influence through `endorse`, budget is {b}", fmt_bits(endorsed_bits)));
                for c in casts.iter().filter(|c| c.bits > 0.0) {
                    d = d.label(c.span, format!("{} bit(s) x {}", fmt_bits(c.bits), c.times));
                }
                an.push(d);
            }
        }
        report.fns.push(FnAudit {
            name: name.clone(),
            root: !called.contains(name),
            sinks: an.sinks.values().cloned().collect(),
            sources: an.sources.values().cloned().collect(),
            casts,
            endorsed_bits,
            budget_bits,
        });
    }
    report.fns.sort_by_key(|f| f.name.clone());
    (report, an.diags)
}

pub fn fmt_bits(b: f64) -> String {
    if (b - b.round()).abs() < 1e-9 {
        format!("{}", b.round() as i64)
    } else {
        format!("{b:.2}")
    }
}

/// The value a parameter starts with when its function is analysed as an entry point:
/// trusted and public unless the type says otherwise.
fn initial(t: &TypeExpr) -> FVal {
    let l = Label { untrusted: has(t, "untrusted"), secret: has(t, "secret"), u_src: if has(t, "untrusted") { Some(t.span) } else { None }, s_src: if has(t, "secret") { Some(t.span) } else { None } };
    match &t.kind {
        TypeExprKind::Ctx(_) => FVal::Ctx(vec![Seg { name: None, kv: l }]),
        TypeExprKind::Tuple(ts) if !ts.is_empty() => FVal::Tuple(ts.iter().map(|x| initial(x).taint(l)).collect()),
        _ => FVal::data(l),
    }
}

impl<'a> An<'a> {
    fn fn_decl(&self, name: &str) -> Option<&'a FnDecl> {
        self.prog.items.iter().find_map(|i| match i {
            Item::Fn(f) if f.name.name == name => Some(f),
            _ => None,
        })
    }

    fn push(&mut self, mut d: Diagnostic) {
        if !self.reported.insert((d.code, d.span)) {
            return;
        }
        if let Some(c) = self.calls.last() {
            d = d.label(*c, "in this call");
        }
        self.diags.push(d);
    }

    fn key(&self, s: Span) -> (Vec<Span>, Span) {
        (self.calls.clone(), s)
    }

    /// Check a value against the labels written on a type (`@trusted`, `@public`), recursively through tuples.
    fn check_req(&mut self, v: &FVal, t: &TypeExpr, span: Span, what: &str) {
        let l = v.label();
        if has(t, "trusted") && l.untrusted {
            let mut d = Diagnostic::error("V0400", span, format!("{what} is marked `@trusted` but may be controlled by untrusted data"));
            if let Some(s) = l.u_src {
                d = d.label(s, "untrusted data enters here");
            }
            self.push(d);
        }
        if has(t, "public") && l.secret {
            let mut d = Diagnostic::error("V0402", span, format!("{what} is marked `@public` but may contain secret data"));
            if let Some(s) = l.s_src {
                d = d.label(s, "secret data enters here");
            }
            self.push(d);
        }
        if let (TypeExprKind::Tuple(ts), FVal::Tuple(vs)) = (&t.kind, v) {
            for (te, x) in ts.iter().zip(vs) {
                self.check_req(x, te, span, what);
            }
        }
    }

    /// Apply the labels a type declares as assumptions (`@untrusted`, `@secret`).
    fn assume(&self, v: FVal, t: &TypeExpr) -> FVal {
        let l = Label { untrusted: has(t, "untrusted"), secret: has(t, "secret"), u_src: Some(t.span), s_src: Some(t.span) };
        let l = Label { u_src: if l.untrusted { l.u_src } else { None }, s_src: if l.secret { l.s_src } else { None }, ..l };
        let v = v.taint(l);
        match (&t.kind, v) {
            (TypeExprKind::Tuple(ts), FVal::Tuple(vs)) if ts.len() == vs.len() => FVal::Tuple(vs.into_iter().zip(ts).map(|(x, te)| self.assume(x, te)).collect()),
            (_, v) => v,
        }
    }

    fn run_fn(&mut self, f: &FnDecl, args: Vec<FVal>, call: Option<Span>, pc: Label) -> FVal {
        let mut scope = HashMap::new();
        for (p, v) in f.params.iter().zip(args) {
            let v = self.assume(v, &p.ty);
            scope.insert(p.name.name.clone(), v);
        }
        if let Some(c) = call {
            self.calls.push(c);
        }
        self.frames.push(Frame { returns: vec![], ret_ty: f.ret.clone() });
        let mut st = St { scopes: vec![scope], pc, sticky: Label::BOTTOM, diverged: false };
        self.block_stmts(&mut st, &f.body);
        let fr = self.frames.pop().unwrap();
        if call.is_some() {
            self.calls.pop();
        }
        let mut it = fr.returns.into_iter();
        match it.next() {
            None => FVal::data(Label::BOTTOM),
            Some(first) => it.fold(first, |a, b| a.join(&b)),
        }
    }

    fn block(&mut self, st: &mut St, b: &Block) {
        st.scopes.push(HashMap::new());
        self.block_stmts(st, b);
        st.scopes.pop();
    }

    fn block_stmts(&mut self, st: &mut St, b: &Block) {
        for s in &b.stmts {
            if st.diverged {
                break;
            }
            self.stmt(st, s);
        }
    }

    fn bind(&mut self, st: &mut St, pat: &Pattern, v: FVal) {
        match pat {
            Pattern::Ident(id) => {
                st.scopes.last_mut().unwrap().insert(id.name.clone(), v);
            }
            Pattern::Wildcard(_) => {}
            Pattern::Tuple(ps, _) => match v {
                FVal::Tuple(vs) if vs.len() == ps.len() => {
                    for (p, x) in ps.iter().zip(vs) {
                        self.bind(st, p, x);
                    }
                }
                other => {
                    let l = other.label();
                    for p in ps {
                        self.bind(st, p, FVal::data(l));
                    }
                }
            },
        }
    }

    fn assign(&mut self, st: &mut St, pat: &Pattern, v: FVal) {
        match pat {
            Pattern::Ident(id) => {
                for s in st.scopes.iter_mut().rev() {
                    if let Some(slot) = s.get_mut(&id.name) {
                        *slot = v;
                        return;
                    }
                }
            }
            Pattern::Wildcard(_) => {}
            Pattern::Tuple(ps, _) => match v {
                FVal::Tuple(vs) if vs.len() == ps.len() => {
                    for (p, x) in ps.iter().zip(vs) {
                        self.assign(st, p, x);
                    }
                }
                other => {
                    let l = other.label();
                    for p in ps {
                        self.assign(st, p, FVal::data(l));
                    }
                }
            },
        }
    }

    fn stmt(&mut self, st: &mut St, s: &Stmt) {
        match &s.kind {
            StmtKind::Let { pat, ty, value } => {
                let mut v = self.eval(st, value).taint_data(st.eff_pc());
                if let Some(t) = ty {
                    self.check_req(&v, t, value.span, "this binding");
                    v = self.assume(v, t);
                }
                self.bind(st, pat, v);
            }
            StmtKind::Assign { pat, value } => {
                let v = self.eval(st, value).taint_data(st.eff_pc());
                self.assign(st, pat, v);
            }
            StmtKind::Expr(e) => {
                self.eval(st, e);
            }
            StmtKind::Drop(es) => {
                for e in es {
                    self.eval(st, e);
                }
            }
            StmtKind::Return(e) => {
                let v = match e {
                    Some(e) => self.eval(st, e),
                    None => FVal::data(Label::BOTTOM),
                };
                let v = v.taint(st.eff_pc());
                if let Some(rt) = self.frames.last().and_then(|f| f.ret_ty.clone()) {
                    self.check_req(&v, &rt, e.as_ref().map(|e| e.span).unwrap_or(s.span), "the return value");
                }
                if let Some(fr) = self.frames.last_mut() {
                    fr.returns.push(v);
                }
                st.diverged = true;
            }
            StmtKind::If { cond, then, els } => {
                let cl = self.eval(st, cond).label();
                let outer_pc = st.pc;
                let branch_pc = st.eff_pc().join(cl);
                let mut a = st.clone();
                a.pc = outer_pc.join(cl);
                self.block(&mut a, then);
                let mut b = st.clone();
                b.pc = outer_pc.join(cl);
                if let Some(e) = els {
                    self.block(&mut b, e);
                }
                *st = match (a.diverged, b.diverged) {
                    (true, true) => a,
                    (true, false) | (false, true) => {
                        let mut live = if a.diverged { b } else { a };
                        // The rest of the function only runs because the other branch was not taken.
                        live.sticky = live.sticky.join(branch_pc);
                        live
                    }
                    (false, false) => merge(&a, &b, branch_pc),
                };
                st.pc = outer_pc;
            }
            StmtKind::For { var, iter, body } => {
                let (il, trips) = match iter {
                    ForIter::Range(a, b) => {
                        let la = self.eval(st, a).label();
                        let lb = self.eval(st, b).label();
                        let trips = match (const_eval(&self.ck.consts, a), const_eval(&self.ck.consts, b)) {
                            (Some(ConstVal::Int(x)), Some(ConstVal::Int(y))) => (y - x).max(0) as u64,
                            _ => 1,
                        };
                        (la.join(lb), trips)
                    }
                    ForIter::Each(e) => {
                        let l = self.eval(st, e).label();
                        let trips = match self.ck.ty(e) {
                            Ty::List(_, n) => *n,
                            _ => 1,
                        };
                        (l, trips)
                    }
                };
                let outer_pc = st.pc;
                let saved_mult = self.mult;
                self.mult = self.mult.saturating_mul(trips.max(1));
                let mut cur = st.clone();
                for _ in 0..64 {
                    let mut b = cur.clone();
                    b.pc = outer_pc.join(il);
                    b.scopes.push(HashMap::new());
                    let lv = FVal::data(il.join(b.eff_pc()));
                    self.bind(&mut b, &Pattern::Ident(var.clone()), lv);
                    self.block_stmts(&mut b, body);
                    b.scopes.pop();
                    if b.diverged {
                        // The loop may exit the function; whatever follows depends on that decision.
                        b = cur.clone();
                        b.sticky = b.sticky.join(il.join(outer_pc));
                    }
                    b.pc = outer_pc;
                    let next = merge(&cur, &b, il);
                    if next == cur {
                        break;
                    }
                    cur = next;
                }
                self.mult = saved_mult;
                cur.pc = outer_pc;
                *st = cur;
            }
            StmtKind::Commit(e) => {
                let v = self.eval(st, e);
                let pc = st.eff_pc();
                let (c, x) = match v {
                    FVal::Tuple(mut vs) if vs.len() == 2 => {
                        let x = vs.pop().unwrap();
                        (vs.pop().unwrap(), x)
                    }
                    other => (FVal::Ctx(other.segs()), FVal::data(other.label())),
                };
                if let Some(ex) = self.specs.last_mut() {
                    ex.push(Exit { st: st.clone(), committed: Some(c.taint(pc)), value: x.taint(pc) });
                }
                st.diverged = true;
            }
            StmtKind::Rollback(e) => {
                let v = self.eval(st, e).taint(st.eff_pc());
                if let Some(ex) = self.specs.last_mut() {
                    ex.push(Exit { st: st.clone(), committed: None, value: v });
                }
                st.diverged = true;
            }
        }
    }

    fn lookup(&self, st: &St, name: &str) -> Option<FVal> {
        st.scopes.iter().rev().find_map(|s| s.get(name).cloned())
    }

    fn eval(&mut self, st: &mut St, e: &Expr) -> FVal {
        match &e.kind {
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) => FVal::data(Label::BOTTOM),
            ExprKind::Str(s) => FVal::Data { label: Label::BOTTOM, choices: Some(BTreeSet::from([s.clone()])) },
            ExprKind::Var(id) => self.lookup(st, &id.name).unwrap_or(FVal::data(Label::BOTTOM)),
            ExprKind::Binary(_, a, b) => {
                let l = self.eval(st, a).label().join(self.eval(st, b).label());
                FVal::data(l)
            }
            ExprKind::Unary(_, x) | ExprKind::Field(x, _) => FVal::data(self.eval(st, x).label()),
            ExprKind::Index(a, b) => {
                let l = self.eval(st, a).label().join(self.eval(st, b).label());
                FVal::data(l)
            }
            ExprKind::Tuple(es) => {
                if es.is_empty() {
                    FVal::data(Label::BOTTOM)
                } else {
                    FVal::Tuple(es.iter().map(|x| self.eval(st, x)).collect())
                }
            }
            ExprKind::List(_) => FVal::data(Label::BOTTOM),
            ExprKind::Call(name, args) => self.call(st, name, args, e.span),
            ExprKind::Open(_) => FVal::Ctx(vec![Seg { name: None, kv: st.eff_pc() }]),
            ExprKind::Append { ctx, label, text } => {
                let mut segs = self.eval(st, ctx).segs();
                let t = self.eval(st, text).label();
                let kv = last_kv(&segs).join(t).join(st.eff_pc());
                segs.push(Seg { name: label.as_ref().map(|l| l.name.clone()), kv });
                FVal::Ctx(segs)
            }
            ExprKind::Fork(c) => {
                let c = FVal::Ctx(self.eval(st, c).segs());
                FVal::Tuple(vec![c.clone(), c])
            }
            ExprKind::Gen { ctx, max, constraint, stop } => {
                let mut segs = self.eval(st, ctx).segs();
                self.eval(st, max);
                let sl = match stop {
                    Some(s) => self.eval(st, s).label(),
                    None => Label::BOTTOM,
                };
                // The model is an arbitrary function of its whole context.
                let kv = last_kv(&segs).join(st.eff_pc()).join(sl);
                segs.push(Seg { name: None, kv });
                let choices = match constraint {
                    Some(Constraint::Choice(opts)) => Some(opts.iter().cloned().collect()),
                    _ => None,
                };
                FVal::Tuple(vec![FVal::Ctx(segs), FVal::Data { label: kv, choices }])
            }
            ExprKind::Splice { ctx, span_name, new_label, text } => {
                let mut segs = self.eval(st, ctx).segs();
                let t = self.eval(st, text).label().join(st.eff_pc());
                match segs.iter().position(|s| s.name.as_deref() == Some(span_name.name.as_str())) {
                    Some(i) => {
                        let before = last_kv(&segs[..i]);
                        let kv = before.join(t);
                        segs[i] = Seg { name: new_label.as_ref().map(|l| l.name.clone()), kv };
                        // Later KV was computed while attending to the old span: it keeps that
                        // taint (already folded in, cumulatively) and gains the new text's.
                        for s in &mut segs[i + 1..] {
                            s.kv = s.kv.join(kv);
                        }
                    }
                    None => {
                        let total = last_kv(&segs).join(t);
                        for s in &mut segs {
                            s.kv = total;
                        }
                        if let Some(l) = new_label {
                            segs.push(Seg { name: Some(l.name.clone()), kv: total });
                        }
                    }
                }
                FVal::Ctx(segs)
            }
            ExprKind::Speculate { ctx, body } => {
                let c = FVal::Ctx(self.eval(st, ctx).segs());
                let ExprKind::Var(name) = &ctx.kind else { return c };
                let depth = st.scopes.len();
                let entry = st.clone();
                self.specs.push(Vec::new());
                let mut inner = st.clone();
                inner.scopes.push(HashMap::from([(name.name.clone(), c.clone())]));
                inner.sticky = entry.sticky;
                self.block_stmts(&mut inner, body);
                let exits = self.specs.pop().unwrap();
                if exits.is_empty() {
                    return FVal::Tuple(vec![c, FVal::data(Label::BOTTOM)]);
                }
                let decision = exits.iter().fold(Label::BOTTOM, |a, ex| a.join(ex.st.eff_pc()));
                let mut ctx_out: Option<FVal> = None;
                let mut val_out: Option<FVal> = None;
                let mut st_out: Option<St> = None;
                for ex in exits {
                    // Rollback restores the exact prior KV state whatever path led there. A
                    // committed context already carries the decision's label, so mixing
                    // commit and rollback paths still taints the result through the join.
                    let cv = ex.committed.unwrap_or_else(|| c.clone());
                    ctx_out = Some(match ctx_out {
                        None => cv,
                        Some(p) => p.join(&cv),
                    });
                    val_out = Some(match val_out {
                        None => ex.value,
                        Some(p) => p.join(&ex.value),
                    });
                    let mut s = ex.st;
                    s.scopes.truncate(depth);
                    st_out = Some(match st_out {
                        None => s,
                        Some(p) => merge(&p, &s, decision),
                    });
                }
                let mut s = st_out.unwrap();
                s.pc = entry.pc;
                s.sticky = entry.sticky;
                s.diverged = false;
                *st = s;
                FVal::Tuple(vec![ctx_out.unwrap(), val_out.unwrap()])
            }
        }
    }

    fn call(&mut self, st: &mut St, name: &Ident, args: &[Expr], span: Span) -> FVal {
        let n = name.name.as_str();
        if let Some(f) = self.fn_decl(n).filter(|_| self.ck.fns.contains_key(n)) {
            let mut vals = Vec::new();
            for (a, p) in args.iter().zip(&f.params) {
                let v = self.eval(st, a);
                self.check_req(&v, &p.ty, a.span, &format!("argument `{}` of `{n}`", p.name.name));
                vals.push(v);
            }
            return self.run_fn(f, vals, Some(span), st.eff_pc());
        }
        if let Some(tf) = self.tools.get(n).cloned() {
            let pc = st.eff_pc();
            let mut joined = Label::BOTTOM;
            let mut arg_labels = Vec::new();
            let mut ok = true;
            for (a, (pname, req)) in args.iter().zip(&tf.params) {
                let l = self.eval(st, a).label();
                joined = joined.join(l);
                arg_labels.push((pname.clone(), l));
                if req.trusted && l.untrusted {
                    ok = false;
                    let why = if tf.privileged { format!("`{n}` is privileged") } else { format!("parameter `{pname}` is `@trusted`") };
                    let mut d = Diagnostic::error("V0400", a.span, format!("argument `{pname}` of `{n}` may be controlled by untrusted data ({why})"));
                    if let Some(s) = l.u_src {
                        d = d.label(s, "untrusted data enters here");
                    }
                    self.push(d.note("a prompt injection in that data could choose this value; `endorse` a constrained choice or take the value from trusted input"));
                }
                if req.public && l.secret {
                    ok = false;
                    let why = if tf.egress { format!("`{n}` is egress") } else { format!("parameter `{pname}` is `@public`") };
                    let mut d = Diagnostic::error("V0402", a.span, format!("argument `{pname}` of `{n}` may contain secret data ({why})"));
                    if let Some(s) = l.s_src {
                        d = d.label(s, "secret data enters here");
                    }
                    self.push(d.note("this would let an attacker exfiltrate it; `declassify` trusted data if releasing it is intended"));
                }
            }
            if tf.privileged && pc.untrusted {
                ok = false;
                let mut d = Diagnostic::error("V0401", span, format!("whether privileged tool `{n}` runs depends on untrusted data"));
                if let Some(s) = pc.u_src {
                    d = d.label(s, "untrusted data enters here");
                }
                self.push(d.note("decide on trusted data, or `endorse` a constrained choice first"));
            }
            if tf.egress && pc.secret {
                ok = false;
                let mut d = Diagnostic::error("V0403", span, format!("whether egress tool `{n}` runs depends on secret data"));
                if let Some(s) = pc.s_src {
                    d = d.label(s, "secret data enters here");
                }
                self.push(d);
            }
            let key = self.key(span);
            if tf.privileged || tf.egress {
                let entry = self.sinks.entry(key.clone()).or_insert(SinkUse { tool: n.to_string(), span, privileged: tf.privileged, egress: tf.egress, pc: Label::BOTTOM, args: arg_labels.iter().map(|(p, _)| (p.clone(), Label::BOTTOM)).collect(), ok: true });
                entry.pc = entry.pc.join(pc);
                for (slot, (_, l)) in entry.args.iter_mut().zip(&arg_labels) {
                    slot.1 = slot.1.join(*l);
                }
                entry.ok &= ok;
            }
            let src = Label { u_src: if tf.ret.untrusted { Some(span) } else { None }, s_src: if tf.ret.secret { Some(span) } else { None }, ..tf.ret };
            if (tf.ret.untrusted || tf.ret.secret) && !(tf.privileged || tf.egress) {
                self.sources.entry(key).or_insert(SourceUse { tool: n.to_string(), span, label: src });
            }
            return FVal::data(src.join(joined));
        }
        match n {
            "endorse" | "declassify" if args.len() == 1 => {
                let v = self.eval(st, &args[0]);
                let l = v.label();
                let pc = st.eff_pc();
                if pc.untrusted {
                    let mut d = Diagnostic::error("V0405", span, format!("`{n}` runs on a path that depends on untrusted data"));
                    if let Some(s) = pc.u_src {
                        d = d.label(s, "untrusted data enters here");
                    }
                    self.push(d.note("robust declassification: an attacker must not decide what gets upgraded"));
                }
                let choices = match &v {
                    FVal::Data { choices, .. } => choices.clone(),
                    _ => None,
                };
                if n == "endorse" {
                    let options = match (&choices, self.ck.ty(&args[0])) {
                        (Some(c), _) => Some(c.len()),
                        (None, Ty::Bool) => Some(2),
                        _ => None,
                    };
                    if l.untrusted && options.is_none() {
                        let mut d = Diagnostic::error("V0404", args[0].span, "cannot endorse untrusted data that can take unboundedly many values");
                        if let Some(s) = l.u_src {
                            d = d.label(s, "untrusted data enters here");
                        }
                        self.push(d.note("constrain it first, e.g. `gen c ... as choice[\"archive\", \"reply\"]`, and endorse the choice"));
                    }
                    let bits = if l.untrusted { options.map(|k| (k.max(1) as f64).log2()).unwrap_or(f64::INFINITY) } else { 0.0 };
                    let key = self.key(span);
                    let mult = self.mult;
                    let c = self.casts.entry(key).or_insert(CastUse { kind: "endorse", span, options, bits: 0.0, times: mult });
                    c.bits = c.bits.max(if bits.is_finite() { bits } else { 0.0 });
                    c.times = c.times.max(mult);
                    FVal::Data { label: Label { untrusted: false, u_src: None, ..l }, choices }
                } else {
                    if l.untrusted {
                        let mut d = Diagnostic::error("V0407", args[0].span, "cannot declassify untrusted data");
                        if let Some(s) = l.u_src {
                            d = d.label(s, "untrusted data enters here");
                        }
                        self.push(d.note("an attacker could choose what gets released"));
                    }
                    let key = self.key(span);
                    let mult = self.mult;
                    self.casts.entry(key).or_insert(CastUse { kind: "declassify", span, options: None, bits: 0.0, times: mult });
                    FVal::Data { label: Label { secret: false, s_src: None, ..l }, choices }
                }
            }
            _ => {
                let mut l = Label::BOTTOM;
                for a in args {
                    if matches!(a.kind, ExprKind::List(_)) {
                        continue;
                    }
                    l = l.join(self.eval(st, a).label());
                }
                FVal::data(l)
            }
        }
    }
}

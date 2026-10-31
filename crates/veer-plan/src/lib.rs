//! The Veer static planner.
//!
//! An abstract interpreter over a type-checked program. Every context is
//! modelled as a chain of KV *segments* with token upper bounds; `fork` shares
//! segments, `++`/`gen` add segments, `splice` copies the tail. Walking all
//! paths (both sides of every `if`, every loop iteration, every speculation
//! outcome, every inlined call) yields sound upper bounds on:
//!
//! * peak resident KV-cache bytes (segment-aligned paging, page = model `block`),
//! * prompt tokens prefilled and tokens decoded,
//! * re-prefill cost of each `splice`,
//! * the length of every context (checked against `max_ctx`).
//!
//! It also emits the context DAG and per-segment reuse steps, which a serving
//! engine can use for eviction (as in KVFlow's steps-to-execution).

use serde::Serialize;
use std::collections::{BTreeSet, HashMap, HashSet};
use veer_syntax::ast::*;
use veer_syntax::{Diagnostic, Source, Span};
use veer_types::check::tokens_of;
use veer_types::{Checked, Ty};

pub type SegId = u64;
pub type CtxId = u64;

#[derive(Clone, Debug, Serialize)]
pub struct Seg {
    pub id: SegId,
    pub model: String,
    pub kind: &'static str,
    pub tokens: u64,
    /// Tokens of page-aligned allocation (≥ tokens).
    pub alloc: u64,
    pub labels: BTreeSet<String>,
    #[serde(skip)]
    pub pinned: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct ACtx {
    model: String,
    segs: Vec<SegId>,
}

#[derive(Clone, Debug)]
enum AVal {
    Ctx(CtxId),
    Str(u64),
    Int(Option<i64>),
    Tuple(Vec<AVal>),
    Other,
}

#[derive(Clone, Default)]
struct State {
    scopes: Vec<HashMap<String, AVal>>,
    ctxs: HashMap<CtxId, ACtx>,
    hidden: Vec<CtxId>,
    prefill: u64,
    decode: u64,
    diverged: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct TraceOp {
    pub step: usize,
    pub op: &'static str,
    pub line: usize,
    pub detail: String,
    pub inputs: Vec<CtxId>,
    pub outputs: Vec<CtxId>,
    pub tokens: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct LiveCtx {
    pub var: String,
    pub tokens: u64,
    pub shared_tokens: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SpliceCost {
    pub line: usize,
    pub span_label: String,
    pub reprefill_tokens: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ForkShare {
    pub line: usize,
    pub shared_tokens: u64,
    pub saved_bytes: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReuseHint {
    pub segment: SegId,
    pub kind: &'static str,
    pub tokens: u64,
    pub use_steps: Vec<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelSummary {
    pub name: String,
    pub kv_format: String,
    pub kv_bytes_per_token: f64,
    pub block: u64,
    pub max_ctx: Option<u64>,
    pub longest_ctx_tokens: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Plan {
    pub entry: String,
    pub peak_kv_bytes: f64,
    pub peak_line: Option<usize>,
    pub live_at_peak: Vec<LiveCtx>,
    pub resident_prefix_bytes: f64,
    pub prefill_tokens: u64,
    pub decode_tokens: u64,
    pub budget_kv_bytes: Option<u64>,
    pub budget_prefill: Option<u64>,
    pub models: Vec<ModelSummary>,
    pub forks: Vec<ForkShare>,
    pub splices: Vec<SpliceCost>,
    pub trace: Vec<TraceOp>,
    pub reuse_hints: Vec<ReuseHint>,
    pub segments: Vec<Seg>,
    #[serde(skip)]
    pub diags: Vec<Diagnostic>,
}

const MAX_STEPS: usize = 2_000_000;

struct Planner<'a> {
    prog: &'a Program,
    ck: &'a Checked,
    src: &'a Source,
    segs: HashMap<SegId, Seg>,
    next_id: u64,
    prefix_segs: HashMap<String, SegId>,
    peak: f64,
    peak_span: Option<Span>,
    live_at_peak: Vec<LiveCtx>,
    trace: Vec<TraceOp>,
    seg_uses: HashMap<SegId, Vec<usize>>,
    forks: Vec<ForkShare>,
    splices: HashMap<Span, SpliceCost>,
    longest: HashMap<String, u64>,
    diags: Vec<Diagnostic>,
    reported: HashSet<(Span, &'static str)>,
    steps: usize,
    aborted: bool,
    /// Results collected by `return` / `commit` / `rollback`.
    returns: Vec<Vec<(State, AVal)>>,
    specs: Vec<(Vec<(State, AVal)>, usize, CtxId)>,
}

/// Plan one function as an entry point.
pub fn plan_fn(prog: &Program, ck: &Checked, src: &Source, name: &str) -> Option<Plan> {
    let f = prog.items.iter().find_map(|i| match i {
        Item::Fn(f) if f.name.name == name => Some(f),
        _ => None,
    })?;
    let info = ck.fns.get(name)?;
    let mut p = Planner {
        prog,
        ck,
        src,
        segs: HashMap::new(),
        next_id: 1,
        prefix_segs: HashMap::new(),
        peak: 0.0,
        peak_span: None,
        live_at_peak: vec![],
        trace: vec![],
        seg_uses: HashMap::new(),
        forks: vec![],
        splices: HashMap::new(),
        longest: HashMap::new(),
        diags: vec![],
        reported: HashSet::new(),
        steps: 0,
        aborted: false,
        returns: vec![],
        specs: vec![],
    };
    let mut st = State::default();
    // Prefixes are registered once and stay resident (they are shared roots).
    let mut prefixes: Vec<_> = ck.prefixes.values().collect();
    prefixes.sort_by(|a, b| a.name.cmp(&b.name));
    for pre in prefixes {
        let model = &ck.models[&pre.model];
        let t = 1 + tokens_of(&pre.text);
        let id = p.new_seg(&pre.model, "prefix", t, model.block, BTreeSet::new());
        p.segs.get_mut(&id).unwrap().pinned = true;
        p.prefix_segs.insert(pre.name.clone(), id);
        st.prefill += t;
    }
    let resident_prefix_bytes = p.memory(&st);
    // Parameters: strings at their bounds, contexts as opaque external handles.
    let mut scope = HashMap::new();
    for (pname, ty) in &info.sig.params {
        let v = match ty {
            Ty::Ctx { model, .. } => {
                let s = p.new_seg(model, "external", 0, 1, BTreeSet::new());
                let c = p.new_ctx(&mut st, model, vec![s]);
                AVal::Ctx(c)
            }
            t => aval_of_ty(t),
        };
        scope.insert(pname.clone(), v);
    }
    st.scopes.push(scope);
    p.returns.push(vec![]);
    p.observe(&st, f.span);
    p.block(&mut st, &f.body);
    let mut outs = p.returns.pop().unwrap();
    if !st.diverged {
        outs.push((st, AVal::Other));
    }
    let final_state = p.join_all(outs.into_iter().map(|(s, v)| (s, v)).collect());
    if let Some((s, _)) = &final_state {
        p.observe(s, f.span);
    }
    if p.aborted {
        p.diags.push(Diagnostic::error("V0200", f.name.span, "program too large to plan (more than 2,000,000 abstract steps)"));
    }
    let (prefill, decode) = final_state.as_ref().map(|(s, _)| (s.prefill, s.decode)).unwrap_or((0, 0));

    // Budgets.
    if let Some(b) = info.budget.kv_bytes {
        if p.peak > b as f64 {
            let live: Vec<String> = p.live_at_peak.iter().map(|l| format!("{} ({} tokens)", l.var, l.tokens)).collect();
            let mut d = Diagnostic::error("V0200", f.name.span, format!("worst-case KV memory {} exceeds the budget of {}", human_bytes(p.peak), human_bytes(b as f64)));
            if let Some(sp) = p.peak_span {
                d = d.label(sp, "peak reached here");
            }
            p.diags.push(d.note(format!("live at peak: {}", if live.is_empty() { "-".into() } else { live.join(", ") })));
        }
    }
    if let Some(b) = info.budget.prefill {
        if prefill > b {
            p.diags.push(Diagnostic::error("V0201", f.name.span, format!("worst-case prefill of {prefill} tokens exceeds the budget of {b}")));
        }
    }
    let mut splices: Vec<SpliceCost> = p.splices.values().cloned().collect();
    splices.sort_by_key(|s| s.line);
    let mut models: Vec<ModelSummary> = ck
        .models
        .values()
        .map(|m| ModelSummary {
            name: m.name.clone(),
            kv_format: m.kv_fmt.name().to_string(),
            kv_bytes_per_token: m.kv_bytes_per_token(),
            block: m.block,
            max_ctx: m.max_ctx,
            longest_ctx_tokens: p.longest.get(&m.name).copied().unwrap_or(0),
        })
        .collect();
    models.sort_by(|a, b| a.name.cmp(&b.name));
    let mut hints: Vec<ReuseHint> = p
        .seg_uses
        .iter()
        .filter(|(_, u)| u.len() > 1)
        .map(|(id, u)| {
            let s = &p.segs[id];
            ReuseHint { segment: *id, kind: s.kind, tokens: s.tokens, use_steps: u.clone() }
        })
        .collect();
    hints.sort_by_key(|h| h.segment);
    let mut segments: Vec<Seg> = p.segs.values().cloned().collect();
    segments.sort_by_key(|s| s.id);
    // Dedupe fork reports per line (unrolled loops repeat them): keep the largest.
    let mut fork_by_line: HashMap<usize, ForkShare> = HashMap::new();
    for fs in p.forks.drain(..) {
        let e = fork_by_line.entry(fs.line).or_insert(fs.clone());
        if fs.shared_tokens > e.shared_tokens {
            *e = fs;
        }
    }
    let mut forks: Vec<ForkShare> = fork_by_line.into_values().collect();
    forks.sort_by_key(|f| f.line);
    Some(Plan {
        entry: name.to_string(),
        peak_kv_bytes: p.peak,
        peak_line: p.peak_span.map(|s| src.line_col(s.start).0 + 1),
        live_at_peak: p.live_at_peak.clone(),
        resident_prefix_bytes,
        prefill_tokens: prefill,
        decode_tokens: decode,
        budget_kv_bytes: info.budget.kv_bytes,
        budget_prefill: info.budget.prefill,
        models,
        forks,
        splices,
        trace: p.trace,
        reuse_hints: hints,
        segments,
        diags: p.diags,
    })
}

/// Plan every function that declares a `@budget`, plus `main`. Returns the
/// plans and all planner diagnostics.
pub fn plan_program(prog: &Program, ck: &Checked, src: &Source) -> (Vec<Plan>, Vec<Diagnostic>) {
    let mut plans = Vec::new();
    let mut diags = Vec::new();
    for name in &ck.fn_order {
        let info = &ck.fns[name];
        if name == "main" || info.budget.kv_bytes.is_some() || info.budget.prefill.is_some() {
            if let Some(p) = plan_fn(prog, ck, src, name) {
                diags.extend(p.diags.iter().cloned());
                plans.push(p);
            }
        }
    }
    // Same warning may be produced for several entries.
    let mut seen = HashSet::new();
    diags.retain(|d| seen.insert((d.code, d.span, d.message.clone())));
    (plans, diags)
}

fn aval_of_ty(t: &Ty) -> AVal {
    match t {
        Ty::Str(b) => AVal::Str(b.unwrap_or(0)),
        Ty::Int => AVal::Int(None),
        Ty::Tuple(ts) => AVal::Tuple(ts.iter().map(aval_of_ty).collect()),
        _ => AVal::Other,
    }
}

fn round_up(t: u64, block: u64) -> u64 {
    t.div_ceil(block) * block
}

impl<'a> Planner<'a> {
    fn line(&self, s: Span) -> usize {
        self.src.line_col(s.start).0 + 1
    }

    fn new_seg(&mut self, model: &str, kind: &'static str, tokens: u64, block: u64, labels: BTreeSet<String>) -> SegId {
        let id = self.next_id;
        self.next_id += 1;
        self.segs.insert(id, Seg { id, model: model.to_string(), kind, tokens, alloc: round_up(tokens, block), labels, pinned: false });
        id
    }

    fn new_ctx(&mut self, st: &mut State, model: &str, segs: Vec<SegId>) -> CtxId {
        let id = self.next_id;
        self.next_id += 1;
        st.ctxs.insert(id, ACtx { model: model.to_string(), segs });
        id
    }

    fn block_of(&self, model: &str) -> u64 {
        self.ck.models[model].block
    }

    fn memory(&self, st: &State) -> f64 {
        let mut seen = HashSet::new();
        let mut bytes = 0.0;
        let mut add = |sid: SegId, seen: &mut HashSet<SegId>| {
            if seen.insert(sid) {
                let s = &self.segs[&sid];
                bytes += s.alloc as f64 * self.ck.models[&s.model].kv_bytes_per_token();
            }
        };
        for &sid in self.prefix_segs.values() {
            add(sid, &mut seen);
        }
        for c in st.ctxs.values() {
            for &sid in &c.segs {
                add(sid, &mut seen);
            }
        }
        bytes
    }

    fn ctx_tokens(&self, c: &ACtx) -> u64 {
        c.segs.iter().map(|s| self.segs[s].tokens).sum()
    }

    fn observe(&mut self, st: &State, span: Span) {
        let m = self.memory(st);
        if m > self.peak {
            self.peak = m;
            self.peak_span = Some(span);
            // Describe live contexts by variable name where possible.
            let mut names: HashMap<CtxId, String> = HashMap::new();
            for sc in &st.scopes {
                for (n, v) in sc {
                    collect_ctx_names(n, v, &mut names);
                }
            }
            for h in &st.hidden {
                names.entry(*h).or_insert_with(|| "(speculation original)".into());
            }
            let mut counts: HashMap<SegId, usize> = HashMap::new();
            for c in st.ctxs.values() {
                for s in &c.segs {
                    *counts.entry(*s).or_default() += 1;
                }
            }
            let mut live: Vec<LiveCtx> = st
                .ctxs
                .iter()
                .map(|(id, c)| LiveCtx {
                    var: names.get(id).cloned().unwrap_or_else(|| "(temporary)".into()),
                    tokens: self.ctx_tokens(c),
                    shared_tokens: c.segs.iter().filter(|s| counts[s] > 1 || self.segs[s].pinned).map(|s| self.segs[s].tokens).sum(),
                })
                .collect();
            live.sort_by(|a, b| a.var.cmp(&b.var));
            self.live_at_peak = live;
        }
    }

    fn check_len(&mut self, c: &ACtx, span: Span) {
        let t = self.ctx_tokens(c);
        let e = self.longest.entry(c.model.clone()).or_default();
        *e = (*e).max(t);
        if let Some(max) = self.ck.models[&c.model].max_ctx {
            if t > max && self.reported.insert((span, "V0202")) {
                self.diags.push(Diagnostic::error("V0202", span, format!("this context may reach {t} tokens, beyond max_ctx = {max} for `{}`", c.model)));
            }
        }
    }

    fn record(&mut self, op: &'static str, span: Span, detail: String, inputs: Vec<CtxId>, outputs: Vec<CtxId>, tokens: u64, st: &State) {
        let step = self.trace.len();
        for o in &outputs {
            if let Some(c) = st.ctxs.get(o) {
                for s in &c.segs {
                    let u = self.seg_uses.entry(*s).or_default();
                    if u.last() != Some(&step) {
                        u.push(step);
                    }
                }
            }
        }
        if self.trace.len() < 5000 {
            self.trace.push(TraceOp { step, op, line: self.line(span), detail, inputs, outputs, tokens });
        }
    }

    fn take_ctx(&mut self, st: &mut State, v: &AVal) -> Option<(CtxId, ACtx)> {
        match v {
            AVal::Ctx(id) => st.ctxs.remove(id).map(|c| (*id, c)),
            _ => None,
        }
    }

    fn tick(&mut self) -> bool {
        self.steps += 1;
        if self.steps > MAX_STEPS {
            self.aborted = true;
        }
        !self.aborted
    }

    // ------------------------------------------------------------ joins

    fn join_all(&mut self, outs: Vec<(State, AVal)>) -> Option<(State, AVal)> {
        let mut it = outs.into_iter();
        let first = it.next()?;
        Some(it.fold(first, |(sa, va), (sb, vb)| self.join(sa, va, sb, vb)))
    }

    fn join(&mut self, a: State, va: AVal, b: State, vb: AVal) -> (State, AVal) {
        let mut out = State { scopes: vec![], ctxs: HashMap::new(), hidden: vec![], prefill: a.prefill.max(b.prefill), decode: a.decode.max(b.decode), diverged: false };
        for (id, c) in &a.ctxs {
            if b.ctxs.get(id) == Some(c) {
                out.ctxs.insert(*id, c.clone());
            }
        }
        let mut memo = HashMap::new();
        for (sa, sb) in a.scopes.iter().zip(&b.scopes) {
            let mut scope = HashMap::new();
            for (n, x) in sa {
                let v = match sb.get(n) {
                    Some(y) => self.join_val(x, y, &a, &b, &mut out, &mut memo),
                    None => x.clone(),
                };
                scope.insert(n.clone(), v);
            }
            out.scopes.push(scope);
        }
        for (x, y) in a.hidden.iter().zip(&b.hidden) {
            if let AVal::Ctx(id) = self.join_val(&AVal::Ctx(*x), &AVal::Ctx(*y), &a, &b, &mut out, &mut memo) {
                out.hidden.push(id);
            }
        }
        let v = self.join_val(&va, &vb, &a, &b, &mut out, &mut memo);
        (out, v)
    }

    fn join_val(&mut self, x: &AVal, y: &AVal, a: &State, b: &State, out: &mut State, memo: &mut HashMap<(CtxId, CtxId), CtxId>) -> AVal {
        match (x, y) {
            (AVal::Ctx(i), AVal::Ctx(j)) => {
                let (ci, cj) = (a.ctxs.get(i), b.ctxs.get(j));
                match (ci, cj) {
                    (Some(ci), Some(cj)) => {
                        if i == j {
                            out.ctxs.insert(*i, ci.clone());
                            return AVal::Ctx(*i);
                        }
                        if let Some(id) = memo.get(&(*i, *j)) {
                            return AVal::Ctx(*id);
                        }
                        let common = ci.segs.iter().zip(&cj.segs).take_while(|(p, q)| p == q).count();
                        let rest = |c: &ACtx| -> (u64, u64, BTreeSet<String>) {
                            let mut t = 0;
                            let mut al = 0;
                            let mut labels = BTreeSet::new();
                            for s in &c.segs[common..] {
                                let s = &self.segs[s];
                                t += s.tokens;
                                al += s.alloc;
                                labels.extend(s.labels.iter().cloned());
                            }
                            (t, al, labels)
                        };
                        let (t1, a1, l1) = rest(ci);
                        let (t2, a2, l2) = rest(cj);
                        let mut segs = ci.segs[..common].to_vec();
                        if t1 > 0 || t2 > 0 || a1 > 0 || a2 > 0 {
                            let id = self.next_id;
                            self.next_id += 1;
                            let labels: BTreeSet<String> = l1.union(&l2).cloned().collect();
                            self.segs.insert(id, Seg { id, model: ci.model.clone(), kind: "join", tokens: t1.max(t2), alloc: a1.max(a2), labels, pinned: false });
                            segs.push(id);
                        }
                        let cid = self.next_id;
                        self.next_id += 1;
                        out.ctxs.insert(cid, ACtx { model: ci.model.clone(), segs });
                        memo.insert((*i, *j), cid);
                        AVal::Ctx(cid)
                    }
                    (Some(ci), None) => {
                        out.ctxs.insert(*i, ci.clone());
                        AVal::Ctx(*i)
                    }
                    (None, Some(cj)) => {
                        out.ctxs.insert(*j, cj.clone());
                        AVal::Ctx(*j)
                    }
                    (None, None) => x.clone(),
                }
            }
            (AVal::Str(p), AVal::Str(q)) => AVal::Str(*p.max(q)),
            (AVal::Int(p), AVal::Int(q)) => AVal::Int(if p == q { *p } else { None }),
            (AVal::Tuple(ps), AVal::Tuple(qs)) if ps.len() == qs.len() => AVal::Tuple(ps.iter().zip(qs).map(|(p, q)| self.join_val(p, q, a, b, out, memo)).collect()),
            _ => x.clone(),
        }
    }

    // ------------------------------------------------------------ statements

    fn block(&mut self, st: &mut State, b: &Block) {
        st.scopes.push(HashMap::new());
        for s in &b.stmts {
            if st.diverged || self.aborted {
                break;
            }
            self.stmt(st, s);
        }
        st.scopes.pop();
    }

    fn bind(&mut self, st: &mut State, pat: &Pattern, v: AVal) {
        match (pat, v) {
            (Pattern::Ident(id), v) => {
                st.scopes.last_mut().unwrap().insert(id.name.clone(), v);
            }
            (Pattern::Tuple(ps, _), AVal::Tuple(vs)) => {
                for (p, v) in ps.iter().zip(vs) {
                    self.bind(st, p, v);
                }
            }
            (Pattern::Tuple(ps, _), _) => {
                for p in ps {
                    self.bind(st, p, AVal::Other);
                }
            }
            (Pattern::Wildcard(_), _) => {}
        }
    }

    fn assign(&mut self, st: &mut State, pat: &Pattern, v: AVal) {
        match (pat, v) {
            (Pattern::Ident(id), v) => {
                for sc in st.scopes.iter_mut().rev() {
                    if let Some(slot) = sc.get_mut(&id.name) {
                        *slot = v;
                        return;
                    }
                }
            }
            (Pattern::Tuple(ps, _), AVal::Tuple(vs)) => {
                for (p, v) in ps.iter().zip(vs) {
                    self.assign(st, p, v);
                }
            }
            _ => {}
        }
    }

    fn stmt(&mut self, st: &mut State, s: &Stmt) {
        if !self.tick() {
            return;
        }
        match &s.kind {
            StmtKind::Let { pat, value, .. } => {
                let v = self.eval(st, value);
                self.bind(st, pat, v);
            }
            StmtKind::Assign { pat, value } => {
                let v = self.eval(st, value);
                self.assign(st, pat, v);
            }
            StmtKind::Expr(e) => {
                self.eval(st, e);
            }
            StmtKind::Drop(es) => {
                for e in es {
                    let v = self.eval(st, e);
                    if let Some((id, c)) = self.take_ctx(st, &v) {
                        let t = self.ctx_tokens(&c);
                        self.record("drop", e.span, String::new(), vec![id], vec![], t, st);
                    }
                }
            }
            StmtKind::Return(e) => {
                let v = e.as_ref().map(|e| self.eval(st, e)).unwrap_or(AVal::Other);
                let snapshot = st.clone();
                self.returns.last_mut().unwrap().push((snapshot, v));
                st.diverged = true;
            }
            StmtKind::If { cond, then, els } => {
                self.eval(st, cond);
                let mut a = st.clone();
                self.block(&mut a, then);
                let mut b = std::mem::take(st);
                if let Some(e) = els {
                    self.block(&mut b, e);
                }
                *st = match (a.diverged, b.diverged) {
                    (true, true) => {
                        let mut s = a;
                        s.diverged = true;
                        s
                    }
                    (true, false) => b,
                    (false, true) => a,
                    (false, false) => {
                        let (s, _) = self.join(a, AVal::Other, b, AVal::Other);
                        self.observe(&s, s_span(s.scopes.len(), cond.span));
                        s
                    }
                };
            }
            StmtKind::For { var, iter, body } => match iter {
                ForIter::Range(a, b) => {
                    self.eval(st, a);
                    self.eval(st, b);
                    let lo = self.ck.const_eval(a).map(|v| v.as_f64() as i64).unwrap_or(0);
                    let hi = self.ck.const_eval(b).map(|v| v.as_f64() as i64).unwrap_or(0);
                    for k in lo..hi {
                        if st.diverged || self.aborted {
                            break;
                        }
                        st.scopes.push(HashMap::from([(var.name.clone(), AVal::Int(Some(k)))]));
                        self.block(st, body);
                        st.scopes.pop();
                    }
                    // A body that returns on some path is handled by the return collector.
                    st.diverged = false;
                }
                ForIter::Each(e) => {
                    self.eval(st, e);
                    let n = match self.ck.ty(e) {
                        Ty::List(_, n) => *n,
                        _ => 0,
                    };
                    let elem = match self.ck.ty(e) {
                        Ty::List(t, _) => aval_of_ty(t),
                        _ => AVal::Other,
                    };
                    // Unknown trip count in 0..=n: join the states after every count.
                    let mut acc = st.clone();
                    let mut cur = st.clone();
                    for _ in 0..n {
                        if self.aborted {
                            break;
                        }
                        cur.scopes.push(HashMap::from([(var.name.clone(), elem.clone())]));
                        self.block(&mut cur, body);
                        cur.scopes.pop();
                        if cur.diverged {
                            break;
                        }
                        let (j, _) = self.join(acc, AVal::Other, cur.clone(), AVal::Other);
                        acc = j;
                    }
                    self.observe(&acc, e.span);
                    *st = acc;
                }
            },
            StmtKind::Commit(e) => {
                let v = self.eval(st, e);
                let committed = match &v {
                    AVal::Tuple(vs) => match vs.first() {
                        Some(AVal::Ctx(id)) => Some(*id),
                        _ => None,
                    },
                    _ => None,
                };
                self.finish_spec(st, committed, v, e.span);
            }
            StmtKind::Rollback(e) => {
                let v = self.eval(st, e);
                self.finish_spec(st, None, v, e.span);
            }
        }
    }

    /// Terminate the innermost speculation path: auto-release body contexts,
    /// then either keep the committed context or restore the original.
    fn finish_spec(&mut self, st: &mut State, committed: Option<CtxId>, v: AVal, span: Span) {
        let (_, depth, orig) = {
            let f = self.specs.last().unwrap();
            (0, f.1, f.2)
        };
        let mut out = st.clone();
        // Release every context bound in the speculation scopes, except the committed one.
        let mut release = Vec::new();
        for sc in &out.scopes[depth..] {
            for val in sc.values() {
                let mut ids = vec![];
                collect_ids(val, &mut ids);
                release.extend(ids);
            }
        }
        for id in release {
            if Some(id) != committed {
                out.ctxs.remove(&id);
            }
        }
        out.scopes.truncate(depth);
        out.hidden.pop();
        let result = match committed {
            Some(c) => {
                out.ctxs.remove(&orig);
                let vv = match v {
                    AVal::Tuple(vs) => vs.get(1).cloned().unwrap_or(AVal::Other),
                    _ => AVal::Other,
                };
                self.record("commit", span, String::new(), vec![orig], vec![c], 0, &out);
                AVal::Tuple(vec![AVal::Ctx(c), vv])
            }
            None => {
                self.record("rollback", span, String::new(), vec![], vec![orig], 0, &out);
                AVal::Tuple(vec![AVal::Ctx(orig), v])
            }
        };
        self.specs.last_mut().unwrap().0.push((out, result));
        st.diverged = true;
    }

    // ------------------------------------------------------------ expressions

    fn eval(&mut self, st: &mut State, e: &Expr) -> AVal {
        if !self.tick() {
            return AVal::Other;
        }
        match &e.kind {
            ExprKind::Int(i) => AVal::Int(Some(*i)),
            ExprKind::Var(id) => {
                for sc in st.scopes.iter().rev() {
                    if let Some(v) = sc.get(&id.name) {
                        return v.clone();
                    }
                }
                match self.ck.const_eval(e) {
                    Some(v) => AVal::Int(Some(v.as_f64() as i64)),
                    None => aval_of_ty(self.ck.ty(e)),
                }
            }
            ExprKind::Binary(_, a, b) => {
                self.eval(st, a);
                self.eval(st, b);
                aval_of_ty(self.ck.ty(e))
            }
            ExprKind::Unary(_, a) | ExprKind::Field(a, _) => {
                self.eval(st, a);
                aval_of_ty(self.ck.ty(e))
            }
            ExprKind::Index(a, b) => {
                self.eval(st, a);
                self.eval(st, b);
                aval_of_ty(self.ck.ty(e))
            }
            ExprKind::Tuple(es) => AVal::Tuple(es.iter().map(|x| self.eval(st, x)).collect()),
            ExprKind::Call(name, args) => {
                let is_fn = self.ck.fns.contains_key(&name.name);
                let mut vals = Vec::new();
                for a in args {
                    if matches!(a.kind, ExprKind::List(_)) || (!is_fn && matches!(self.ck.ty(a), Ty::Format(_))) {
                        vals.push(AVal::Other);
                    } else {
                        vals.push(self.eval(st, a));
                    }
                }
                if is_fn {
                    self.inline(st, &name.name, vals, e)
                } else {
                    aval_of_ty(self.ck.ty(e))
                }
            }
            ExprKind::Open(name) => {
                let (model, segs, tokens, what) = if let Some(&sid) = self.prefix_segs.get(&name.name) {
                    let m = self.ck.prefixes[&name.name].model.clone();
                    (m, vec![sid], 0, format!("prefix {}", name.name))
                } else {
                    let block = self.block_of(&name.name);
                    let s = self.new_seg(&name.name, "bos", 1, block, BTreeSet::new());
                    (name.name.clone(), vec![s], 1, format!("model {}", name.name))
                };
                st.prefill += tokens;
                let c = self.new_ctx(st, &model, segs);
                self.record("open", e.span, what, vec![], vec![c], tokens, st);
                self.observe(st, e.span);
                AVal::Ctx(c)
            }
            ExprKind::Append { ctx, label, text } => {
                let cv = self.eval(st, ctx);
                self.eval(st, text);
                let ub = self.ck.ty(text).str_bound().unwrap_or(0);
                let Some((cid, c)) = self.take_ctx(st, &cv) else { return AVal::Other };
                let labels: BTreeSet<String> = label.iter().map(|l| l.name.clone()).collect();
                let block = self.block_of(&c.model);
                let s = self.new_seg(&c.model, "text", ub, block, labels);
                let mut segs = c.segs.clone();
                segs.push(s);
                st.prefill += ub;
                let nc = self.new_ctx(st, &c.model, segs);
                let nctx = st.ctxs[&nc].clone();
                self.check_len(&nctx, e.span);
                let detail = label.as_ref().map(|l| format!("#[{}]", l.name)).unwrap_or_default();
                self.record("append", e.span, detail, vec![cid], vec![nc], ub, st);
                self.observe(st, e.span);
                AVal::Ctx(nc)
            }
            ExprKind::Fork(ctx) => {
                let cv = self.eval(st, ctx);
                let Some((cid, c)) = self.take_ctx(st, &cv) else { return AVal::Other };
                let a = self.new_ctx(st, &c.model, c.segs.clone());
                let b = self.new_ctx(st, &c.model, c.segs.clone());
                let shared: u64 = c.segs.iter().map(|s| self.segs[s].alloc).sum();
                let bpt = self.ck.models[&c.model].kv_bytes_per_token();
                self.forks.push(ForkShare { line: self.line(e.span), shared_tokens: self.ctx_tokens(&c), saved_bytes: shared as f64 * bpt });
                self.record("fork", e.span, String::new(), vec![cid], vec![a, b], 0, st);
                self.observe(st, e.span);
                AVal::Tuple(vec![AVal::Ctx(a), AVal::Ctx(b)])
            }
            ExprKind::Gen { ctx, max, stop, .. } => {
                let cv = self.eval(st, ctx);
                if let Some(s) = stop {
                    self.eval(st, s);
                }
                let maxv = self.ck.const_eval(max).map(|v| v.as_f64() as u64).unwrap_or(0);
                let out_val = match self.ck.ty(e) {
                    Ty::Tuple(ts) if ts.len() == 2 => aval_of_ty(&ts[1]),
                    _ => AVal::Other,
                };
                let Some((cid, c)) = self.take_ctx(st, &cv) else { return AVal::Other };
                let block = self.block_of(&c.model);
                let s = self.new_seg(&c.model, "gen", maxv, block, BTreeSet::new());
                let mut segs = c.segs.clone();
                segs.push(s);
                st.decode += maxv;
                let nc = self.new_ctx(st, &c.model, segs);
                let nctx = st.ctxs[&nc].clone();
                self.check_len(&nctx, e.span);
                self.record("gen", e.span, format!("max {maxv}"), vec![cid], vec![nc], maxv, st);
                self.observe(st, e.span);
                AVal::Tuple(vec![AVal::Ctx(nc), out_val])
            }
            ExprKind::Splice { ctx, span_name, new_label, text } => {
                let cv = self.eval(st, ctx);
                self.eval(st, text);
                let ub = self.ck.ty(text).str_bound().unwrap_or(0);
                let Some((cid, c)) = self.take_ctx(st, &cv) else { return AVal::Other };
                let idx = c.segs.iter().position(|s| self.segs[s].labels.contains(&span_name.name)).unwrap_or(c.segs.len().saturating_sub(1));
                let block = self.block_of(&c.model);
                let mut segs = c.segs[..idx].to_vec();
                // The replacement carries only the new label (if any); a join
                // segment keeps its other labels since it may hold them anywhere.
                let mut lbl: BTreeSet<String> = new_label.iter().map(|l| l.name.clone()).collect();
                if let Some(old) = c.segs.get(idx) {
                    if self.segs[old].kind == "join" {
                        let mut keep = self.segs[old].labels.clone();
                        keep.remove(&span_name.name);
                        lbl.extend(keep);
                    }
                }
                let ns = self.new_seg(&c.model, "splice", ub, block, lbl);
                segs.push(ns);
                let mut tail = 0;
                let tail_ids: Vec<SegId> = c.segs.get(idx + 1..).map(|s| s.to_vec()).unwrap_or_default();
                let joined_old = c.segs.get(idx).map(|o| self.segs[o].kind == "join").unwrap_or(false);
                if joined_old {
                    // Conservatively re-prefill everything the join segment might hold after the label.
                    tail += self.segs[&c.segs[idx]].tokens;
                }
                for t in tail_ids {
                    let old = self.segs[&t].clone();
                    tail += old.tokens;
                    let copy = self.new_seg(&c.model, old.kind, old.tokens, block, old.labels.clone());
                    self.segs.get_mut(&copy).unwrap().alloc = old.alloc;
                    segs.push(copy);
                }
                st.prefill += ub + tail;
                // Allocate the new chain before releasing the old one.
                st.ctxs.insert(cid, c.clone());
                let nc = self.new_ctx(st, &c.model, segs);
                self.observe(st, e.span);
                st.ctxs.remove(&cid);
                let nctx = st.ctxs[&nc].clone();
                self.check_len(&nctx, e.span);
                let entry = self.splices.entry(e.span).or_insert(SpliceCost { line: 0, span_label: span_name.name.clone(), reprefill_tokens: 0 });
                entry.reprefill_tokens = entry.reprefill_tokens.max(tail);
                let line = self.line(e.span);
                self.splices.get_mut(&e.span).unwrap().line = line;
                if tail > 0 && self.reported.insert((e.span, "W0210")) {
                    self.diags.push(
                        Diagnostic::warning("W0210", e.span, format!("splicing `{}` re-prefills up to {tail} following tokens", span_name.name))
                            .note("engines with position-corrected KV editing (Leyline-style) can avoid this; splice near the end to avoid it everywhere"),
                    );
                }
                self.record("splice", e.span, format!("at {} (+{} re-prefill)", span_name.name, tail), vec![cid], vec![nc], ub + tail, st);
                AVal::Ctx(nc)
            }
            ExprKind::Speculate { ctx, body } => {
                let cv = self.eval(st, ctx);
                let ExprKind::Var(name) = &ctx.kind else { return AVal::Other };
                let AVal::Ctx(orig) = cv else { return AVal::Other };
                let Some(oc) = st.ctxs.get(&orig).cloned() else { return AVal::Other };
                st.hidden.push(orig);
                let spec = self.new_ctx(st, &oc.model, oc.segs.clone());
                self.record("speculate", e.span, String::new(), vec![orig], vec![spec], 0, st);
                let depth = st.scopes.len();
                self.specs.push((vec![], depth, orig));
                let mut body_state = st.clone();
                body_state.scopes.push(HashMap::from([(name.name.clone(), AVal::Ctx(spec))]));
                self.observe(&body_state, e.span);
                self.block(&mut body_state, body);
                let (outs, _, _) = self.specs.pop().unwrap();
                match self.join_all(outs) {
                    Some((s, v)) => {
                        *st = s;
                        self.observe(st, e.span);
                        v
                    }
                    None => AVal::Other,
                }
            }
            _ => aval_of_ty(self.ck.ty(e)),
        }
    }

    fn inline(&mut self, st: &mut State, name: &str, args: Vec<AVal>, call: &Expr) -> AVal {
        let f = self.prog.items.iter().find_map(|i| match i {
            Item::Fn(f) if f.name.name == name => Some(f),
            _ => None,
        });
        let Some(f) = f else { return AVal::Other };
        let caller = std::mem::take(&mut st.scopes);
        let mut scope = HashMap::new();
        for (p, v) in f.params.iter().zip(args) {
            let v = match v {
                AVal::Str(_) => aval_of_ty(&self.ck.fns[name].sig.params.iter().find(|x| x.0 == p.name.name).unwrap().1),
                v => v,
            };
            scope.insert(p.name.name.clone(), v);
        }
        st.scopes.push(scope);
        self.returns.push(vec![]);
        let mut callee = std::mem::take(st);
        self.block(&mut callee, &f.body);
        let mut outs = self.returns.pop().unwrap();
        if !callee.diverged {
            outs.push((callee, AVal::Other));
        }
        let ret_ty = self.ck.ty(call).clone();
        match self.join_all(outs) {
            Some((mut s, v)) => {
                s.scopes = caller;
                s.diverged = false;
                *st = s;
                self.observe(st, call.span);
                match v {
                    AVal::Other => aval_of_ty(&ret_ty),
                    v => v,
                }
            }
            None => {
                st.scopes = caller;
                aval_of_ty(&ret_ty)
            }
        }
    }
}

fn s_span(_depth: usize, s: Span) -> Span {
    s
}

fn collect_ids(v: &AVal, out: &mut Vec<CtxId>) {
    match v {
        AVal::Ctx(id) => out.push(*id),
        AVal::Tuple(vs) => vs.iter().for_each(|x| collect_ids(x, out)),
        _ => {}
    }
}

fn collect_ctx_names(name: &str, v: &AVal, out: &mut HashMap<CtxId, String>) {
    match v {
        AVal::Ctx(id) => {
            out.insert(*id, name.to_string());
        }
        AVal::Tuple(vs) => vs.iter().for_each(|x| collect_ctx_names(name, x, out)),
        _ => {}
    }
}

pub fn human_bytes(b: f64) -> String {
    let units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b;
    let mut i = 0;
    while v >= 1024.0 && i < units.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{v:.0} {}", units[i])
    } else {
        format!("{v:.2} {}", units[i])
    }
}

fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

impl Plan {
    /// Human-readable report.
    pub fn render(&self) -> String {
        use std::fmt::Write;
        let mut o = String::new();
        let _ = writeln!(o, "plan for `{}`", self.entry);
        for m in &self.models {
            let _ = writeln!(
                o,
                "  model {:<10} kv={} · {}/token · page {} tok{}",
                m.name,
                m.kv_format,
                human_bytes(m.kv_bytes_per_token),
                m.block,
                m.max_ctx.map(|x| format!(" · max_ctx {}", thousands(x))).unwrap_or_default()
            );
        }
        let budget = match self.budget_kv_bytes {
            Some(b) => format!("  (budget {} {})", human_bytes(b as f64), if self.peak_kv_bytes <= b as f64 { "✓" } else { "✗ EXCEEDED" }),
            None => String::new(),
        };
        let _ = writeln!(o, "  peak KV memory    {}{}{}", human_bytes(self.peak_kv_bytes), budget, self.peak_line.map(|l| format!("  at line {l}")).unwrap_or_default());
        if !self.live_at_peak.is_empty() {
            let live: Vec<String> = self
                .live_at_peak
                .iter()
                .map(|l| if l.shared_tokens > 0 { format!("{} ({} tok, {} shared)", l.var, thousands(l.tokens), thousands(l.shared_tokens)) } else { format!("{} ({} tok)", l.var, thousands(l.tokens)) })
                .collect();
            let _ = writeln!(o, "  live at peak      {}", live.join(", "));
        }
        if self.resident_prefix_bytes > 0.0 {
            let _ = writeln!(o, "  resident prefixes {}", human_bytes(self.resident_prefix_bytes));
        }
        let pb = match self.budget_prefill {
            Some(b) => format!("  (budget {} {})", thousands(b), if self.prefill_tokens <= b { "✓" } else { "✗ EXCEEDED" }),
            None => String::new(),
        };
        let _ = writeln!(o, "  prefill tokens    ≤ {}{}", thousands(self.prefill_tokens), pb);
        let _ = writeln!(o, "  decode tokens     ≤ {}", thousands(self.decode_tokens));
        for m in &self.models {
            if m.longest_ctx_tokens > 0 {
                let _ = writeln!(o, "  longest context   ≤ {} tokens ({})", thousands(m.longest_ctx_tokens), m.name);
            }
        }
        for f in &self.forks {
            let _ = writeln!(o, "  fork  line {:<4}  shares {} tokens, saves {} vs copying", f.line, thousands(f.shared_tokens), human_bytes(f.saved_bytes));
        }
        for s in &self.splices {
            let _ = writeln!(o, "  splice line {:<3}  at `{}` re-prefills ≤ {} tokens", s.line, s.span_label, thousands(s.reprefill_tokens));
        }
        if !self.trace.is_empty() {
            let _ = writeln!(o, "  context DAG (first {} of {} ops):", self.trace.len().min(24), self.trace.len());
            for t in self.trace.iter().take(24) {
                let ins: Vec<String> = t.inputs.iter().map(|i| format!("c{i}")).collect();
                let outs: Vec<String> = t.outputs.iter().map(|i| format!("c{i}")).collect();
                let _ = writeln!(
                    o,
                    "    {:>3}  L{:<4} {:<9} {:>12} -> {:<12} {}{}",
                    t.step,
                    t.line,
                    t.op,
                    ins.join(","),
                    outs.join(","),
                    if t.tokens > 0 { format!("{} tok ", thousands(t.tokens)) } else { String::new() },
                    t.detail
                );
            }
        }
        if !self.reuse_hints.is_empty() {
            let shown: Vec<String> = self.reuse_hints.iter().take(6).map(|h| format!("seg{} ({}, {} tok) used at steps {:?}", h.segment, h.kind, h.tokens, &h.use_steps[..h.use_steps.len().min(8)])).collect();
            let _ = writeln!(o, "  reuse hints       {}", shown.join("\n                    "));
        }
        o
    }
}

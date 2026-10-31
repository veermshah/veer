//! Tree-walking interpreter for type-checked Veer programs.

use crate::engine::{Engine, GenConstraint, GenRequest, JsonShape, ModelRef};
use crate::store::{CtxId, ModelGeom};
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use veer_numerics::tensor::Tensor;
use veer_numerics::Format;
use veer_syntax::ast::*;
use veer_syntax::Span;
use veer_types::check::tokens_of;
use veer_types::ty::fmt_num;
use veer_types::{Checked, Ty};

#[derive(Clone, Debug)]
pub enum Value {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Unit,
    Ctx(CtxId),
    Tuple(Vec<Value>),
    Record(String, Vec<(String, Value)>),
    List(Vec<Value>),
    Tensor(Rc<Tensor>),
    Moved,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Str(s) => write!(f, "{s}"),
            Value::Int(i) => write!(f, "{i}"),
            Value::Float(x) => write!(f, "{}", fmt_num(*x)),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Unit => write!(f, "()"),
            Value::Ctx(c) => write!(f, "<ctx c{c}>"),
            Value::Moved => write!(f, "<moved>"),
            Value::Tuple(vs) => {
                write!(f, "(")?;
                for (i, v) in vs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, ")")
            }
            Value::Record(n, fs) => {
                write!(f, "{n} {{ ")?;
                for (i, (k, v)) in fs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    match v {
                        Value::Str(s) => write!(f, "{k}: {s:?}")?,
                        v => write!(f, "{k}: {v}")?,
                    }
                }
                write!(f, " }}")
            }
            Value::List(vs) => {
                write!(f, "[")?;
                for (i, v) in vs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            Value::Tensor(t) => write!(f, "Tensor<{:?}, {}>", t.shape, t.fmt.name()),
        }
    }
}

#[derive(Debug)]
pub struct RuntimeError {
    pub message: String,
    pub span: Option<Span>,
}

impl RuntimeError {
    fn at(span: Span, m: impl Into<String>) -> Self {
        RuntimeError { message: m.into(), span: Some(span) }
    }
}

type R<T> = Result<T, RuntimeError>;

enum Flow {
    Normal,
    Return(Value),
    Commit(Value),
    Rollback(Value),
}

/// How a declared `tool` is executed.
pub enum ToolImpl {
    /// Run a shell command; arguments arrive as a JSON object on stdin (and as
    /// `VEER_ARG_<name>` environment variables); stdout is the result.
    Command(String),
    /// Deterministic stand-in used when no command is bound.
    Echo,
}

pub struct Interp<'a> {
    prog: &'a Program,
    ck: &'a Checked,
    pub engine: &'a mut dyn Engine,
    tools: HashMap<String, ToolImpl>,
    scopes: Vec<HashMap<String, Value>>,
    specs: Vec<usize>,
    pub output: Vec<String>,
    pub echo: bool,
}

impl<'a> Interp<'a> {
    pub fn new(prog: &'a Program, ck: &'a Checked, engine: &'a mut dyn Engine) -> Self {
        Interp { prog, ck, engine, tools: HashMap::new(), scopes: vec![], specs: vec![], output: vec![], echo: true }
    }

    pub fn bind_tool(&mut self, name: &str, imp: ToolImpl) {
        self.tools.insert(name.to_string(), imp);
    }

    fn model_ref(&self, name: &str) -> ModelRef<'a> {
        let m = &self.ck.models[name];
        ModelRef { name: &m.name, hf_id: &m.hf_id, geom: ModelGeom { name: m.name.clone(), bytes_per_token: m.kv_bytes_per_token(), block: m.block } }
    }

    fn emit(&mut self, s: String) {
        if self.echo {
            println!("{s}");
        }
        self.output.push(s);
    }

    /// Register prefixes and run `entry` with the given arguments.
    pub fn run(&mut self, entry: &str, args: Vec<Value>) -> R<Value> {
        let mut prefixes: Vec<_> = self.ck.prefixes.values().collect();
        prefixes.sort_by(|a, b| a.name.cmp(&b.name));
        for p in prefixes {
            let mr = self.model_ref(&p.model);
            self.engine.register_prefix(&p.name, &mr, &p.text).map_err(|m| RuntimeError { message: m, span: None })?;
        }
        self.call_fn(entry, args, Span::default())
    }

    fn call_fn(&mut self, name: &str, args: Vec<Value>, span: Span) -> R<Value> {
        let f = self
            .prog
            .items
            .iter()
            .find_map(|i| match i {
                Item::Fn(f) if f.name.name == name => Some(f),
                _ => None,
            })
            .ok_or_else(|| RuntimeError::at(span, format!("no function `{name}`")))?;
        let saved = std::mem::take(&mut self.scopes);
        let saved_specs = std::mem::take(&mut self.specs);
        self.scopes.push(f.params.iter().map(|p| p.name.name.clone()).zip(args).collect());
        let flow = self.block(&f.body);
        self.scopes = saved;
        self.specs = saved_specs;
        match flow? {
            Flow::Return(v) => Ok(v),
            _ => Ok(Value::Unit),
        }
    }

    fn lookup_mut(&mut self, name: &str) -> Option<&mut Value> {
        self.scopes.iter_mut().rev().find_map(|s| s.get_mut(name))
    }

    fn block(&mut self, b: &Block) -> R<Flow> {
        self.scopes.push(HashMap::new());
        let mut flow = Flow::Normal;
        for s in &b.stmts {
            match self.stmt(s) {
                Ok(Flow::Normal) => {}
                Ok(f) => {
                    flow = f;
                    break;
                }
                Err(e) => {
                    self.scopes.pop();
                    return Err(e);
                }
            }
        }
        self.scopes.pop();
        Ok(flow)
    }

    fn bind(&mut self, pat: &Pattern, v: Value) {
        match (pat, v) {
            (Pattern::Ident(id), v) => {
                self.scopes.last_mut().unwrap().insert(id.name.clone(), v);
            }
            (Pattern::Tuple(ps, _), Value::Tuple(vs)) => {
                for (p, v) in ps.iter().zip(vs) {
                    self.bind(p, v);
                }
            }
            _ => {}
        }
    }

    fn assign(&mut self, pat: &Pattern, v: Value) {
        match (pat, v) {
            (Pattern::Ident(id), v) => {
                if let Some(slot) = self.lookup_mut(&id.name) {
                    *slot = v;
                }
            }
            (Pattern::Tuple(ps, _), Value::Tuple(vs)) => {
                for (p, v) in ps.iter().zip(vs) {
                    self.assign(p, v);
                }
            }
            _ => {}
        }
    }

    /// Release every context bound in speculation scopes (except `keep`).
    fn release_spec_scopes(&mut self, keep: Option<CtxId>) {
        let depth = *self.specs.last().unwrap();
        let mut ids = Vec::new();
        for s in &mut self.scopes[depth..] {
            for v in s.values_mut() {
                collect_ctx(v, &mut ids);
            }
        }
        for id in ids {
            if Some(id) != keep {
                self.engine.release(id);
            }
        }
    }

    fn stmt(&mut self, s: &Stmt) -> R<Flow> {
        match &s.kind {
            StmtKind::Let { pat, value, .. } => {
                let v = self.eval(value)?;
                self.bind(pat, v);
            }
            StmtKind::Assign { pat, value } => {
                let v = self.eval(value)?;
                self.assign(pat, v);
            }
            StmtKind::Expr(e) => {
                self.eval(e)?;
            }
            StmtKind::Drop(es) => {
                for e in es {
                    if let Value::Ctx(c) = self.eval(e)? {
                        self.engine.release(c);
                    }
                }
            }
            StmtKind::Return(e) => {
                let v = match e {
                    Some(e) => self.eval(e)?,
                    None => Value::Unit,
                };
                return Ok(Flow::Return(v));
            }
            StmtKind::If { cond, then, els } => {
                let c = matches!(self.eval(cond)?, Value::Bool(true));
                if c {
                    return self.block(then);
                } else if let Some(e) = els {
                    return self.block(e);
                }
            }
            StmtKind::For { var, iter, body } => {
                let items: Vec<Value> = match iter {
                    ForIter::Range(a, b) => {
                        let (Value::Int(lo), Value::Int(hi)) = (self.eval(a)?, self.eval(b)?) else {
                            return Err(RuntimeError::at(s.span, "range bounds must be integers"));
                        };
                        (lo..hi).map(Value::Int).collect()
                    }
                    ForIter::Each(e) => match self.eval(e)? {
                        Value::List(vs) => vs,
                        _ => return Err(RuntimeError::at(e.span, "expected a list")),
                    },
                };
                for it in items {
                    self.scopes.push(HashMap::from([(var.name.clone(), it)]));
                    let f = self.block(body);
                    self.scopes.pop();
                    match f? {
                        Flow::Normal => {}
                        other => return Ok(other),
                    }
                }
            }
            StmtKind::Commit(e) => {
                let v = self.eval(e)?;
                let keep = match &v {
                    Value::Tuple(vs) => match vs.first() {
                        Some(Value::Ctx(c)) => Some(*c),
                        _ => None,
                    },
                    _ => None,
                };
                self.release_spec_scopes(keep);
                return Ok(Flow::Commit(v));
            }
            StmtKind::Rollback(e) => {
                let v = self.eval(e)?;
                self.release_spec_scopes(None);
                return Ok(Flow::Rollback(v));
            }
        }
        Ok(Flow::Normal)
    }

    fn take_ctx(&mut self, e: &Expr) -> R<CtxId> {
        match self.eval(e)? {
            Value::Ctx(c) => Ok(c),
            v => Err(RuntimeError::at(e.span, format!("expected a context, found {v}"))),
        }
    }

    fn str_of(&mut self, e: &Expr) -> R<String> {
        match self.eval(e)? {
            Value::Str(s) => Ok(s),
            v => Err(RuntimeError::at(e.span, format!("expected a string, found {v}"))),
        }
    }

    fn eval(&mut self, e: &Expr) -> R<Value> {
        let sp = e.span;
        let eng = |m: String| RuntimeError::at(sp, m);
        Ok(match &e.kind {
            ExprKind::Int(i) => Value::Int(*i),
            ExprKind::Float(f) => Value::Float(*f),
            ExprKind::Str(s) => Value::Str(s.clone()),
            ExprKind::Bool(b) => Value::Bool(*b),
            ExprKind::Var(id) => {
                if let Some(slot) = self.lookup_mut(&id.name) {
                    if matches!(slot, Value::Ctx(_)) {
                        return Ok(std::mem::replace(slot, Value::Moved));
                    }
                    return Ok(slot.clone());
                }
                match self.ck.consts.get(&id.name) {
                    Some(veer_types::ConstVal::Int(i)) => Value::Int(*i),
                    Some(veer_types::ConstVal::Float(f)) => Value::Float(*f),
                    None => return Err(RuntimeError::at(sp, format!("unbound `{}`", id.name))),
                }
            }
            ExprKind::Unary(op, x) => match (op, self.eval(x)?) {
                (UnOp::Not, Value::Bool(b)) => Value::Bool(!b),
                (UnOp::Neg, Value::Int(i)) => Value::Int(-i),
                (UnOp::Neg, Value::Float(f)) => Value::Float(-f),
                (UnOp::Neg, Value::Tensor(t)) => Value::Tensor(Rc::new(t.scale(-1.0))),
                (_, v) => return Err(RuntimeError::at(sp, format!("bad operand {v}"))),
            },
            ExprKind::Binary(op, a, b) => {
                let (x, y) = (self.eval(a)?, self.eval(b)?);
                binary(*op, x, y).map_err(|m| RuntimeError::at(sp, m))?
            }
            ExprKind::Call(name, args) => self.call(name, args, sp)?,
            ExprKind::Field(x, f) => match self.eval(x)? {
                Value::Record(_, fs) => fs.into_iter().find(|(k, _)| *k == f.name).map(|(_, v)| v).ok_or_else(|| RuntimeError::at(sp, "no such field"))?,
                v => return Err(RuntimeError::at(sp, format!("{v} has no fields"))),
            },
            ExprKind::Index(x, i) => match (self.eval(x)?, self.eval(i)?) {
                (Value::List(vs), Value::Int(i)) => {
                    let n = vs.len();
                    vs.into_iter().nth(i as usize).ok_or_else(|| RuntimeError::at(sp, format!("index {i} out of bounds (len {n})")))?
                }
                _ => return Err(RuntimeError::at(sp, "bad index")),
            },
            ExprKind::Tuple(es) => {
                if es.is_empty() {
                    Value::Unit
                } else {
                    Value::Tuple(es.iter().map(|x| self.eval(x)).collect::<R<_>>()?)
                }
            }
            ExprKind::List(_) => return Err(RuntimeError::at(sp, "list literal outside a shape")),
            ExprKind::Open(name) => {
                let (model, prefix) = match self.ck.prefixes.get(&name.name) {
                    Some(p) => (p.model.clone(), Some(name.name.as_str())),
                    None => (name.name.clone(), None),
                };
                let mr = self.model_ref(&model);
                Value::Ctx(self.engine.open(&mr, prefix).map_err(eng)?)
            }
            ExprKind::Append { ctx, label, text } => {
                let c = self.take_ctx(ctx)?;
                let t = self.str_of(text)?;
                let model = self.ctx_model(ctx);
                let mr = self.model_ref(&model);
                Value::Ctx(self.engine.append(&mr, c, &t, label.as_ref().map(|l| l.name.as_str())).map_err(eng)?)
            }
            ExprKind::Fork(ctx) => {
                let c = self.take_ctx(ctx)?;
                let (a, b) = self.engine.fork(c).map_err(eng)?;
                Value::Tuple(vec![Value::Ctx(a), Value::Ctx(b)])
            }
            ExprKind::Gen { ctx, max, constraint, stop } => {
                let c = self.take_ctx(ctx)?;
                let model = self.ctx_model(ctx);
                let max_tokens = match self.eval(max)? {
                    Value::Int(i) => i as u64,
                    _ => return Err(RuntimeError::at(max.span, "max must be an integer")),
                };
                let stop = match stop {
                    Some(s) => Some(self.str_of(s)?),
                    None => None,
                };
                let cons = match constraint {
                    None => GenConstraint::None,
                    Some(Constraint::Choice(o)) => GenConstraint::Choice(o.clone()),
                    Some(Constraint::Json(s)) => GenConstraint::Json(self.shape_of(&Ty::Record(s.name.clone()))),
                };
                let req = GenRequest { max_tokens, stop, constraint: cons.clone() };
                let mr = self.model_ref(&model);
                let (nc, out) = self.engine.gen(&mr, c, &req).map_err(eng)?;
                let v = match constraint {
                    Some(Constraint::Json(s)) => {
                        let j: serde_json::Value = serde_json::from_str(out.trim()).map_err(|err| RuntimeError::at(sp, format!("model output is not valid JSON for `{}`: {err}\n  output: {}", s.name, out.chars().take(200).collect::<String>())))?;
                        self.from_json(&Ty::Record(s.name.clone()), &j).map_err(|m| RuntimeError::at(sp, format!("model output does not match `{}`: {m}", s.name)))?
                    }
                    _ => Value::Str(out),
                };
                Value::Tuple(vec![Value::Ctx(nc), v])
            }
            ExprKind::Splice { ctx, span_name, new_label, text } => {
                let c = self.take_ctx(ctx)?;
                let t = self.str_of(text)?;
                let model = self.ctx_model(ctx);
                let mr = self.model_ref(&model);
                Value::Ctx(self.engine.splice(&mr, c, &span_name.name, new_label.as_ref().map(|l| l.name.as_str()), &t).map_err(eng)?)
            }
            ExprKind::Speculate { ctx, body } => {
                let ExprKind::Var(name) = &ctx.kind else { return Err(RuntimeError::at(sp, "speculate needs a variable")) };
                let orig = self.take_ctx(ctx)?;
                let (keep, spec) = self.engine.fork(orig).map_err(eng)?;
                self.scopes.push(HashMap::from([(name.name.clone(), Value::Ctx(spec))]));
                self.specs.push(self.scopes.len() - 1);
                let flow = self.block(body);
                self.specs.pop();
                self.scopes.pop();
                match flow? {
                    Flow::Commit(v) => {
                        self.engine.release(keep);
                        v
                    }
                    Flow::Rollback(v) => Value::Tuple(vec![Value::Ctx(keep), v]),
                    _ => return Err(RuntimeError::at(sp, "speculation ended without commit or rollback")),
                }
            }
        })
    }

    fn ctx_model(&self, ctx_expr: &Expr) -> String {
        match self.ck.ty(ctx_expr) {
            Ty::Ctx { model, .. } => model.clone(),
            _ => String::new(),
        }
    }

    fn shape_of(&self, t: &Ty) -> JsonShape {
        match t {
            Ty::Str(b) => JsonShape::Str(*b),
            Ty::Int => JsonShape::Int,
            Ty::Float => JsonShape::Float,
            Ty::Bool => JsonShape::Bool,
            Ty::List(e, n) => JsonShape::List(Box::new(self.shape_of(e)), *n),
            Ty::Record(n) => JsonShape::Object(self.ck.schemas[n].iter().map(|(k, v)| (k.clone(), self.shape_of(v))).collect()),
            _ => JsonShape::Str(None),
        }
    }

    fn from_json(&self, t: &Ty, j: &serde_json::Value) -> Result<Value, String> {
        use serde_json::Value as J;
        Ok(match (t, j) {
            (Ty::Str(_), J::String(s)) => Value::Str(s.clone()),
            (Ty::Int, J::Number(n)) => Value::Int(n.as_i64().ok_or("expected an integer")?),
            (Ty::Float, J::Number(n)) => Value::Float(n.as_f64().unwrap_or(0.0)),
            (Ty::Bool, J::Bool(b)) => Value::Bool(*b),
            (Ty::List(e, n), J::Array(xs)) => {
                if xs.len() as u64 > *n {
                    return Err(format!("list has {} items, at most {n} allowed", xs.len()));
                }
                Value::List(xs.iter().map(|x| self.from_json(e, x)).collect::<Result<_, _>>()?)
            }
            (Ty::Record(name), J::Object(m)) => {
                let mut fs = Vec::new();
                for (k, ft) in &self.ck.schemas[name] {
                    let v = m.get(k).ok_or_else(|| format!("missing field `{k}`"))?;
                    fs.push((k.clone(), self.from_json(ft, v)?));
                }
                Value::Record(name.clone(), fs)
            }
            (t, j) => return Err(format!("expected {t}, got {j}")),
        })
    }

    fn call(&mut self, name: &Ident, args: &[Expr], sp: Span) -> R<Value> {
        let n = name.name.as_str();
        if self.ck.fns.contains_key(n) {
            let vals = args.iter().map(|a| self.eval(a)).collect::<R<Vec<_>>>()?;
            return self.call_fn(n, vals, sp);
        }
        if let Some(sig) = self.ck.tools.get(n).cloned() {
            let vals = args.iter().map(|a| self.eval(a)).collect::<R<Vec<_>>>()?;
            let bound = sig.ret.str_bound();
            let out = self.run_tool(n, &sig.params, &vals).map_err(|m| RuntimeError::at(sp, m))?;
            return Ok(match bound {
                Some(b) => Value::Str(truncate_tokens(&out, b)),
                None => Value::Str(out),
            });
        }
        let tensor = |v: Value, s: Span| -> R<Rc<Tensor>> {
            match v {
                Value::Tensor(t) => Ok(t),
                v => Err(RuntimeError::at(s, format!("expected a tensor, found {v}"))),
            }
        };
        Ok(match n {
            "print" => {
                let parts = args.iter().map(|a| self.eval(a).map(|v| v.to_string())).collect::<R<Vec<_>>>()?;
                self.emit(parts.join(" "));
                Value::Unit
            }
            "len" => match self.eval(&args[0])? {
                Value::Str(s) => Value::Int(tokens_of(&s) as i64),
                Value::List(v) => Value::Int(v.len() as i64),
                v => return Err(RuntimeError::at(sp, format!("len of {v}"))),
            },
            "contains" => {
                let a = self.str_of(&args[0])?;
                let b = self.str_of(&args[1])?;
                Value::Bool(a.contains(&b))
            }
            "to_str" => Value::Str(truncate_tokens(&self.eval(&args[0])?.to_string(), 24)),
            "trim" => Value::Str(self.str_of(&args[0])?.trim().to_string()),
            "rand_uniform" => {
                let shape = self.shape(&args[0]);
                let lo = self.num(&args[1])?;
                let hi = self.num(&args[2])?;
                let seed = match self.eval(&args[3])? {
                    Value::Int(i) => i as u64,
                    _ => 0,
                };
                Value::Tensor(Rc::new(Tensor::rand_uniform(shape, lo, hi, seed)))
            }
            "zeros" => Value::Tensor(Rc::new(Tensor::zeros(self.shape(&args[0])))),
            "quant" => {
                let t = tensor(self.eval(&args[0])?, args[0].span)?;
                let f = match &args[1].kind {
                    ExprKind::Var(id) => Format::parse(&id.name).ok_or_else(|| RuntimeError::at(sp, "unknown format"))?,
                    _ => return Err(RuntimeError::at(sp, "expected a format")),
                };
                Value::Tensor(Rc::new(t.quant(f)))
            }
            "matmul" => {
                let a = tensor(self.eval(&args[0])?, args[0].span)?;
                let b = tensor(self.eval(&args[1])?, args[1].span)?;
                Value::Tensor(Rc::new(a.matmul(&b)))
            }
            "add" => {
                let a = tensor(self.eval(&args[0])?, args[0].span)?;
                let b = tensor(self.eval(&args[1])?, args[1].span)?;
                Value::Tensor(Rc::new(a.add(&b)))
            }
            "scale" => {
                let a = tensor(self.eval(&args[0])?, args[0].span)?;
                let c = self.num(&args[1])?;
                Value::Tensor(Rc::new(a.scale(c)))
            }
            "relu" => Value::Tensor(Rc::new(tensor(self.eval(&args[0])?, args[0].span)?.relu())),
            "silu" => Value::Tensor(Rc::new(tensor(self.eval(&args[0])?, args[0].span)?.silu())),
            "measured_err" => Value::Float(tensor(self.eval(&args[0])?, args[0].span)?.measured_err()),
            "report" => {
                let t = tensor(self.eval(&args[0])?, args[0].span)?;
                let (sb, se) = match self.ck.ty(&args[0]) {
                    Ty::Tensor { bound, .. } => (bound.m, bound.e),
                    _ => (f64::NAN, f64::NAN),
                };
                let measured = t.measured_err();
                let ok = if measured <= se * (1.0 + 1e-9) { "✓ within bound" } else { "✗ BOUND VIOLATED" };
                let label = expr_label(&args[0]);
                self.emit(format!(
                    "report {label}: shape {:?} fmt {} | max|x| {} (static ≤ {}) | error {} (static ≤ {}) {ok}",
                    t.shape,
                    t.fmt.name(),
                    fmt_num(t.max_abs()),
                    fmt_num(sb),
                    fmt_num(measured),
                    fmt_num(se)
                ));
                Value::Unit
            }
            _ => return Err(RuntimeError::at(sp, format!("unknown function `{n}`"))),
        })
    }

    fn num(&mut self, e: &Expr) -> R<f64> {
        match self.eval(e)? {
            Value::Int(i) => Ok(i as f64),
            Value::Float(f) => Ok(f),
            v => Err(RuntimeError::at(e.span, format!("expected a number, found {v}"))),
        }
    }

    fn shape(&self, e: &Expr) -> Vec<usize> {
        match &e.kind {
            ExprKind::List(es) => es.iter().map(|d| self.ck.const_eval(d).map(|v| v.as_f64() as usize).unwrap_or(1)).collect(),
            _ => vec![],
        }
    }

    fn run_tool(&mut self, name: &str, params: &[(String, Ty)], vals: &[Value]) -> Result<String, String> {
        let args: serde_json::Map<String, serde_json::Value> = params.iter().zip(vals).map(|((k, _), v)| (k.clone(), value_to_json(v))).collect();
        match self.tools.get(name).unwrap_or(&ToolImpl::Echo) {
            ToolImpl::Echo => {
                let parts: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
                Ok(format!("[{name} result for: {}]", parts.join(", ")))
            }
            ToolImpl::Command(cmd) => {
                use std::io::Write;
                use std::process::{Command, Stdio};
                let mut c = if cfg!(windows) {
                    let mut c = Command::new("cmd");
                    c.args(["/C", cmd]);
                    c
                } else {
                    let mut c = Command::new("sh");
                    c.args(["-c", cmd]);
                    c
                };
                for ((k, _), v) in params.iter().zip(vals) {
                    c.env(format!("VEER_ARG_{}", k.to_uppercase()), v.to_string());
                }
                let mut child = c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().map_err(|e| format!("tool `{name}`: {e}"))?;
                child.stdin.take().unwrap().write_all(serde_json::Value::Object(args).to_string().as_bytes()).map_err(|e| e.to_string())?;
                let out = child.wait_with_output().map_err(|e| e.to_string())?;
                if !out.status.success() {
                    return Err(format!("tool `{name}` exited with {}", out.status));
                }
                Ok(String::from_utf8_lossy(&out.stdout).trim_end().to_string())
            }
        }
    }
}

fn expr_label(e: &Expr) -> String {
    match &e.kind {
        ExprKind::Var(id) => format!("`{}`", id.name),
        _ => "expr".into(),
    }
}

fn value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Str(s) => serde_json::Value::String(s.clone()),
        Value::Int(i) => (*i).into(),
        Value::Float(f) => serde_json::json!(f),
        Value::Bool(b) => (*b).into(),
        other => serde_json::Value::String(other.to_string()),
    }
}

/// Truncate to at most `n` tokens under the byte measure (on a char boundary).
pub fn truncate_tokens(s: &str, n: u64) -> String {
    if s.len() as u64 <= n {
        return s.to_string();
    }
    let mut end = n as usize;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

fn collect_ctx(v: &mut Value, out: &mut Vec<CtxId>) {
    match v {
        Value::Ctx(c) => {
            out.push(*c);
            *v = Value::Moved;
        }
        Value::Tuple(vs) => vs.iter_mut().for_each(|x| collect_ctx(x, out)),
        _ => {}
    }
}

fn binary(op: BinOp, x: Value, y: Value) -> Result<Value, String> {
    use BinOp::*;
    use Value::*;
    Ok(match (op, x, y) {
        (Add, Int(a), Int(b)) => Int(a.wrapping_add(b)),
        (Sub, Int(a), Int(b)) => Int(a.wrapping_sub(b)),
        (Mul, Int(a), Int(b)) => Int(a.wrapping_mul(b)),
        (Div, Int(_), Int(0)) => return Err("division by zero".into()),
        (Div, Int(a), Int(b)) => Int(a / b),
        (Add, Str(a), Str(b)) => Str(a + &b),
        (Add, Tensor(a), Tensor(b)) => Tensor(Rc::new(a.add(&b))),
        (Sub, Tensor(a), Tensor(b)) => Tensor(Rc::new(a.add(&b.scale(-1.0)))),
        (MatMul, Tensor(a), Tensor(b)) => Tensor(Rc::new(a.matmul(&b))),
        (Mul, Tensor(a), Int(c)) | (Mul, Int(c), Tensor(a)) => Tensor(Rc::new(a.scale(c as f64))),
        (Mul, Tensor(a), Float(c)) | (Mul, Float(c), Tensor(a)) => Tensor(Rc::new(a.scale(c))),
        (Eq, Str(a), Str(b)) => Bool(a == b),
        (Ne, Str(a), Str(b)) => Bool(a != b),
        (Eq, Bool(a), Bool(b)) => Bool(a == b),
        (Ne, Bool(a), Bool(b)) => Bool(a != b),
        (And, Bool(a), Bool(b)) => Bool(a && b),
        (Or, Bool(a), Bool(b)) => Bool(a || b),
        (op, a, b) => {
            let (fa, fb) = match (&a, &b) {
                (Int(x), Int(y)) => (*x as f64, *y as f64),
                (Int(x), Float(y)) => (*x as f64, *y),
                (Float(x), Int(y)) => (*x, *y as f64),
                (Float(x), Float(y)) => (*x, *y),
                _ => return Err(format!("cannot apply `{}` to {a} and {b}", op.as_str())),
            };
            match op {
                Add => Float(fa + fb),
                Sub => Float(fa - fb),
                Mul => Float(fa * fb),
                Div => Float(fa / fb),
                Eq => Bool(fa == fb),
                Ne => Bool(fa != fb),
                Lt => Bool(fa < fb),
                Le => Bool(fa <= fb),
                Gt => Bool(fa > fb),
                Ge => Bool(fa >= fb),
                _ => return Err(format!("cannot apply `{}`", op.as_str())),
            }
        }
    })
}

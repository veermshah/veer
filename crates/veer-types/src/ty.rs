use std::collections::BTreeSet;
use std::fmt;
use veer_numerics::{Bound, Format};

/// Span labels a context is known to contain (`must`, used to allow `splice`)
/// and may contain (`may`, used to reject duplicate labels).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Spans {
    pub must: BTreeSet<String>,
    pub may: BTreeSet<String>,
}

impl Spans {
    /// Add a label; false if it may already be present.
    pub fn add(&mut self, l: &str) -> bool {
        if self.may.contains(l) {
            return false;
        }
        self.must.insert(l.to_string());
        self.may.insert(l.to_string());
        true
    }
    pub fn remove(&mut self, l: &str) {
        self.must.remove(l);
        self.may.remove(l);
    }
    pub fn has(&self, l: &str) -> bool {
        self.must.contains(l)
    }
    pub fn join(&self, o: &Spans) -> Spans {
        Spans { must: self.must.intersection(&o.must).cloned().collect(), may: self.may.union(&o.may).cloned().collect() }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Ty {
    /// String with an optional static upper bound on its token count.
    Str(Option<u64>),
    Int,
    Float,
    Bool,
    Unit,
    /// A linear handle to a KV-cache context. `spans` are the labels known to be present.
    Ctx { model: String, spans: Spans },
    Tuple(Vec<Ty>),
    Record(String),
    List(Box<Ty>, u64),
    Tensor { shape: Vec<u64>, fmt: Format, bound: Bound },
    /// A number format used as a value, e.g. the `mxfp4` in `quant(w, mxfp4)`.
    Format(Format),
    /// Poison type after an error, compatible with everything.
    Error,
}

impl Ty {
    pub fn ctx(model: &str) -> Ty {
        Ty::Ctx { model: model.to_string(), spans: Spans::default() }
    }
    pub fn is_ctx(&self) -> bool {
        matches!(self, Ty::Ctx { .. })
    }
    pub fn contains_ctx(&self) -> bool {
        match self {
            Ty::Ctx { .. } => true,
            Ty::Tuple(ts) => ts.iter().any(|t| t.contains_ctx()),
            Ty::List(t, _) => t.contains_ctx(),
            _ => false,
        }
    }
    pub fn is_numeric(&self) -> bool {
        matches!(self, Ty::Int | Ty::Float | Ty::Error)
    }
    pub fn str_bound(&self) -> Option<u64> {
        match self {
            Ty::Str(b) => *b,
            _ => None,
        }
    }
}

/// Is a value of type `actual` acceptable where `expected` is required?
/// Returns Err(reason) if not.
pub fn compatible(expected: &Ty, actual: &Ty) -> Result<(), Mismatch> {
    use Ty::*;
    match (expected, actual) {
        (Error, _) | (_, Error) => Ok(()),
        (Str(None), Str(_)) => Ok(()),
        (Str(Some(e)), Str(Some(a))) => {
            if a <= e {
                Ok(())
            } else {
                Err(Mismatch::Bound { expected: *e, actual: Some(*a) })
            }
        }
        (Str(Some(e)), Str(None)) => Err(Mismatch::Bound { expected: *e, actual: None }),
        (Float, Int) => Ok(()),
        (Ctx { model: a, .. }, Ctx { model: b, .. }) => {
            if a == b {
                Ok(())
            } else {
                Err(Mismatch::Model(a.clone(), b.clone()))
            }
        }
        (Tuple(es), Tuple(acts)) if es.len() == acts.len() => {
            for (e, a) in es.iter().zip(acts) {
                compatible(e, a)?;
            }
            Ok(())
        }
        (List(e, n), List(a, m)) if m <= n => compatible(e, a),
        (Tensor { shape: s1, fmt: f1, bound: b1 }, Tensor { shape: s2, fmt: f2, bound: b2 }) => {
            if s1 != s2 || f1 != f2 {
                return Err(Mismatch::Plain);
            }
            if b2.m > b1.m * (1.0 + 1e-12) || b2.e > b1.e * (1.0 + 1e-12) {
                return Err(Mismatch::TensorBound { expected: *b1, actual: *b2 });
            }
            Ok(())
        }
        (a, b) if a == b => Ok(()),
        _ => Err(Mismatch::Plain),
    }
}

#[derive(Debug)]
pub enum Mismatch {
    Plain,
    Bound { expected: u64, actual: Option<u64> },
    Model(String, String),
    TensorBound { expected: Bound, actual: Bound },
}

pub fn fmt_num(x: f64) -> String {
    if x.is_infinite() {
        "∞".into()
    } else if x == 0.0 {
        "0".into()
    } else if (1e-3..1e5).contains(&x.abs()) {
        let s = format!("{x:.4}");
        let s = s.trim_end_matches('0').trim_end_matches('.');
        s.to_string()
    } else {
        format!("{x:.3e}")
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Ty::Str(None) => write!(f, "str"),
            Ty::Str(Some(n)) => write!(f, "str<={n}"),
            Ty::Int => write!(f, "int"),
            Ty::Float => write!(f, "float"),
            Ty::Bool => write!(f, "bool"),
            Ty::Unit => write!(f, "()"),
            Ty::Ctx { model, .. } => write!(f, "Ctx<{model}>"),
            Ty::Tuple(ts) => {
                write!(f, "(")?;
                for (i, t) in ts.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{t}")?;
                }
                write!(f, ")")
            }
            Ty::Record(n) => write!(f, "{n}"),
            Ty::List(t, n) => write!(f, "[{t}; <={n}]"),
            Ty::Tensor { shape, fmt, bound } => {
                let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
                write!(f, "Tensor<[{}], {}, abs<={}, err<={}>", dims.join(", "), fmt.name(), fmt_num(bound.m), fmt_num(bound.e))
            }
            Ty::Format(fm) => write!(f, "format {}", fm.name()),
            Ty::Error => write!(f, "{{error}}"),
        }
    }
}

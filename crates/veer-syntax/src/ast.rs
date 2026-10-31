use crate::span::Span;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Clone, Debug)]
pub enum Item {
    Model(ModelDecl),
    Prefix(PrefixDecl),
    Tool(ToolDecl),
    Schema(SchemaDecl),
    Const(ConstDecl),
    Fn(FnDecl),
}

impl Item {
    pub fn name(&self) -> &Ident {
        match self {
            Item::Model(d) => &d.name,
            Item::Prefix(d) => &d.name,
            Item::Tool(d) => &d.name,
            Item::Schema(d) => &d.name,
            Item::Const(d) => &d.name,
            Item::Fn(d) => &d.name,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ModelVal {
    Int(u64, Span),
    Ident(Ident),
}

#[derive(Clone, Debug)]
pub struct ModelDecl {
    pub name: Ident,
    pub hf_id: String,
    pub fields: Vec<(Ident, ModelVal)>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct PrefixDecl {
    pub name: Ident,
    pub model: Ident,
    pub text: String,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct ToolDecl {
    pub name: Ident,
    pub params: Vec<Param>,
    pub ret: TypeExpr,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct SchemaDecl {
    pub name: Ident,
    pub fields: Vec<(Ident, TypeExpr)>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct ConstDecl {
    pub name: Ident,
    pub value: Expr,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: Ident,
    pub ty: TypeExpr,
}

#[derive(Clone, Debug)]
pub struct Attr {
    pub name: Ident,
    pub args: Vec<AttrArg>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct AttrArg {
    pub key: Ident,
    pub value: AttrVal,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AttrVal {
    Int(u64, Option<String>),
    Float(f64),
}

#[derive(Clone, Debug)]
pub struct FnDecl {
    pub name: Ident,
    pub attrs: Vec<Attr>,
    pub params: Vec<Param>,
    pub ret: Option<TypeExpr>,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum TypeExprKind {
    /// `str` or `str<=N`
    Str(Option<u64>),
    Int,
    Float,
    Bool,
    Ctx(Ident),
    Tuple(Vec<TypeExpr>),
    Named(Ident),
    /// `[T; <=N]`
    List(Box<TypeExpr>, u64),
    /// `Tensor<[d, ...], fmt, abs<=M, err<=E>`
    Tensor { shape: Vec<u64>, fmt: Ident, abs: Option<f64>, err: Option<f64> },
}

#[derive(Clone, Debug)]
pub struct TypeExpr {
    pub kind: TypeExprKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum Pattern {
    Ident(Ident),
    Wildcard(Span),
    Tuple(Vec<Pattern>, Span),
}

impl Pattern {
    pub fn span(&self) -> Span {
        match self {
            Pattern::Ident(i) => i.span,
            Pattern::Wildcard(s) | Pattern::Tuple(_, s) => *s,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ForIter {
    Range(Expr, Expr),
    Each(Expr),
}

#[derive(Clone, Debug)]
pub enum StmtKind {
    Let { pat: Pattern, ty: Option<TypeExpr>, value: Expr },
    Assign { pat: Pattern, value: Expr },
    Expr(Expr),
    Drop(Vec<Expr>),
    Return(Option<Expr>),
    If { cond: Expr, then: Block, els: Option<Block> },
    For { var: Ident, iter: ForIter, body: Block },
    Commit(Expr),
    Rollback(Expr),
}

#[derive(Clone, Debug)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

impl BinOp {
    pub fn as_str(self) -> &'static str {
        use BinOp::*;
        match self {
            Add => "+",
            Sub => "-",
            Mul => "*",
            Div => "/",
            MatMul => "@",
            Eq => "==",
            Ne => "!=",
            Lt => "<",
            Le => "<=",
            Gt => ">",
            Ge => ">=",
            And => "&&",
            Or => "||",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Clone, Debug)]
pub enum Constraint {
    Json(Ident),
    Choice(Vec<String>),
}

pub type ExprId = u32;

#[derive(Clone, Debug)]
pub struct Expr {
    pub id: ExprId,
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Var(Ident),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Call(Ident, Vec<Expr>),
    Field(Box<Expr>, Ident),
    Index(Box<Expr>, Box<Expr>),
    Tuple(Vec<Expr>),
    /// `[a, b, c]` — only used for tensor shapes.
    List(Vec<Expr>),
    Open(Ident),
    Append { ctx: Box<Expr>, label: Option<Ident>, text: Box<Expr> },
    Fork(Box<Expr>),
    Gen { ctx: Box<Expr>, max: Box<Expr>, constraint: Option<Constraint>, stop: Option<Box<Expr>> },
    Splice { ctx: Box<Expr>, span_name: Ident, new_label: Option<Ident>, text: Box<Expr> },
    Speculate { ctx: Box<Expr>, body: Block },
}

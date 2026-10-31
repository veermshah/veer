//! Type checking for Veer: linear contexts, token bounds, spans, speculation,
//! and the precision (number format + error bound) layer.

pub mod check;
pub mod ty;

pub use check::{check, Accuracy, Budget, Checked, ConstVal, FnInfo, ModelInfo, PrefixInfo, Sig};
pub use ty::Ty;

use veer_syntax::ast::Program;
use veer_syntax::Diagnostic;

/// Parse and check a source file. Warnings are returned alongside a successful result.
pub fn frontend(src: &str) -> Result<(Program, Checked, Vec<Diagnostic>), Vec<Diagnostic>> {
    let prog = veer_syntax::parse(src).map_err(|d| vec![d])?;
    let (checked, diags) = check(&prog);
    if diags.iter().any(|d| d.is_error()) {
        Err(diags)
    } else {
        Ok((prog, checked, diags))
    }
}

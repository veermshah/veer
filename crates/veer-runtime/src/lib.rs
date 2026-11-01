//! Veer runtime: the paged context store, execution engines (simulated and
//! OpenAI-compatible HTTP), and the interpreter.

pub mod engine;
pub mod interp;
pub mod store;

#[cfg(feature = "http")]
pub use engine::http_engine;
pub use engine::{sim_engine, Dialect, Engine, EngineStats};
pub use interp::{Interp, RuntimeError, ToolImpl, Value};

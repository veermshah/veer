//! The single registry of every diagnostic Veer can emit.
//! The docs site's error index is generated from this table (`veer explain --all-markdown`).

pub struct CodeInfo {
    pub code: &'static str,
    pub title: &'static str,
    pub explanation: &'static str,
    pub example: &'static str,
}

macro_rules! codes {
    ($( $code:literal, $title:literal, $expl:literal, $ex:literal; )*) => {
        pub static CODES: &[CodeInfo] = &[ $( CodeInfo { code: $code, title: $title, explanation: $expl, example: $ex } ),* ];
    };
}

codes! {
"V0001", "invalid token",
"The lexer found a character or literal it does not understand, for example an unterminated string or a stray `$`.",
"let s = \"unterminated";

"V0002", "syntax error",
"The parser expected a different token here. The message says what was expected.",
"fn main( {";

"V0100", "unknown name",
"A variable, function, model, prefix, tool, schema or constant was used but never declared in scope.",
"fn main() { print(nope) }";

"V0101", "use of moved context",
"Contexts (`Ctx<M>`) are linear: every context operation (`++`, `gen`, `fork`, `splice`, `drop`, passing to a function) consumes its input and returns a new handle. Using the old handle afterwards would read KV-cache state that no longer exists, so it is rejected. Rebind the result instead: `let c = c ++ \"x\"`.",
"model m = \"x\" { layers: 1, kv_heads: 1, head_dim: 8 }\nfn main() {\n  let c = open m\n  let d = c ++ \"hi\"\n  let e = c ++ \"again\"   // c was moved into d\n  drop d, e\n}";

"V0102", "context is never consumed",
"Every context must be consumed exactly once: returned, passed on, committed, or released with `drop`. A context that silently goes out of scope would leak KV-cache memory on the serving engine, so Veer makes the release explicit.",
"model m = \"x\" { layers: 1, kv_heads: 1, head_dim: 8 }\nfn main() {\n  let c = open m   // never dropped\n}";

"V0103", "branches disagree about a context",
"After an `if`/`else`, every context that existed before must be in the same state (live or consumed) on both paths. Otherwise the program would not know, after the branch, whether the KV cache still exists.",
"model m = \"x\" { layers: 1, kv_heads: 1, head_dim: 8 }\nfn main() {\n  let c = open m\n  if true { drop c }\n}";

"V0104", "loop changes context state",
"A loop body must leave every outer context in the state it found it. To grow a context in a loop, consume and re-assign it: `c = c ++ \"step\"`.",
"model m = \"x\" { layers: 1, kv_heads: 1, head_dim: 8 }\nfn main() {\n  let c = open m\n  for i in 0..3 { drop c }\n}";

"V0105", "context cannot be duplicated",
"A context may not be stored twice, put in a non-destructured tuple, or used where a copy would be made. Use `fork` to get two handles that share the KV-cache prefix.",
"model m = \"x\" { layers: 1, kv_heads: 1, head_dim: 8 }\nfn main() { let c = open m\n let t = (c, 1) }";

"V0106", "model mismatch",
"Two contexts that must belong to the same model (for example the result of `commit` and the speculated context, or a function argument and parameter) were opened on different models.",
"model a = \"x\" { layers: 1, kv_heads: 1, head_dim: 8 }\nmodel b = \"y\" { layers: 1, kv_heads: 1, head_dim: 8 }\nfn f(c: Ctx<a>) { drop c }\nfn main() { f(open b) }";

"V0110", "type mismatch",
"An expression has a different type than the position requires.",
"fn main() { let x: int = \"text\" }";

"V0111", "wrong number of arguments",
"A function, tool, or builtin was called with the wrong number of arguments.",
"fn f(x: int) -> int { return x }\nfn main() { print(f(1, 2)) }";

"V0112", "string bound exceeded",
"Strings carry a static upper bound on their token count (`str<=N`). An expression whose bound is larger than the declared bound cannot be stored there. Increase the declared bound or shorten the expression.",
"fn f() -> str<=3 { return \"far too long\" }";

"V0113", "unbounded loop",
"Loops must have a statically known trip count (`for i in 0..N` with constant `N`, or iteration over a bounded list). This is what lets the planner bound KV memory and prefill cost.",
"fn f(n: int) { for i in 0..n { } }";

"V0114", "recursion is not allowed",
"Veer forbids recursive functions so that every program has a finite, statically analyzable context graph.",
"fn f() { f() }";

"V0115", "unknown field",
"The schema or model declaration has no field with this name.",
"schema S { a: int }\nfn g(s: S) -> int { return s.b }";

"V0116", "structured output does not fit",
"The smallest valid output for this `json<T>` schema or `choice[...]` needs more tokens than `max` allows, so the generation could never succeed.",
"schema S { name: str<=10, age: int }\nmodel m = \"x\" { layers: 1, kv_heads: 1, head_dim: 8 }\nfn main() { let (c, s) = gen open m max 2 as json<S>\n drop c }";

"V0117", "unbounded string appended to a context",
"Only strings with a static bound (`str<=N`) may be appended to a context, otherwise KV memory could not be bounded.",
"model m = \"x\" { layers: 1, kv_heads: 1, head_dim: 8 }\nfn f(s: str) { let c = open m ++ s\n drop c }";

"V0118", "invalid declaration",
"A model, schema, tool or attribute declaration is malformed, for example a model missing `layers`, an unknown KV format, or an unknown attribute.",
"model m = \"x\" { layers: 1 }";

"V0119", "duplicate definition",
"Two items, parameters, or fields have the same name.",
"fn f() {}\nfn f() {}";

"V0120", "unknown span",
"`splice c at name` requires that `name` labels a region (`#[name]`) that is present in the context `c`.",
"model m = \"x\" { layers: 1, kv_heads: 1, head_dim: 8 }\nfn main() { let c = open m ++ \"hi\"\n let c = splice c at doc with \"x\"\n drop c }";

"V0121", "duplicate span label",
"A context may contain each span label at most once, so that `splice` is unambiguous.",
"model m = \"x\" { layers: 1, kv_heads: 1, head_dim: 8 }\nfn main() { let c = open m ++ #[a] \"x\" ++ #[a] \"y\"\n drop c }";

"V0130", "speculation must end in commit or rollback",
"Every path through a `speculate` block must end with `commit (ctx, value)` or `rollback value`. That is what makes speculation transactional.",
"model m = \"x\" { layers: 1, kv_heads: 1, head_dim: 8 }\nfn main() {\n  let c = open m\n  let (c, v) = speculate c { let x = 1 }\n  drop c\n}";

"V0131", "outer context used inside speculation",
"Inside `speculate`, only the speculated context and contexts created inside the block are accessible. Other outer contexts are frozen so that a rollback can restore the exact prior state.",
"";

"V0132", "statement not allowed here",
"`return` is not allowed inside `speculate`, and `commit`/`rollback` are only allowed inside `speculate`.",
"fn main() { rollback 1 }";

"V0200", "KV memory budget exceeded",
"The planner computed a worst-case peak KV-cache footprint larger than the function's `@budget(kv = ...)`. The plan output shows which contexts are live at the peak. Reduce `max`, drop contexts earlier, use `fork` to share prefixes, or use a smaller KV format.",
"";

"V0201", "prefill budget exceeded",
"The worst-case number of prompt tokens the engine must prefill exceeds `@budget(prefill = ...)`.",
"";

"V0202", "context length exceeded",
"Some context could grow longer than the model's `max_ctx`.",
"";

"W0210", "splice forces re-prefill",
"Splicing a span that is followed by other content forces every following token to be re-prefilled on engines without position-corrected KV editing. The warning reports the worst-case token count.",
"";

"V0300", "tensor shape mismatch",
"The static shapes of tensors are incompatible, for example `[1, 256] @ [128, 64]`.",
"fn main() { let a = zeros([1, 4])\n let b = zeros([3, 2])\n let c = a @ b }";

"V0301", "accuracy bound violated",
"The compiler's sound, worst-case error bound for the returned tensor is larger than the function's `@accuracy(...)` contract. The error is tracked through every quantization, matmul, and elementwise op.",
"";

"V0302", "possible overflow in number format",
"A tensor whose static magnitude bound exceeds the largest finite value of a non-scaled format (for example 448 for fp8_e4m3) is being quantized. Scale it first or use a block-scaled format (mxfp4, nvfp4, int8).",
"fn main() { let t = rand_uniform([4], -1000.0, 1000.0, 1)\n let q = quant(t, fp8_e4m3) }";

"V0303", "unknown number format",
"Formats are: f32, bf16, f16, fp8_e4m3, fp8_e5m2, int8, mxfp4, mxfp4_oas, nvfp4.",
"";
}

pub fn lookup(code: &str) -> Option<&'static CodeInfo> {
    CODES.iter().find(|c| c.code == code)
}

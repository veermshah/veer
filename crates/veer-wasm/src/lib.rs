//! Browser entry points for the Veer playground: check, plan, types, and run
//! (on the simulated engine). Each returns a JSON string `{ ok, output }`.

use veer_plan::{human_bytes, plan_fn, plan_program};
use veer_runtime::{sim_engine, Engine, Interp, Value};
use veer_syntax::diag::render_all;
use veer_syntax::Source;
use wasm_bindgen::prelude::*;

fn reply(ok: bool, output: String) -> String {
    serde_json::json!({ "ok": ok, "output": output }).to_string()
}

fn frontend(src: &Source) -> Result<(veer_syntax::ast::Program, veer_types::Checked, String), String> {
    match veer_types::frontend(&src.text) {
        Ok((p, c, w)) => Ok((p, c, render_all(&w, src))),
        Err(d) => Err(render_all(&d, src)),
    }
}

/// `veer check`
#[wasm_bindgen]
pub fn check(code: &str) -> String {
    let src = Source::new("playground.veer", code);
    let (prog, ck, warns) = match frontend(&src) {
        Ok(x) => x,
        Err(e) => return reply(false, e),
    };
    let (plans, diags) = plan_program(&prog, &ck, &src);
    let mut out = warns;
    out.push_str(&render_all(&diags, &src));
    if diags.iter().any(|d| d.is_error()) {
        return reply(false, out);
    }
    let plans: Vec<_> = plans.into_iter().filter(|p| !p.trace.is_empty()).collect();
    out.push_str(&format!("ok: {} function(s) checked, {} planned\n", ck.fns.len(), plans.len()));
    for p in &plans {
        let budget = p.budget_kv_bytes.map(|b| format!(" / budget {}", human_bytes(b as f64))).unwrap_or_default();
        out.push_str(&format!("  {:<16} peak KV ≤ {}{budget} · prefill ≤ {} tok · decode ≤ {} tok\n", p.entry, human_bytes(p.peak_kv_bytes), p.prefill_tokens, p.decode_tokens));
    }
    reply(true, out)
}

/// `veer plan`
#[wasm_bindgen]
pub fn plan(code: &str) -> String {
    let src = Source::new("playground.veer", code);
    let (prog, ck, mut out) = match frontend(&src) {
        Ok(x) => x,
        Err(e) => return reply(false, e),
    };
    let (plans, diags) = plan_program(&prog, &ck, &src);
    out.push_str(&render_all(&diags, &src));
    if plans.is_empty() {
        out.push_str("nothing to plan: add `fn main()` or a function with @budget\n");
    }
    for p in &plans {
        out.push_str(&p.render());
        out.push('\n');
    }
    reply(!diags.iter().any(|d| d.is_error()), out)
}

/// `veer types`
#[wasm_bindgen]
pub fn types(code: &str) -> String {
    let src = Source::new("playground.veer", code);
    let (_, ck, mut out) = match frontend(&src) {
        Ok(x) => x,
        Err(e) => return reply(false, e),
    };
    for name in &ck.fn_order {
        let f = &ck.fns[name];
        let ps: Vec<String> = f.sig.params.iter().map(|(n, t)| format!("{n}: {t}")).collect();
        out.push_str(&format!("fn {name}({}) -> {}\n", ps.join(", "), f.sig.ret));
    }
    reply(true, out)
}

/// `veer run --stats` on the simulated engine (entry `main`, no arguments).
#[wasm_bindgen]
pub fn run(code: &str) -> String {
    let src = Source::new("playground.veer", code);
    let (prog, ck, mut out) = match frontend(&src) {
        Ok(x) => x,
        Err(e) => return reply(false, e),
    };
    let Some(info) = ck.fns.get("main") else { return reply(false, "no `fn main()` to run".into()) };
    if !info.sig.params.is_empty() {
        return reply(false, "the playground runs `fn main()` without parameters".into());
    }
    let plan = plan_fn(&prog, &ck, &src, "main");
    if let Some(p) = &plan {
        let errs: Vec<_> = p.diags.iter().filter(|d| d.is_error()).cloned().collect();
        if !errs.is_empty() {
            out.push_str(&render_all(&errs, &src));
            return reply(false, out);
        }
    }
    let mut eng = sim_engine();
    let mut it = Interp::new(&prog, &ck, &mut eng);
    it.echo = false;
    let result = it.run("main", vec![]);
    for line in &it.output {
        out.push_str(line);
        out.push('\n');
    }
    match result {
        Ok(Value::Unit) => {}
        Ok(v) => out.push_str(&format!("=> {v}\n")),
        Err(e) => {
            let at = e.span.map(|s| {
                let (l, c) = src.line_col(s.start);
                format!(" at {}:{}", l + 1, c + 1)
            });
            out.push_str(&format!("runtime error{}: {}\n", at.unwrap_or_default(), e.message));
            return reply(false, out);
        }
    }
    let s = eng.stats();
    out.push_str("\n── engine: sim ──\n");
    if let Some(p) = &plan {
        let row = |name: &str, m: String, pl: String, ok: bool| format!("  {name:<16} {m:>14}   planned ≤ {pl:<14} {}\n", if ok { "✓" } else { "✗" });
        out.push_str(&row("peak KV", human_bytes(s.store.peak_kv_bytes), human_bytes(p.peak_kv_bytes), s.store.peak_kv_bytes <= p.peak_kv_bytes + 1e-6));
        out.push_str(&row("prefill tokens", s.store.prefill_tokens.to_string(), p.prefill_tokens.to_string(), s.store.prefill_tokens <= p.prefill_tokens));
        out.push_str(&row("decode tokens", s.store.decode_tokens.to_string(), p.decode_tokens.to_string(), s.store.decode_tokens <= p.decode_tokens));
    }
    out.push_str(&format!("  {:<16} {:>14}\n  {:<16} {:>14}\n", "reused tokens", s.store.reused_tokens, "gen calls", s.gen_calls));
    reply(true, out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn entry_points_work_natively() {
        let code = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/review.veer")).unwrap();
        for f in [super::check, super::plan, super::types, super::run] {
            let v: serde_json::Value = serde_json::from_str(&f(&code)).unwrap();
            assert_eq!(v["ok"], true, "{}", v["output"]);
        }
        let v: serde_json::Value = serde_json::from_str(&super::check("fn main() { let x: int = \"s\" }")).unwrap();
        assert_eq!(v["ok"], false);
        assert!(v["output"].as_str().unwrap().contains("V0110"));
    }
}

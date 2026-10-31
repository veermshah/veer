//! Soundness of the static planner: for randomly generated well-typed context
//! programs, the simulated engine's measured peak KV memory, prefill tokens and
//! decode tokens never exceed the planner's static bounds.

use proptest::prelude::*;
use std::collections::BTreeSet;
use veer_numerics::tensor::Rng;
use veer_runtime::{sim_engine, Engine, Interp};
use veer_syntax::Source;

struct Gen {
    rng: Rng,
    out: String,
    next: usize,
    /// live ctx vars with their "must" labels
    live: Vec<(String, BTreeSet<String>)>,
}

impl Gen {
    fn r(&mut self, n: u64) -> u64 {
        self.rng.next_u64() % n.max(1)
    }
    fn fresh(&mut self, p: &str) -> String {
        self.next += 1;
        format!("{p}{}", self.next)
    }
    fn lit(&mut self) -> String {
        let n = self.r(60) as usize;
        let words = ["alpha", "beta", "kv", "page", "a", "zz", "token"];
        let mut s = String::new();
        while s.len() < n {
            s.push_str(words[self.r(words.len() as u64) as usize]);
            s.push(' ');
        }
        s
    }
    fn line(&mut self, indent: usize, s: &str) {
        self.out.push_str(&"  ".repeat(indent));
        self.out.push_str(s);
        self.out.push('\n');
    }
    fn pick(&mut self) -> Option<usize> {
        if self.live.is_empty() {
            None
        } else {
            Some(self.r(self.live.len() as u64) as usize)
        }
    }

    fn stmt(&mut self, depth: usize, indent: usize) {
        let choice = self.r(if depth < 2 { 11 } else { 8 });
        match (choice, self.pick()) {
            (0, _) | (_, None) => {
                let v = self.fresh("v");
                let src = if self.r(2) == 0 { "m" } else { "P" };
                self.line(indent, &format!("let {v} = open {src}"));
                self.live.push((v, BTreeSet::new()));
            }
            (1 | 2, Some(i)) => {
                let lit = self.lit();
                let v = self.live[i].0.clone();
                if self.r(3) == 0 {
                    let l = self.fresh("l");
                    self.line(indent, &format!("{v} = {v} ++ #[{l}] \"{lit}\""));
                    self.live[i].1.insert(l);
                } else {
                    self.line(indent, &format!("{v} = {v} ++ \"{lit}\""));
                }
            }
            (3, Some(i)) => {
                let (v, labels) = self.live.remove(i);
                let (a, b) = (self.fresh("v"), self.fresh("v"));
                self.line(indent, &format!("let ({a}, {b}) = fork {v}"));
                self.live.push((a, labels.clone()));
                self.live.push((b, labels));
            }
            (4 | 5, Some(i)) => {
                let k = 1 + self.r(64);
                let v = self.live[i].0.clone();
                self.line(indent, &format!("({v}, s) = gen {v} ++ \"q\" max {k}"));
            }
            (6, Some(i)) => {
                if let Some(l) = self.live[i].1.iter().next().cloned() {
                    let lit = self.lit();
                    let v = self.live[i].0.clone();
                    self.line(indent, &format!("{v} = splice {v} at {l} with \"{lit}\""));
                    self.live[i].1.remove(&l);
                }
            }
            (7, Some(i)) => {
                if self.live.len() > 1 {
                    let (v, _) = self.live.remove(i);
                    self.line(indent, &format!("drop {v}"));
                }
            }
            (8, Some(i)) => {
                // Branches grow the same variable differently.
                let v = self.live[i].0.clone();
                let k = 1 + self.r(64);
                let (l1, l2) = (self.lit(), self.lit());
                self.line(indent, "if contains(s, \"a\") {");
                self.line(indent + 1, &format!("{v} = {v} ++ \"{l1}\""));
                self.line(indent + 1, &format!("({v}, s) = gen {v} max {k}"));
                self.line(indent, "} else {");
                self.line(indent + 1, &format!("{v} = {v} ++ \"{l2}\""));
                self.line(indent, "}");
            }
            (9, Some(i)) => {
                let v = self.live[i].0.clone();
                let n = 1 + self.r(4);
                let k = 1 + self.r(40);
                let l = self.lit();
                self.line(indent, &format!("for it in 0..{n} {{"));
                self.line(indent + 1, &format!("{v} = {v} ++ \"{l}\""));
                self.line(indent + 1, &format!("({v}, s) = gen {v} max {k}"));
                self.line(indent, "}");
            }
            (10, Some(i)) => {
                let (v, labels) = self.live.remove(i);
                let w = self.fresh("v");
                let t = self.fresh("t");
                let k = 1 + self.r(64);
                self.line(indent, &format!("let ({w}, {t}) = speculate {v} {{"));
                self.line(indent + 1, &format!("let ({v}, x) = gen {v} ++ \"try\" max {k}"));
                self.line(indent + 1, &format!("if contains(x, \"a\") {{ commit ({v}, x) }} else {{ rollback \"\" }}"));
                self.line(indent, "}");
                self.live.push((w, labels));
            }
            _ => {}
        }
    }
}

fn program(seed: u64) -> String {
    let mut g = Gen { rng: Rng::new(seed), out: String::new(), next: 0, live: vec![] };
    let fmts = ["bf16", "fp8_e4m3", "f32", "nvfp4"];
    let fmt = fmts[g.r(4) as usize];
    let block = [1, 8, 16, 64][g.r(4) as usize];
    g.out.push_str(&format!("model m = \"x\" {{ layers: 2, kv_heads: 2, head_dim: 16, kv: {fmt}, block: {block} }}\n"));
    g.out.push_str("prefix P: m = \"You are a test prefix with some words.\"\n");
    g.out.push_str("fn main() {\n  let s: str<=64 = \"\"\n");
    let n = 3 + g.r(14);
    for _ in 0..n {
        g.stmt(0, 1);
    }
    let live: Vec<String> = g.live.iter().map(|(v, _)| v.clone()).collect();
    if !live.is_empty() {
        g.out.push_str(&format!("  drop {}\n", live.join(", ")));
    }
    g.out.push_str("}\n");
    g.out
}

fn check_one(seed: u64) -> Result<(), String> {
    let text = program(seed);
    let src = Source::new("gen.veer", text.clone());
    let (prog, ck, _) = veer_types::frontend(&text).map_err(|d| format!("generated program failed to check:\n{text}\n{}", veer_syntax::diag::render_all(&d, &src)))?;
    let plan = veer_plan::plan_fn(&prog, &ck, &src, "main").unwrap();
    if plan.diags.iter().any(|d| d.is_error()) {
        return Err(format!("planner error\n{text}"));
    }
    let mut eng = sim_engine();
    let mut it = Interp::new(&prog, &ck, &mut eng);
    it.echo = false;
    it.run("main", vec![]).map_err(|e| format!("runtime error {}\n{text}", e.message))?;
    let s = eng.stats();
    if s.store.peak_kv_bytes > plan.peak_kv_bytes + 1e-6 {
        return Err(format!("peak {} > planned {}\n{text}", s.store.peak_kv_bytes, plan.peak_kv_bytes));
    }
    if s.store.prefill_tokens > plan.prefill_tokens {
        return Err(format!("prefill {} > planned {}\n{text}", s.store.prefill_tokens, plan.prefill_tokens));
    }
    if s.store.decode_tokens > plan.decode_tokens {
        return Err(format!("decode {} > planned {}\n{text}", s.store.decode_tokens, plan.decode_tokens));
    }
    if s.store.kv_bytes_now > plan.resident_prefix_bytes + 1e-6 {
        return Err(format!("leak: {} bytes still resident after main\n{text}", s.store.kv_bytes_now - plan.resident_prefix_bytes));
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]
    #[test]
    fn planner_bounds_hold(seed in any::<u64>()) {
        if let Err(e) = check_one(seed) {
            prop_assert!(false, "{}", e);
        }
    }
}

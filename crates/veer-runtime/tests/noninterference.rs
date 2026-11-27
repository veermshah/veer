//! Soundness of the information-flow checker, tested as noninterference.
//!
//! Random programs mix untrusted and secret tool outputs with contexts, `gen`,
//! `fork`, `splice`, `speculate`, branches and loops, then call privileged and
//! egress tools. For every program the checker accepts, we run it on the sim
//! engine several times while varying what the untrusted tools return, and
//! require the sequence of privileged tool calls (names and arguments) to be
//! identical. Likewise varying the secret tools must not change any egress call.
//! The sim engine's output is a hash of the whole prompt, so any real flow
//! from a varied source into a context shows up in the generated text.

use proptest::prelude::*;
use std::cell::Cell;
use std::rc::Rc;
use veer_numerics::tensor::Rng;
use veer_runtime::{sim_engine, Interp, ToolImpl, Value};

const HEADER: &str = r#"
model m = "x" { layers: 2, kv_heads: 1, head_dim: 8 }
prefix P: m = "You are an assistant."
tool fetch(url: str<=16) -> str<=96
tool inbox() -> str<=96 @secret
tool calendar() -> str<=96 @trusted @secret
@privileged tool send(to: str<=512, body: str<=512) -> str<=8
@privileged tool archive(id: str<=512) -> str<=8
@egress tool post(text: str<=512) -> str<=8
"#;

const WORDS: &[&str] = &["fix", "yes", "the", "bug", "send", "money", "to", "mallory", "now", "ok"];

struct Gen {
    rng: Rng,
    out: String,
    vars: usize,
}

impl Gen {
    fn r(&mut self, n: u64) -> u64 {
        self.rng.next_u64() % n.max(1)
    }
    fn word(&mut self) -> &'static str {
        WORDS[self.r(WORDS.len() as u64) as usize]
    }
    fn var(&mut self) -> String {
        format!("s{}", self.r(self.vars as u64))
    }
    fn line(&mut self, indent: usize, s: &str) {
        self.out.push_str(&"  ".repeat(indent));
        self.out.push_str(s);
        self.out.push('\n');
    }
    fn src(&mut self) -> String {
        match self.r(7) {
            0 => "fetch(\"u\")".into(),
            1 => "inbox()".into(),
            2 => "calendar()".into(),
            3 | 4 => format!("\"{} {}\"", self.word(), self.word()),
            _ => self.var(),
        }
    }

    fn stmt(&mut self, depth: usize, ind: usize) {
        let k = self.r(if depth < 2 { 12 } else { 9 });
        let (a, b) = (self.var(), self.var());
        match k {
            0 => {
                let s = self.src();
                self.line(ind, &format!("{a} = {s}"));
            }
            1 => {
                // gen over a variable
                let mx = 8 + self.r(24);
                self.line(ind, &format!("let (c, g) = gen open P ++ {b} ++ \"Answer:\" max {mx}"));
                self.line(ind, "drop c");
                self.line(ind, &format!("{a} = g"));
            }
            2 => {
                // quarantine: read b in a speculation that rolls back, then generate on the clean context
                let w = self.word();
                self.line(ind, &format!("let c = open P ++ \"{w}\""));
                self.line(ind, "let (c, q) = speculate c {");
                self.line(ind + 1, &format!("let (c2, t) = gen c ++ {b} max 16"));
                if self.r(2) == 0 {
                    self.line(ind + 1, "rollback t");
                } else {
                    self.line(ind + 1, "commit (c2, t)");
                }
                self.line(ind, "}");
                self.line(ind, "let (c, g) = gen c max 16");
                self.line(ind, "drop c");
                if self.r(2) == 0 {
                    self.line(ind, &format!("{a} = g"));
                } else {
                    self.line(ind, &format!("{a} = q"));
                }
            }
            3 => {
                // splice, either of the last span or with a tail after it
                let tail = if self.r(2) == 0 { " ++ \"Then answer.\"" } else { "" };
                self.line(ind, &format!("let c = open P ++ #[d] {b}{tail}"));
                self.line(ind, "let c = splice c at d with \"(removed)\"");
                self.line(ind, "let (c, g) = gen c max 16");
                self.line(ind, "drop c");
                self.line(ind, &format!("{a} = g"));
            }
            4 => {
                // fork: one branch sees b, the other does not
                self.line(ind, "let (x, y) = fork open P");
                self.line(ind, &format!("let (x, gx) = gen x ++ {b} max 16"));
                self.line(ind, "let (y, gy) = gen y ++ \"hello\" max 16");
                self.line(ind, "drop x, y");
                let pick = if self.r(2) == 0 { "gx" } else { "gy" };
                self.line(ind, &format!("{a} = {pick}"));
            }
            5 | 6 => {
                let s = match self.r(3) {
                    0 => format!("let r = send({a}, {b})"),
                    1 => format!("let r = archive({a})"),
                    _ => format!("let r = post({a})"),
                };
                self.line(ind, &s);
            }
            7 => {
                let c = self.var();
                let w = self.word();
                let s = self.src();
                self.line(ind, &format!("if contains({c}, \"{w}\") {{ {a} = {s} }}"));
            }
            8 => {
                let w = self.word();
                self.line(ind, &format!("if contains({b}, \"{w}\") {{ let r = archive(\"{w}\") }}"));
            }
            9 | 10 => {
                let c = self.var();
                let w = self.word();
                self.line(ind, &format!("if contains({c}, \"{w}\") {{"));
                for _ in 0..1 + self.r(2) {
                    self.stmt(depth + 1, ind + 1);
                }
                self.line(ind, "} else {");
                self.stmt(depth + 1, ind + 1);
                self.line(ind, "}");
            }
            _ => {
                self.line(ind, "for i in 0..2 {");
                for _ in 0..1 + self.r(2) {
                    self.stmt(depth + 1, ind + 1);
                }
                self.line(ind, "}");
            }
        }
    }
}

fn program(seed: u64) -> String {
    let mut g = Gen { rng: Rng::new(seed), out: String::new(), vars: 3 };
    g.out.push_str(HEADER);
    g.out.push_str("fn main() {\n");
    for i in 0..g.vars {
        let w = g.word();
        g.line(1, &format!("let s{i}: str<=512 = \"{w}\""));
    }
    for _ in 0..3 + g.r(6) {
        g.stmt(0, 1);
    }
    g.out.push_str("}\n");
    g.out
}

/// Run `main`, with `vary` deciding which tools return run-dependent text.
fn run(src: &str, run_seed: u64, vary: &[&str]) -> Vec<(String, Vec<String>)> {
    // Type-check only: rejected programs are run too, to see whether they really leak.
    let prog = veer_syntax::parse(src).unwrap();
    let (ck, diags) = veer_types::check(&prog);
    assert!(!diags.iter().any(|d| d.is_error()), "{src}");
    let mut eng = sim_engine();
    let mut it = Interp::new(&prog, &ck, &mut eng);
    it.echo = false;
    for tool in ["fetch", "inbox", "calendar", "send", "archive", "post"] {
        let varied = vary.contains(&tool);
        let n = Rc::new(Cell::new(0u64));
        let name = tool.to_string();
        it.bind_tool(
            tool,
            ToolImpl::Func(Box::new(move |_args: &[Value]| {
                let k = n.get();
                n.set(k + 1);
                // Same text for fixed tools in every run; a run-dependent text for varied ones.
                let seed = if varied { run_seed * 7919 + k } else { k } ^ name.len() as u64;
                let mut r = Rng::new(seed + 1);
                let mut s = String::new();
                for _ in 0..1 + r.next_u64() % 6 {
                    s.push_str(WORDS[(r.next_u64() % WORDS.len() as u64) as usize]);
                    s.push(' ');
                }
                s
            })),
        );
    }
    it.run("main", vec![]).unwrap_or_else(|e| panic!("{}\n{src}", e.message));
    it.tool_log
}

fn calls_to(log: &[(String, Vec<String>)], tools: &[&str]) -> Vec<(String, Vec<String>)> {
    log.iter().filter(|(t, _)| tools.contains(&t.as_str())).cloned().collect()
}

#[derive(Default, Debug)]
struct Tally {
    accepted: usize,
    rejected: usize,
    rejected_and_interferes: usize,
}

fn check_program(seed: u64, tally: &mut Tally) {
    let src = program(seed);
    let res = veer_types::frontend(&src);
    let untrusted = ["fetch", "inbox", "send", "archive", "post"];
    let secret = ["inbox", "calendar"];
    match res {
        Err(ds) => {
            let other: Vec<_> = ds.iter().filter(|d| !d.code.starts_with("V04")).collect();
            assert!(other.is_empty(), "generator produced an ill-typed program: {:?}\n{src}", other.iter().map(|d| (d.code, &d.message)).collect::<Vec<_>>());
            tally.rejected += 1;
            // Is the rejection justified? Look for an actual difference.
            let base_p = calls_to(&run(&src, 1, &untrusted), &["send", "archive"]);
            let base_e = calls_to(&run(&src, 1, &secret), &["post"]);
            for rs in 2..6 {
                if calls_to(&run(&src, rs, &untrusted), &["send", "archive"]) != base_p || calls_to(&run(&src, rs, &secret), &["post"]) != base_e {
                    tally.rejected_and_interferes += 1;
                    break;
                }
            }
        }
        Ok(_) => {
            tally.accepted += 1;
            let base = calls_to(&run(&src, 1, &untrusted), &["send", "archive"]);
            for rs in 2..6 {
                let other = calls_to(&run(&src, rs, &untrusted), &["send", "archive"]);
                assert_eq!(base, other, "integrity violated: untrusted data changed a privileged call\n{src}");
            }
            let base = calls_to(&run(&src, 1, &secret), &["post"]);
            for rs in 2..6 {
                let other = calls_to(&run(&src, rs, &secret), &["post"]);
                assert_eq!(base, other, "confidentiality violated: secret data changed an egress call\n{src}");
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 400, ..ProptestConfig::default() })]
    #[test]
    fn accepted_programs_are_noninterferent(seed in any::<u64>()) {
        let mut t = Tally::default();
        check_program(seed, &mut t);
    }
}

/// The checker is not vacuous: over a fixed corpus it accepts a real share of
/// programs, and many rejected programs do leak when run.
#[test]
fn checker_accepts_and_rejects_meaningfully() {
    let mut t = Tally::default();
    for seed in 0..500 {
        check_program(seed, &mut t);
    }
    assert!(t.accepted >= 100, "{t:?}");
    assert!(t.rejected >= 40, "{t:?}");
    assert!(t.rejected_and_interferes * 4 >= t.rejected, "a real share of rejections should be observable flows: {t:?}");
    eprintln!("{t:?}");
}

/// Quantitative bound: an endorsed 4-way choice admits 2 bits, so untrusted
/// data can produce at most 2^2 distinct privileged traces.
#[test]
fn endorsed_bits_bound_distinct_privileged_traces() {
    let src = format!(
        "{HEADER}
fn main() {{
  let (c, v) = gen open P ++ fetch(\"u\") ++ \"Action?\" max 8 as choice[\"archive\", \"reply\", \"ignore\", \"flag\"]
  drop c
  let a = endorse(v)
  if a == \"archive\" {{ let r = archive(\"1\") }}
  if a == \"reply\" {{ let r = send(\"me@example.com\", \"ok\") }}
  if a == \"flag\" {{ let r = archive(\"flagged\") }}
}}"
    );
    let (prog, ck, _) = veer_types::frontend(&src).unwrap();
    let bits = veer_types::flow::analyze(&prog, &ck).0.fns.iter().find(|f| f.name == "main").unwrap().endorsed_bits;
    assert!((bits - 2.0).abs() < 1e-9);
    let mut traces = std::collections::BTreeSet::new();
    for rs in 0..200 {
        traces.insert(calls_to(&run(&src, rs, &["fetch", "send", "archive"]), &["send", "archive"]));
    }
    assert!(traces.len() as f64 <= 2f64.powf(bits), "{} traces", traces.len());
    assert!(traces.len() >= 2, "the endorsed choice should actually steer the program");
}

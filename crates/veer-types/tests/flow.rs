//! Information-flow rules: each test is a small program and the codes it must
//! (or must not) produce.

const HEADER: &str = r#"
model m = "x" { layers: 1, kv_heads: 1, head_dim: 8 }
prefix SYS: m = "You are an email assistant."
tool fetch(url: str<=64) -> str<=256
tool inbox() -> str<=512 @secret
tool calendar() -> str<=256 @trusted @secret
@privileged tool send(to: str<=512, body: str<=512) -> str<=8
@egress tool post(text: str<=512) -> str<=8
@privileged tool archive(id: str<=256) -> str<=8
"#;

fn codes(body: &str) -> Vec<&'static str> {
    let src = format!("{HEADER}\n{body}");
    let mut v: Vec<&'static str> = match veer_types::frontend(&src) {
        Ok((_, _, ds)) => ds.iter().map(|d| d.code).collect(),
        Err(ds) => ds.iter().map(|d| d.code).collect(),
    };
    v.sort();
    v.dedup();
    v
}

fn ok(body: &str) {
    let c = codes(body);
    assert!(c.is_empty(), "expected no diagnostics, got {c:?}\n{body}");
}

fn err(body: &str, code: &str) {
    let c = codes(body);
    assert!(c.contains(&code), "expected {code}, got {c:?}\n{body}");
}

fn report(body: &str) -> veer_types::flow::FlowReport {
    let src = format!("{HEADER}\n{body}");
    let (prog, ck, _) = veer_types::frontend(&src).unwrap_or_else(|d| panic!("{d:?}"));
    veer_types::flow::analyze(&prog, &ck).0
}

#[test]
fn trusted_constants_reach_privileged_tools() {
    ok(r#"fn main() { let r = send("bob@example.com", "hi") }"#);
}

#[test]
fn tool_output_is_untrusted() {
    err(r#"fn main() { let r = send(fetch("u"), "hi") }"#, "V0400");
}

#[test]
fn llm_output_over_untrusted_context_is_untrusted() {
    err(
        r#"fn main() {
  let (c, to) = gen open SYS ++ fetch("u") ++ "Recipient:" max 16
  drop c
  let r = send(to, "hi")
}"#,
        "V0400",
    );
}

#[test]
fn llm_output_over_trusted_context_is_trusted() {
    ok(r#"fn main() {
  let (c, body) = gen open SYS ++ "Write a greeting." max 64
  drop c
  let r = send("bob@example.com", body)
}"#);
}

#[test]
fn implicit_flow_through_if_is_caught() {
    err(r#"fn main() { if contains(fetch("u"), "x") { let r = archive("1") } }"#, "V0401");
    err(
        r#"fn main() {
  let to: str<=64 = "alice@example.com"
  if contains(fetch("u"), "x") { to = "mallory@example.com" }
  let r = send(to, "hi")
}"#,
        "V0400",
    );
}

#[test]
fn early_return_on_untrusted_data_taints_the_rest() {
    err(
        r#"fn main() {
  if contains(fetch("u"), "stop") { return }
  let r = archive("1")
}"#,
        "V0401",
    );
}

#[test]
fn speculate_rollback_is_a_quarantine() {
    // The page is read in a speculation that always rolls back: the context
    // comes out exactly as it went in, so later generations stay trusted.
    ok(r#"fn main() {
  let c = open SYS ++ "Draft a status update."
  let (c, note) = speculate c {
    let (c2, s) = gen c ++ fetch("u") ++ "Summarize:" max 64
    rollback s
  }
  print(note)
  let (c, body) = gen c max 64
  drop c
  let r = send("bob@example.com", body)
}"#);
    // ... but the value carried out of the quarantine is still untrusted.
    err(
        r#"fn main() {
  let c = open SYS
  let (c, note) = speculate c {
    let (c2, s) = gen c ++ fetch("u") ++ "Summarize:" max 64
    rollback s
  }
  drop c
  let r = send("bob@example.com", note)
}"#,
        "V0400",
    );
}

#[test]
fn speculate_commit_keeps_taint() {
    err(
        r#"fn main() {
  let c = open SYS
  let (c, s) = speculate c {
    let (c2, s) = gen c ++ fetch("u") max 16
    commit (c2, s)
  }
  let (c, body) = gen c max 64
  drop c
  let r = send("bob@example.com", body)
}"#,
        "V0400",
    );
}

#[test]
fn splice_of_the_last_span_removes_taint() {
    ok(r#"fn main() {
  let c = open SYS ++ #[page] fetch("u")
  let c = splice c at page with "(page removed)"
  let (c, body) = gen c ++ "Write a greeting." max 64
  drop c
  let r = send("bob@example.com", body)
}"#);
}

#[test]
fn splice_cannot_launder_kv_computed_after_the_span() {
    // The instruction after the page was prefilled while attending to the
    // page; replacing the page does not recompute that KV.
    err(
        r#"fn main() {
  let c = open SYS ++ #[page] fetch("u") ++ "Write a greeting."
  let c = splice c at page with "(page removed)"
  let (c, body) = gen c max 64
  drop c
  let r = send("bob@example.com", body)
}"#,
        "V0400",
    );
}

#[test]
fn fork_branches_are_independent() {
    ok(r#"fn main() {
  let (a, b) = fork open SYS
  let (a, s) = gen a ++ fetch("u") max 32
  print(s)
  drop a
  let (b, body) = gen b ++ "Write a greeting." max 64
  drop b
  let r = send("bob@example.com", body)
}"#);
}

#[test]
fn endorse_finite_choice() {
    ok(r#"fn main() {
  let (c, verdict) = gen open SYS ++ fetch("u") ++ "Is this spam?" max 4 as choice["spam", "ham"]
  drop c
  let v = endorse(verdict)
  if v == "spam" { let r = archive("1") }
}"#);
    let rep = report(
        r#"fn main() {
  let (c, verdict) = gen open SYS ++ fetch("u") ++ "Action?" max 8 as choice["archive", "reply", "ignore", "flag"]
  drop c
  let v = endorse(verdict)
  if v == "archive" { let r = archive("1") }
}"#,
    );
    let main = rep.fns.iter().find(|f| f.name == "main").unwrap();
    assert_eq!(main.casts.len(), 1);
    assert!((main.endorsed_bits - 2.0).abs() < 1e-9);
    assert!(main.sinks.iter().all(|s| s.ok));
}

#[test]
fn endorse_free_text_is_rejected() {
    err(r#"fn main() { let p = endorse(fetch("u")) }"#, "V0404");
}

#[test]
fn literals_chosen_between_branches_can_be_endorsed() {
    ok(r#"fn main() {
  let action = "ignore"
  if contains(fetch("u"), "urgent") { action = "flag" }
  let a = endorse(action)
  let r = archive(a)
}"#);
}

#[test]
fn endorse_must_not_be_decided_by_untrusted_data() {
    err(r#"fn main() { if contains(fetch("u"), "x") { let b = endorse(true) } }"#, "V0405");
}

#[test]
fn endorsement_budget_counts_loop_iterations() {
    let prog = r#"@budget(endorse = 4bits)
fn main() {
  for i in 0..4 {
    let (c, v) = gen open SYS ++ fetch("u") max 8 as choice["spam", "ham", "unsure"]
    drop c
    let e = endorse(v)
    if e == "spam" { let r = archive("1") }
  }
}"#;
    err(prog, "V0406");
    let rep = report(&prog.replace("4bits", "7bits"));
    let main = rep.fns.iter().find(|f| f.name == "main").unwrap();
    assert!((main.endorsed_bits - 4.0 * 3f64.log2()).abs() < 1e-9, "{}", main.endorsed_bits);
}

#[test]
fn functions_are_label_polymorphic() {
    ok(r#"fn pass(x: str<=256) -> str<=256 { return x }
fn main() { let r = send(pass("bob@example.com"), "hi") }"#);
    err(
        r#"fn pass(x: str<=256) -> str<=256 { return x }
fn main() { let r = send(pass(fetch("u")), "hi") }"#,
        "V0400",
    );
}

#[test]
fn trusted_parameter_and_return_annotations() {
    err(
        r#"fn greet(to: str<=256 @trusted) { let r = send(to, "hi") }
fn main() { greet(fetch("u")) }"#,
        "V0400",
    );
    err(r#"fn f() -> str<=256 @trusted { return fetch("u") }"#, "V0400");
    err(r#"fn f(doc: str<=256 @untrusted) { let r = send(doc, "hi") }"#, "V0400");
    ok(r#"fn f(doc: str<=256) { let r = send(doc, "hi") }"#);
}

#[test]
fn trusted_tools_can_be_declared() {
    ok(r#"tool lookup(name: str<=256) -> str<=64 @trusted
fn main() { let r = send(lookup("bob"), "hi") }"#);
    err(
        r#"tool lookup(name: str<=256) -> str<=64 @trusted
fn main() { let r = send(lookup(fetch("u")), "hi") }"#,
        "V0400",
    );
}

#[test]
fn secret_data_cannot_reach_egress() {
    err(r#"fn main() { let r = post(inbox()) }"#, "V0402");
    err(
        r#"fn main() {
  let (c, s) = gen open SYS ++ inbox() ++ "Summarize:" max 64
  drop c
  let r = post(s)
}"#,
        "V0402",
    );
    err(r#"fn main() { if contains(inbox(), "bank") { let r = post("hello") } }"#, "V0403");
}

#[test]
fn declassify_requires_trusted_data() {
    err(r#"fn main() { let r = post(declassify(inbox())) }"#, "V0407");
    ok(r#"fn main() { let r = post(declassify(calendar())) }"#);
}

#[test]
fn privileged_tools_reject_secret_free_but_untrusted_bodies() {
    // `send` is privileged but not egress: secret bodies are allowed, untrusted ones are not.
    ok(r#"fn main() { let r = send("me@example.com", calendar()) }"#);
    err(r#"fn main() { let r = send("me@example.com", inbox()) }"#, "V0400");
}

#[test]
fn untrusted_parameters_of_privileged_tools_can_be_opted_in() {
    ok(r#"@privileged tool reply(thread: str<=16, body: str<=256 @untrusted) -> str<=8
fn main() { let r = reply("t1", fetch("u")) }"#);
}

#[test]
fn loop_accumulation_reaches_a_fixed_point() {
    err(
        r#"fn main() {
  let to: str<=256 = "bob@example.com"
  let note: str<=256 = "start"
  for i in 0..3 {
    to = note
    note = fetch("u")
  }
  let r = send(to, "hi")
}"#,
        "V0400",
    );
}

#[test]
fn bad_labels_and_attributes_are_rejected() {
    err(r#"fn f(x: str<=8 @trusted @untrusted) { }"#, "V0118");
    err(
        r#"@sudo tool x() -> str<=8
fn main() { }"#,
        "V0118",
    );
}

#[test]
fn existing_programs_without_sinks_are_unaffected() {
    ok(r#"tool search(q: str<=32) -> str<=128
fn main() {
  let (c, q) = gen open SYS ++ search("x") max 16
  let r = search(q)
  drop c
}"#);
}

#[test]
fn audit_lists_sources_and_sinks() {
    let rep = report(
        r#"fn main() {
  let page = fetch("u")
  let r = send("bob@example.com", "hi")
  let p = post("hello")
}"#,
    );
    let main = rep.fns.iter().find(|f| f.name == "main").unwrap();
    assert!(main.root);
    assert_eq!(main.sources.len(), 1);
    assert_eq!(main.sources[0].tool, "fetch");
    assert_eq!(main.sinks.len(), 2);
    assert!(main.sinks.iter().all(|s| s.ok && !s.pc.untrusted));
}

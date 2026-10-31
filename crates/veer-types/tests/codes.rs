//! Every example in the diagnostic registry must trigger exactly its own code.
use veer_syntax::codes::CODES;

#[test]
fn every_registry_example_triggers_its_code() {
    let mut failures = Vec::new();
    for c in CODES {
        if c.example.is_empty() {
            continue;
        }
        let codes: Vec<&str> = match veer_types::frontend(c.example) {
            Ok((_, _, warns)) => warns.iter().map(|d| d.code).collect(),
            Err(ds) => ds.iter().map(|d| d.code).collect(),
        };
        if !codes.contains(&c.code) {
            failures.push(format!("{}: got {:?}\n{}", c.code, codes, c.example));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n\n"));
}

use std::path::PathBuf;
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn veer(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_veer")).args(args).current_dir(root()).output().unwrap();
    (out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stdout).into(), String::from_utf8_lossy(&out.stderr).into())
}

fn examples() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(root().join("examples")).unwrap().map(|e| format!("examples/{}", e.unwrap().file_name().to_string_lossy())).filter(|p| p.ends_with(".veer")).collect();
    v.sort();
    v
}

#[test]
fn every_example_checks() {
    for ex in examples() {
        let (code, out, err) = veer(&["check", &ex]);
        assert_eq!(code, 0, "{ex}\n{out}\n{err}");
        assert!(out.starts_with("ok:"), "{ex}: {out}");
    }
}

#[test]
fn every_example_runs_within_its_plan() {
    for ex in examples() {
        let (code, _out, err) = veer(&["run", &ex, "--stats"]);
        assert_eq!(code, 0, "{ex}\n{err}");
        assert!(!err.contains('✗'), "{ex}: measured exceeded plan\n{err}");
    }
}

#[test]
fn plan_json_is_valid() {
    let (code, out, _) = veer(&["plan", "examples/review.veer", "--entry", "review", "--json"]);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert!(v[0]["peak_kv_bytes"].as_f64().unwrap() > 0.0);
    assert!(!v[0]["trace"].as_array().unwrap().is_empty());
}

#[test]
fn ui_tests_report_expected_codes() {
    let dir = root().join("tests/ui");
    let mut n = 0;
    for e in std::fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().map(|x| x != "veer").unwrap_or(true) {
            continue;
        }
        let text = std::fs::read_to_string(&p).unwrap();
        let expect = text.lines().next().unwrap().trim_start_matches("// expect:").trim().to_string();
        let rel = format!("tests/ui/{}", p.file_name().unwrap().to_string_lossy());
        let (code, _, err) = veer(&["check", &rel]);
        assert_eq!(code, 1, "{rel} should fail\n{err}");
        assert!(err.contains(&format!("[{expect}]")), "{rel}: expected {expect}\n{err}");
        n += 1;
    }
    assert!(n >= 5);
}

#[test]
fn explain_and_version() {
    let (code, out, _) = veer(&["explain", "V0101"]);
    assert_eq!(code, 0);
    assert!(out.contains("use of moved context"));
    let (code, out, _) = veer(&["--version"]);
    assert_eq!(code, 0);
    assert!(out.starts_with("veer "));
}

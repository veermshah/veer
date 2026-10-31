//! The HTTP engine against a tiny mock OpenAI-compatible server.
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use veer_runtime::{http_engine, Dialect, Engine, Interp};

fn serve(bodies: Arc<Mutex<Vec<serde_json::Value>>>, replies: Vec<&'static str>) -> String {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = format!("http://{}", l.local_addr().unwrap());
    std::thread::spawn(move || {
        for reply in replies {
            let (mut s, _) = l.accept().unwrap();
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut len = 0;
            loop {
                let mut line = String::new();
                r.read_line(&mut line).unwrap();
                if line.to_ascii_lowercase().starts_with("content-length:") {
                    len = line[15..].trim().parse().unwrap();
                }
                if line == "\r\n" {
                    break;
                }
            }
            let mut buf = vec![0; len];
            r.read_exact(&mut buf).unwrap();
            bodies.lock().unwrap().push(serde_json::from_slice(&buf).unwrap());
            let body = serde_json::json!({"choices": [{"text": reply}], "usage": {"prompt_tokens": 20, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 12}}}).to_string();
            write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
        }
    });
    addr
}

#[test]
fn http_engine_sends_full_context_and_constraints() {
    let src = r#"
model m = "org/model-1" { layers: 2, kv_heads: 2, head_dim: 16 }
prefix SYS: m = "System."
schema Ans { ok: bool, why: str<=40 }
fn main() {
  let c = open SYS ++ "\nQ: hi"
  let (a, b) = fork c
  let (a, x) = gen a max 8
  let (b, y) = gen b max 32 as json<Ans>
  let (b, z) = gen b max 4 as choice["yes", "no"]
  print(x, y.ok, y.why, z)
  drop a, b
}
"#;
    let (prog, ck, _) = veer_types::frontend(src).unwrap();
    let bodies = Arc::new(Mutex::new(vec![]));
    let addr = serve(bodies.clone(), vec![" hello", r#"{"ok": true, "why": "fine"}"#, "\"no\""]);
    let mut eng = http_engine(&addr, Dialect::LlamaCpp, None, None);
    let mut it = Interp::new(&prog, &ck, &mut eng);
    it.echo = false;
    it.run("main", vec![]).unwrap();
    assert_eq!(it.output, vec![" hello true fine no".to_string()]);
    let b = bodies.lock().unwrap();
    assert_eq!(b[0]["model"], "org/model-1");
    assert_eq!(b[0]["prompt"], "System.\nQ: hi");
    assert_eq!(b[0]["max_tokens"], 8);
    assert_eq!(b[1]["json_schema"]["properties"]["why"]["type"], "string");
    assert!(b[2]["prompt"].as_str().unwrap().ends_with(r#"{"ok": true, "why": "fine"}"#));
    assert_eq!(b[2]["json_schema"]["enum"][1], "no");
    let s = eng.stats();
    assert_eq!(s.server_cached_tokens, Some(36));
    assert_eq!(s.gen_calls, 3);
}

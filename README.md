# Veer

**A language where LLM contexts are typed, linear values, and the compiler proves KV-cache memory, prefill cost, and numeric accuracy before anything runs.**

📖 Docs and downloads: **https://veer-lang.vercel.app** (site source: [veermshah/veer-site](https://github.com/veermshah/veer-site))

```veer
model llama = "meta-llama/Llama-3.1-8B-Instruct" { layers: 32, kv_heads: 8, head_dim: 128, kv: fp8_e4m3 }
prefix SYS: llama = "You are a careful senior code reviewer."

@budget(kv = 2GiB, prefill = 12000)
fn review(doc: str<=4096) -> str<=384 {
  let c = open SYS ++ "\n<document>\n" ++ #[doc] doc ++ "\n</document>\n"
  let (a, b) = fork c                                   // shared prefix, proven zero-copy
  let (a, summary) = gen a ++ "Summarize the change:" max 128
  drop a
  let b = splice b at doc with "<document elided>"      // typed KV-cache edit
  let (b, patch) = speculate b {                         // transactional commit/rollback
    let (b, p) = gen b ++ "Propose a one-line patch:" max 64
    if contains(p, "fix") { commit (b, p) } else { rollback "no patch" }
  }
  drop b
  return summary + " | patch: " + patch
}
```

## Why

LLM programs (agents, RAG, tree search) are bottlenecked by KV-cache memory and prefill, but serving engines only *guess* at reuse (LRU over radix trees), editing an agent's history silently re-prefills thousands of tokens, and quantization accuracy is checked by running evals. Veer makes the structure explicit:

- **Linear contexts.** `Ctx<M>` values are consumed by every operation (`++`, `gen`, `fork`, `splice`, `speculate`, `drop`). Leaks, use-after-free, and ambiguous edits are compile errors.
- **Static KV planner.** An abstract interpreter proves sound upper bounds on peak KV bytes, prefill and decode tokens, context length, and splice re-prefill cost, and emits the context DAG plus reuse hints. `@budget(kv = …, prefill = …)` is a compile-time contract.
- **Precision types.** `Tensor<[64, 32], nvfp4, abs<=…, err<=…>` carries a proven error bound through quantization, matmul, and activations. `@accuracy(rel <= 0.25)` is a compile-time contract across FP8, INT8, MXFP4, and NVFP4.

See the [research survey](https://veer-lang.vercel.app/research/) for the papers behind each feature (SGLang, KVFlow, Leyline, AAFLOW+, transactional KV for speculative decoding, MXFP4/NVFP4 error analyses).

## Layout

| crate | role |
|---|---|
| `veer-syntax` | lexer, parser, AST, diagnostics, error-code registry |
| `veer-numerics` | number formats, bit-exact quantization simulators, sound error-bound algebra |
| `veer-types` | type checker: linearity, token bounds, spans (must/may), speculation, tensor bounds |
| `veer-plan` | static planner (abstract interpretation over paged KV segments) |
| `veer-runtime` | paged context store, `sim` and OpenAI-compatible `http` engines, interpreter |
| `veer-cli` | the `veer` binary: `check`, `plan`, `run`, `types`, `explain` |

## Develop

```sh
cargo test --workspace                 # unit, soundness property tests, CLI, HTTP mock
cargo run -p veer-cli -- run examples/review.veer --stats
python3 scripts/gen_site.py            # regenerate cookbook + error pages in ../veer-site
python3 scripts/check_docs.py          # compile every program in the docs
```

## Releasing

Push a version tag; the release workflow builds macOS (arm64, x64), Windows (x64, MSI), and Linux (x64, arm64) binaries and publishes them, with shell and PowerShell installers, to the public [veer-site releases](https://github.com/veermshah/veer-site/releases):

```sh
# bump `version` in Cargo.toml, then
git tag v0.1.0 && git push origin v0.1.0
```

Requires the `GH_RELEASES_TOKEN` repository secret (a token with `repo` access to `veermshah/veer-site`).

## License

MIT OR Apache-2.0

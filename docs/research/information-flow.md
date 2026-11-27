# Research note: why Veer's next feature is static information-flow control

*November 2026. This note records the survey behind the information-flow
feature (see [`docs/information-flow.md`](../information-flow.md)): what was
read, which gaps it showed, and why this one was chosen.*

## Method

Veer's code was mapped first (syntax, linear type checker, abstract-interpretation
planner, sim/HTTP runtime, precision types). Then four surveys ran in parallel
over 2024 to late-2026 work, reading arXiv abstracts and, where available, full
text:

1. LLM serving and agent systems (KV reuse, workflow compilers, cost),
2. security of LLM agents (prompt injection, information flow),
3. programming-languages work on LLM programs (DSLs, typed decoding, effects, resource types),
4. low-precision numerics (FP4 formats, KV quantization, sound error analysis).

Each survey ranked the five most valuable gaps a small Rust compiler could fill.
Key citations were then re-checked against their arXiv abstract pages. Numbers
below are the papers' own claims; most 2026 items are preprints.

## The candidates

| candidate | ranked by | why not first |
|---|---|---|
| **Static information-flow / prompt-injection safety** | security #1, PL #1, serving #4 | chosen |
| Cache-aware dollar bounds (`@budget(usd = ...)`) | serving #1, PL #2 | valuable and small; a natural follow-up that reuses the planner, but an incremental extension of an existing analysis |
| Probabilistic (sqrt n) error bounds for FP4 | numerics #1 | strong, but serves a narrower audience (kernel/quantization engineers) |
| Record/replay testing | PL #3 | useful, not a guarantee only a language can give |
| Effect rows and trace policies on tools | PL #4 | overlaps heavily with IFC sinks; IFC is the more fundamental piece |
| Particle inference (SMC) over `fork` | PL #5 | research-heavy, smaller market today |

## Why information flow

**The problem is the top unsolved risk for agents.** Prompt injection is first
on OWASP's LLM Top 10. Model-level defenses keep failing under adaptive attack:
*The Attacker Moves Second* (Nasr, Carlini, Tramèr et al.,
[arXiv 2510.09023](https://arxiv.org/abs/2510.09023)) bypasses 12 recent
defenses, most at over 90% attack success. The consensus answer is to constrain
the *system* so that untrusted data cannot trigger consequential actions,
whatever the model does (Beurer-Kellner et al., *Design Patterns for Securing
LLM Agents against Prompt Injections*,
[arXiv 2506.08837](https://arxiv.org/abs/2506.08837)).

**The working systems all enforce it at run time.**

* **CaMeL** (Debenedetti et al., *Defeating Prompt Injections by Design*,
  [arXiv 2503.18813](https://arxiv.org/abs/2503.18813)): a privileged LLM writes
  a program, an interpreter tracks data-flow capabilities, policies are checked
  at tool calls. 77% of AgentDojo tasks with provable security vs 84%
  undefended, at roughly 2.8x the tokens.
* **Fides** (Costa et al., *Securing AI Agents with Information-Flow Control*,
  [arXiv 2505.23643](https://arxiv.org/abs/2505.23643)): confidentiality and
  integrity labels with dynamic taint tracking in the planner, variable hiding,
  and policies such as "consequential tools need a trusted context".
* **APPA** ([arXiv 2607.24625](https://arxiv.org/abs/2607.24625)): disposable
  child branches absorb taint and return only through constrained channels;
  dynamic.
* Restricting data types between a quarantined and a privileged side
  (Jacob et al., *Preventing Prompt Injection with Type-Directed Privilege
  Separation*, [arXiv 2509.25926](https://arxiv.org/abs/2509.25926)).
* Benchmarks: AgentDojo ([arXiv 2406.13352](https://arxiv.org/abs/2406.13352)).

**Static approaches exist but miss what Veer has.**

* *Tracking Capabilities for Safer Agents* (Odersky et al.,
  [arXiv 2603.00991](https://arxiv.org/abs/2603.00991)): Scala 3 capture
  checking for capabilities and confidentiality. No integrity lattice, no
  model of LLM contexts.
* *Language-Based Agent Control* (Zhou, D'Antoni, Polikarpova,
  [arXiv 2605.12863](https://arxiv.org/abs/2605.12863)): agent-generated code
  must type-check against developer policies, including information flow.
* *ETAS* ([arXiv 2607.17780](https://arxiv.org/abs/2607.17780)): an effect-typed
  agent language with trusted/untrusted prompts and trace specifications.

None of them types the **KV cache**. In an LLM program the context is where
untrusted text and the model meet, and the operations on it (fork, splice,
speculate) decide what the model can be influenced by. Veer already makes
contexts linear values, which is exactly what flow-sensitive labelling needs:
every operation consumes a context and returns a new one, so its label can
change without aliasing.

## What is new in Veer's design

1. **Labels on KV operations.** A context is a chain of segments with cumulative
   labels. `fork` gives independent branches; `gen` treats the model as an
   arbitrary function of its context (the CaMeL/Fides rule, applied statically).
2. **Splice cannot launder.** Context editing (Leyline, *KV Cache Directives for
   Agentic Inference*, [arXiv 2606.01065](https://arxiv.org/abs/2606.01065))
   keeps the KV that followed an edited span, and that KV attended to the old
   content. Veer keeps the taint on later segments, and only a splice of the
   last span removes it. We found no prior work that states this rule.
3. **`speculate` + `rollback` is a static quarantine.** Rollback restores the
   exact KV state, so the dual-LLM / taint-absorbing-branch pattern becomes a
   typing rule instead of a runtime mechanism.
4. **Endorsement with a bit budget.** `endorse` only accepts values with a
   finite, statically known set of options (robust and nonmalleable
   endorsement: Myers, Sabelfeld and Zdancewic, *Enforcing Robust
   Declassification*, CSFW 2004; Cecchetti, Myers and Arden, *Nonmalleable
   Information Flow Control*, CCS 2017). The compiler multiplies log2(options)
   by statically known trip counts and checks `@budget(endorse = N bits)`,
   alongside the KV and prefill budgets Veer already proves.
5. **Checked before any token is spent**, with a machine-readable certificate
   (`veer audit --json`).

The core typing discipline is classical: Denning's lattice model, the
Volpano, Smith and Irvine type system (*A Sound Type System for Secure Flow
Analysis*, JCS 1996) with a program-counter label for implicit flows, and
flow-sensitive labels (Hunt and Sands, *On Flow-Sensitive Security Types*,
POPL 2006).

## Evidence that it works

* 27 rule tests (`crates/veer-types/tests/flow.rs`) and 9 diagnostic UI tests.
* **Noninterference property test** (`crates/veer-runtime/tests/noninterference.rs`):
  random programs mixing untrusted/secret tools, contexts, `gen`, `fork`,
  `splice`, `speculate`, branches and loops. Every accepted program is run with
  varied untrusted and secret tool outputs on the sim engine (whose output is a
  hash of the whole prompt) and its privileged and egress call logs must not
  change. 400 random cases plus a 500-program corpus pass; in that corpus the
  checker accepts 433 programs and rejects 67, of which 21 visibly leak within
  four runs (the rest are rejected conservatively or leak only on inputs the
  test did not hit).
* **Quantitative check:** an endorsed 4-way choice (2 bits) produces at most
  2^2 distinct privileged traces over 200 runs.

## Commercial angle

Enterprises are blocked on agent deployment by prompt injection, and runtime
defenses cost tokens and fail at run time. A compiler that rejects unsafe agent
programs before deployment and emits a per-program certificate (which tools
untrusted data can steer, how many bits, plus the KV and token budgets) is
something a security or platform team can put in CI and an auditor can read.

## Other findings worth keeping

* **Dollar budgets.** Cache reads dominate real agent spend
  (*How Do AI Agents Spend Your Money?*, [arXiv 2604.22750](https://arxiv.org/abs/2604.22750));
  the only dollar-budget type discipline found checks caps at run time and ignores caching
  (*Token Budgets*, [arXiv 2606.04056](https://arxiv.org/abs/2606.04056)). Veer's
  planner already proves prefill, decode and reuse, so `@budget(usd = ...)`
  with cached-input pricing is the natural next step.
* **FP4 numerics.** Worst-case bounds grow like n while probabilistic rounding
  analysis gives sqrt(n) (Higham and Mary); blockwise Hadamard transforms change
  the bound's shape (Tseng et al., *Training LLMs with MXFP4*,
  [arXiv 2502.20586](https://arxiv.org/abs/2502.20586); NVIDIA, *Pretraining
  Large Language Models with NVFP4*, [arXiv 2509.25149](https://arxiv.org/abs/2509.25149)).
  A dual-mode (worst-case plus probabilistic) bound would make Veer's
  `@accuracy` contracts usable at real matrix sizes.
* **Retries and replay.** Typed retry with cost accounting and record/replay
  test engines were the other strong PL-side suggestions.

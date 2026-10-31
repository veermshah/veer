#!/usr/bin/env python3
"""Generate the docs pages that are derived from the toolchain:

* cookbook/*.md  — each example's source, its static plan, and a simulated run
* errors/*.md    — one page per diagnostic (via `veer explain --markdown-dir`)

Usage: python3 scripts/gen_site.py [path/to/veer]
"""
import pathlib, subprocess, sys

root = pathlib.Path(__file__).resolve().parent.parent
veer = sys.argv[1] if len(sys.argv) > 1 else str(root / "target" / "debug" / "veer")
import os
site = pathlib.Path(os.environ.get("VEER_SITE", root.parent / "veer-site"))
docs = site / "src" / "content" / "docs"


def sh(*args):
    r = subprocess.run([veer, *args], capture_output=True, text=True, cwd=root)
    return (r.stdout + r.stderr).rstrip()


EXAMPLES = [
    ("review", "Code review with sharing, editing, and speculation", 1,
     "A reviewer that reads a document once, branches into a summary and a structured bug list that share the document's KV cache, "
     "edits the document out of the cache with `splice`, and tries a patch inside a `speculate` transaction. "
     "It uses every context feature in about 30 lines.",
     ["`fork` shares the system prompt and the whole 4,096-token document between the two branches. The plan shows how much memory that saves.",
      "`as json<Bugs>` returns a typed record. The checker proves the smallest valid `Bugs` fits in 256 tokens.",
      "`splice b at doc` removes the document from `b`'s cache once it's no longer needed. `W0210` reports the worst-case re-prefill of what followed it.",
      "`speculate` keeps the original context alive until the patch is accepted (`commit`) or rejected (`rollback`).",
      "`@budget(kv = 2GiB, prefill = 12000)` is checked at compile time."]),
    ("agent_edit_history", "Agent loop with KV editing", 2,
     "A three-step debugging agent whose tool returns up to 2,048 tokens per call. After each step the observation is replaced in the cache "
     "by a short note, in the style of Leyline's KV directives, so the context grows by about 100 tokens per step instead of 2,000.",
     ["Planned with the splice: **peak 346 MiB, longest context 2,579 tokens, prefill ≤ 6,674**.",
      "The same program without the splice plans at **peak 854 MiB, longest context 6,769 tokens, prefill ≤ 6,401**. The splice trades about 270 tokens of re-prefill for 2.5× less memory, and both numbers are known before running.",
      "The label `obs` is added and spliced away within each iteration, which is what the loop rule (`V0104`) requires.",
      "Bind the tool to a real command with `--tool search='./search.sh'`."]),
    ("tree_of_thought", "Tree-of-thought search", 3,
     "Breadth-limited tree search over a shared problem statement: each level forks `WIDTH` branches from the root, generates a thought, "
     "and judges it with a constrained `choice`.",
     ["`fork root` + `root = keep` is the idiom for branching repeatedly from one context.",
      "`as choice[\"yes\", \"no\"]` constrains the verdict. Its type is `str<=3`.",
      "Loops have constant bounds (`DEPTH`, `WIDTH`), so the planner unrolls them and proves the 512 MiB budget."]),
    ("rag", "Many questions over one document", 4,
     "Retrieval-augmented QA where the retrieved document is prefilled once and each question forks from it inside a helper function that "
     "takes and returns the context.",
     ["`fn ask(doc_ctx: Ctx<mistral>, q: str<=200) -> (Ctx<mistral>, str<=160)` shows linearity through function calls: the context goes in and comes back.",
      "Each `ask` shares the up-to-3,000-token document instead of re-prefilling it: total prefill is ≤ 3,666 tokens, where three independent requests would prefill the document three times.",
      "`kv: fp8_e4m3` halves bytes per token compared to bf16."]),
    ("quant_mlp", "Precision types: choosing a 4-bit format", 5,
     "The same tiny MLP with weights in NVFP4, INT8, MXFP4, and overflow-aware MXFP4. The types carry proven worst-case error bounds, "
     "and `@accuracy` contracts turn them into compile-time checks.",
     ["`veer types` prints each function's inferred bound. NVFP4 is about 0.20 relative, MXFP4 about 0.28, INT8 about 0.009.",
      "Adding `@accuracy(rel <= 0.25)` to `mlp_mxfp4` is a compile error (`V0301`): power-of-two block scales cost accuracy that E4M3 scales don't.",
      "`report` shows measured error next to the bound. The bound is a guarantee over all inputs in range, so it's larger than typical error."]),
    ("hello", "Hello, Veer", 6,
     "The smallest complete program, also used in the [Quickstart](/start/quickstart/).", []),
]

cook = docs / "cookbook"
cook.mkdir(parents=True, exist_ok=True)
for slug, title, order, intro, notes in EXAMPLES:
    src = (root / "examples" / f"{slug}.veer").read_text().rstrip()
    extra = sh("types", f"examples/{slug}.veer") if slug == "quant_mlp" else sh("plan", f"examples/{slug}.veer", "--entry", "main")
    run = sh("run", f"examples/{slug}.veer", "--stats")
    notes_md = "\n".join(f"- {n}" for n in notes)
    page = f"""---
title: "{title}"
description: "Veer cookbook: {title.lower()}."
sidebar:
  order: {order}
---

{intro}

File: `examples/{slug}.veer` (included in every [release download](/download/))

```veer
{src}
```

## {"Inferred types" if slug == "quant_mlp" else "Static plan"}

```text
$ veer {"types" if slug == "quant_mlp" else "plan"} examples/{slug}.veer{"" if slug == "quant_mlp" else " --entry main"}
{extra}
```

## Simulated run

```text
$ veer run examples/{slug}.veer --stats
{run}
```

{"## Notes" if notes else ""}

{notes_md}
"""
    (cook / f"{slug.replace('_', '-')}.md").write_text(page)

# Error index
err = docs / "errors"
err.mkdir(parents=True, exist_ok=True)
for p in err.glob("*.md"):
    p.unlink()
print(sh("explain", "--markdown-dir", str(err)))
rows = []
for line in sh("explain").splitlines():
    code, title = line.split("  ", 1)
    rows.append(f"| [{code}](/errors/{code.lower()}/) | {title.strip()} |")
(err / "index.md").write_text(
    "---\ntitle: Error index\ndescription: Every diagnostic Veer can emit.\nsidebar:\n  order: 0\n---\n\n"
    "Every error and warning has a code. Run `veer explain CODE` in a terminal, or open its page here. Codes `V00xx` are syntax, "
    "`V01xx` types and linearity, `V02xx` planner budgets, `V03xx` the precision layer, and `W` codes are warnings.\n\n"
    "| Code | Meaning |\n|---|---|\n" + "\n".join(rows) + "\n"
)
print(f"generated {len(EXAMPLES)} cookbook pages and the error index")

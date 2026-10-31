#!/usr/bin/env python3
"""Check every complete ```veer program in the docs with `veer check`.

A block is a complete program if it declares a `fn`. If it contains a comment
like `// error[V0301]`, it must fail with that code; otherwise it must pass.
Usage: python3 scripts/check_docs.py [path/to/veer]
"""
import pathlib, re, subprocess, sys, tempfile

veer = sys.argv[1] if len(sys.argv) > 1 else "target/debug/veer"
root = pathlib.Path(__file__).resolve().parent.parent
import os
site = pathlib.Path(os.environ.get("VEER_SITE", root.parent / "veer-site"))
docs = site / "src" / "content" / "docs"
fails, n = [], 0
for md in sorted(p for p in docs.rglob("*.md*") if "errors" not in p.parts):
    text = md.read_text()
    for i, block in enumerate(re.findall(r"```veer\n(.*?)```", text, re.S)):
        if not re.search(r"^\s*(@\w+.*\n\s*)?fn ", block, re.M):
            continue
        n += 1
        expect = re.search(r"//\s*error\[(V\d{4})\]", block)
        with tempfile.NamedTemporaryFile("w", suffix=".veer", delete=False) as f:
            f.write(block)
        r = subprocess.run([veer, "check", f.name], capture_output=True, text=True)
        out = r.stdout + r.stderr
        where = f"{md.relative_to(site)} block {i}"
        if expect:
            if r.returncode == 0 or f"[{expect.group(1)}]" not in out:
                fails.append(f"{where}: expected {expect.group(1)}\n{out}")
        elif r.returncode != 0:
            fails.append(f"{where}: should pass\n{block}\n{out}")
print(f"checked {n} doc programs, {len(fails)} failure(s)")
for f in fails:
    print("\n---", f)
sys.exit(1 if fails else 0)

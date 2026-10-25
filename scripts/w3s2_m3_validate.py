"""Validate M3 run outputs before analysis (read-only; reads no results beyond structure).

Usage: python scripts/w3s2_m3_validate.py <expected cases file> <out json> <run dir> [<run dir> ...]
Checks per case: listed in done_cases.txt (once), at most one row per (case, rank, condition), every lesion has a base
row, every edit/control arm has both an edit and a control row with the base row's arm, tokens/<case>_r<rank>.npz is
loadable (base, edit, n; equal shapes; finite) exactly for the lesions with an edit row, no rows for cases not marked
done, and every expected case is done. `rerun` lists the cases to delete from the run and re-run."""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

expected = [l.strip() for l in open(sys.argv[1]) if l.strip()]; out = sys.argv[2]; dirs = sys.argv[3:]
done = defaultdict(int); rows = defaultdict(list); problems = defaultdict(list); tokfiles = {}
for d in dirs:
    p = os.path.join(d, "done_cases.txt")
    for c in (l.strip() for l in open(p)) if os.path.exists(p) else []:
        done[c] += 1
    p = os.path.join(d, "m3_lesions.csv")
    for r in csv.DictReader(open(p)) if os.path.exists(p) else []:
        rows[r["case"]].append(r)
    td = os.path.join(d, "tokens")
    for f in os.listdir(td) if os.path.isdir(td) else []:
        tokfiles[f] = os.path.join(td, f)
for c in expected:
    if done[c] == 0:
        problems[c].append("not done")
    if done[c] > 1:
        problems[c].append(f"listed done {done[c]}x (in more than one run dir or duplicated)")
for c in set(rows) - set(c for c in done if done[c]):
    problems[c].append("rows present but case not marked done (partial write)")
for c in set(done) - set(expected):
    problems[c].append("done but not in the expected list")
need_tok = set()
for c, rs in rows.items():
    by = defaultdict(list)
    for r in rs:
        by[(r["rank"], r["condition"])].append(r)
    for k, v in by.items():
        if len(v) > 1:
            problems[c].append(f"duplicate rows for rank {k[0]} {k[1]} (x{len(v)})")
    for rank in sorted({r["rank"] for r in rs}):
        base = by.get((rank, "base"), [])
        if len(base) != 1:
            problems[c].append(f"rank {rank}: {len(base)} base rows"); continue
        has_e, has_c = (rank, "edit") in by, (rank, "control") in by
        if has_e != has_c:
            problems[c].append(f"rank {rank}: edit/control incomplete (edit={has_e}, control={has_c})")
        if has_e and has_c:
            arm = base[0]["arm"]
            if arm not in ("A", "B") or any(x["arm"] != arm for x in by[(rank, "edit")] + by[(rank, "control")]):
                problems[c].append(f"rank {rank}: arm inconsistent ({arm!r})")
            need_tok.add(f"{c}_r{rank}.npz")
        elif base[0]["arm"] in ("A", "B"):
            problems[c].append(f"rank {rank}: arm {base[0]['arm']} on base but no edit/control rows")
        for r in by.get((rank, "base"), []) + by.get((rank, "edit"), []) + by.get((rank, "control"), []):
            try:
                if not np.isfinite(float(r["coverage"])):
                    raise ValueError
            except (ValueError, KeyError):
                problems[c].append(f"rank {rank} {r['condition']}: coverage not finite")
for f in sorted(need_tok):
    case = f.rsplit("_r", 1)[0]
    if f not in tokfiles:
        problems[case].append(f"missing tokens file {f}"); continue
    try:
        z = np.load(tokfiles[f])
        b, e, n = z["base"], z["edit"], z["n"]
        if b.shape != e.shape or not (np.isfinite(b).all() and np.isfinite(e).all()):
            raise ValueError("shape mismatch or non-finite")
    except Exception as ex:  # noqa: BLE001
        problems[case].append(f"tokens file {f} not loadable/valid: {ex}")
for f in sorted(set(tokfiles) - need_tok):
    problems[f.rsplit("_r", 1)[0]].append(f"orphan tokens file {f} (no edit row)")
res = {"n_expected": len(expected), "n_done": sum(1 for c in expected if done[c] >= 1), "n_csv_rows": sum(len(v) for v in rows.values()),
       "n_token_files": len(tokfiles), "problems": {k: v for k, v in sorted(problems.items())}, "rerun": sorted(problems), "ok": not problems}
json.dump(res, open(out, "w"), indent=1)
print(f"expected {res['n_expected']} done {res['n_done']} rows {res['n_csv_rows']} token files {res['n_token_files']} problems in {len(problems)} cases -> {'OK' if res['ok'] else 'NOT OK'}")
for k, v in sorted(problems.items()):
    print(" ", k, v)
sys.exit(0 if res["ok"] else 1)

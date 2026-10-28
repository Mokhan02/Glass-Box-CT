"""W5 Amendment 3: merge the two shard output folders of scripts/w3s2_m2_run.py into one run folder (CPU). Shards write disjoint cases, so the merge is a concatenation:
lesions.csv and cases.csv (headers must be identical; a case appearing in two shards is an error), done_cases.txt, maps/*.npz, and determinism.json (lists concatenated, summary recomputed with the runner's formula).
Usage: python scripts/w5a3_merge_m2.py <out dir> <shard dir> [<shard dir> ...]"""
import csv
import json
import os
import shutil
import sys


def merge(out, shards):
    if len(set(shards)) != len(shards):
        raise SystemExit("the same shard folder was given twice")
    os.makedirs(os.path.join(out, "maps"), exist_ok=True); seen = {}
    for name in ("lesions.csv", "cases.csv"):
        header, rows = None, []
        for s in shards:
            p = os.path.join(s, name)
            if not os.path.exists(p):
                continue
            rd = csv.DictReader(open(p)); h = rd.fieldnames
            if header is None:
                header = h
            elif h != header:
                raise SystemExit(f"{name}: header of {s} differs from the first shard")
            for r in rd:
                if name == "cases.csv":
                    seen.setdefault(r["case"], s)
                    if seen[r["case"]] != s:
                        raise SystemExit(f"case {r['case']} appears in two shards ({seen[r['case']]} and {s})")
                rows.append(r)
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows)
    det = {"base_vs_stage1": [], "identity_vs_base": []}; done = []
    for s in shards:
        p = os.path.join(s, "determinism.json")
        if os.path.exists(p):
            d = json.load(open(p))
            for k in det:
                det[k] += d.get(k, [])
        if os.path.exists(os.path.join(s, "done_cases.txt")):
            done += [l.strip() for l in open(os.path.join(s, "done_cases.txt")) if l.strip()]
        for fn in (os.listdir(os.path.join(s, "maps")) if os.path.isdir(os.path.join(s, "maps")) else []):
            shutil.copy2(os.path.join(s, "maps", fn), os.path.join(out, "maps", fn))
    det["summary"] = {"max_base_vs_stage1_cov_diff": max((d["abs_cov_diff"] for d in det["base_vs_stage1"]), default=None), "detection_flag_mismatches": sum(not d["flag_same"] for d in det["base_vs_stage1"]),
                      "max_identity_vs_base_cov_diff": max((d["abs_cov_diff"] for d in det["identity_vs_base"]), default=None)}
    json.dump(det, open(os.path.join(out, "determinism.json"), "w"), indent=1); open(os.path.join(out, "done_cases.txt"), "w").write("\n".join(sorted(set(done))) + "\n")
    return {"cases": len(set(done)), "summary": det["summary"]}


if __name__ == "__main__":
    print(json.dumps(merge(sys.argv[1], sys.argv[2:])))

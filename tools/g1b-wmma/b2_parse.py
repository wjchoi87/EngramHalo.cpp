#!/usr/bin/env python3
"""§127 B2 측정 파서: 서버 로그를 요청 경계(print_timing)로 분해해
요청별 total wall / capture 수·wall / eager wall / replay wall / KEYDIFF n_kv churn 집계."""
import re, json, sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ab_n256.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/b2_decomp.json"

reqs = []
cur = {"eager": [], "replay": [], "capw": [], "kd": []}
for line in open(LOG, errors="replace"):
    m = re.search(r"WALLMODE mode=(\w+) n_nodes=\d+ us=(\d+)", line)
    if m:
        cur[m.group(1)].append(int(m.group(2)))
        continue
    m = re.search(r"CAPWALL graph=\S+ n_nodes=\d+ recapture_us=(\d+)", line)
    if m:
        cur["capw"].append(int(m.group(1)))
        continue
    m = re.search(r"KEYDIFF\[\d+\] nodes=\d+ chg=(\d+) shape_chg=(\d+).*tail=(\d+) kvshape=(\d+) kvderived=(\d+)", line)
    if m:
        cur["kd"].append(tuple(int(x) for x in m.groups()))
        continue
    m = re.search(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens", line)
    if m:
        cur["server_ms"] = float(m.group(1))
        cur["ptok"] = int(m.group(2))
        reqs.append(cur)
        cur = {"eager": [], "replay": [], "capw": [], "kd": []}

def s(x): return sum(x) / 1000.0  # ms->s

labels = []
for r in reqs:
    if r.get("ptok", 0) > 100000: labels.append("FILL")
    else: labels.append(f"req{r.get('ptok')}")

out = []
for r, lab in zip(reqs, labels):
    d = {
        "label": lab, "ptok": r.get("ptok"), "server_ms": r.get("server_ms", 0),
        "eager_n": len(r["eager"]), "eager_s": round(s(r["eager"]), 3),
        "replay_n": len(r["replay"]), "replay_s": round(s(r["replay"]), 3),
        "cap_n": len(r["capw"]), "cap_s": round(s(r["capw"]), 3),
        "kd_last": r["kd"][-1] if r["kd"] else None,
        "kd_kvderived_sum": sum(k[4] for k in r["kd"]),
        "kd_lines": len(r["kd"]),
    }
    out.append(d)
json.dump(out, open(OUT, "w"), indent=1)
print(f"{'label':>12} {'srv_ms':>10} {'eager_n':>7} {'eager_s':>8} {'rep_n':>6} {'rep_s':>7} {'cap_n':>6} {'cap_s':>8} {'kd_kvder':>8}")
for d in out:
    print(f"{d['label']:>12} {d['server_ms']:>10.0f} {d['eager_n']:>7} {d['eager_s']:>8.2f} "
          f"{d['replay_n']:>6} {d['replay_s']:>7.3f} {d['cap_n']:>6} {d['cap_s']:>8.2f} {d['kd_kvderived_sum']:>8}")

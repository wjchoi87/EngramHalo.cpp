#!/usr/bin/env python3
"""§125 NPAD bucket A/B — fresh 프로세스 per bucket.
측정: factual correctness, 256K fill, +256/+1K/+4K ×3 TTFT, capture/replay 수, KEYDIFF churn.
게이트: 모든 요청 n_probs logprob null = NaN → FAIL.
"""
import json, os, re, subprocess, sys, time, urllib.request, random

MODEL = "/home/wjchoi/models/flashnext-gguf/Qwen3.8-Flash-Next-ROCmFP4-direct.gguf"
BIN = "/home/wjchoi/workspace/baselines/EngramHalo.cpp/build-g1a/bin/llama-server"
PORT = 8054
PARA = ("The study of renewable energy sources has become one of the most important scientific "
        "and engineering challenges of the twenty-first century. Solar panels convert sunlight directly "
        "into electricity through the photovoltaic effect, while wind turbines capture kinetic energy "
        "from moving air masses. Hydroelectric power plants harness the potential energy of water stored "
        "at elevation, and geothermal systems tap into the heat of the Earth's interior. Each technology "
        "has distinct advantages and limitations regarding cost, efficiency, environmental impact, and "
        "geographic suitability. Understanding these tradeoffs is essential for policymakers, engineers, "
        "and citizens who must decide how to allocate limited resources toward a sustainable future. ")
WORDS = ("energy policy solar wind hydro grid storage climate carbon efficiency turbine panel "
         "battery hydrogen reactor thermal fusion transmission demand supply research engineer "
         "science data model system network signal process control machine learning inference "
         "market grid balance region seasonal output capacity maintenance lifetime recycle ").split()

def suffix(n_tokens, seed):
    rng = random.Random(seed)
    words = [rng.choice(WORDS) for _ in range(int(n_tokens * 0.93) + 8)]
    sents = [" ".join(words[i:i+12]) + "." for i in range(0, len(words), 12)]
    return " " + " ".join(sents)

def start_server(npad, graphs_on=True, log="/tmp/ab_srv.log"):
    subprocess.run("pgrep -x llama-server | xargs -r kill", shell=True)
    time.sleep(3)
    env = dict(os.environ)
    env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    env["LLAMA_KV_NPAD"] = str(npad)
    env["GGML_CUDA_GRAPH_DEBUG"] = "1"
    env["GGML_CUDA_GRAPH_KEYDIFF"] = "1"
    if not graphs_on:
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    fh = open(log, "w")
    p = subprocess.Popen([BIN, "-m", MODEL, "-ngl", "99", "-c", "262144", "-b", "512",
                          "-ub", "128", "-np", "1", "--port", str(PORT), "--host", "127.0.0.1"],
                         stdout=fh, stderr=fh, env=env)
    for _ in range(300):
        time.sleep(3)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            return p
        except Exception:
            pass
    raise RuntimeError("server start failed")

def probe(prompt, mx=1, timeout=7200):
    body = json.dumps({"prompt": prompt, "n_predict": mx, "temperature": 0,
                       "cache_prompt": True, "n_probs": 1}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    wall = time.time() - t0
    probs = d.get("completion_probabilities") or []
    lp = probs[0]["top_logprobs"][0]["logprob"] if probs and probs[0].get("top_logprobs") else None
    return {"ptok": d["tokens_evaluated"], "nan": lp is None,
            "out": (d.get("content") or "")[:40], "wall_ms": round(wall*1000)}

def aggregate(log):
    cap = rep = 0
    keydiff = []
    try:
        for line in open(log, errors="replace"):
            if "GRAPHPATH" in line:
                if "update_req=1" in line: cap += 1
                elif "update_req=0" in line: rep += 1
            elif line.startswith("KEYDIFF["):
                m = re.search(r"chg=(\d+)", line)
                if m: keydiff.append(int(m.group(1)))
    except FileNotFoundError:
        pass
    return cap, rep, (sum(keydiff[-64:])/max(1,len(keydiff[-64:])), len(keydiff))

def run_bucket(npad):
    log = f"/tmp/ab_n{npad}.log"
    p = start_server(npad, True, log)
    R = {"bucket": npad, "facts": [], "fill": None, "incr": []}
    for q, tag in [("The capital of France is", "Paris"),
                   ("The capital of Italy is", "Rome"),
                   ("The largest planet in our solar system is", "Jupiter")]:
        r = probe(q, 4)
        r["tag"] = tag
        R["facts"].append(r)
        print(f"  n{npad} fact {tag}: nan={r['nan']} out={r['out']!r}", flush=True)
    state = PARA * 2080
    r = probe(state + " Summarize:", 1)
    R["fill"] = r
    print(f"  n{npad} FILL ptok={r['ptok']} wall={r['wall_ms']}ms nan={r['nan']}", flush=True)
    cum = state; nxt = 1
    for size in [256, 1024, 4096]:
        for rep in (1, 2, 3):
            cum = cum + suffix(size, nxt); nxt += 1
            try:
                r = probe(cum + " Summarize:", 1)
            except urllib.error.HTTPError as e:
                r = {"ptok": -1, "nan": None, "out": f"HTTP{e.code}", "wall_ms": -1}
            r["size"], r["rep"] = size, rep
            R["incr"].append(r)
            print(f"  n{npad} +{size} rep{rep}: ptok={r['ptok']} wall={r['wall_ms']}ms nan={r['nan']}", flush=True)
    cap, rep, (avgchg, nkd) = aggregate(log)
    R["captures"], R["replays"], R["keydiff_avg_tail"], R["keydiff_lines"] = cap, rep, avgchg, nkd
    p.terminate(); p.wait()
    return R

def run_nanscan_gate(npad):
    log = f"/tmp/ab_ns_{npad}.log"
    p = start_server(npad, False, log)
    env_cmd = None
    # restart with NANSCAN env (needs env at server start)
    p.terminate(); p.wait()
    subprocess.run("pgrep -x llama-server | xargs -r kill", shell=True); time.sleep(3)
    env = dict(os.environ)
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    env["LLAMA_KV_NPAD"] = str(npad)
    env["GGML_CUDA_GRAPH_NANSCAN_V4"] = "1"
    fh = open(log, "w")
    p = subprocess.Popen([BIN, "-m", MODEL, "-ngl", "99", "-c", "8192", "-b", "512",
                          "-ub", "128", "-np", "1", "--port", str(PORT), "--host", "127.0.0.1"],
                         stdout=fh, stderr=fh, env=env)
    for _ in range(300):
        time.sleep(3)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3); break
        except Exception: pass
    r = probe("The capital of France is", 1)
    # trigger 길이도 1회 (부분 ubatch M=9 경계)
    F = PARA * 48
    r2 = probe(F + " abc"*33, 1, timeout=900)
    p.terminate(); p.wait()
    nan_lines = 0
    try:
        for line in open(log, errors="replace"):
            if "NANSCAN4" in line and "nancnt=0" not in line:
                nan_lines += 1
    except FileNotFoundError:
        pass
    return {"bucket": npad, "probe_nan": r["nan"], "trigger5901_nan": r2["nan"], "v4_genuine_nan_lines": nan_lines}

if __name__ == "__main__":
    buckets = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [256, 2048, 512, 1024]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ab_results.json"
    allR = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for b in buckets:
        print(f"=== BUCKET {b} perf run ===", flush=True)
        R = run_bucket(b)
        allR[str(b)] = R
        json.dump(allR, open(out_path, "w"), indent=1)
        print(f"=== BUCKET {b} nanscan gate ===", flush=True)
        if len(sys.argv) <= 3:
            NS = run_nanscan_gate(b)
            allR[str(b)]["nanscan"] = NS
            print(f"bucket {b} nanscan: {NS}", flush=True)
        json.dump(allR, open(out_path, "w"), indent=1)
        print(f"bucket {b} done: cap={R['captures']} rep={R['replays']}", flush=True)
    print("ALL DONE")

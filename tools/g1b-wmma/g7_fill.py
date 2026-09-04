#!/usr/bin/env python3
"""G-7: 256K 세션 프리필 체움 + 증분 TTFT 측정. TTFT/prefill 전용 (UMA decode 비사용)."""
import json, urllib.request, time, sys, random

BASE = "http://127.0.0.1:8054"
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
         "science data model system network signal process control machine learning inference ").split()

def request(prompt, n_predict=1, cache=True, timeout=7200):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0,
                       "cache_prompt": cache}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    return resp, time.time() - t0

def base_text(paras):
    return PARA * paras

def suffix_text(n_tokens, seed):
    rng = random.Random(seed)
    words = [rng.choice(WORDS) for _ in range(max(8, int(n_tokens * 1.35)))]
    out = []
    for i in range(0, len(words), 12):
        out.append(" ".join(words[i:i+12]) + ".")
    return " " + " ".join(out)

def main():
    mode = sys.argv[1]
    if mode == "probe":  # paras당 토큰 비율 측정
        resp, wall = request(base_text(200), cache=True)
        t = resp["timings"]
        print(json.dumps({"paras": 200, "tokens": t["prompt_n"],
                          "tpp": t["prompt_n"]/200.0, "pp": t["prompt_per_second"]}))
    elif mode == "fill":  # 단일 대형 요청
        paras = int(sys.argv[2])
        resp, wall = request(base_text(paras), cache=True)
        t = resp["timings"]
        print(json.dumps({"paras": paras, "prompt_n": t["prompt_n"],
                          "pp": t["prompt_per_second"], "ms": t["prompt_ms"], "wall": wall}))
    elif mode == "incr":  # base + 고유 suffix 증분 TTFT
        paras, n_tok, seed = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        prompt = base_text(paras) + suffix_text(n_tok, seed)
        resp, wall = request(prompt, cache=True)
        t = resp["timings"]
        print(json.dumps({"incr_target": n_tok, "evaluated": t["prompt_n"],
                          "cache_n": t.get("cache_n", -1), "ttft_ms": t["prompt_ms"],
                          "pp": t["prompt_per_second"], "wall": wall}))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""G-7 append-only 증분 TTFT: 각 요청이 이전 텍스트에 접합 → LCP=이전 전체 보장.
시퀀스: +256×3 → +1K×3 → +4K×3, 누적 ctx ≤ 262,144 검증."""
import json, sys, urllib.request, time, random

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
         "science data model system network signal process control machine learning inference "
         "market grid balance region seasonal output capacity maintenance lifetime recycle ").split()

def request(prompt, timeout=7200):
    body = json.dumps({"prompt": prompt, "n_predict": 1, "temperature": 0,
                       "cache_prompt": True}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def suffix(n_tokens, seed):
    rng = random.Random(seed)
    words = [rng.choice(WORDS) for _ in range(int(n_tokens * 0.93) + 8)]
    sents = [" ".join(words[i:i+12]) + "." for i in range(0, len(words), 12)]
    return " " + " ".join(sents)

def main():
    paras = int(sys.argv[1])
    state = PARA * paras
    cum = None  # 서버 cache_n으로 검증
    results = []
    for size in [256, 1024, 4096]:
        for rep in (1, 2, 3):
            state += suffix(size, seed=size * 1000 + rep)
            resp = request(state)
            t = resp["timings"]
            results.append({"size": size, "rep": rep, "evaluated": t["prompt_n"],
                            "cache_n": t.get("cache_n", -1), "ttft_ms": t["prompt_ms"],
                            "pp": t["prompt_per_second"]})
            r = results[-1]
            print(f"+{size:5d} r{rep}: evaluated={r['evaluated']:6d} cache_n={r['cache_n']:7d} "
                  f"TTFT={r['ttft_ms']:9.0f} ms pp={r['pp']:6.1f}", flush=True)
    print("JSON:" + json.dumps(results))

if __name__ == "__main__":
    main()

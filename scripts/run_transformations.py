#!/usr/bin/env python3
"""
ARDIP-Bench transformation engine.
Tiers:
  human   : deterministic rule-based human-style edits (all docs, no LLM)     [T family F6]
  tier1   : phi3 on ALL docs x {paraphrase, summarize_50}
  tier2   : phi3 on every 3rd doc x {rewrite_lay, translate_es, regenerate}
  tier3   : frontier-class engines {DeepSeek-V4-Pro, Qwen3.8, Kimi-K3, GLM} (remote)
            x every 3rd doc x {paraphrase, summarize_50}, 6-way parallel
  tier4   : phi3 on every 12th doc x {translate_fr, translate_de, summarize_20}
  chain   : phi3 on every 10th doc: paraphrase -> summarize_50 (depth 2)
  backtr  : phi3 on every 20th doc: translate_es -> back-translation (depth 2)
Resumable via results/transformations_bench.jsonl keyed by (tier, engine, doc_id, family).
"""
import argparse, hashlib, json, os, re, sys, time, datetime, threading
from pathlib import Path
import requests
import concurrent.futures

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus/seeds/ardip_bench_corpus.json"
RAW = ROOT / "results/transformations_bench.jsonl"
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# OpenAI-compatible frontier endpoint (configured via environment variables)
FRONTIER = None
if os.environ.get("FRONTIER_BASE_URL") and os.environ.get("FRONTIER_API_KEY"):
    FRONTIER = {"base": os.environ["FRONTIER_BASE_URL"], "key": os.environ["FRONTIER_API_KEY"]}

import urllib3
urllib3.disable_warnings()

PROMPTS = {
    "paraphrase": "Rewrite the following text in different words while preserving all factual details and meaning exactly. Output only the rewritten text, no preamble.\n\nTEXT:\n{t}",
    "summarize_50": "Summarize the following text in about 50 percent of its original length, preserving the key facts and requirements. Output only the summary, no preamble.\n\nTEXT:\n{t}",
    "summarize_20": "Summarize the following text in about 20 percent of its original length, keeping only the most essential facts. Output only the summary, no preamble.\n\nTEXT:\n{t}",
    "rewrite_lay": "Rewrite the following text for a general non-expert audience while preserving the key facts. Output only the rewritten text, no preamble.\n\nTEXT:\n{t}",
    "translate_es": "Translate the following text into Spanish. Output only the translation, no preamble.\n\nTEXT:\n{t}",
    "translate_fr": "Translate the following text into French. Output only the translation, no preamble.\n\nTEXT:\n{t}",
    "translate_de": "Translate the following text into German. Output only the translation, no preamble.\n\nTEXT:\n{t}",
    "back_en": "Translate the following Spanish text into English. Output only the translation, no preamble.\n\nTEXT:\n{t}",
    "regenerate": "Read the following text carefully, then put it aside and rewrite its complete content from memory in your own words, without copying phrases. Output only the rewritten text.\n\nTEXT:\n{t}",
}

FILLER = " The guidance applies prospectively from the date of publication."

def _frontier_call(model, prompt, max_tok, temp, timeout):
    max_tok = 3000  # reasoning models (Qwen3.8, Kimi-K3) spend tokens on internal reasoning
    timeout = 600
    r = requests.post(f"{FRONTIER['base']}/chat/completions",
        headers={"Authorization": f"Bearer {FRONTIER['key']}", "Content-Type": "application/json"},
        json={"model": model.split(":", 1)[1],
              "messages": [{"role": "system", "content": "You are a text rewriter. Respond with ONLY the requested text. No task description, no preamble."},
                           {"role": "user", "content": prompt}],
              "max_tokens": max_tok, "temperature": temp},
        timeout=timeout, verify=False)
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content")

def gen(model, prompt, max_tok=700, temp=0.7, timeout=420):
    if model.startswith("frontier:"):
        if FRONTIER is None:
            raise RuntimeError("frontier endpoint not configured (set FRONTIER_BASE_URL and FRONTIER_API_KEY)")
        last_err = None
        for attempt in range(3):
            try:
                content = _frontier_call(model, prompt, max_tok, temp, timeout)
                if content and content.strip():
                    return content.strip()
                last_err = "empty content"
            except Exception as e:
                last_err = repr(e)
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"frontier gen failed after 3 tries: {last_err}")
    r = requests.post(f"{OLLAMA}/api/generate", json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temp, "num_predict": max_tok},
    }, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) > 10]

def human_edit(text, seed):
    """Deterministic 'human-style' edit: swap two adjacent sentences, delete one, add filler."""
    ss = sentences(text)
    if len(ss) < 6:
        return text + FILLER
    i = 2 + seed % (len(ss) - 4)
    ss[i], ss[i + 1] = ss[i + 1], ss[i]
    del ss[1 + seed % (len(ss) - 2)]
    return " ".join(ss) + FILLER

def joblist(corpus, tier):
    docs = corpus["documents"]
    jobs = []
    if tier == "human":
        for d in docs:
            jobs.append(("human", "rule-edit", d["doc_id"], "human_edit"))
    elif tier == "tier1":
        for d in docs:
            for f in ("paraphrase", "summarize_50"):
                jobs.append(("tier1", "phi3:latest", d["doc_id"], f))
    elif tier == "tier2":
        for d in docs[::3]:
            for f in ("rewrite_lay", "translate_es", "regenerate"):
                jobs.append(("tier2", "phi3:latest", d["doc_id"], f))
    elif tier == "tier3":
        for eng in ("frontier:DeepSeek-V4-Pro", "frontier:Qwen3.8",
                    "frontier:Kimi-K3", "frontier:GLM"):
            for d in docs[::3]:
                for f in ("paraphrase", "summarize_50"):
                    jobs.append(("tier3", eng, d["doc_id"], f))
    elif tier == "tier5":
        # frontier full coverage: engines x 2 core families x ALL docs
        for eng in ("frontier:DeepSeek-V4-Pro", "frontier:Qwen3.8",
                    "frontier:Kimi-K3", "frontier:GLM", "frontier:GLM-Flash"):
            for d in docs:
                for f in ("paraphrase", "summarize_50"):
                    jobs.append(("tier5", eng, d["doc_id"], f))
    elif tier == "tier6":
        # frontier extended families on every 3rd doc
        for eng in ("frontier:DeepSeek-V4-Pro", "frontier:Qwen3.8",
                    "frontier:Kimi-K3", "frontier:GLM", "frontier:GLM-Flash"):
            for d in docs[::3]:
                for f in ("rewrite_lay", "translate_es", "regenerate"):
                    jobs.append(("tier6", eng, d["doc_id"], f))
    elif tier == "tier4":
        for d in docs[::12]:
            for f in ("translate_fr", "translate_de", "summarize_20"):
                jobs.append(("tier4", "phi3:latest", d["doc_id"], f))
    elif tier == "tier7":
        # frontier depth-2 chains + back-translation, every 3rd doc
        for eng in ("frontier:DeepSeek-V4-Pro", "frontier:Qwen3.8",
                    "frontier:Kimi-K3", "frontier:GLM", "frontier:GLM-Flash"):
            for d in docs[::3]:
                for f in ("paraphrase+summarize_50", "translate_es+back_en"):
                    jobs.append(("tier7", eng, d["doc_id"], f))
    elif tier == "chain":
        for d in docs[::10]:
            jobs.append(("chain", "phi3:latest", d["doc_id"], "paraphrase+summarize_50"))
    elif tier == "backtr":
        for d in docs[::20]:
            jobs.append(("backtr", "phi3:latest", d["doc_id"], "translate_es+back_en"))
    return jobs

def run_job(engine, text, family):
    """Returns dict(output=..., extra={step outputs for chains}, seconds=...)."""
    t0 = time.time(); extra = {}
    if "+" in family:  # two-step chain
        f1, f2 = family.split("+")
        o1 = gen(engine, PROMPTS[f1].format(t=text))
        extra["step1"] = o1
        o2 = gen(engine, PROMPTS[f2].format(t=o1))
        return {"output": o2, "extra": extra, "seconds": round(time.time() - t0, 1)}
    return {"output": gen(engine, PROMPTS[family].format(t=text)), "extra": extra,
            "seconds": round(time.time() - t0, 1)}

def available_models():
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=15)
        return {m["name"] for m in r.json().get("models", [])}
    except Exception:
        return set()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True,
                    choices=["human", "tier1", "tier2", "tier3", "tier5", "tier6", "tier7", "tier4", "chain", "backtr"])
    ap.add_argument("--limit", type=int, default=None, help="max jobs (for staged runs)")
    ap.add_argument("--workers", type=int, default=3, help="parallel LLM requests")
    ap.add_argument("--engine", default=None, help="restrict jobs to one engine")
    ap.add_argument("--every", type=int, default=None, help="thin tier5 docs to every Nth")
    args = ap.parse_args()

    corpus = json.loads(CORPUS.read_text())
    by_id = {d["doc_id"]: d for d in corpus["documents"]}
    jobs = joblist(corpus, args.tier)
    if args.engine:
        jobs = [j for j in jobs if j[1] == args.engine]
    if args.every and args.tier == "tier5":
        keep = {d["doc_id"] for d in corpus["documents"][::args.every]}
        jobs = [j for j in jobs if j[2] in keep]
    if args.limit: jobs = jobs[:args.limit]

    done = set()
    if RAW.exists():
        for line in RAW.read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                if j.get("error") or not j.get("output"):
                    continue  # failed jobs remain eligible for retry
                done.add((j.get("tier"), j.get("engine"), j.get("doc_id"), j.get("family")))

    RAW.parent.mkdir(parents=True, exist_ok=True)
    counters = {"ok": 0, "err": 0}
    lock = threading.Lock()
    out_f = RAW.open("a")

    def do_job(job):
        tier, engine, doc_id, family = job
        d = by_id[doc_id]
        rec = {"tier": tier, "engine": engine, "doc_id": doc_id, "family": family,
               "ts": datetime.datetime.now().isoformat(timespec="seconds")}
        try:
            if engine == "rule-edit":
                seed = int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16)
                rec["output"], rec["extra"], rec["seconds"] = human_edit(d["text"], seed), {}, 0.0
            else:
                rec.update(run_job(engine, d["text"], family))
            rec["error"] = None
        except Exception as e:
            rec["output"], rec["extra"], rec["seconds"], rec["error"] = "", {}, 0.0, repr(e)
        with lock:
            out_f.write(json.dumps(rec) + "\n"); out_f.flush()
            counters["ok" if rec["error"] is None else "err"] += 1
            tot = counters["ok"] + counters["err"]
            if tot % 10 == 0:
                print(f"[{args.tier}] {tot}/{len(jobs)} ok={counters['ok']} err={counters['err']}", flush=True)

    pending = [j for j in jobs if (j[0], j[1], j[2], j[3]) not in done]
    if args.tier == "human":
        workers = 1
    elif args.workers != 3:
        workers = args.workers
    else:
        workers = 6 if args.tier in ("tier3", "tier5", "tier6", "tier7") else 3
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(do_job, pending))
    out_f.close()
    print(f"[{args.tier}] DONE ok={counters['ok']} err={counters['err']}")

if __name__ == "__main__":
    main()

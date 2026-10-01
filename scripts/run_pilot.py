#!/usr/bin/env python3
"""
ARDIPF pilot study runner (v2: full-length generations + hierarchical beacon detector).
Channels:
  B1  exact SHA-256 match (classical byte binding)
  B2  Winnowing-style 5-shingle Jaccard attribution (lexical)
  e3  semantic: doc-level cosine (attribution) + claim-weighted SIPS (integrity)
  e2  L3 semantic beacons, hierarchical detector:
        L0 exact phrase            -> 1.00
        L1 ordered token subsequence (window +6) -> 0.90
        L2 semantic-span match (window embedding cosine >= BEACON_SEM_THR) -> 0.75
        L3 rare-signature tokens (>=2 of beacon's distinctive tokens) -> 0.50
Decision (gated attribution, fixed a priori):
  ATTRIBUTE iff doc_cos(best registered match) >= COS_THR and BSR >= BSR_THR
  (semantic similarity alone never attributes -- keyed content evidence required)
Forgery probes: false ownership claims must be REFUSED.
Outputs: results/pilot_results.json (+ make_report.py -> results/pilot_report.md)
"""
import argparse, hashlib, json, os, re, time, datetime
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus/seeds/pilot_corpus.json"
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
RAW = ROOT / "results/transformations.jsonl"

SENT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

COS_THR = 0.75     # theta_sem for attribution
BSR_THR = 0.5/3    # >= one L3-level hit (0.5/3)
BEACON_SEM_THR = 0.78
MAX_TOK = 600

PROMPTS = {
    "paraphrase": "Rewrite the following text in different words while preserving all factual details and meaning exactly. Output only the rewritten text, no preamble.\n\nTEXT:\n{t}",
    "summarize": "Summarize the following text in about 40 percent of its original length, preserving the key facts and requirements. Output only the summary, no preamble.\n\nTEXT:\n{t}",
    "rewrite_lay": "Rewrite the following text for a general non-expert audience while preserving the key facts. Output only the rewritten text, no preamble.\n\nTEXT:\n{t}",
    "translate_es": "Translate the following text into Spanish. Output only the translation, no preamble.\n\nTEXT:\n{t}",
}

STOP = set("""a an and are as at be been but by for from had has have in is it its of on or
that the their then there these they this to was were will with within where which while
any all also may might shall should must can could our we you your this following text""".split())

def gen(model, prompt, max_tok=MAX_TOK, temp=0.7, timeout=240):
    r = requests.post(f"{OLLAMA}/api/generate", json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temp, "num_predict": max_tok},
    }, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def norm(s): return re.sub(r"\s+", " ", s).strip().lower()

def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?;:])\s+", t) if len(s.strip()) > 10]

def shingles(t, k=5):
    toks = re.findall(r"[a-z0-9]+", norm(t))
    return {tuple(toks[i:i+k]) for i in range(len(toks)-k+1)} if len(toks) >= k else set()

CLAIM_WEIGHT_KW = ["shall", "must", "may not", "prohibited", "mitigation", "impact",
                   "severity", "retention", "dispose", "disposition", "revoked", "required"]

def claim_weight(s):
    sl = s.lower()
    return 1.5 if any(k in sl for k in CLAIM_WEIGHT_KW) else 1.0

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def sips(c_emb, c_w, d_emb, match_thr=0.35):
    if len(c_emb) == 0:
        return 0.0, 0, 0
    if len(d_emb) == 0:
        return 0.0, 0, len(c_emb)
    S = np.array([[cosine(a, b) for b in d_emb] for a in c_emb])
    order = sorted(((S[i, j], i, j) for i in range(S.shape[0]) for j in range(S.shape[1])), reverse=True)
    picked, used, tot_w, acc, matched = {}, set(), 0.0, 0.0, 0
    for v, i, j in order:
        if v < match_thr: break
        if i in picked or j in used: continue
        picked[i] = v; used.add(j)
    for i, w in enumerate(c_w):
        tot_w += w
        if i in picked:
            acc += w * picked[i]; matched += 1
    return acc / max(tot_w, 1e-12), matched, len(c_emb) - matched

def sig_tokens(beacon):
    return [t for t in re.findall(r"[a-z0-9']+", beacon.lower()) if len(t) >= 6 and t not in STOP]

def beacon_hits(text, beacons, beacon_embs, encoder):
    """Hierarchical detector. Returns per-beacon level and score."""
    t = norm(text)
    toks = re.findall(r"[a-z0-9']+", t)
    # candidate windows for semantic matching: sentences + 8-token sliding windows
    windows = sentences(text)
    for i in range(0, max(0, len(toks)-7), 4):
        windows.append(" ".join(toks[i:i+8]))
    win_embs = encoder.encode(windows, normalize_embeddings=True, show_progress_bar=False) if windows else None
    out = {}
    for bi, b in enumerate(beacons):
        bt = re.findall(r"[a-z0-9']+", b.lower())
        level, score = "miss", 0.0
        if b.lower() in t:
            level, score = "exact", 1.0
        else:
            pos = -1; ok = True
            for x in bt:
                try: pos = toks.index(x, pos + 1)
                except ValueError: ok = False; break
            if ok:
                level, score = "subseq", 0.9
            else:
                best = 0.0
                if win_embs is not None and len(win_embs):
                    sims = win_embs @ beacon_embs[bi]
                    best = float(sims.max())
                if best >= BEACON_SEM_THR:
                    level, score = f"semantic({best:.2f})", 0.75
                else:
                    sig = sig_tokens(b)
                    hits = sum(1 for s in sig if s in t)
                    if sig and hits >= min(2, len(sig)):
                        level, score = "sigtokens", 0.5
        out[b] = {"level": level, "score": score}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/pilot_results.json"))
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    corpus = json.loads(CORPUS.read_text())
    docs = corpus["documents"]; owners = corpus["owners"]; forgeries = corpus["forgeries"]

    jobs = []
    fams = ["paraphrase", "summarize"] if args.quick else ["paraphrase", "summarize", "rewrite_lay", "translate_es"]
    for d in docs:
        for fam in fams:
            jobs.append(("phi3:latest", d["doc_id"], fam))
    if not args.quick:
        for d in docs[:3]:
            jobs.append(("tinyllama:latest", d["doc_id"], "paraphrase"))

    if not args.skip_gen:
        done = set()
        if RAW.exists():
            for line in RAW.read_text().splitlines():
                if line.strip():
                    j = json.loads(line); done.add((j["engine"], j["doc_id"], j["family"]))
        with RAW.open("a") as f:
            for engine, doc_id, fam in jobs:
                if (engine, doc_id, fam) in done: continue
                doc = next(d for d in docs if d["doc_id"] == doc_id)
                t0 = time.time()
                try:
                    out = gen(engine, PROMPTS[fam].format(t=doc["text"]))
                    err = None
                except Exception as e:
                    out, err = "", repr(e)
                rec = {"engine": engine, "doc_id": doc_id, "family": fam,
                       "output": out, "error": err, "max_tokens": MAX_TOK,
                       "seconds": round(time.time() - t0, 1),
                       "ts": datetime.datetime.now().isoformat(timespec="seconds")}
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"[gen] {engine:18s} {doc_id:15s} {fam:12s} {rec['seconds']:6.1f}s "
                      f"{'OK' if not err else 'ERR: ' + err}", flush=True)

    print("[emb] loading sentence transformer ...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SENT_MODEL)

    def embed_sents(t):
        ss = sentences(t)
        if not ss: return [], np.zeros((0, model.get_sentence_embedding_dimension()))
        return ss, np.asarray(model.encode(ss, normalize_embeddings=True, show_progress_bar=False))

    def embed_doc(t):
        ss, em = embed_sents(t)
        if em.size == 0: return np.zeros(model.get_sentence_embedding_dimension())
        v = em.mean(axis=0); return v / (np.linalg.norm(v) + 1e-12)

    # beacon embeddings per owner
    owner_beacon_embs = {}
    for o, meta in owners.items():
        owner_beacon_embs[o] = np.asarray(model.encode(meta["beacons"], normalize_embeddings=True,
                                                       show_progress_bar=False))

    registry = {}
    for d in docs:
        ss, em = embed_sents(d["text"])
        registry[d["doc_id"]] = {
            "doc": d, "embs": em, "weights": [claim_weight(s) for s in ss],
            "docvec": embed_doc(d["text"]),
            "hash": hashlib.sha256(d["text"].encode()).hexdigest(),
            "shingles": shingles(d["text"]),
        }

    tx = []
    if RAW.exists():
        for line in RAW.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("output") and not r.get("error"):
                    tx.append(r)

    rows = []
    for r in tx:
        d = next(x for x in docs if x["doc_id"] == r["doc_id"])
        out = r["output"]
        o_ss, o_em = embed_sents(out)
        o_doc = embed_doc(out)
        o_sh = shingles(out)
        o_hash = hashlib.sha256(out.encode()).hexdigest()

        best_doc_id, best_cos, best_sips, best_match = None, 0.0, 0.0, (0, 0)
        best_jac, best_jac_id = 0.0, None
        for reg_id, reg in registry.items():
            ds = cosine(reg["docvec"], o_doc)
            sc, m, lost = sips(reg["embs"], reg["weights"], o_em)
            jac = len(reg["shingles"] & o_sh) / max(1, len(reg["shingles"] | o_sh))
            if ds > best_cos: best_doc_id, best_cos = reg_id, ds
            if sc > best_sips: best_sips, best_match = sc, (m, lost)
            if jac > best_jac: best_jac, best_jac_id = jac, reg_id

        best_owner = registry[best_doc_id]["doc"]["owner"]
        hits = beacon_hits(out, owners[best_owner]["beacons"], owner_beacon_embs[best_owner], model)
        bsr = float(np.mean([h["score"] for h in hits.values()]))

        gated = (best_cos >= COS_THR) and (bsr >= BSR_THR)
        tau = round(0.6 * best_cos + 0.4 * bsr, 3)
        sh_attr_owner = registry[best_jac_id]["doc"]["owner"] if best_jac_id and best_jac >= 0.05 else None

        rows.append({
            "engine": r["engine"], "doc_id": r["doc_id"], "family": r["family"],
            "true_owner": d["owner"],
            "B1_hash_match": o_hash == registry[best_doc_id]["hash"],
            "B2_shingle_owner": sh_attr_owner, "B2_jaccard": round(best_jac, 3),
            "B2_correct": bool(sh_attr_owner == d["owner"]),
            "doc_cosine": round(best_cos, 3),
            "SIPS": round(best_sips, 3), "claims_matched": best_match[0], "claims_lost": best_match[1],
            "beacon_hits": {k: v["level"] for k, v in hits.items()},
            "BSR": round(bsr, 3), "tau": tau,
            "verdict": "ATTRIBUTE" if gated else "REFUSE",
            "attributed_owner": best_owner if gated else None,
            "ORA_hit": bool(gated and best_owner == d["owner"]),
            "best_match_doc": best_doc_id, "gen_seconds": r["seconds"],
        })

    forge_rows = []
    for fg in forgeries:
        f_ss, f_em = embed_sents(fg["text"])
        f_doc = embed_doc(fg["text"])
        best_doc_id, best_cos = None, 0.0
        for reg_id, reg in registry.items():
            ds = cosine(reg["docvec"], f_doc)
            if ds > best_cos: best_doc_id, best_cos = reg_id, ds
        match_owner = registry[best_doc_id]["doc"]["owner"]
        hits = beacon_hits(fg["text"], owners[fg["claimed_owner"]]["beacons"],
                           owner_beacon_embs[fg["claimed_owner"]], model)
        bsr = float(np.mean([h["score"] for h in hits.values()]))
        gated = (best_cos >= COS_THR) and (bsr >= BSR_THR)
        forge_rows.append({
            "doc_id": fg["doc_id"], "claimed_owner": fg["claimed_owner"],
            "semantic_best_match_owner": match_owner, "doc_cosine": round(best_cos, 3),
            "beacon_hits_claimed_owner": {k: v["level"] for k, v in hits.items()},
            "BSR_vs_claimed": round(bsr, 3),
            "tau": round(0.6 * best_cos + 0.4 * bsr, 3),
            "verdict": "ATTRIBUTE" if gated else "REFUSE",
            "correctly_refused": not gated,
            "note": "similarity-only would attribute" if best_cos >= COS_THR else "low similarity too",
        })

    def agg(rows_, key=None):
        sub = [r for r in rows_ if r["family"] == key] if key else rows_
        if not sub: return None
        return {
            "n": len(sub),
            "doc_cosine_mean": round(float(np.mean([r["doc_cosine"] for r in sub])), 3),
            "SIPS_mean": round(float(np.mean([r["SIPS"] for r in sub])), 3),
            "BSR_mean": round(float(np.mean([r["BSR"] for r in sub])), 3),
            "ORA_fused": round(float(np.mean([r["ORA_hit"] for r in sub])), 3),
            "ORA_shingle_baseline": round(float(np.mean([r["B2_correct"] for r in sub])), 3),
            "ORA_hash_baseline": round(float(np.mean([r["B1_hash_match"] for r in sub])), 3),
        }

    # separation of legit vs forged by tau (rank statistic = ROC-AUC)
    def auc(pos, neg):
        if not pos or not neg: return None
        pos, neg = list(pos), list(neg)
        rank_sum = 0.0
        for p in pos:
            rank_sum += sum(1 for n in neg if n < p) + 0.5 * sum(1 for n in neg if n == p)
        return round(rank_sum / (len(pos) * len(neg)), 3)
    tau_legit = [r["tau"] for r in rows]
    tau_forge = [f["tau"] for f in forge_rows]
    cos_legit = [r["doc_cosine"] for r in rows]
    cos_forge = [f["doc_cosine"] for f in forge_rows]

    out = {
        "meta": {
            "corpus": "ARDIP-Pilot-v1", "model": SENT_MODEL, "ollama": OLLAMA,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "decision_rule": {"cos_thr": COS_THR, "bsr_thr": round(BSR_THR, 3),
                              "beacon_sem_thr": BEACON_SEM_THR, "max_tokens": MAX_TOK},
            "beacon_level_weights": {"exact": 1.0, "subseq": 0.9, "semantic": 0.75, "sigtokens": 0.5},
        },
        "per_query": rows,
        "forgery_queries": forge_rows,
        "aggregate_by_family": {f: agg(rows, f) for f in ["paraphrase", "summarize", "rewrite_lay", "translate_es"]},
        "aggregate_by_engine": {e: agg([r for r in rows if r["engine"] == e]) for e in sorted({r["engine"] for r in rows})},
        "headline": {
            "ORA_fused_all": round(float(np.mean([r["ORA_hit"] for r in rows])), 3) if rows else None,
            "ORA_hash_baseline_all": round(float(np.mean([r["B1_hash_match"] for r in rows])), 3) if rows else None,
            "ORA_shingle_baseline_all": round(float(np.mean([r["B2_correct"] for r in rows])), 3) if rows else None,
            "forgeries_refused": sum(1 for f in forge_rows if f["correctly_refused"]),
            "forgeries_total": len(forge_rows),
            "similarity_would_misattribute": sum(1 for f in forge_rows if f["doc_cosine"] >= COS_THR),
            "AUC_tau_legit_vs_forged": auc(tau_legit, tau_forge),
            "AUC_cosine_alone": auc(cos_legit, cos_forge),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["headline"], indent=2))
    print("saved ->", args.out)

if __name__ == "__main__":
    main()

import json, hashlib, re, time, datetime
from pathlib import Path
import numpy as np, requests

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT/"corpus/seeds/ardip_bench_corpus.json"
import sys
sys.path.insert(0, str(ROOT/"scripts"))
OLLAMA=None
from run_transformations import gen as rgen

STOP=set("""a an and are as at be but by for from had has have in is it its of on or that the
their then there these they this to was were will with within where which while any all also
may might shall should must can could our we you your following text""".split())
CLAIM_KW=["shall","must","may not","prohibited","mitigation","impact","severity","retention",
          "required","should","vulnerability","exploit"]
def norm(s): return re.sub(r"\s+"," ",s).strip().lower()
def sentences(t): return [s.strip() for s in re.split(r"(?<=[.!?;:])\s+",t) if len(s.strip())>10]
def cosine(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def claim_weight(s):
    sl=s.lower(); return 1.5 if any(k in sl for k in CLAIM_KW) else 1.0
def shingles(t,k=5):
    toks=re.findall(r"[a-z0-9]+",norm(t))
    return {tuple(toks[i:i+k]) for i in range(len(toks)-k+1)} if len(toks)>=k else set()
def sips(c_emb,c_w,d_emb,thr=0.35):
    if len(c_emb)==0 or len(d_emb)==0: return 0.0
    S=np.array([[cosine(a,b) for b in d_emb] for a in c_emb])
    order=sorted(((S[i,j],i,j) for i in range(S.shape[0]) for j in range(S.shape[1])),reverse=True)
    picked,used,tot,acc={},set(),0.0,0.0
    for v,i,j in order:
        if v<thr: break
        if i in picked or j in used: continue
        picked[i]=v; used.add(j)
    for i,w in enumerate(c_w):
        tot+=w
        if i in picked: acc+=w*picked[i]
    return acc/max(tot,1e-12)
def sig_tokens(b): return [t for t in re.findall(r"[a-z0-9']+",b.lower()) if len(t)>=6 and t not in STOP]

def main():
    corpus=json.loads(CORPUS.read_text())
    docs,owners=corpus["documents"],corpus["owners"]
    rng=np.random.default_rng(42)
    sample=[docs[i] for i in rng.choice(len(docs),size=15,replace=False)]
    print("[cp] loading encoder...",flush=True)
    from sentence_transformers import SentenceTransformer
    enc=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    def embed_doc(t):
        ss=sentences(t)
        if not ss: return np.zeros(enc.get_sentence_embedding_dimension())
        em=np.asarray(enc.encode(ss,normalize_embeddings=True,show_progress_bar=False))
        v=em.mean(axis=0); return v/(np.linalg.norm(v)+1e-12)
    def embed_claims(t):
        ss=sentences(t)
        if not ss: return np.zeros((0,enc.get_sentence_embedding_dimension())),[]
        em=np.asarray(enc.encode(ss,normalize_embeddings=True,show_progress_bar=False))
        return em,[claim_weight(s) for s in ss]
    print("[cp] embedding registry...",flush=True)
    reg_vec=np.asarray([embed_doc(d["text"]) for d in docs])
    reg_claims=[embed_claims(d["text"]) for d in docs]
    beacon_embs={o:np.asarray(enc.encode(m["beacons"],normalize_embeddings=True,show_progress_bar=False)) for o,m in owners.items()}
    def detect(text,owner):
        t=norm(text); toks=re.findall(r"[a-z0-9']+",t)
        wins=sentences(text)
        for i in range(0,max(0,len(toks)-7),4): wins.append(" ".join(toks[i:i+8]))
        wem=enc.encode(wins,normalize_embeddings=True,show_progress_bar=False) if wins else None
        sc=[]
        for bi,b in enumerate(owners[owner]["beacons"]):
            if b.lower() in t: sc.append(1.0); continue
            bt=re.findall(r"[a-z0-9']+",b.lower()); pos,ok=-1,True
            for x in bt:
                try: pos=toks.index(x,pos+1)
                except ValueError: ok=False;break
            if ok: sc.append(0.9); continue
            best=float((wem@beacon_embs[owner][bi]).max()) if wem is not None and len(wem) else 0.0
            if best>=0.78: sc.append(0.75); continue
            sig=sig_tokens(b)
            sc.append(0.5 if sig and sum(1 for s in sig if s in t)>=min(2,len(sig)) else 0.0)
        return float(np.mean(sc))
    def verify(text):
        v=embed_doc(text); sims=reg_vec@v; top=int(np.argmax(sims)); cos=float(sims[top])
        qc,qw=embed_claims(text); rc,rw=reg_claims[top]
        s=sips(rc,rw,qc); owner=docs[top]["owner"]; b=detect(text,owner)
        keyed=(b>=0.5 and cos>=0.60); cons=(cos>=0.96 and b>=0.167)
        return {"best_doc":docs[top]["doc_id"],"owner":owner,"true_owner":None,"cos":round(cos,3),
                "sips":round(s,3),"bsr":round(b,3),"attributed":bool(keyed or cons)}
    results=[]
    for d in sample:
        # 1) verbatim copy-paste (content only, new file, no metadata)
        v=verify(d["text"]); v.update({"doc":d["doc_id"],"true_owner":d["owner"],"attack":"verbatim_copypaste"})
        v["correct_owner"]= (v["attributed"] and v["owner"]==d["owner"])
        results.append(v)
        # 2) copy-paste then AI paraphrase (beacon scrub attempt) via DeepSeek
        try:
            para=rgen("frontier:DeepSeek-V4-Pro",
                "Rewrite the following text in different words while preserving all factual details and meaning exactly. Output only the rewritten text, no preamble.\n\nTEXT:\n"+d["text"],max_tok=3000)
            p=verify(para); p.update({"doc":d["doc_id"],"true_owner":d["owner"],"attack":"copypaste_then_paraphrase"})
            p["correct_owner"]=(p["attributed"] and p["owner"]==d["owner"])
            results.append(p)
        except Exception as e:
            results.append({"doc":d["doc_id"],"attack":"copypaste_then_paraphrase","error":repr(e)})
    out={"generated_at":datetime.datetime.now().isoformat(timespec="seconds"),"n_docs":len(sample),"results":results}
    (ROOT/"results/copypaste_attack_results.json").write_text(json.dumps(out,indent=2))
    verb=[r for r in results if r.get("attack")=="verbatim_copypaste"]
    para=[r for r in results if r.get("attack")=="copypaste_then_paraphrase" and "cos" in r]
    print("\n=== COPY-PASTE ATTACK RESULTS ===")
    print(f"verbatim copy-paste : attributed-to-true-owner {sum(r['correct_owner'] for r in verb)}/{len(verb)}  (mean cos {np.mean([r['cos'] for r in verb]):.3f}, mean BSR {np.mean([r['bsr'] for r in verb]):.3f})")
    if para:
        print(f"copy+paraphrase     : attributed-to-true-owner {sum(r['correct_owner'] for r in para)}/{len(para)}  (mean cos {np.mean([r['cos'] for r in para]):.3f}, mean BSR {np.mean([r['bsr'] for r in para]):.3f})")
    print("saved -> results/copypaste_attack_results.json")

if __name__=="__main__":
    main()

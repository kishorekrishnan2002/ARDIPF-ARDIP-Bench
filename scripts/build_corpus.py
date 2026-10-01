#!/usr/bin/env python3
"""
Build ARDIP-Bench-Lite corpus: 1,200 documents across six public domains from REAL sources.

  1. research papers    -> arXiv abstracts (cs.CR/CL/LG/AI)        owner ACAD-LAB-Z   (300)
  2. government reports -> Federal Register rules/notices          owner GOV-AGENCY-X (200)
  3. technical standards-> IETF RFC excerpts                       owner STD-BODY-Y   (200)
  4. CERT advisories    -> NVD CVE data (real vulns, structured)   owner CERT-EXAMPLE (200)
  5. policy documents   -> Federal Register PRESDOCU + NIST pubs   owner GOV-POLICY-B (150)
  6. academic articles  -> Europe PMC open-access abstracts        owner ACAD-MED-C   (150)

Beacon sentences are inserted at simulated registration (ARDIPF semantics: beacons are
embedded when a document is registered). Raw sources snapshotted under corpus/sources/.
GAO/CRS endpoints returned HTTP 403 from this network; Federal Register substitutes for
government reports (documented in paper §Methodology).
"""
import json, random, re, sys, time, html as htmllib
import urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "corpus/sources"
OUT = ROOT / "corpus/seeds/ardip_bench_corpus.json"
RNG = random.Random(20260929)

OWNERS = {
    "CERT-EXAMPLE": {"beacons": [
        "This advisory remains authoritative notwithstanding any prior circular on the same component.",
        "Operational findings herein shall be forthwith transmitted to constituency members through the established secure channel.",
        "Any remediation guidance countermanded by subsequent bulletin is void from the date of issue."]},
    "GOV-AGENCY-X": {"beacons": [
        "It is expressly incumbent upon the accountable officer to certify compliance with this directive.",
        "The permitted actions are delineated in Annex C of this directive and supersede earlier schedules.",
        "These requirements apply unless superseded by subsequent instruction from the governance board."]},
    "GOV-POLICY-B": {"beacons": [
        "This policy is binding on all subordinate offices notwithstanding any prior memorandum.",
        "Deviations require the documented concurrence of the policy secretariat and shall be logged accordingly.",
        "The provisions herein remain in force unless rescinded by subsequent executive action."]},
    "STD-BODY-Y": {"beacons": [
        "Conformant implementations SHALL exhibit auditable behavior for every event covered by this section.",
        "Non-conformant behavior MUST be surfaced through the designated fault channel without delay.",
        "Auditable evidence MUST be retained for the period mandated by the deploying jurisdiction."]},
    "ACAD-LAB-Z": {"beacons": [
        "We operationalize this notion formally and release the artifacts for replication.",
        "Our ablations isolate this effect from confounding factors across model scales.",
        "The corpus is released for replication together with the evaluation harness."]},
    "ACAD-MED-C": {"beacons": [
        "The study protocol was registered beforehand and the dataset is released for replication.",
        "Our sensitivity analyses isolate this effect from confounding factors.",
        "Ethics approval and informed consent documentation accompany the released corpus."]},
}

UA = {"User-Agent": "ARDIPF-research-corpus/1.0 (contact: research@example.org)"}

def fetch(url, dest=None, timeout=40):
    if dest and dest.exists() and dest.stat().st_size > 500:
        return dest.read_bytes()
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if dest:
        dest.write_bytes(data)
    return data

def fetch_json(url, dest=None):
    return json.loads(fetch(url, dest).decode("utf-8"))

def clean_text(t):
    return re.sub(r"\s+", " ", t).strip()

def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) > 10]

# ---------------- 1. arXiv research papers ----------------
ARXIV = [("cat:cs.CR", 130), ("cat:cs.CL", 95), ("cat:cs.LG", 50), ("cat:cs.AI", 35)]

def collect_arxiv():
    docs, ns = [], {"a": "http://www.w3.org/2005/Atom"}
    for cat, want in ARXIV:
        url = f"https://export.arxiv.org/api/query?search_query={urllib.parse.quote(cat)}&start=0&max_results={want}"
        try:
            data = fetch(url, SRC / "arxiv" / f"batch_{cat.replace(':','_')}.xml")
        except Exception as e:
            print(f"[arxiv] {cat} FAILED: {e}", file=sys.stderr); continue
        root = ET.fromstring(data)
        for e in root.findall("a:entry", ns):
            summ = clean_text(e.findtext("a:summary", "", ns))
            title = clean_text(e.findtext("a:title", "", ns))
            aid = e.findtext("a:id", "", ns)
            if 100 <= len(summ.split()) <= 330 and len(sentences(summ)) >= 4:
                docs.append({"title": title, "text": summ, "source": aid})
        time.sleep(3)
    return docs

# ---------------- 2/5. Federal Register ----------------
def fr_candidates(doc_type, per_page=200, pages=2):
    out = []
    for pg in range(1, pages + 1):
        url = ("https://www.federalregister.gov/api/v1/documents.json?"
               f"conditions[type]={doc_type}&per_page={per_page}&page={pg}"
               "&fields[]=title&fields[]=raw_text_url&fields[]=publication_date")
        try:
            j = fetch_json(url, SRC / "fr" / f"{doc_type}_p{pg}.json")
        except Exception as e:
            print(f"[fr] {doc_type} p{pg} FAILED: {e}", file=sys.stderr); continue
        out.extend(j.get("results", []))
        time.sleep(0.4)
    return out

def fr_excerpt(raw, lo=100, hi=220):
    txt = raw.decode("utf-8", "ignore")
    txt = re.sub(r"(?im)^\[FR Doc.*$", " ", txt)
    paras = [clean_text(p) for p in re.split(r"\n\s*\n", txt)]
    cand = [p for p in paras if lo <= len(p.split()) <= hi and len(sentences(p)) >= 3
            and not re.match(r"^(AGENCY|ACTION|SUMMARY|DATES|ADDRESSES|FOR FURTHER|SUPPLEMENTARY)", p)]
    if not cand: return None
    return cand[len(cand) // 2]  # middle of document

def collect_fedreg(doc_type, n, tag):
    cands = [c for c in fr_candidates(doc_type) if c.get("raw_text_url")]
    RNG.shuffle(cands)
    docs = []
    for c in cands:
        if len(docs) >= n: break
        key = re.sub(r"[^A-Za-z0-9]+", "_", c["raw_text_url"])[-40:]
        try:
            raw = fetch(c["raw_text_url"], SRC / "fr" / f"{tag}_{key}.txt")
        except Exception:
            continue
        ex = fr_excerpt(raw)
        if ex:
            docs.append({"title": clean_text(c.get("title", ""))[:110], "text": ex,
                         "source": c["raw_text_url"]})
        time.sleep(0.15)
    return docs

# ---------------- 3. RFCs ----------------
RFC_LIST = [1034,1035,1122,1123,1321,1421,1422,1423,1510,1750,1918,1928,1939,1994,2045,
            2046,2104,2119,2131,2246,2315,2401,2402,2404,2406,2410,2474,2560,2616,2712,2818,
            2828,2830,2898,2986,3029,3161,3268,3279,3280,3394,3447,3501,3513,3526,3546,3552,
            3602,3610,3713,3720,3810,3986,3987,4033,4034,4035,4086,4107,4121,4162,4210,4211,
            4226,4251,4252,4253,4279,4301,4302,4303,4346,4357,4366,4418,4492,4493,4494,4507,
            4627,4634,4647,4648,4764,4868,4880,4949,4960,5054,5055,5056,5077,5116,5191,5216,
            5226,5229,5246,5277,5280,5281,5288,5289,5321,5424,5425,5480,5487,5506,5647,5649,
            5652,5702,5735,5758,5759,5869,5905,5934,5990,6024,6066,6091,6101,6112,6125,6176,
            6187,6209,6234,6265,6347,6367,6402,6454,6455,6585,6637,6698,6749,6797,6819,6943,
            6960,6961,6962,6973,6991,7027,7030,7120,7230,7231,7233,7296,7301,7435,7457,7468,
            7469,7507,7515,7519,7525,7540,7616,7627,7633,7641,7672,7685,7748,7761,7807,7831,
            7850,7905,7906,7919,7924,7932,7942,8005,8017,8032,8053,8055,8162,8226,8246,8247,
            8301,8410,8411,8422,8446,8452,8457,8492,8555,8576,8613,8615,8674,8705,8725,8747,
            8808,8890,8906,8907,8908,9001,9028,9068,9101,9110,9111,9155,9162,9163,9180,9231,
            9335,9336,9420,9421,9425,9458,9460,9474,9524,9536,9539,9576,9619,9620,9645,9661,
            9728,9738,9758,9839]

PAGE_HDR = re.compile(r"^(RFC \d+|.*Standards Track|.*Informational|.*Experimental|.*Best Current Practice|\[Page \d+\])\s*$")

def rfc_paragraphs(txt):
    body, buf = [], []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln:
            if buf: body.append(" ".join(buf)); buf = []
            continue
        if PAGE_HDR.match(ln): continue
        buf.append(ln)
    if buf: body.append(" ".join(buf))
    paras = [clean_text(p) for p in body]
    return [p for p in paras if 100 <= len(p.split()) <= 230 and len(sentences(p)) >= 3]

def collect_rfc(n):
    docs = []
    for num in RFC_LIST:
        if len(docs) >= n: break
        dest = SRC / "rfc" / f"rfc{num}.txt"
        try:
            data = fetch(f"https://www.rfc-editor.org/rfc/rfc{num}.txt", dest)
        except Exception:
            continue
        paras = rfc_paragraphs(data.decode("utf-8", "ignore"))
        if paras:
            strong = [p for p in paras if re.search(r"\b(MUST|SHALL|SHOULD|security|threat)\b", p)]
            pool = strong if strong else paras
            p = pool[num % len(pool)]
            docs.append({"title": f"RFC {num} excerpt", "text": p,
                         "source": f"https://www.rfc-editor.org/rfc/rfc{num}.txt"})
        time.sleep(0.12)
    return docs

# ---------------- 4. CERT advisories from NVD ----------------
def collect_nvd(n):
    docs = []
    windows = [("2026-05-01T00:00:00.000", "2026-07-01T00:00:00.000"),
               ("2026-07-01T00:00:00.000", "2026-09-01T00:00:00.000"),
               ("2026-03-01T00:00:00.000", "2026-05-01T00:00:00.000")]
    seen = set()
    for s, e in windows:
        url = ("https://services.nvd.nist.gov/rest/json/cves/2.0?resultsPerPage=200"
               f"&pubStartDate={s}&pubEndDate={e}")
        try:
            j = fetch_json(url, SRC / "nvd" / f"cves_{s[:7]}.json")
        except Exception as ex:
            print(f"[nvd] window {s[:7]} FAILED: {ex}", file=sys.stderr)
            time.sleep(30); continue
        for it in j.get("vulnerabilities", []):
            c = it.get("cve", {})
            cid = c.get("id", "")
            if cid in seen: continue
            desc = next((d["value"] for d in c.get("descriptions", []) if d["lang"] == "en"), "")
            desc = clean_text(desc)
            if len(desc.split()) < 40 or desc.lower().startswith("** rejected"): continue
            score, sev = None, "Medium"
            m = c.get("metrics", {})
            for k in ("cvssMetricV31", "cvssMetricV40", "cvssMetricV30"):
                if m.get(k):
                    cd = m[k][0]["cvssData"]; score = cd.get("baseScore"); sev = cd.get("baseSeverity", sev)
                    break
            cwes = [w["description"][0]["value"] for w in c.get("weaknesses", [])
                    if w.get("description") and w["description"][0]["lang"] == "en"][:2]
            txt = (f"Advisory on {cid}. Severity: {str(sev).title()}"
                   + (f" (CVSS {score})" if score else "") + ". " + desc + " "
                   + (f"Weakness classification: {'; '.join(cwes)}. " if cwes else "")
                   + "Affected deployments should assess exposure, apply vendor fixes when available, "
                     "restrict access to affected interfaces in the interim, and monitor for exploitation "
                     "attempts consistent with this issue. Confirmed incidents should be reported through "
                     "the established secure channel.")
            docs.append({"title": f"Advisory {cid}", "text": clean_text(txt),
                         "source": f"NVD:{cid} (services.nvd.nist.gov, public domain)"})
            seen.add(cid)
            if len(docs) >= n: break
        if len(docs) >= n: break
        time.sleep(31)  # NVD rate limit without API key
    return docs

# ---------------- 6. Europe PMC open-access abstracts ----------------
def collect_pmc(n):
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
           + urllib.parse.quote("OPEN_ACCESS:y AND PUB_YEAR:[2021 TO 2026]")
           + "&format=json&pageSize=1000&resultType=core")
    try:
        j = fetch_json(url, SRC / "pmc" / "pmc_batch1.json")
    except Exception as e:
        print(f"[pmc] FAILED: {e}", file=sys.stderr); return []
    docs = []
    for r in j.get("resultList", {}).get("result", []):
        ab = clean_text(r.get("abstractText", ""))
        ab = re.sub(r"<[^>]+>", " ", ab)
        ab = clean_text(ab)
        title = clean_text(r.get("title", ""))
        pmcid = r.get("pmcid") or r.get("id", "")
        if 110 <= len(ab.split()) <= 330 and len(sentences(ab)) >= 4:
            docs.append({"title": title[:110], "text": ab,
                         "source": f"EuropePMC:{pmcid} (open access)"})
        if len(docs) >= n: break
    return docs

# ---------------- 5b. NIST policy paragraphs ----------------
NIST = [("https://csrc.nist.gov/pubs/sp/800/63/b/upd1/final", "sp800-63b-upd1.html"),
        ("https://csrc.nist.gov/pubs/sp/800/207/final", "sp800-207.html"),
        ("https://csrc.nist.gov/pubs/sp/800/218/final", "sp800-218.html"),
        ("https://csrc.nist.gov/pubs/ai/100/1/final", "ai100-1.html")]

def nist_paragraphs(htm):
    htm = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", htm)
    txt = re.sub(r"(?s)<[^>]+>", "\n", htm)
    txt = htmllib.unescape(txt)
    paras = [clean_text(p) for p in txt.split("\n")]
    return [p for p in paras if 90 <= len(p.split()) <= 220 and len(sentences(p)) >= 3]

def collect_nist():
    docs = []
    for url, fname in NIST:
        try:
            data = fetch(url, SRC / "nist" / fname)
        except Exception as e:
            print(f"[nist] FAILED {fname}: {e}", file=sys.stderr); continue
        paras = nist_paragraphs(data.decode("utf-8", "ignore"))
        if len(paras) > 6:
            idxs = sorted({int(i * (len(paras) - 1) / 7) for i in range(8)})
            for j, i in enumerate(idxs):
                docs.append({"title": f"{fname} excerpt {j+1}", "text": paras[i], "source": url})
    return docs

# ---------------- registration (beacon insertion) ----------------
def insert_beacons(text, beacons):
    ss = sentences(text)
    positions = sorted({max(1, int(f * len(ss))) for f in (0.25, 0.55, 0.85)}, reverse=True)
    for b, pos in zip(beacons, positions):
        ss.insert(min(pos, len(ss)), b)
    return " ".join(ss)

SYNONYMS = {"immediately": "promptly", "utilize": "use", "shall": "will", "must": "has to",
            "however": "nevertheless", "therefore": "thus", "additional": "extra",
            "demonstrate": "show", "method": "technique", "system": "platform",
            "approach": "technique", "significant": "substantial"}

def strip_beacons(text, beacons):
    for b in beacons: text = text.replace(b, " ")
    return clean_text(text)

def light_paraphrase(text):
    for a, b in SYNONYMS.items():
        text = re.sub(rf"\b{a}\b", b, text)
    return text

def main():
    for d in ("arxiv", "fr", "rfc", "nvd", "pmc", "nist", "generated"):
        (SRC / d).mkdir(parents=True, exist_ok=True)

    targets = [("ACAD-LAB-Z", "research_paper", collect_arxiv, 300),
               ("GOV-AGENCY-X", "gov_report", lambda: collect_fedreg("RULE", 130, "rule") + collect_fedreg("NOTICE", 90, "notice"), 200),
               ("STD-BODY-Y", "technical_standard", lambda: collect_rfc(215), 200),
               ("CERT-EXAMPLE", "cert_advisory", lambda: collect_nvd(215), 200),
               ("GOV-POLICY-B", "policy_document", lambda: collect_fedreg("PRESDOCU", 135, "presdocu") + collect_nist(), 150),
               ("ACAD-MED-C", "academic_article", lambda: collect_pmc(165), 150)]

    corpus_docs = []
    for owner, register, fn, tgt in targets:
        try:
            items = fn()
        except Exception as e:
            print(f"[build] {register} collector crashed: {e}", file=sys.stderr); items = []
        items = items[:tgt]
        print(f"[build] {register:20s} collected {len(items)}/{tgt}")
        for it in items:
            did = f"B1-{owner.split('-')[0]}-{len(corpus_docs)+1:04d}"
            corpus_docs.append({"doc_id": did, "owner": owner, "register": register,
                                "title": it["title"][:110],
                                "text": insert_beacons(it["text"], OWNERS[owner]["beacons"]),
                                "provenance": it["source"]})

    # forgeries: 30 stripped near-duplicates (across owners) + 2 hand-crafted
    forgeries = []
    samples = RNG.sample(corpus_docs, 30)
    for i, d in enumerate(samples):
        beacons = OWNERS[d["owner"]]["beacons"]
        forgeries.append({"doc_id": f"B1-FORGE-{i+1:02d}", "claimed_owner": d["owner"],
                          "true_owner": None, "derived_from": d["doc_id"],
                          "attack": "beacon-stripped near-duplicate claiming victim ownership",
                          "text": light_paraphrase(strip_beacons(d["text"], beacons))})
    v1 = json.loads((ROOT / "corpus/seeds/pilot_corpus.json").read_text())
    for fg in v1["forgeries"]:
        fg["doc_id"] = fg["doc_id"].replace("FORGE", "B1-FORGE-HC")
        fg["attack"] = "hand-crafted stylistic imitation"
        forgeries.append(fg)

    out = {"corpus": "ARDIP-Bench-Lite v1", "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "note": ("Real public sources: arXiv, Federal Register, IETF RFCs, NVD, Europe PMC, NIST. "
                    "Beacons inserted at simulated registration. GAO/CRS unreachable (HTTP 403); "
                    "Federal Register substitutes for government reports."),
           "owners": OWNERS, "documents": corpus_docs, "forgeries": forgeries}
    OUT.write_text(json.dumps(out, indent=2))
    from collections import Counter
    print("[build] TOTAL:", len(corpus_docs), "| forgeries:", len(forgeries))
    print("[build] by owner:", dict(Counter(d["owner"] for d in corpus_docs)))
    print("[build] saved ->", OUT)

if __name__ == "__main__":
    main()

# ARDIPF Pilot Study Report

Generated: 2026-09-29T10:15:25 · encoder: `sentence-transformers/all-MiniLM-L6-v2` · decision rule: {'cos_thr': 0.75, 'bsr_thr': 0.167, 'beacon_sem_thr': 0.78, 'max_tokens': 600} · beacon levels: {'exact': 1.0, 'subseq': 0.9, 'semantic': 0.75, 'sigtokens': 0.5}

## Headline

- **ORA (gated semantic+beacon attribution): 0.257**
- ORA exact-hash baseline (B1): 0.0
- ORA lexical-shingle baseline (B2): 0.086
- Forged ownership claims correctly refused: 2/2
- Forgeries that semantic-similarity ALONE would have wrongly attributed: 2
- AUC separating legit vs forged by fused τ: 0.357
- AUC separating legit vs forged by cosine alone: 0.243

> **Key finding:** semantic similarity by itself does NOT separate legitimate from forged documents (forgeries reach cosine 0.83–0.91, so a similarity-only system would misattribute them). Attribution is only sound when gated by keyed in-content evidence (beacon recovery) and/or registered lineage — which is exactly what the forgery probes demonstrate.

## By transformation family

| Family | n | SIPS | doc-cos | BSR | ORA fused | ORA shingle | ORA hash |
|---|---|---|---|---|---|---|---|
| paraphrase | 11 | 0.59 | 0.896 | 0.136 | 0.545 | 0.273 | 0.0 |
| summarize | 8 | 0.33 | 0.846 | 0.063 | 0.375 | 0.0 | 0.0 |
| rewrite_lay | 8 | 0.447 | 0.771 | 0.0 | 0.0 | 0.0 | 0.0 |
| translate_es | 8 | 0.25 | 0.429 | 0.0 | 0.0 | 0.0 | 0.0 |

## By engine

| Engine | n | SIPS mean | ORA fused |
|---|---|---|---|
| phi3:latest | 32 | 0.388 | 0.188 |
| tinyllama:latest | 3 | 0.761 | 1.0 |

## Per-query detail

| engine | doc | family | doc-cos | SIPS | BSR | τ | verdict | attributed | correct | shingle |
|---|---|---|---|---|---|---|---|---|---|---|
| phi3 | PILOT-CERT-001 | paraphrase | 0.894 | 0.398 | 0.0 | 0.537 | REFUSE | — | ✘ | ✘ (J=0.014) |
| phi3 | PILOT-CERT-001 | summarize | 0.829 | 0.238 | 0.0 | 0.497 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-CERT-001 | rewrite_lay | 0.717 | 0.197 | 0.0 | 0.43 | REFUSE | — | ✘ | ✘ (J=0.004) |
| phi3 | PILOT-CERT-001 | translate_es | 0.403 | 0.303 | 0.0 | 0.242 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-CERT-002 | paraphrase | 0.845 | 0.455 | 0.0 | 0.507 | REFUSE | — | ✘ | ✘ (J=0.003) |
| phi3 | PILOT-CERT-002 | summarize | 0.837 | 0.26 | 0.0 | 0.502 | REFUSE | — | ✘ | ✘ (J=0.008) |
| phi3 | PILOT-CERT-002 | rewrite_lay | 0.686 | 0.291 | 0.0 | 0.411 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-CERT-002 | translate_es | 0.424 | 0.284 | 0.0 | 0.254 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-GOV-001 | paraphrase | 0.883 | 0.64 | 0.0 | 0.53 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-GOV-001 | summarize | 0.825 | 0.28 | 0.167 | 0.562 | ATTRIBUTE | GOV-AGENCY-X | ✔ | ✘ (J=0.0) |
| phi3 | PILOT-GOV-001 | rewrite_lay | 0.784 | 0.392 | 0.0 | 0.47 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-GOV-001 | translate_es | 0.382 | 0.097 | 0.0 | 0.229 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-GOV-002 | paraphrase | 0.867 | 0.499 | 0.167 | 0.587 | ATTRIBUTE | GOV-AGENCY-X | ✔ | ✘ (J=0.0) |
| phi3 | PILOT-GOV-002 | summarize | 0.829 | 0.218 | 0.0 | 0.497 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-GOV-002 | rewrite_lay | 0.816 | 0.66 | 0.0 | 0.49 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-GOV-002 | translate_es | 0.457 | 0.326 | 0.0 | 0.274 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-STD-001 | paraphrase | 0.886 | 0.575 | 0.167 | 0.598 | ATTRIBUTE | STD-BODY-Y | ✔ | ✘ (J=0.0) |
| phi3 | PILOT-STD-001 | summarize | 0.886 | 0.446 | 0.167 | 0.598 | ATTRIBUTE | STD-BODY-Y | ✔ | ✘ (J=0.009) |
| phi3 | PILOT-STD-001 | rewrite_lay | 0.765 | 0.478 | 0.0 | 0.459 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-STD-001 | translate_es | 0.468 | 0.227 | 0.0 | 0.281 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-STD-002 | paraphrase | 0.895 | 0.479 | 0.25 | 0.637 | ATTRIBUTE | STD-BODY-Y | ✔ | ✘ (J=0.006) |
| phi3 | PILOT-STD-002 | summarize | 0.904 | 0.508 | 0.167 | 0.609 | ATTRIBUTE | STD-BODY-Y | ✔ | ✘ (J=0.004) |
| phi3 | PILOT-STD-002 | rewrite_lay | 0.796 | 0.579 | 0.0 | 0.478 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-STD-002 | translate_es | 0.426 | 0.26 | 0.0 | 0.255 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-ACAD-001 | paraphrase | 0.901 | 0.659 | 0.0 | 0.541 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-ACAD-001 | summarize | 0.754 | 0.184 | 0.0 | 0.452 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-ACAD-001 | rewrite_lay | 0.81 | 0.486 | 0.0 | 0.486 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-ACAD-001 | translate_es | 0.551 | 0.375 | 0.0 | 0.331 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-ACAD-002 | paraphrase | 0.887 | 0.5 | 0.0 | 0.532 | REFUSE | — | ✘ | ✘ (J=0.009) |
| phi3 | PILOT-ACAD-002 | summarize | 0.904 | 0.502 | 0.0 | 0.542 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-ACAD-002 | rewrite_lay | 0.795 | 0.496 | 0.0 | 0.477 | REFUSE | — | ✘ | ✘ (J=0.0) |
| phi3 | PILOT-ACAD-002 | translate_es | 0.325 | 0.13 | 0.0 | 0.195 | REFUSE | — | ✘ | ✘ (J=0.0) |
| tinyllama | PILOT-CERT-001 | paraphrase | 0.898 | 0.69 | 0.167 | 0.606 | ATTRIBUTE | CERT-EXAMPLE | ✔ | ✔ (J=0.129) |
| tinyllama | PILOT-CERT-002 | paraphrase | 0.941 | 0.807 | 0.333 | 0.698 | ATTRIBUTE | CERT-EXAMPLE | ✔ | ✔ (J=0.153) |
| tinyllama | PILOT-GOV-001 | paraphrase | 0.957 | 0.786 | 0.417 | 0.741 | ATTRIBUTE | GOV-AGENCY-X | ✔ | ✔ (J=0.253) |

## Forgery probes

| forged doc | claimed owner | best semantic match owner | doc-cos | BSR vs claimed | τ | verdict | correctly refused |
|---|---|---|---|---|---|---|---|
| FORGE-001 | GOV-AGENCY-X | GOV-AGENCY-X | 0.908 | 0.0 | 0.545 | REFUSE | ✔ |
| FORGE-002 | CERT-EXAMPLE | CERT-EXAMPLE | 0.831 | 0.0 | 0.498 | REFUSE | ✔ |

## Interpretation

- B1 (exact hash) scores 0 on every transformed document: any LLM rewrite changes all bytes (G1 byte-anchor collapse).
- B2 (lexical shingles) is classical plagiarism-style attribution; it degrades as semantic distance grows.
- doc-cosine (semantic channel) is strong for paraphrase/summarize, weaker for lay-rewrite, and low for translation (monolingual encoder).
- SIPS (claim-weighted) measures semantic *integrity* and drops under summarization as claims are dropped — by design.
- **Beacon survival is capacity-dependent:** the weak model (tinyllama) preserves beacons (BSR up to ~0.42, incl. exact hits); the stronger model (phi3) scrubs most (BSR ~0). This is the honest, central pilot observation motivating stronger beacons + lineage.
- Attribution rule = semantic match AND keyed beacon evidence. The forgery probes show this gate is *necessary*: similarity alone would have attributed both forgeries.
- The low absolute ORA (0.26) is expected for a deliberately minimal beacon set and is NOT the framework's projected performance; the full ARDIPF adds lineage/registration and hardened beacons (projections in docs/10).

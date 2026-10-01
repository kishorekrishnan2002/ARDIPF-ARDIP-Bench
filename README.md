# ARDIP-Bench: AI-Resilient Document Integrity and Provenance Benchmark

Data and code accompanying the manuscript:

> **ARDIPF: An AI-Resilient Framework for Document Integrity, Ownership, and Provenance Under Generative AI Transformation**

## What is included

```
data/
  ardip_bench_corpus.json        1,204 registered documents, 6 domains, 6 owner identities,
                                 owner-linked beacons embedded at simulated registration
  transformations_bench.jsonl     23,098 AI transformation records (7 engines x 8 families)
  results/
    bench_results_full.json       primary evaluation metrics (full run)
    bench_results_fast.json       stratified-sample evaluation (incl. T6/T9 overlays)
    copypaste_attack_results.json copy-paste attribution attack study
    pilot_results.json            8-document pilot study
    pilot_report.md               pilot report (human-readable)
scripts/
  build_corpus.py                 corpus builder (re-fetches from the public sources below)
  run_transformations.py          transformation generation (LLM engines)
  evaluate.py                     evaluation pipeline (all metrics, ablations, overlays)
  run_pilot.py                    pilot study pipeline
  test_copypaste_attack.py        copy-paste attack experiment
  requirements.txt                Python dependencies
docs/
  SOURCE_MANIFEST.md              public sources and licenses for every corpus domain
```

## Corpus domains (all public sources)

| Domain | Source | n |
|---|---|---|
| Research papers | arXiv abstracts (cs.CR/CL/LG/AI) | 319 |
| Government reports | U.S. Federal Register rules/notices | 200 |
| Technical standards | IETF RFC excerpts | 200 |
| CERT-style advisories | NVD CVE records (public vulnerability data) | 200 |
| Policy documents | Federal Register presidential docs + NIST SP text | 135 |
| Academic articles | Europe PMC open-access abstracts | 150 |

The advisory-style documents are reconstructed from public NVD vulnerability descriptions;
no non-public advisory content is included anywhere in this release.

## Transformation engines

DeepSeek-V4-Pro, Qwen3.8, Kimi-K3, GLM, GLM-Flash (frontier-class, accessed through an
OpenAI-compatible research endpoint), Phi-3-mini-3.8B and TinyLlama-1.1B (local Ollama),
and a deterministic rule-based human-edit baseline. Engine endpoint URLs and credentials are
**not** included; `run_transformations.py` reads them from environment variables
(`OLLAMA_HOST`, `FRONTIER_BASE_URL`, `FRONTIER_API_KEY`).

## Reproducing the reported results

The evaluation pipeline needs no LLM access — it runs entirely over the bundled data:

```bash
pip install -r scripts/requirements.txt
# metrics, ablations, and attack overlays over the bundled transformations:
python scripts/evaluate.py            # reads data/ and writes results
```

Regenerating the transformations requires engine access (see script header).

## Key metrics in this release

| Metric | Value |
|---|---|
| Ownership Recovery Accuracy (overall) | 0.835 |
| ORA on attested (gateway) flows | 0.962 |
| False Attribution Rate | 0.000 |
| Fused legit-vs-forged AUC (vs cosine-only) | 0.999 (0.52) |
| Rollback detection (T12) | 0.967 |
| Verification latency p50/p95 | 0.70 / 1.52 s |

## Privacy / sensitivity statement

This package was scanned before release: it contains **no credentials, no internal
hostnames or addresses, no personal data, and no non-public organizational documents**.
All corpus content derives from the public sources listed in `docs/SOURCE_MANIFEST.md`.

## License

Code: MIT License (see `LICENSE`).
Data: released under CC BY 4.0; underlying corpus documents retain the licenses of their
public sources (see `docs/SOURCE_MANIFEST.md`).

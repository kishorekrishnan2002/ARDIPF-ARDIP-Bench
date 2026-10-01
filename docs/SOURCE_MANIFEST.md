# Source Manifest — ARDIP-Bench corpus

Every corpus domain is built from public sources. The builder (`scripts/build_corpus.py`)
re-fetches from these origins; URLs of all fetched items were snapshotted during corpus
construction.

| Domain | Origin | Access | License / status |
|---|---|---|---|
| Research papers | arXiv API (`export.arxiv.org/api/query`, categories cs.CR, cs.CL, cs.LG, cs.AI) | public | Abstracts; per-paper arXiv licenses (see arXiv.org) |
| Government reports | U.S. Federal Register API (`federalregister.gov/api/v1`) | public | U.S. public domain |
| Technical standards | IETF RFC editor (`rfc-editor.org/rfc/rfcNNNN.txt`) | public | Freely redistributable (IETF) |
| Advisory documents | NIST National Vulnerability Database API (`services.nvd.nist.gov`) | public | U.S. public domain |
| Policy documents | Federal Register presidential documents + NIST Special Publication text (`csrc.nist.gov`) | public | U.S. public domain |
| Academic articles | Europe PMC REST API (`ebi.ac.uk/europepmc/webservices/rest`) | public | Open-access articles (per-article licenses) |

Notes:
- Advisory-style corpus documents are reconstructed from public NVD CVE descriptions; no
  non-public advisory material is included.
- No author, institutional, or government-internal documents are part of this corpus.
- Engine endpoints and credentials are intentionally NOT part of this release.

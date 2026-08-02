#!/usr/bin/env python3
"""Bake-off RQ-b: source-concentration analysis (framework x backend).

Pre-registered prediction: the SEARCH BACKEND governs source concentration
(domain-HHI / arXiv-share), while overall quality stays bounded across
frameworks. We have one framework crossed on backend (Open Deep Research on
Azure-web-search vs Tavily), plus 5 other frameworks on their default backend.

Outputs staging/bakeoff_concentration.json:
  - per-arm: mean HHI, arXiv-share, academic-share, n_sources, n_unique_domains
  - backend contrast (ODR azure vs ODR tavily, query-paired): dHHI, darXiv, paired CI
  - framework spread on HHI (default-backend arms) for a backend-vs-framework read

Pure CPU: parses report markdown for URLs. Deterministic; seed=20260712 for the
paired bootstrap. Writes to STAGING only (main session merges + reconciles).
"""
import json, re, glob, os, statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo root
BASE = ROOT / "results" / "experiments_bakeoff"
OUT = ROOT / "papers" / "paper_a_bounded_returns" / "analysis" / "staging" / "bakeoff_concentration.json"
SEED = 20260712
N_BOOT = 10000

ARMS = ["gpt_researcher", "open_deep_research", "open_deep_research_tavily",
        "storm", "ii_researcher", "owl", "deerflow"]

URL_RE = re.compile(r'https?://([^/\s)\]}"\'>,]+)', re.I)
ACADEMIC = ("arxiv.org", "scholar.google", "pubmed", "ncbi.nlm.nih.gov", "doi.org",
            "aclanthology.org", "openreview.net", "semanticscholar.org", "acm.org",
            "ieee.org", "nature.com", "sciencedirect.com", "springer.com", "ssrn.com",
            "biorxiv.org", "medrxiv.org", "proceedings.")


def registrable(host: str) -> str:
    host = host.lower().lstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def report_domains(text: str) -> list[str]:
    return [registrable(m.group(1)) for m in URL_RE.finditer(text)]


def hhi(domains: list[str]) -> float | None:
    """Herfindahl index on domain shares (1.0 = single domain, ->0 = diffuse)."""
    if not domains:
        return None
    c = Counter(domains)
    n = len(domains)
    return sum((v / n) ** 2 for v in c.values())


def arxiv_share(domains: list[str]) -> float | None:
    if not domains:
        return None
    return sum(1 for d in domains if "arxiv.org" in d) / len(domains)


def academic_share(domains: list[str]) -> float | None:
    if not domains:
        return None
    return sum(1 for d in domains if any(a in d for a in ACADEMIC)) / len(domains)


def load_arm(arm: str) -> dict[str, dict]:
    """query_id -> per-report concentration metrics."""
    out = {}
    for f in sorted(glob.glob(str(BASE / arm / "*.md"))):
        qid = Path(f).stem
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        doms = report_domains(text)
        out[qid] = {
            "n_urls": len(doms),
            "n_unique_domains": len(set(doms)),
            "hhi": hhi(doms),
            "arxiv_share": arxiv_share(doms),
            "academic_share": academic_share(doms),
        }
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def paired_bootstrap_diff(pairs: list[tuple[float, float]]):
    """Deterministic paired bootstrap of mean(b - a). Returns (mean, lo, hi)."""
    import random
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    diffs = [b - a for a, b in pairs]
    obs = statistics.mean(diffs)
    rng = random.Random(SEED)
    boots = []
    n = len(diffs)
    for _ in range(N_BOOT):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(statistics.mean(s))
    boots.sort()
    lo = boots[int(0.025 * N_BOOT)]
    hi = boots[int(0.975 * N_BOOT)]
    return {"mean_diff": round(obs, 4), "ci95": [round(lo, 4), round(hi, 4)], "n_pairs": n}


def main():
    arms = {a: load_arm(a) for a in ARMS}

    per_arm = {}
    for a, reps in arms.items():
        per_arm[a] = {
            "n_reports": len(reps),
            "mean_hhi": _mean([r["hhi"] for r in reps.values()]),
            "mean_arxiv_share": _mean([r["arxiv_share"] for r in reps.values()]),
            "mean_academic_share": _mean([r["academic_share"] for r in reps.values()]),
            "mean_n_urls": _mean([r["n_urls"] for r in reps.values()]),
            "mean_n_unique_domains": _mean([r["n_unique_domains"] for r in reps.values()]),
        }

    # Backend contrast: ODR azure vs ODR tavily, paired on query_id
    az, tv = arms["open_deep_research"], arms["open_deep_research_tavily"]
    common = sorted(set(az) & set(tv))
    backend = {
        "framework": "open_deep_research",
        "n_common_queries": len(common),
        "hhi_azure_vs_tavily": paired_bootstrap_diff([(az[q]["hhi"], tv[q]["hhi"]) for q in common]),
        "arxiv_azure_vs_tavily": paired_bootstrap_diff([(az[q]["arxiv_share"], tv[q]["arxiv_share"]) for q in common]),
        "academic_azure_vs_tavily": paired_bootstrap_diff([(az[q]["academic_share"], tv[q]["academic_share"]) for q in common]),
        "note": "positive mean_diff = Tavily MORE concentrated / more arXiv than Azure web-search",
    }

    # Framework spread on HHI (default-backend arms only: exclude the tavily twin)
    default_arms = [a for a in ARMS if a != "open_deep_research_tavily"]
    fw_hhi = {a: per_arm[a]["mean_hhi"] for a in default_arms if per_arm[a]["mean_hhi"] is not None}
    fw_spread = {
        "arms": fw_hhi,
        "range": round(max(fw_hhi.values()) - min(fw_hhi.values()), 4) if fw_hhi else None,
        "note": "framework main-effect proxy on default backends; compare its size to the within-ODR backend dHHI",
    }

    result = {
        "experiment": "bakeoff_concentration_rqb",
        "date": "2026-07-12",
        "method": "domain-HHI + arXiv/academic share from report-markdown URL extraction; paired bootstrap seed=%d N=%d" % (SEED, N_BOOT),
        "caveat": "only Open Deep Research is crossed on backend, so a full 2-way framework x backend variance partition is not identified; we report the clean within-ODR backend contrast plus the across-framework HHI spread as a size comparison.",
        "per_arm": per_arm,
        "backend_contrast": backend,
        "framework_hhi_spread": fw_spread,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps({"per_arm_hhi": {a: per_arm[a]["mean_hhi"] for a in ARMS},
                      "per_arm_arxiv": {a: per_arm[a]["mean_arxiv_share"] for a in ARMS},
                      "backend_dHHI": backend["hhi_azure_vs_tavily"],
                      "backend_dArxiv": backend["arxiv_azure_vs_tavily"],
                      "framework_hhi_range": fw_spread["range"]}, indent=2))


if __name__ == "__main__":
    main()

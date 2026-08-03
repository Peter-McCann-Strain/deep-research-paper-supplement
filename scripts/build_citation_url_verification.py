#!/usr/bin/env python3
"""Build the `citation_url_verification` canonical key.

PURPOSE
-------
Defend the headline claim that orchestration's gain "lands on citation quality,
not factual accuracy" against the standard reviewer critique that "LLM judges
reward citation FORM over substance, and 3-13% of cited URLs are fabricated"
(arXiv:2604.03173-style).

It does this with two *independent* evidence channels over the canonical
P0-P10 base report corpus:

  (1) URL RESOLUTION (the form/fabrication channel).
      Cited URLs are extracted from the reports (handled upstream by the
      phase-7a citation extractor, which parses BOTH inline `[n]` markers and
      `[n] Title — URL` reference sections, and writes
      data/analysis/df_citations.parquet). A stratified-across-pattern SAMPLE of
      the unique URLs is resolved over HTTP (HEAD, GET fallback, follow
      redirects, short timeout, polite per-host rate limiting). This measures
      the resolve rate and the dead/fabricated rate empirically, instead of
      asserting an external paper's number.

  (2) ENTAILMENT SUPPORT (the substance channel).
      To rebut "form over substance" we report what fraction of cited claims are
      actually *entailed* by the source they cite. We reuse the on-disk
      claim-level entailment verdicts (data/analysis/df_c0_verdicts.parquet,
      the same C0/FActScore pipeline behind canonical['citation_faithfulness']
      and the e14 oracle entailment snapshot), restricted to claims with a
      non-null citation_idx (the cited subset, n=176 — matches
      citation_faithfulness['proxy_lowerbound']['n_cited_total']). SUPPORT rate =
      share of cited claims whose verdict == 'supports'.

HONESTY / COVERAGE
------------------
The substance channel has limited coverage: only 176 of ~22.9k citation slots
have an entailment verdict attached (the C0 run sampled claims per report and
only a minority of sampled claims carried an inline citation marker). The URL
channel covers far more (a few hundred resolved out of 10,235 unique URLs). We
report both coverage figures explicitly and DO NOT extrapolate either beyond its
measured base. We also do NOT call an external 3-13% number "confirmed"; we
report the rate we measure here.

CONVENTIONS
-----------
Atomic merge-preserving write mirrors build_frozen_vintage.py:_atomic_append
(line ~400): load store, set one key, json.dump to a tempfile in the analysis
dir, os.replace. THIS SCRIPT DEFAULTS TO --dry-run AND, AS REQUESTED, ONLY
PRINTS THE KEY JSON. --write is wired but intentionally not used for this task.

Usage:
  ./venv/bin/python scripts/build_citation_url_verification.py            # dry-run (default): print key, no write
  ./venv/bin/python scripts/build_citation_url_verification.py --sample 400 --seed 7
  ./venv/bin/python scripts/build_citation_url_verification.py --offline  # skip HTTP, entailment channel only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import ssl
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

ROOT = Path(".")
ANA = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
CANON = ANA / "canonical_numbers.json"
CITES_PARQUET = ROOT / "data" / "analysis" / "df_citations.parquet"
C0_VERDICTS_PARQUET = ROOT / "data" / "analysis" / "df_c0_verdicts.parquet"

KEY = "citation_url_verification"

# Verdict labels from the C0/FActScore entailment pipeline.
SUPPORT_LABELS = {"supports"}
NON_ENTAIL_LABELS = {"neutral", "no_source", "contradicts"}

# Categories from the phase-7a classifier that correspond to an actual http(s)
# URL we can try to resolve. 'placeholder' = web-search-synthesis / empty / no
# scheme, so it is not resolvable by construction.
RESOLVABLE_CATEGORIES = {"real_url", "academic", "suspicious"}

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# Treat these as "resolved / live" vs "dead/fabricated".
# 2xx and 3xx -> live. 401/403/405/406/429/451 are access-restricted, NOT
# fabricated (the host exists and answered), so we count them as RESOLVED-LIVE
# but flag them separately. 404/410 and DNS/connection failures -> dead.
LIVE_STATUS = set(range(200, 400))
ACCESS_RESTRICTED_STATUS = {401, 403, 405, 406, 429, 451, 999}
DEAD_STATUS = {400, 404, 410, 451}  # 451 also restricted; kept in restricted above (takes precedence)


# ---------------------------------------------------------------------------
# Polite, rate-limited HTTP resolver
# ---------------------------------------------------------------------------
class HostThrottle:
    """Enforce a minimum gap between requests to the same host."""

    def __init__(self, min_gap_s: float = 1.0):
        self.min_gap_s = min_gap_s
        self._last: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                gap = now - self._last[host]
                if gap >= self.min_gap_s:
                    self._last[host] = now
                    return
                sleep_for = self.min_gap_s - gap
            time.sleep(min(sleep_for, self.min_gap_s))


def _host(url: str) -> str:
    try:
        return urlsplit(url).netloc.lower()
    except Exception:
        return ""


def resolve_one(url: str, throttle: HostThrottle, timeout: float = 5.0) -> dict:
    """Resolve a single URL. Returns dict with status classification."""
    host = _host(url)
    throttle.wait(host)
    sess_headers = {"User-Agent": UA, "Accept": "*/*"}
    result = {"url": url, "host": host, "status": None,
              "outcome": None, "error": None}
    if requests is None:
        result["outcome"] = "no_http_client"
        return result
    for method in ("HEAD", "GET"):
        try:
            r = requests.request(
                method, url, headers=sess_headers, timeout=timeout,
                allow_redirects=True, stream=(method == "GET"),
                verify=True)
            code = r.status_code
            if method == "GET":
                # don't download the body
                try:
                    r.close()
                except Exception:
                    pass
            result["status"] = code
            if code in ACCESS_RESTRICTED_STATUS:
                result["outcome"] = "access_restricted"  # host exists, answered
                return result
            if code in LIVE_STATUS:
                result["outcome"] = "live"
                return result
            if code in DEAD_STATUS or code >= 400:
                # a HEAD 405 already handled above; a real 404 here is dead.
                if method == "HEAD" and code in (400, 403, 405, 406, 501):
                    # some servers reject HEAD; retry with GET
                    continue
                result["outcome"] = "dead"
                return result
        except requests.exceptions.SSLError as e:
            result["error"] = "ssl:" + type(e).__name__
            # SSL failure: host exists but cert/handshake issue -> not fabricated
            result["outcome"] = "tls_error"
            return result
        except requests.exceptions.ConnectionError as e:
            result["error"] = "conn:" + type(e).__name__
            result["outcome"] = "dead"  # DNS / refused / unreachable
            return result
        except requests.exceptions.Timeout:
            result["error"] = "timeout"
            result["outcome"] = "timeout"
            return result
        except Exception as e:
            result["error"] = type(e).__name__
            result["outcome"] = "error"
            return result
    # HEAD said retry-with-GET but GET loop fell through
    if result["outcome"] is None:
        result["outcome"] = "dead" if result["status"] else "error"
    return result


# ---------------------------------------------------------------------------
# Entailment SUPPORT channel (substance)
# ---------------------------------------------------------------------------
def compute_support_channel() -> dict:
    """Of cited claims (citation_idx non-null), fraction entailed by their cite."""
    if not C0_VERDICTS_PARQUET.exists():
        return {"status": "data_insufficient",
                "note": f"missing {C0_VERDICTS_PARQUET}"}
    v = pd.read_parquet(C0_VERDICTS_PARQUET)
    cited = v[v["citation_idx"].notna()].copy()
    n_cited = len(cited)
    if n_cited == 0:
        return {"status": "no_cited_claims"}
    n_support = int((cited["verdict"].isin(SUPPORT_LABELS)).sum())
    support_rate = n_support / n_cited
    # Wilson-ish CI via bootstrap (cheap, deterministic)
    rng = np.random.default_rng(0)
    sup = cited["verdict"].isin(SUPPORT_LABELS).to_numpy().astype(float)
    boots = [rng.choice(sup, size=len(sup), replace=True).mean()
             for _ in range(2000)]
    ci = [round(float(np.percentile(boots, 2.5)), 4),
          round(float(np.percentile(boots, 97.5)), 4)]

    # Join to citation URLs so we can report support among URL-bearing cites.
    out = {"status": "ok",
           "n_cited_claims_with_verdict": n_cited,
           "support_rate_of_cited": round(support_rate, 4),
           "support_rate_ci95_bootstrap": ci,
           "verdict_breakdown": {k: int((cited["verdict"] == k).sum())
                                 for k in sorted(cited["verdict"].unique())},
           "method": ("C0/FActScore claim-level entailment "
                      "(df_c0_verdicts.parquet); same pipeline behind "
                      "canonical['citation_faithfulness'] and e14 oracle "
                      "entailment. Cited subset = claims with non-null "
                      "citation_idx."),
           "support_label": sorted(SUPPORT_LABELS)}

    if CITES_PARQUET.exists():
        c = pd.read_parquet(CITES_PARQUET)
        cited = cited.copy()
        cited["citation_idx"] = cited["citation_idx"].astype(int)
        j = cited.merge(
            c[["pattern", "query_id", "citation_index", "cited_url",
               "category"]],
            left_on=["pattern", "query_id", "citation_idx"],
            right_on=["pattern", "query_id", "citation_index"], how="left")
        url_bearing = j[j["cited_url"].fillna("").str.startswith("http")]
        if len(url_bearing):
            out["n_cited_with_resolvable_url"] = int(len(url_bearing))
            out["support_rate_among_url_bearing_cites"] = round(
                float(url_bearing["verdict"].isin(SUPPORT_LABELS).mean()), 4)
    return out


# ---------------------------------------------------------------------------
# URL RESOLUTION channel (form / fabrication)
# ---------------------------------------------------------------------------
def build_url_sample(sample_n: int, seed: int) -> pd.DataFrame:
    """Stratified-across-pattern sample of unique resolvable URLs."""
    c = pd.read_parquet(CITES_PARQUET)
    res = c[c["category"].isin(RESOLVABLE_CATEGORIES)
            & c["cited_url"].fillna("").str.startswith("http")].copy()
    # one row per (pattern, url) so per-pattern stratification is on uniques
    uniq = res.drop_duplicates(subset=["pattern", "cited_url"])[
        ["pattern", "cited_url", "category", "domain"]].reset_index(drop=True)
    pats = sorted(uniq["pattern"].unique())
    per = max(1, sample_n // len(pats))
    rng = random.Random(seed)
    chunks = []
    for p in pats:
        sub = uniq[uniq["pattern"] == p]
        take = min(per, len(sub))
        idx = rng.sample(list(sub.index), take)
        chunks.append(sub.loc[idx])
    out = pd.concat(chunks).drop_duplicates(subset=["cited_url"]).reset_index(
        drop=True)
    return out


def compute_url_channel(sample_n: int, seed: int, workers: int,
                        host_gap_s: float, timeout: float,
                        offline: bool) -> dict:
    if not CITES_PARQUET.exists():
        return {"status": "data_insufficient",
                "note": f"missing {CITES_PARQUET}"}
    c = pd.read_parquet(CITES_PARQUET)
    res_mask = (c["category"].isin(RESOLVABLE_CATEGORIES)
                & c["cited_url"].fillna("").str.startswith("http"))
    n_citation_slots = int(len(c))
    n_resolvable_slots = int(res_mask.sum())
    n_unique_urls = int(c.loc[res_mask, "cited_url"].nunique())

    base = {
        "n_citation_slots_total": n_citation_slots,
        "n_resolvable_url_slots": n_resolvable_slots,
        "n_unique_resolvable_urls": n_unique_urls,
        "placeholder_slots": int((c["category"] == "placeholder").sum()),
    }
    if offline or requests is None:
        base["status"] = "skipped_offline" if offline else "no_http_client"
        return base

    sample = build_url_sample(sample_n, seed)
    base["n_urls_sampled"] = int(len(sample))
    base["sampling"] = (f"stratified across {sample['pattern'].nunique()} "
                        f"patterns; seed={seed}; unique URLs only")

    throttle = HostThrottle(min_gap_s=host_gap_s)
    rows = []
    urls = list(sample.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(resolve_one, r.cited_url, throttle, timeout): r
                for r in urls}
        done = 0
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                rr = fut.result()
            except Exception as e:
                rr = {"url": r.cited_url, "host": _host(r.cited_url),
                      "status": None, "outcome": "error",
                      "error": type(e).__name__}
            rr["pattern"] = r.pattern
            rr["category"] = r.category
            rows.append(rr)
            done += 1
            if done % 25 == 0:
                print(f"  resolved {done}/{len(urls)} ...", file=sys.stderr)

    df = pd.DataFrame(rows)
    base["n_urls_checked"] = int(len(df))

    live = df["outcome"].isin(["live", "access_restricted", "tls_error"])
    dead = df["outcome"].isin(["dead"])
    indeterminate = df["outcome"].isin(["timeout", "error"])

    n = len(df)
    base["resolve_rate"] = round(float(live.mean()), 4)
    base["dead_or_fabricated_rate"] = round(float(dead.mean()), 4)
    base["indeterminate_rate"] = round(float(indeterminate.mean()), 4)
    base["access_restricted_rate"] = round(
        float((df["outcome"] == "access_restricted").mean()), 4)
    # Dead rate among DETERMINATE attempts (excludes timeout/error noise) -- the
    # honest "fabrication-or-rot" estimate that the reviewer critique is about.
    det = df[~indeterminate]
    if len(det):
        base["dead_rate_among_determinate"] = round(
            float((det["outcome"] == "dead").mean()), 4)
        base["n_determinate"] = int(len(det))
    base["outcome_breakdown"] = {k: int((df["outcome"] == k).sum())
                                 for k in sorted(df["outcome"].dropna().unique())}

    # cheap per-pattern resolve/dead
    per_pattern = {}
    for p, sub in df.groupby("pattern"):
        per_pattern[p] = {
            "n_checked": int(len(sub)),
            "resolve_rate": round(float(
                sub["outcome"].isin(["live", "access_restricted",
                                     "tls_error"]).mean()), 4),
            "dead_rate": round(float((sub["outcome"] == "dead").mean()), 4),
        }
    base["per_pattern"] = per_pattern
    base["status"] = "ok"
    base["resolver_notes"] = (
        "HEAD then GET fallback; follow redirects; timeout="
        f"{timeout}s; per-host min gap={host_gap_s}s; verify=True. "
        "2xx/3xx and 401/403/405/406/429/451 (access-restricted) and TLS "
        "handshake errors are counted RESOLVE-LIVE (host exists & answered / "
        "is reachable); 404/410 and DNS/connection failures are DEAD/fabricated; "
        "timeouts and other exceptions are INDETERMINATE and excluded from the "
        "determinate dead rate.")
    return base


# ---------------------------------------------------------------------------
def build(args) -> dict:
    url_ch = compute_url_channel(
        sample_n=args.sample, seed=args.seed, workers=args.workers,
        host_gap_s=args.host_gap, timeout=args.timeout, offline=args.offline)
    sup_ch = compute_support_channel()

    n_reports_sampled = None
    if CITES_PARQUET.exists():
        c = pd.read_parquet(CITES_PARQUET)
        n_reports_sampled = int(
            c.drop_duplicates(["pattern", "query_id"]).shape[0])

    out = {
        "_purpose": ("Empirical rebuttal to the 'LLM judges reward citation FORM "
                     "over substance; 3-13% of cited URLs are fabricated' critique "
                     "(arXiv:2604.03173-style). Two channels: (1) HTTP resolution "
                     "of a stratified sample of cited URLs measures the actual "
                     "dead/fabricated rate; (2) claim-level entailment of the cited "
                     "subset measures SUPPORT (substance). Numbers are MEASURED here, "
                     "not imported."),
        "corpus": "artifacts/experiments/canonical/base_p0..p11 (90 queries/pattern)",
        "n_reports_sampled": n_reports_sampled,
        "n_citations": url_ch.get("n_citation_slots_total"),
        "n_urls_checked": url_ch.get("n_urls_checked"),
        "resolve_rate": url_ch.get("resolve_rate"),
        "dead_or_fabricated_rate": url_ch.get("dead_or_fabricated_rate"),
        "dead_rate_among_determinate": url_ch.get("dead_rate_among_determinate"),
        "support_rate_of_resolved": sup_ch.get("support_rate_of_cited"),
        "support_rate_among_url_bearing_cites": sup_ch.get(
            "support_rate_among_url_bearing_cites"),
        "coverage_note": (
            "URL channel: {checked} URLs resolved out of {uniq} unique "
            "resolvable URLs ({slots} total citation slots; {ph} are "
            "non-resolvable web-search-synthesis/placeholder slots). "
            "SUPPORT channel coverage is LIMITED: only {ncit} citation slots "
            "carry a claim-level entailment verdict (C0 sampled claims per "
            "report; few sampled claims bore an inline [n] marker), so the "
            "support rate is over n={ncit} cited claims, NOT the full corpus. "
            "Both rates are reported on their own measured base; neither is "
            "extrapolated, and no external fabrication % is asserted as "
            "confirmed.").format(
                checked=url_ch.get("n_urls_checked"),
                uniq=url_ch.get("n_unique_resolvable_urls"),
                slots=url_ch.get("n_citation_slots_total"),
                ph=url_ch.get("placeholder_slots"),
                ncit=sup_ch.get("n_cited_claims_with_verdict")),
        "per_pattern": url_ch.get("per_pattern"),
        "url_resolution_channel": url_ch,
        "entailment_support_channel": sup_ch,
        "interpretation": (
            "If the measured dead/fabricated rate is in the same low band the "
            "critique cites (single-digit %) the corpus is NOT pathologically "
            "fabricated; and if the SUPPORT rate of cited claims is materially "
            "above zero, citations carry substance, not just form. Conversely a "
            "LOW support rate would CONCEDE the substance critique while still "
            "showing URLs mostly resolve -- the honest framing the paper should "
            "adopt. Read both channels together."),
    }
    return out


def _print_key(out):
    print(f"[{KEY}] DRY-RUN — computed, NOTHING written to canonical_numbers.json.\n")
    print(json.dumps({KEY: out}, indent=2))


def _atomic_append(out, force):  # wired but unused for this task
    cn = json.load(open(CANON))
    if KEY in cn and not force:
        print(f"[{KEY}] REFUSING to overwrite existing key (use --force).")
        return 1
    cn[KEY] = out
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=str(ANA),
                                   prefix="canonical_numbers.",
                                   suffix=".json.tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(cn, f, indent=1)
        os.replace(tmp, CANON)
        tmp = None
    except BaseException:
        if tmp is not None and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise
    print(f"[{KEY}] WROTE key -> {CANON} (store now {len(cn)} keys)")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + print the key, write nothing (DEFAULT)")
    ap.add_argument("--write", action="store_true",
                    help="atomically append the key (NOT used for this task)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sample", type=int, default=420,
                    help="approx number of unique URLs to resolve (stratified)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--host-gap", type=float, default=1.0,
                    help="min seconds between requests to the same host")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--offline", action="store_true",
                    help="skip HTTP; report entailment channel + counts only")
    args = ap.parse_args()

    if not CITES_PARQUET.exists():
        print(f"[{KEY}] citations parquet missing at {CITES_PARQUET}; "
              "run scripts/phase7a_citation_extraction.py first.")
        return 1

    out = build(args)

    if args.write:
        return _atomic_append(out, args.force)
    _print_key(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

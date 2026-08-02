#!/usr/bin/env python3
"""Tool-health scanner for deep-research generation logs.

Scans one or more structlog-style generation logs and prints a per-log +
aggregate HEALTH REPORT so a batch of generated reports can be TRUSTED or
DISTRUSTED before it is judged/analysed. It answers the question the 6-week
programme keeps asking: "were the search/extraction TOOLS actually working
while these reports were produced, or were they silently broken?"

WHAT IT LOOKS FOR (robust to partial lines / interleaved tracebacks):
  * total searches + % empty (a *_results event with count=0)
  * HTTP 429 throttling per host: s2 (Semantic Scholar), arxiv, bing, tavily,
    plus PTU/LLM 429s (rate_limited) reported separately (those are expected,
    short-lived, and NOT a search-tool fault)
  * auth failures (401 / 403 / AuthenticationError / DeploymentNotFound 404) —
    a single one usually invalidates the whole batch (wrong endpoint/key)
  * extraction health: trafilatura/bs4 successes, timeouts, errors, low-char
    (thin) extractions, mean chars
  * academic-channel YIELD: mean `academic=` per `search_done` — the canary for
    "Semantic Scholar is 429-throttled so the academic channel is ~0"
  * backbone-mismatch aborts and Python tracebacks

USAGE:
  [ -f venv/bin/activate ] && source venv/bin/activate
  python scripts/tool_health_report.py path/to/run.log
  python scripts/tool_health_report.py 'artifacts/**/logs/*.log'        # glob
  python scripts/tool_health_report.py a.log b.log --json
  python scripts/tool_health_report.py run.log --minutes 1              # first minute only

Exit code: 0 = healthy, 1 = degraded, 2 = broken (auth dead / backbone mismatch
/ >50% empty searches). CI/automation can gate a batch on the exit code.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# ── Known search hosts (event-name prefix -> canonical host label) ────────────
HOSTS = ("s2", "arxiv", "bing", "tavily")
HOST_LABEL = {
    "s2": "s2 (Semantic Scholar)",
    "arxiv": "arxiv",
    "bing": "bing (web)",
    "tavily": "tavily (web)",
}

# ── Line parsing ──────────────────────────────────────────────────────────────
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})")
_LEVEL_RE = re.compile(r"\[(debug|info|warning|error|critical)\s*\]")
# event = first whitespace-delimited token after the "[level]" bracket
_EVENT_RE = re.compile(r"\[(?:debug|info|warning|error|critical)\s*\]\s+([^\s]+)")


def _ts_seconds(line: str) -> Optional[float]:
    """Seconds-of-day for the leading timestamp (for --minutes windowing)."""
    m = _TS_RE.match(line)
    if not m:
        return None
    hh, mm, ss = m.group(2).split(":")
    return int(hh) * 3600 + int(mm) * 60 + int(ss)


def _kv_int(line: str, key: str) -> Optional[int]:
    m = re.search(rf"(?<![\w-]){re.escape(key)}=(\d+)", line)
    return int(m.group(1)) if m else None


def _event(line: str) -> Optional[str]:
    m = _EVENT_RE.search(line)
    return m.group(1) if m else None


def _host_of(event: str) -> Optional[str]:
    for h in HOSTS:
        if event.startswith(h + "_"):
            return h
    return None


# ── 429 / auth detection (keyword based, resilient to query-text false hits) ──
_429_MARKERS = ("status=429", "http 429", "error code: 429", "429 (",
                "too many requests", "'429'", '"429"')
_AUTH_MARKERS = ("status=401", "error code: 401", "http 401", "401 unauthorized",
                 "status=403", "error code: 403", "http 403", "unauthorized",
                 "authenticationerror", "invalid api key", "invalid subscription key",
                 "deploymentnotfound", "'401'", "'403'", "permissiondenied")


def _is_429(low: str) -> bool:
    return any(m in low for m in _429_MARKERS)


def _is_auth_fail(low: str) -> bool:
    return any(m in low for m in _AUTH_MARKERS)


# ── Per-log accumulator ───────────────────────────────────────────────────────
class Health:
    def __init__(self, name: str):
        self.name = name
        self.lines = 0
        self.parsed = 0            # lines with a recognisable [level] event
        self.t_first: Optional[float] = None
        self.t_last: Optional[float] = None

        # search results per host: attempts (terminal result events) + empties
        self.results = Counter()   # host -> count of LIVE *_results events
        self.empty = Counter()     # host -> LIVE *_results with count=0
        self.search_fail = Counter()  # host -> terminal failure (exception/exhausted/error)
        self.h429 = Counter()      # host -> 429-bearing lines
        self.cache_hits = Counter()  # host -> *_cache_hit (served from cache, no live req)
        self.ptu_429 = 0           # LLM/PTU rate_limited events (expected, separate)

        # per-QUERY source yield (the metric that decides trust): a search_done
        # with academic+web == 0 is a genuinely dry query (got NO sources).
        self.query_searches = 0    # count of search_done events
        self.query_dry = 0         # search_done with academic==0 AND web==0

        # academic-channel yield (mean academic= per search_done)
        self.academic_vals: list[int] = []
        self.web_vals: list[int] = []
        self.dedup_vals: list[int] = []

        # extraction
        self.extract_ok = 0        # trafilatura_ok
        self.extract_bs4 = 0       # bs4_fallback_ok
        self.extract_timeout = 0   # url_extract_timeout
        self.extract_error = 0     # url_extract_error
        self.extract_lowchar = 0   # trafilatura_ok/bs4 with chars < threshold
        self.char_vals: list[int] = []

        # fatal / correctness signals
        self.auth_fail = 0
        self.backbone_mismatch = 0
        self.tracebacks = 0
        self.pattern_starts = Counter()  # p0_start etc.

        # cost/volume context
        self.cost_seen = 0.0
        self.cost_events = 0
        self.llm_complete = 0
        self.no_source_events = 0  # no_docs_from_search / no_relevant_sources*

        self.warnings: list[str] = []
        self._severity = "healthy"

    # ── ingest one raw line ──
    def feed(self, line: str, low: str, low_char_thresh: int) -> None:
        self.lines += 1

        # timestamp window tracking
        ts = _ts_seconds(line)
        if ts is not None:
            if self.t_first is None:
                self.t_first = ts
            self.t_last = ts

        # tracebacks (multi-line; count the header)
        if "traceback (most recent call last):" in low:
            self.tracebacks += 1
        if "backbone mismatch" in low:
            self.backbone_mismatch += 1

        event = _event(line)
        if event is None:
            # No structured event on this line, but it may still carry a
            # traceback body, a bare 429/401, etc. Auth failures are critical
            # enough to catch even off-event.
            if _is_auth_fail(low):
                self.auth_fail += 1
            return
        self.parsed += 1

        # ── 429 attribution ──
        if _is_429(low):
            if event == "rate_limited":            # LLM/PTU generation endpoint
                self.ptu_429 += 1
            else:
                host = _host_of(event) or (
                    "arxiv" if "arxiv" in low else
                    "s2" if ("semanticscholar" in low or "semantic_scholar" in low) else
                    "bing" if "bing" in low else
                    "tavily" if "tavily" in low else None)
                if host:
                    self.h429[host] += 1
                else:
                    self.ptu_429 += 1  # unattributable 429 -> bucket as generic

        # ── auth failures (on any event line) ──
        if _is_auth_fail(low):
            self.auth_fail += 1

        # ── search result terminal events ──
        host = _host_of(event)
        if event.endswith("_results") and host in HOSTS:
            self.results[host] += 1
            cnt = _kv_int(line, "count")
            if cnt == 0:
                self.empty[host] += 1
        elif host in HOSTS and (
            event.endswith("_exception")
            or event.endswith("_retries_exhausted")
            or event.endswith("_batch_error")
            or (event.endswith("_error") and not event.endswith("_batch_error"))
        ):
            self.search_fail[host] += 1

        # ── cache hits (served without a live request) ──
        if event.endswith("_cache_hit") and host in HOSTS:
            self.cache_hits[host] += 1

        # ── aggregate search_done channel yields (per-query source yield) ──
        if event.endswith("search_done"):
            a = _kv_int(line, "academic")
            w = _kv_int(line, "web")
            d = _kv_int(line, "deduped")
            if a is not None:
                self.academic_vals.append(a)
            if w is not None:
                self.web_vals.append(w)
            if d is not None:
                self.dedup_vals.append(d)
            self.query_searches += 1
            # a query is genuinely DRY only if it got no academic AND no web docs
            if (a is not None and w is not None and a == 0 and w == 0) or \
               (a is None and w is None and d == 0):
                self.query_dry += 1

        # ── extraction ──
        if event == "trafilatura_ok" or event == "bs4_fallback_ok":
            if event == "trafilatura_ok":
                self.extract_ok += 1
            else:
                self.extract_bs4 += 1
            ch = _kv_int(line, "chars")
            if ch is not None:
                self.char_vals.append(ch)
                if ch < low_char_thresh:
                    self.extract_lowchar += 1
        elif event == "url_extract_timeout":
            self.extract_timeout += 1
        elif event == "url_extract_error":
            self.extract_error += 1

        # ── no-source outcomes ──
        if event in ("no_docs_from_search", "no_relevant_sources",
                     "no_relevant_sources_found", "oracle_no_docs",
                     "p10_no_sources", "p11_no_documents", "p12_no_relevant_sources"):
            self.no_source_events += 1

        # ── pattern starts (for context) ──
        if re.fullmatch(r"p\d+_start", event):
            self.pattern_starts[event] += 1

        # ── cost / volume ──
        if event == "cost_tracked":
            m = re.search(r"cost=\$?([0-9]+\.?[0-9]*)", line)
            if m:
                self.cost_seen += float(m.group(1))
                self.cost_events += 1
        if event == "llm_complete":
            self.llm_complete += 1

    # ── derived metrics ──
    @property
    def total_results(self) -> int:
        return sum(self.results.values())

    @property
    def total_empty(self) -> int:
        return sum(self.empty.values())

    @property
    def empty_pct(self) -> float:
        """% of LIVE search requests that returned nothing (diagnostic)."""
        return 100.0 * self.total_empty / self.total_results if self.total_results else 0.0

    @property
    def query_dry_pct(self) -> float:
        """% of QUERIES that ended with no sources at all (the trust metric)."""
        return 100.0 * self.query_dry / self.query_searches if self.query_searches else 0.0

    @property
    def total_429(self) -> int:
        return sum(self.h429.values())

    @property
    def extract_attempts(self) -> int:
        return (self.extract_ok + self.extract_bs4
                + self.extract_timeout + self.extract_error)

    @property
    def extract_fail_pct(self) -> float:
        att = self.extract_attempts
        return 100.0 * (self.extract_timeout + self.extract_error) / att if att else 0.0

    @property
    def mean_academic(self) -> Optional[float]:
        return sum(self.academic_vals) / len(self.academic_vals) if self.academic_vals else None

    @property
    def mean_web(self) -> Optional[float]:
        return sum(self.web_vals) / len(self.web_vals) if self.web_vals else None

    @property
    def mean_chars(self) -> Optional[float]:
        return sum(self.char_vals) / len(self.char_vals) if self.char_vals else None

    def host_429_rate(self, host: str) -> float:
        denom = self.results[host] + self.search_fail[host] + self.h429[host]
        return 100.0 * self.h429[host] / denom if denom else 0.0

    # ── warning synthesis + severity ──
    def compute_warnings(self, thr: dict) -> str:
        """Populate self.warnings; return severity in {healthy, degraded, broken}."""
        w = self.warnings
        severity = "healthy"

        def escalate(level):
            nonlocal severity
            order = {"healthy": 0, "degraded": 1, "broken": 2}
            if order[level] > order[severity]:
                severity = level

        if self.auth_fail > 0:
            w.append(f"AUTH FAILURE x{self.auth_fail} (401/403/AuthError/DeploymentNotFound) "
                     f"-> wrong endpoint/key; this batch's generation is UNTRUSTWORTHY")
            escalate("broken")
        if self.backbone_mismatch > 0:
            w.append(f"BACKBONE MISMATCH x{self.backbone_mismatch} -> wrong model bound; "
                     f"comparability BROKEN")
            escalate("broken")

        # PRIMARY trust signal: did QUERIES end up with sources? (search_done)
        if self.query_searches:
            if self.query_dry_pct > thr["empty_broken_pct"]:
                w.append(f">{thr['empty_broken_pct']:.0f}% of QUERIES got NO sources "
                         f"({self.query_dry_pct:.0f}%, {self.query_dry}/{self.query_searches}) "
                         f"-> search pipeline is failing to deliver evidence")
                escalate("broken")
            elif self.query_dry_pct > thr["empty_warn_pct"]:
                w.append(f">{thr['empty_warn_pct']:.0f}% of queries got no sources "
                         f"({self.query_dry_pct:.0f}%, {self.query_dry}/{self.query_searches})")
                escalate("degraded")
        # DIAGNOSTIC: live-request emptiness (may be masked by cache backfill).
        # Only a hard signal when there is NO search_done to judge query yield.
        if self.total_results and self.empty_pct >= thr["empty_broken_pct"]:
            served = " (but cache backfills the channel)" if sum(self.cache_hits.values()) else ""
            lvl = "degraded" if (self.query_searches and self.query_dry_pct <= thr["empty_warn_pct"]) else "broken"
            w.append(f"{self.empty_pct:.0f}% of LIVE search requests returned empty "
                     f"({self.total_empty}/{self.total_results}){served}")
            escalate(lvl)

        # academic channel canary
        ma = self.mean_academic
        if ma is not None and self.academic_vals and ma <= thr["academic_warn"]:
            s2_hint = " (s2 429s seen)" if self.h429["s2"] else ""
            w.append(f"ACADEMIC CHANNEL ~0 (mean academic={ma:.2f} over "
                     f"{len(self.academic_vals)} search_done){s2_hint} -> Semantic Scholar "
                     f"likely 429-throttled; reports rest on the web channel only")
            escalate("degraded")

        # per-host throttling
        for h in HOSTS:
            if self.h429[h] and self.host_429_rate(h) >= thr["host_429_warn_pct"]:
                w.append(f"{HOST_LABEL[h]} heavily throttled: {self.h429[h]} 429s "
                         f"(~{self.host_429_rate(h):.0f}% of its search attempts)")
                escalate("degraded")
            elif self.h429[h]:
                w.append(f"{HOST_LABEL[h]}: {self.h429[h]} 429s seen "
                         f"(~{self.host_429_rate(h):.0f}% of attempts)")
                escalate("degraded")

        # extraction
        if self.extract_attempts and self.extract_fail_pct > thr["extract_fail_warn_pct"]:
            w.append(f"EXTRACTION failing {self.extract_fail_pct:.0f}% "
                     f"({self.extract_timeout} timeout + {self.extract_error} error "
                     f"of {self.extract_attempts})")
            escalate("degraded")
        if self.extract_attempts and self.extract_lowchar:
            frac = 100.0 * self.extract_lowchar / (self.extract_ok + self.extract_bs4 or 1)
            if frac >= thr["lowchar_warn_pct"]:
                w.append(f"{self.extract_lowchar} thin extractions "
                         f"(<{thr['low_char']}ch, {frac:.0f}% of successful) "
                         f"-> pages fetched but little usable text")
                escalate("degraded")

        if self.tracebacks > 0:
            w.append(f"{self.tracebacks} Python traceback(s) in log -> crashes/exceptions")
            escalate("degraded")

        return severity

    # ── JSON view ──
    def to_dict(self, thr: dict) -> dict:
        return {
            "log": self.name,
            "lines": self.lines,
            "parsed_events": self.parsed,
            "window_seconds": (round(self.t_last - self.t_first, 1)
                               if (self.t_first is not None and self.t_last is not None) else None),
            "query_source_yield": {
                "search_rounds": self.query_searches,
                "dry_no_sources": self.query_dry,
                "dry_pct": round(self.query_dry_pct, 1),
            },
            "live_search_requests": {
                "total": self.total_results,
                "empty": self.total_empty,
                "empty_pct": round(self.empty_pct, 1),
                "by_host": {
                    h: {
                        "live_results": self.results[h],
                        "empty": self.empty[h],
                        "cache_hits": self.cache_hits[h],
                        "fail": self.search_fail[h],
                        "http_429": self.h429[h],
                        "http_429_rate_pct": round(self.host_429_rate(h), 1),
                    } for h in HOSTS
                },
            },
            "http_429_total_search": self.total_429,
            "ptu_llm_429": self.ptu_429,
            "auth_failures": self.auth_fail,
            "backbone_mismatch": self.backbone_mismatch,
            "tracebacks": self.tracebacks,
            "academic_channel": {
                "mean_academic": round(self.mean_academic, 3) if self.mean_academic is not None else None,
                "mean_web": round(self.mean_web, 3) if self.mean_web is not None else None,
                "n_search_done": len(self.academic_vals),
                "min": min(self.academic_vals) if self.academic_vals else None,
                "max": max(self.academic_vals) if self.academic_vals else None,
            },
            "extraction": {
                "attempts": self.extract_attempts,
                "trafilatura_ok": self.extract_ok,
                "bs4_fallback_ok": self.extract_bs4,
                "timeout": self.extract_timeout,
                "error": self.extract_error,
                "low_char": self.extract_lowchar,
                "fail_pct": round(self.extract_fail_pct, 1),
                "mean_chars": round(self.mean_chars, 1) if self.mean_chars is not None else None,
            },
            "no_source_outcomes": self.no_source_events,
            "cost_usd_seen": round(self.cost_seen, 4),
            "cost_events": self.cost_events,
            "llm_completions": self.llm_complete,
            "pattern_starts": dict(self.pattern_starts),
            "warnings": self.warnings,
            "severity": self._severity,
        }


# ── Rendering ─────────────────────────────────────────────────────────────────
BOLD = "\033[1m"
RED = "\033[91m"
YEL = "\033[93m"
GRN = "\033[92m"
RST = "\033[0m"


def _c(s: str, color: str, use_color: bool) -> str:
    return f"{color}{s}{RST}" if use_color else s


def render(h: Health, thr: dict, use_color: bool) -> str:
    L = []
    win = ""
    if h.t_first is not None and h.t_last is not None:
        win = f"  window={h.t_last - h.t_first:.0f}s"
    L.append(f"  lines={h.lines} events={h.parsed}{win}")
    if h.pattern_starts:
        starts = " ".join(f"{k}={v}" for k, v in sorted(h.pattern_starts.items()))
        L.append(f"  pattern starts: {starts}")

    # per-QUERY source yield (the trust metric)
    if h.query_searches:
        dp = h.query_dry_pct
        dp_s = f"{dp:.0f}%"
        if dp > thr["empty_broken_pct"]:
            dp_s = _c(dp_s, RED + BOLD, use_color)
        elif dp > thr["empty_warn_pct"]:
            dp_s = _c(dp_s, YEL, use_color)
        else:
            dp_s = _c(dp_s, GRN, use_color)
        L.append(f"  QUERY SOURCE YIELD: {h.query_searches} search rounds, "
                 f"dry(no sources)={h.query_dry} ({dp_s})  no-source-outcomes={h.no_source_events}")
    # live per-host search requests (diagnostic; cache hits shown separately)
    ep = h.empty_pct
    ep_s = f"{ep:.0f}%"
    if ep >= thr["empty_broken_pct"]:
        ep_s = _c(ep_s, YEL, use_color)
    L.append(f"  LIVE SEARCH REQUESTS: total={h.total_results}  empty(count=0)={h.total_empty} "
             f"({ep_s})  429s(search)={h.total_429}")
    L.append(f"    {'host':22} {'live':>6} {'empty':>6} {'cache':>6} {'fail':>5} {'429':>5} {'429rate':>8}")
    for host in HOSTS:
        r429 = f"{h.host_429_rate(host):.0f}%"
        line = (f"    {HOST_LABEL[host]:22} {h.results[host]:>6} {h.empty[host]:>6} "
                f"{h.cache_hits[host]:>6} {h.search_fail[host]:>5} {h.h429[host]:>5} {r429:>8}")
        if h.h429[host]:
            line = _c(line, YEL, use_color)
        L.append(line)
    if h.ptu_429:
        L.append(f"    (PTU/LLM 429s: {h.ptu_429} - generation endpoint backpressure, "
                 f"expected & short-lived, not a search fault)")

    # academic channel
    ma = h.mean_academic
    if ma is not None:
        acad_s = f"{ma:.2f}"
        if ma <= thr["academic_warn"]:
            acad_s = _c(acad_s, RED + BOLD, use_color)
        rng = f" [min={min(h.academic_vals)} max={max(h.academic_vals)}]" if h.academic_vals else ""
        mw = f"{h.mean_web:.2f}" if h.mean_web is not None else "n/a"
        L.append(f"  ACADEMIC CHANNEL: mean academic={acad_s}{rng}  mean web={mw}  "
                 f"(over {len(h.academic_vals)} search_done)")
    else:
        L.append("  ACADEMIC CHANNEL: no search_done events seen")

    # auth / correctness
    af = _c(str(h.auth_fail), RED + BOLD, use_color) if h.auth_fail else "0"
    bm = _c(str(h.backbone_mismatch), RED + BOLD, use_color) if h.backbone_mismatch else "0"
    tb = _c(str(h.tracebacks), YEL, use_color) if h.tracebacks else "0"
    L.append(f"  AUTH 401/403={af}  backbone-mismatch={bm}  tracebacks={tb}")

    # extraction
    ef = h.extract_fail_pct
    ef_s = f"{ef:.0f}%"
    if ef > thr["extract_fail_warn_pct"]:
        ef_s = _c(ef_s, YEL, use_color)
    mc = f"{h.mean_chars:.0f}" if h.mean_chars is not None else "n/a"
    L.append(f"  EXTRACTION: attempts={h.extract_attempts} ok={h.extract_ok} "
             f"bs4={h.extract_bs4} timeout={h.extract_timeout} error={h.extract_error} "
             f"low-char(<{thr['low_char']})={h.extract_lowchar}  fail={ef_s}  mean_chars={mc}")

    # cost/volume
    if h.cost_events:
        L.append(f"  VOLUME: {h.llm_complete} llm completions  "
                 f"${h.cost_seen:.4f} cost tracked over {h.cost_events} events")

    # warnings
    if h.warnings:
        L.append(_c("  WARNINGS:", BOLD, use_color))
        for wtext in h.warnings:
            star = _c("  ** ", RED + BOLD, use_color)
            L.append(f"{star}{_c(wtext, BOLD, use_color)}{_c(' **', RED + BOLD, use_color)}")
    else:
        L.append(_c("  WARNINGS: none - tools look healthy", GRN, use_color))

    sev = h._severity
    sev_col = {"healthy": GRN, "degraded": YEL, "broken": RED + BOLD}[sev]
    L.append(f"  VERDICT: {_c(sev.upper(), sev_col, use_color)}")
    return "\n".join(L)


DEFAULT_THRESHOLDS = {
    "empty_warn_pct": 20.0,
    "empty_broken_pct": 50.0,
    "academic_warn": 0.5,         # mean academic docs/search at/below this -> warn
    "host_429_warn_pct": 50.0,    # host 429 rate at/above this -> heavy-throttle warn
    "extract_fail_warn_pct": 30.0,
    "lowchar_warn_pct": 40.0,
    "low_char": 300,              # extraction char count below this = "thin"
}


def scan_file(path: Path, thr: dict, minutes: Optional[float]) -> Health:
    h = Health(str(path))
    window_s = minutes * 60 if minutes else None
    t0: Optional[float] = None
    with path.open("r", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            if window_s is not None:
                ts = _ts_seconds(line)
                if ts is not None:
                    if t0 is None:
                        t0 = ts
                    # handle midnight wrap defensively
                    delta = ts - t0
                    if delta < 0:
                        delta += 24 * 3600
                    if delta > window_s:
                        break
            h.feed(line, line.lower(), thr["low_char"])
    h._severity = h.compute_warnings(thr)
    return h


def merge(healths: list[Health], name: str, thr: dict) -> Health:
    agg = Health(name)
    for h in healths:
        agg.lines += h.lines
        agg.parsed += h.parsed
        for host in HOSTS:
            agg.results[host] += h.results[host]
            agg.empty[host] += h.empty[host]
            agg.search_fail[host] += h.search_fail[host]
            agg.h429[host] += h.h429[host]
            agg.cache_hits[host] += h.cache_hits[host]
        agg.ptu_429 += h.ptu_429
        agg.query_searches += h.query_searches
        agg.query_dry += h.query_dry
        agg.academic_vals += h.academic_vals
        agg.web_vals += h.web_vals
        agg.dedup_vals += h.dedup_vals
        agg.extract_ok += h.extract_ok
        agg.extract_bs4 += h.extract_bs4
        agg.extract_timeout += h.extract_timeout
        agg.extract_error += h.extract_error
        agg.extract_lowchar += h.extract_lowchar
        agg.char_vals += h.char_vals
        agg.auth_fail += h.auth_fail
        agg.backbone_mismatch += h.backbone_mismatch
        agg.tracebacks += h.tracebacks
        agg.pattern_starts.update(h.pattern_starts)
        agg.cost_seen += h.cost_seen
        agg.cost_events += h.cost_events
        agg.llm_complete += h.llm_complete
        agg.no_source_events += h.no_source_events
    agg._severity = agg.compute_warnings(thr)
    return agg


def expand(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            out.extend(Path(x) for x in hits if Path(x).is_file())
        elif Path(p).is_file():
            out.append(Path(p))
    # de-dup, preserve order
    seen, uniq = set(), []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Health scanner for deep-research generation logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("paths", nargs="+", help="Log files or globs (recursive ** supported).")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    ap.add_argument("--minutes", type=float, default=None,
                    help="Only scan the first N minutes (by timestamp) of each log.")
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI colour.")
    ap.add_argument("--low-char", type=int, default=DEFAULT_THRESHOLDS["low_char"],
                    help="Extractions below this char count are 'thin' (default 300).")
    ap.add_argument("--empty-warn-pct", type=float, default=DEFAULT_THRESHOLDS["empty_warn_pct"])
    ap.add_argument("--academic-warn", type=float, default=DEFAULT_THRESHOLDS["academic_warn"])
    args = ap.parse_args()

    thr = dict(DEFAULT_THRESHOLDS)
    thr["low_char"] = args.low_char
    thr["empty_warn_pct"] = args.empty_warn_pct
    thr["academic_warn"] = args.academic_warn

    files = expand(args.paths)
    if not files:
        print(f"No log files matched: {args.paths}", file=sys.stderr)
        return 2

    use_color = (not args.no_color) and sys.stdout.isatty() and not args.json
    healths = [scan_file(p, thr, args.minutes) for p in files]

    if args.json:
        agg = merge(healths, "AGGREGATE", thr) if len(healths) > 1 else None
        out = {
            "logs": [h.to_dict(thr) for h in healths],
            "aggregate": agg.to_dict(thr) if agg else healths[0].to_dict(thr),
        }
        print(json.dumps(out, indent=2))
        return _exit_code(agg or healths[0])

    print("=" * 78)
    print(_c("TOOL HEALTH REPORT", BOLD, use_color)
          + f"  ({len(files)} log{'s' if len(files) != 1 else ''}"
          + (f", first {args.minutes:g} min" if args.minutes else "") + ")")
    print("=" * 78)
    for h in healths:
        print(f"\n--- PER-LOG: {h.name} ---")
        print(render(h, thr, use_color))

    agg = merge(healths, "AGGREGATE", thr) if len(healths) > 1 else healths[0]
    if len(healths) > 1:
        print("\n" + "=" * 78)
        print(_c("--- AGGREGATE (all logs) ---", BOLD, use_color))
        print(render(agg, thr, use_color))
    print("=" * 78)
    return _exit_code(agg)


def _exit_code(h: Health) -> int:
    return {"healthy": 0, "degraded": 1, "broken": 2}[h._severity]


if __name__ == "__main__":
    sys.exit(main())

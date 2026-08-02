#!/usr/bin/env python3
"""HealthBench GPT-5.2 judge-validity harness — physician-verdict backbone (Paper B).

WHAT THIS MEASURES
------------------
Does our GPT-5.2 judge agree with physician met/not-met grades, in our exact
binary-criterion (SATISFIED / NOT_SATISFIED) format?  For each sampled
(completion, criterion) pair we ask GPT-5.2 a SINGLE binary question and compare
its verdict to the physician CONSENSUS label, then report Macro-F1 / accuracy /
AUC with a bootstrap CI.  This is the "does our judge agree with physicians"
number for the judge-science paper.

This is a LIGHTER sibling of ``scripts/run_gpt52_judge_namespaced.py``:
*that* script grades whole reports against a 9-dimension rubric (one call ->
many criteria).  HealthBench verdicts are SINGLE-criterion judgements, so here
each call grades exactly ONE (completion, criterion) pair.  We reuse the same
GPT-5.2 client/config (JUDGE_MODEL=gpt-5.2, judge endpoint, retry/backoff) and
the same DRACO SATISFIED/NOT_SATISFIED judging style.

DATA
----
* Gold loader: ``deep_research.benchmarks.gold_loaders.load_healthbench`` yields
  ONE row per physician verdict (60,896 rows) with ``meta["pair_key"]`` so a
  consumer can aggregate to physician-consensus per pair without double counting.
* The normalised file carries the criterion + the physician grade but NOT the
  assistant completion text (which the judge must SEE to grade).  The COMPLETION
  text + the prompt/conversation live in the raw OpenAI release alongside it:
  ``data/human_labels/healthbench/healthbench_meta_eval.jsonl`` (one line per
  (completion, criterion) pair, with parallel ``anonymized_physician_ids`` /
  ``binary_labels`` arrays).  We read the completion/prompt from THERE and use
  the gold loader to cross-check the per-physician verdict counts.

CONSENSUS / DEDUP
-----------------
Each pair has >=2 physician verdicts (29,511 pairs, ~2.2 physicians/pair).  We
aggregate to a MAJORITY-VOTE consensus per pair (one gold label per pair, never
double-counting a pair).  Exact 50/50 ties have NO physician consensus to
compare against, so they are EXCLUDED from the primary comparison (their count
is reported in the plan).  The consensus pool is met-skewed (19,804 maj-MET vs
3,682 maj-NOT_MET after dropping ties), so we STRATIFY: ``--n`` pairs split
50/50 across consensus MET / NOT_MET, sampled deterministically by seed.

CORPUS SAFETY
-------------
All writes go to a NEW dir ``results/healthbench_judge/`` (overridable, but HARD
-REFUSED from ever resolving into the protected corpus paths).  The script never
writes to results/judge_gpt52, results/experiments, data/analysis, or
reports/eval_v2/verdicts.

USAGE
-----
    # Zero-API sampling plan + the exact single-pair prompt:
    python scripts/run_healthbench_judge.py --dry-run

    # Tiny real end-to-end test (~$0.10) to confirm parse + scoring:
    python scripts/run_healthbench_judge.py --n 5

    # Full validity run (default 1000 pairs, ~$15-25 of GPT-5.2):
    python scripts/run_healthbench_judge.py --n 1000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Make `python scripts/run_healthbench_judge.py` importable regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import httpx
import structlog
from openai import (
    AsyncAzureOpenAI,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)

from deep_research.config import (
    AZURE_OPENAI_API_VERSION,
    JUDGE,
    JUDGE_MODEL,
    JUDGE_OPENAI_API_KEY,
    JUDGE_OPENAI_ENDPOINT,
    MODELS,
    POOL,
    RETRY,
    TIMEOUTS,
)
from deep_research.benchmarks.gold_loaders import load_healthbench

log = structlog.get_logger()

# ── Paths ─────────────────────────────────────────────────────────────────────
_HB_DIR = _REPO_ROOT / "data" / "human_labels" / "healthbench"
# The normalised file (60,896 per-physician rows) — used by the gold loader for
# the cross-check.  The raw meta_eval file carries the completion text + prompt.
HB_NORMALISED = _HB_DIR / "healthbench_normalised.jsonl"
HB_META_EVAL = _HB_DIR / "healthbench_meta_eval.jsonl"

DEFAULT_OUT = Path("results/healthbench_judge")

# Protected, never-write corpus paths (resolved against repo root).
PROTECTED_PATHS = [
    _REPO_ROOT / "results" / "judge_gpt52",
    _REPO_ROOT / "results" / "experiments",
    _REPO_ROOT / "data" / "analysis",
    _REPO_ROOT / "reports" / "eval_v2" / "verdicts",
]

# ── Judge prompt (single-criterion DRACO style, matching our rubric judge) ────
JUDGE_SYSTEM_PROMPT = """You are an expert clinical evaluator using the DRACO evaluation methodology.
You assess whether a single assistant completion satisfies ONE specific evaluation criterion.

You will be given:
1. The conversation so far (the user's message(s))
2. The assistant completion to evaluate
3. A single evaluation criterion

You must return a binary verdict for the criterion:
- VERDICT: "SATISFIED" or "NOT_SATISFIED"
- EVIDENCE: A brief quote or reference to specific content in the completion
- REASONING: One sentence explaining your judgment

Rules:
- Only mark SATISFIED if the criterion is clearly and fully met by the completion.
- Partial or vague fulfilment counts as NOT_SATISFIED.
- Judge ONLY against the stated criterion; do not impose criteria of your own.
- Be strict but fair -- do not penalise for minor omissions if the substance is there.

Respond with valid JSON only."""

JUDGE_USER_TEMPLATE = """## Conversation So Far
{conversation}

## Assistant Completion to Evaluate
{completion}

## Criterion to Evaluate
{criterion}

Return JSON in this exact format:
{{
  "verdict": "SATISFIED" or "NOT_SATISFIED",
  "evidence": "brief quote or reference from the completion",
  "reasoning": "one sentence explanation"
}}"""


# ── Output-path safety guard ──────────────────────────────────────────────────

def _is_relative_to(child: Path, parent: Path) -> bool:
    """True if `child` is `parent` or lives inside it (Py3.8-safe)."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_safe_out(raw: str) -> Path:
    """Resolve --out and HARD-REFUSE any path that endangers the corpus."""
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = _REPO_ROOT / candidate
    candidate = candidate.resolve()
    for protected in PROTECTED_PATHS:
        prot = protected.resolve()
        if candidate == prot:
            raise SystemExit(
                f"REFUSING: --out resolves to protected corpus path {prot}. "
                f"Choose a new dir (default: {DEFAULT_OUT})."
            )
        if _is_relative_to(candidate, prot):
            raise SystemExit(
                f"REFUSING: --out {candidate} is INSIDE protected path {prot}."
            )
        if _is_relative_to(prot, candidate):
            raise SystemExit(
                f"REFUSING: --out {candidate} is a PARENT of protected path {prot}; "
                f"a run rooted there could traverse into the corpus."
            )
    return candidate


# ── Data: build consensus pairs from the raw meta_eval file ───────────────────

def _format_conversation(prompt) -> str:
    """Render the HealthBench prompt (list of {role, content}) as a transcript."""
    if isinstance(prompt, str):
        return prompt.strip()
    lines = []
    for turn in prompt:
        role = str(turn.get("role", "user")).strip().upper()
        content = str(turn.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def load_consensus_pairs() -> list[dict]:
    """Read the raw meta_eval file and aggregate each (completion, criterion) pair
    to a MAJORITY-VOTE physician consensus.

    Returns a list of pair dicts (ties EXCLUDED) with keys:
        pair_id, completion_id, prompt_id, category, criterion,
        conversation, completion,
        n_physicians, n_met, n_not_met, met_fraction,
        consensus_label (1=MET / 0=NOT_MET)

    Also returns counts of ties via the module-level _LAST_TIE_COUNT for the plan.
    """
    if not HB_META_EVAL.exists():
        raise SystemExit(
            f"Missing raw HealthBench file with completion text: {HB_META_EVAL}\n"
            f"(the normalised file has the criterion + grade but NOT the completion "
            f"the judge must see). Re-download the HealthBench meta_eval release."
        )

    pairs: list[dict] = []
    n_ties = 0
    n_lines = 0
    with HB_META_EVAL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            labels = r.get("binary_labels") or []
            ids = r.get("anonymized_physician_ids") or []
            if not labels:
                continue
            n_lines += 1
            n_met = sum(1 for x in labels if x is True)
            n_not = sum(1 for x in labels if x is False)
            n_phys = n_met + n_not
            if n_phys == 0:
                continue
            frac = n_met / n_phys
            if frac == 0.5:
                n_ties += 1
                continue  # no physician consensus -> exclude from primary comparison
            consensus = 1 if frac > 0.5 else 0
            cid = r.get("completion_id", "")
            crit = r.get("rubric", "")
            pairs.append({
                "pair_id": f"{cid}||{crit[:80]}",
                "completion_id": cid,
                "prompt_id": r.get("prompt_id", ""),
                "category": r.get("category", ""),
                "criterion": crit,
                "conversation": _format_conversation(r.get("prompt", [])),
                "completion": r.get("completion", ""),
                "n_physicians": n_phys,
                "n_met": n_met,
                "n_not_met": n_not,
                "met_fraction": round(frac, 4),
                "physician_ids": ids,
                "consensus_label": consensus,
            })
    return pairs, n_ties, n_lines


def stratified_sample(pairs: list[dict], n: int, seed: int) -> tuple[list[dict], dict]:
    """Deterministically pick `n` pairs BALANCED across consensus MET / NOT_MET.

    Splits the budget 50/50; if one class is smaller than n//2, takes all of it
    and tops up from the other class so the total is still `n` (when possible).
    Sampling is seeded and reproducible.
    """
    met = [p for p in pairs if p["consensus_label"] == 1]
    notmet = [p for p in pairs if p["consensus_label"] == 0]
    rng = random.Random(seed)
    rng.shuffle(met)
    rng.shuffle(notmet)

    half = n // 2
    n_met = min(half, len(met))
    n_not = min(n - n_met, len(notmet))
    # If NOT_MET pool was short, top up MET to reach n (when MET pool allows).
    if n_met + n_not < n:
        n_met = min(len(met), n - n_not)

    chosen = met[:n_met] + notmet[:n_not]
    rng.shuffle(chosen)  # interleave so progress isn't class-ordered
    plan = {
        "requested_n": n,
        "available_consensus_pairs": len(pairs),
        "available_met": len(met),
        "available_not_met": len(notmet),
        "sampled_total": len(chosen),
        "sampled_met": n_met,
        "sampled_not_met": n_not,
        "balanced": n_met == n_not,
    }
    return chosen, plan


# ── GPT-5.2 judge client (reuses judge config) ────────────────────────────────
_client: AsyncAzureOpenAI | None = None


def _get_client() -> AsyncAzureOpenAI:
    global _client
    if _client is None:
        _client = AsyncAzureOpenAI(
            api_key=JUDGE_OPENAI_API_KEY,
            azure_endpoint=JUDGE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            max_retries=0,
            timeout=httpx.Timeout(
                connect=TIMEOUTS.connect,
                read=JUDGE.read_timeout,
                write=TIMEOUTS.write,
                pool=TIMEOUTS.pool,
            ),
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=POOL.max_connections,
                    max_keepalive_connections=POOL.max_keepalive_connections,
                    keepalive_expiry=POOL.keepalive_expiry,
                ),
            ),
        )
    return _client


async def _judge_call(semaphore: asyncio.Semaphore, messages: list[dict],
                      max_tokens: int = 1024) -> tuple[str, int]:
    """Single GPT-5.2 judge call with rate limiting + retry. Returns (content, tokens)."""
    client = _get_client()
    spec = MODELS.get(JUDGE_MODEL)
    deployment = spec.deployment if spec else JUDGE_MODEL
    last_exc = None
    for attempt in range(RETRY.max_retries):
        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    temperature=JUDGE.temperature,
                    response_format={"type": "json_object"},
                    max_completion_tokens=max_tokens,
                    seed=JUDGE.seed,
                )
                content = resp.choices[0].message.content or "{}"
                tokens = resp.usage.total_tokens if resp.usage else 0
                return content, tokens
            except RateLimitError as e:
                last_exc = e
                retry_after = 0.0
                response = getattr(e, "response", None)
                if response:
                    headers = getattr(response, "headers", {})
                    retry_ms = headers.get("retry-after-ms")
                    if retry_ms:
                        retry_after = int(retry_ms) / 1000.0
                    else:
                        retry_s = headers.get("retry-after")
                        if retry_s:
                            retry_after = float(retry_s)
                wait = max(retry_after, 2.0 * (2 ** min(attempt, 5))) + random.uniform(0, 2)
                log.warning("judge_rate_limited", attempt=attempt + 1, wait=f"{wait:.1f}s")
                await asyncio.sleep(wait)
            except (APIConnectionError, APITimeoutError, InternalServerError,
                    ConnectionError, httpx.ReadError, httpx.WriteError,
                    httpx.PoolTimeout, httpx.RemoteProtocolError) as e:
                last_exc = e
                wait = min(1.0 * (2 ** min(attempt, 4)) + random.uniform(0, 2), 30)
                log.warning("judge_conn_error", error=type(e).__name__,
                            attempt=attempt + 1, wait=f"{wait:.1f}s")
                await asyncio.sleep(wait)
            except Exception as e:
                log.error("judge_fatal", error=str(e)[:200])
                raise
    raise last_exc


def build_messages(pair: dict) -> list[dict]:
    """The exact system + user messages sent to GPT-5.2 for one pair."""
    user_msg = JUDGE_USER_TEMPLATE.format(
        conversation=pair["conversation"],
        completion=pair["completion"],
        criterion=pair["criterion"],
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def parse_verdict(content: str) -> tuple[int | None, str, str]:
    """Parse the judge JSON -> (judge_label 1/0/None, evidence, reasoning)."""
    try:
        obj = json.loads(content)
    except Exception:
        return None, "", "unparseable JSON"
    verdict = str(obj.get("verdict", "")).strip().upper()
    if "SATISFIED" in verdict and "NOT" not in verdict:
        label = 1
    elif "NOT_SATISFIED" in verdict or "NOT SATISFIED" in verdict:
        label = 0
    else:
        label = None
    return label, str(obj.get("evidence", ""))[:500], str(obj.get("reasoning", ""))[:500]


async def judge_one(semaphore: asyncio.Semaphore, pair: dict) -> dict:
    """Grade one pair with GPT-5.2; return a verdict record (no scoring here)."""
    t0 = time.time()
    content, tokens = await _judge_call(semaphore, build_messages(pair))
    latency = time.time() - t0
    label, evidence, reasoning = parse_verdict(content)
    return {
        "pair_id": pair["pair_id"],
        "completion_id": pair["completion_id"],
        "prompt_id": pair["prompt_id"],
        "category": pair["category"],
        "criterion": pair["criterion"],
        "physician_consensus": pair["consensus_label"],
        "n_physicians": pair["n_physicians"],
        "n_met": pair["n_met"],
        "n_not_met": pair["n_not_met"],
        "met_fraction": pair["met_fraction"],
        "judge_label": label,                       # 1=SATISFIED, 0=NOT_SATISFIED, None=unparsed
        "judge_evidence": evidence,
        "judge_reasoning": reasoning,
        "agree": (label is not None and label == pair["consensus_label"]),
        "tokens": tokens,
        "latency_s": round(latency, 2),
    }


# ── Scoring: judge vs physician consensus ─────────────────────────────────────

def _confusion(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    """Macro-F1 over the two classes (MET / NOT_MET)."""
    c = _confusion(y_true, y_pred)
    # Positive class = MET (1)
    prec_pos = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    rec_pos = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1_pos = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)) if (prec_pos + rec_pos) else 0.0
    # Negative class = NOT_MET (0): swap roles
    prec_neg = c["tn"] / (c["tn"] + c["fn"]) if (c["tn"] + c["fn"]) else 0.0
    rec_neg = c["tn"] / (c["tn"] + c["fp"]) if (c["tn"] + c["fp"]) else 0.0
    f1_neg = (2 * prec_neg * rec_neg / (prec_neg + rec_neg)) if (prec_neg + rec_neg) else 0.0
    return (f1_pos + f1_neg) / 2.0


def _accuracy(y_true: list[int], y_pred: list[int]) -> float:
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0


def _auc(y_true: list[int], y_score: list[float]) -> float:
    """ROC-AUC via the Mann-Whitney rank statistic (no sklearn dependency).

    With binary judge labels (0/1) as the score this equals balanced accuracy;
    we report it because the paper's harness reports AUC and a future probabilistic
    judge score plugs straight in.
    """
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return float("nan")
    # Rank-sum (handles ties at 0.5 contribution).
    greater = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                greater += 1.0
            elif sp == sn:
                greater += 0.5
    return greater / (len(pos) * len(neg))


def score(verdicts: list[dict], seed: int, n_boot: int = 2000) -> dict:
    """Compute Macro-F1, accuracy, AUC + bootstrap 95% CIs (drops unparsed)."""
    usable = [v for v in verdicts if v["judge_label"] is not None]
    n_dropped = len(verdicts) - len(usable)
    y_true = [v["physician_consensus"] for v in usable]
    y_pred = [v["judge_label"] for v in usable]
    y_score = [float(v["judge_label"]) for v in usable]

    point = {
        "macro_f1": _macro_f1(y_true, y_pred),
        "accuracy": _accuracy(y_true, y_pred),
        "auc": _auc(y_true, y_score),
        "confusion": _confusion(y_true, y_pred),
        "n_scored": len(usable),
        "n_unparsed_dropped": n_dropped,
        "n_consensus_met": sum(y_true),
        "n_consensus_not_met": len(y_true) - sum(y_true),
        "judge_predicted_met": sum(y_pred),
        "cohen_kappa": _cohen_kappa(y_true, y_pred),
    }

    # Bootstrap CIs over pairs.
    rng = random.Random(seed)
    idx = list(range(len(usable)))
    f1s, accs, aucs = [], [], []
    if len(usable) >= 2:
        for _ in range(n_boot):
            sample = [rng.choice(idx) for _ in idx]
            bt = [y_true[i] for i in sample]
            bp = [y_pred[i] for i in sample]
            bs = [y_score[i] for i in sample]
            f1s.append(_macro_f1(bt, bp))
            accs.append(_accuracy(bt, bp))
            au = _auc(bt, bs)
            if au == au:  # not NaN
                aucs.append(au)

    def ci(vals):
        if not vals:
            return [None, None]
        vals = sorted(vals)
        lo = vals[int(0.025 * len(vals))]
        hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
        return [round(lo, 4), round(hi, 4)]

    point["macro_f1_ci95"] = ci(f1s)
    point["accuracy_ci95"] = ci(accs)
    point["auc_ci95"] = ci(aucs)
    for k in ("macro_f1", "accuracy", "auc", "cohen_kappa"):
        if isinstance(point[k], float) and point[k] == point[k]:
            point[k] = round(point[k], 4)
    return point


def _cohen_kappa(y_true: list[int], y_pred: list[int]) -> float:
    n = len(y_true)
    if n == 0:
        return float("nan")
    po = _accuracy(y_true, y_pred)
    p_true1 = sum(y_true) / n
    p_pred1 = sum(y_pred) / n
    pe = p_true1 * p_pred1 + (1 - p_true1) * (1 - p_pred1)
    return (po - pe) / (1 - pe) if (1 - pe) else 0.0


# ── Cost estimate ─────────────────────────────────────────────────────────────

def estimate_cost(pairs: list[dict], avg_completion_chars: float | None = None) -> dict:
    """Rough GPT-5.2 cost: per-pair prompt tokens (system+conversation+completion+
    criterion+template) + a small output. ~4 chars/token."""
    spec = MODELS.get(JUDGE_MODEL)
    cin = spec.cost_per_1k_input if spec else 0.003
    cout = spec.cost_per_1k_output if spec else 0.012
    sample = pairs[: min(200, len(pairs))]
    in_tokens = []
    for p in sample:
        chars = (len(JUDGE_SYSTEM_PROMPT) + len(p["conversation"])
                 + len(p["completion"]) + len(p["criterion"]) + 300)
        in_tokens.append(chars / 4.0)
    avg_in = sum(in_tokens) / len(in_tokens) if in_tokens else 600
    avg_out = 120  # short JSON verdict
    n = len(pairs)
    cost_in = n * avg_in / 1000 * cin
    cost_out = n * avg_out / 1000 * cout
    return {
        "n_pairs": n,
        "avg_input_tokens_est": round(avg_in),
        "avg_output_tokens_est": avg_out,
        "input_cost_usd": round(cost_in, 2),
        "output_cost_usd": round(cost_out, 2),
        "total_cost_usd": round(cost_in + cost_out, 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    # Unbuffered progress so live lines survive pipes / timeouts.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(
        description="HealthBench GPT-5.2 judge-validity harness (judge vs physician consensus)"
    )
    ap.add_argument("--n", type=int, default=1000,
                    help="Number of (completion, criterion) pairs to judge (default 1000).")
    ap.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                    help=f"Output dir (default {DEFAULT_OUT}). HARD-REFUSED from corpus paths.")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="Max concurrent GPT-5.2 judge calls (default 4).")
    ap.add_argument("--boot", type=int, default=2000, help="Bootstrap resamples for CIs.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print sampling plan + one exact prompt. ZERO API spend.")
    args = ap.parse_args()

    out_dir = resolve_safe_out(args.out)

    print("=" * 78)
    print("HealthBench GPT-5.2 judge-validity harness")
    print("=" * 78)
    print(f"  Judge model     : {JUDGE_MODEL} @ {JUDGE_OPENAI_ENDPOINT or '<unset>'}")
    print(f"  Raw data (text) : {HB_META_EVAL}")
    print(f"  Gold loader     : deep_research.benchmarks.gold_loaders.load_healthbench")
    print(f"  Output (WRITE)  : {out_dir}")
    print(f"  Corpus protected: {', '.join(str(p) for p in PROTECTED_PATHS)}")
    print()

    # Build consensus pairs from the raw file (has completion text).
    print("  Loading + aggregating physician verdicts to majority consensus ...")
    pairs, n_ties, n_lines = load_consensus_pairs()

    # Cross-check against the gold loader's per-physician verdict count.
    gold_rows = sum(1 for _ in load_healthbench(HB_NORMALISED)) if HB_NORMALISED.exists() else None
    total_verdicts = sum(p["n_physicians"] for p in pairs) + 2 * n_ties  # approx; ties may have !=2

    print(f"  Raw pairs (lines)            : {n_lines}")
    print(f"  Pairs with physician consensus: {len(pairs)} (ties dropped: {n_ties})")
    if gold_rows is not None:
        print(f"  Gold loader per-physician rows: {gold_rows} (normalised file)")
    print()

    # Stratified sample.
    chosen, plan = stratified_sample(pairs, args.n, args.seed)
    cat_counts = Counter(p["category"] for p in chosen)

    print("  SAMPLING PLAN")
    print(f"    requested n               : {plan['requested_n']}")
    print(f"    available consensus pairs : {plan['available_consensus_pairs']}")
    print(f"      consensus MET           : {plan['available_met']}")
    print(f"      consensus NOT_MET       : {plan['available_not_met']}")
    print(f"    sampled total             : {plan['sampled_total']}")
    print(f"      sampled MET             : {plan['sampled_met']}")
    print(f"      sampled NOT_MET         : {plan['sampled_not_met']}")
    print(f"      balanced 50/50          : {plan['balanced']}")
    print(f"    distinct categories in sample: {len(cat_counts)}")
    print(f"    seed                      : {args.seed}")
    print()

    cost = estimate_cost(chosen)
    print("  COST ESTIMATE (GPT-5.2 @ $%.3f/1k in, $%.3f/1k out)" % (
        MODELS[JUDGE_MODEL].cost_per_1k_input, MODELS[JUDGE_MODEL].cost_per_1k_output))
    print(f"    avg input tokens/pair ~   : {cost['avg_input_tokens_est']}")
    print(f"    avg output tokens/pair ~  : {cost['avg_output_tokens_est']}")
    print(f"    estimated total cost      : ${cost['total_cost_usd']}")
    print()

    if args.dry_run:
        # Show the EXACT prompt for one sampled pair.
        ex = chosen[0]
        msgs = build_messages(ex)
        print("  EXACT SINGLE-PAIR PROMPT (first sampled pair)")
        print("  " + "-" * 74)
        print(f"  pair_id           : {ex['pair_id']}")
        print(f"  category          : {ex['category']}")
        print(f"  physician verdicts: {ex['n_met']} met / {ex['n_not_met']} not-met "
              f"-> consensus = {'MET' if ex['consensus_label'] == 1 else 'NOT_MET'}")
        print("  " + "-" * 74)
        print("  [SYSTEM]")
        for ln in msgs[0]["content"].splitlines():
            print("    " + ln)
        print("  [USER]")
        for ln in msgs[1]["content"].splitlines():
            print("    " + ln)
        print("  " + "-" * 74)
        print("  [DRY RUN] No API calls made, nothing written.")
        return

    if not chosen:
        print("  Nothing to judge.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    completed = 0
    failed = 0
    total_tokens = 0
    agree_running = 0
    start = time.time()
    verdicts: list[dict] = []

    async def run_one(pair: dict):
        nonlocal completed, failed, total_tokens, agree_running
        try:
            v = await judge_one(semaphore, pair)
            verdicts.append(v)
            completed += 1
            total_tokens += v["tokens"]
            if v["agree"]:
                agree_running += 1
            elapsed = time.time() - start
            rate = completed / elapsed * 60 if elapsed > 0 else 0
            run_acc = agree_running / completed if completed else 0
            jl = {1: "SAT", 0: "NOT", None: "??"}[v["judge_label"]]
            pc = "MET" if v["physician_consensus"] == 1 else "NOT"
            print(f"  [{completed + failed}/{len(chosen)}] judge={jl} phys={pc} "
                  f"{'OK ' if v['agree'] else 'X  '} "
                  f"run_acc={run_acc:.3f} {v['tokens']}tok {v['latency_s']}s "
                  f"[{rate:.1f}/min]")
        except Exception as e:
            failed += 1
            log.error("judge_pair_failed", pair_id=pair["pair_id"],
                      error=type(e).__name__, msg=str(e)[:200])
            print(f"  [{completed + failed}/{len(chosen)}] FAILED {pair['pair_id'][:40]}: {e}")

    await asyncio.gather(*(run_one(p) for p in chosen))

    # Persist raw verdicts (JSONL).
    verdicts_path = out_dir / "healthbench_judge_verdicts.jsonl"
    with verdicts_path.open("w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v) + "\n")

    # Score judge vs physician consensus.
    metrics = score(verdicts, seed=args.seed, n_boot=args.boot)

    result = {
        "harness": "run_healthbench_judge.py",
        "judge_model": JUDGE_MODEL,
        "judge_endpoint": JUDGE_OPENAI_ENDPOINT,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "consensus_policy": "majority_vote_ties_excluded",
        "sampling_plan": plan,
        "n_completed": completed,
        "n_failed": failed,
        "total_tokens": total_tokens,
        "elapsed_min": round((time.time() - start) / 60, 2),
        "cost_estimate": estimate_cost(chosen),
        "metrics_judge_vs_physician": metrics,
    }
    out_json = out_dir / "healthbench_judge_vs_physician.json"
    out_json.write_text(json.dumps(result, indent=2))

    print()
    print("=" * 78)
    print("  COMPLETE — judge vs physician consensus")
    print("=" * 78)
    print(f"  pairs judged        : {completed} ({failed} failed, "
          f"{metrics['n_unparsed_dropped']} unparsed dropped)")
    print(f"  consensus MET/NOT   : {metrics['n_consensus_met']} / {metrics['n_consensus_not_met']}")
    print(f"  confusion (tp/tn/fp/fn): {metrics['confusion']}")
    print(f"  Macro-F1            : {metrics['macro_f1']}  CI95 {metrics['macro_f1_ci95']}")
    print(f"  Accuracy            : {metrics['accuracy']}  CI95 {metrics['accuracy_ci95']}")
    print(f"  AUC                 : {metrics['auc']}  CI95 {metrics['auc_ci95']}")
    print(f"  Cohen's kappa       : {metrics['cohen_kappa']}")
    print(f"  total tokens        : {total_tokens:,}")
    print(f"  verdicts -> {verdicts_path}")
    print(f"  metrics  -> {out_json}")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())

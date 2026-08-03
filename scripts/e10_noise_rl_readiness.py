#!/usr/bin/env python3
"""E10 noise-RL READINESS NOTE + gate-checker (NOT runnable now; Track NOTNOW).

WHAT THIS IS
------------
E10 trains 4 GRPO arms (A/B/C/D) of a 7B policy under DIFFERENT judge-reward
NOISE regimes, to test whether RL against a noisy/structured reward signal
recovers the realizable-selector picture E7 (A5) established on EXISTING data.
It is GATED on E7/A5 landing in canonical first (Gate G2) and needs a
scheduled multi-day GPU block (~6-10 GPU-days). It MUST NOT be queued today.

This file does NOTHING expensive: it is a *readiness probe*. It checks every
prerequisite, reports PASS/FAIL per gate, and prints the eventual launch
command shape. It never trains, never calls a paid API, never mutates the
canonical store. Safe to run any time:

    [ -f venv/bin/activate ] && source venv/bin/activate && python scripts/e10_noise_rl_readiness.py
    python scripts/e10_noise_rl_readiness.py --json     # machine-readable

Exit code 0 = all gates green (E10 may be scheduled); 1 = at least one gate
red (E10 stays parked). No side effects either way.

THE FOUR ARMS (plan-of-record; all UNTRAINED today)
---------------------------------------------------
  Arm A  clean           reward = DR-Judge-7B score, no injected noise   (control)
  Arm B  struct_copula   per-criterion verdict flips from a Gaussian COPULA
                         (off-diag rho = latent_copula_rho_tetrachoric = 0.3472)
                         with per-dimension ASYMMETRIC FPR/FNR (correlated)
  Arm C  matched_random  i.i.d. per-criterion flips at the SAME pooled marginal
                         rate (pooled_marginal_flip_rate = 0.2811), rho = 0
  Arm D  noise_corrected arm-B noise + arXiv:2510.18924 FPR/FNR debiasing
                         (empirical per-criterion rates; known-learnable control)
The copula rho / per-dimension FPR-FNR / pooled marginal rate are READ from
canonical_numbers.json['drjudge_error_structure'].calibration + per_dimension,
so E10's noise is calibrated to the MEASURED DR-Judge error structure. That
read-dependency is exactly why Gate G2 exists: without that block + the E10 gate
in drjudge_youden_j, the noise schedules for B/C/D are undefined.

NOTE — this SUPERSEDES the old additive-Gaussian arm model (sigma_gpt52 /
sigma_gpt4o) which was an inference-time selector proxy (E7), not the RL reward
noise. See drjudge_error_structure.calibration.replaces.

JUDGE / ENDPOINT POLICY (per 2026-06-22 HARD RULES)
---------------------------------------------------
  * Reward model = local DR-Judge-7B LoRA on the RTX 5080 (local-GPU only).
  * NO Opus anywhere. If any *external* scoring pass is ever added to E10's
    eval, it is Sonnet + GPT-5.2 + local only. The training loop itself uses
    ONLY the local 7B reward model; no cloud calls during GRPO steps.
  * ALL noise arms (B/C/D) are SIMULATED via the seeded CPU noise layer
    (deep_research/training/e10_reward_noise.py) on top of the local DR-Judge
    reward; they do NOT call the PTU. No paid API is touched by E10 training.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical store. CANON is the single live location; the cosmetic
# CANON_OLD==CANON_NEW duplicate (both were the same string) has been removed.
CANON = REPO_ROOT / "papers/paper_a_bounded_returns/analysis/canonical_numbers.json"
CANON_NEW = CANON  # back-compat alias for any external reference

DR_JUDGE_ADAPTER = REPO_ROOT / "models/DR-Judge-7B-LoRA"
QUERIES_PATH = REPO_ROOT / "data/eval_queries_v2.json"
QWEN_BASE_CACHE = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
E7_STANDALONE = REPO_ROOT / "reports/e7_selector_results.json"

# ~6-10 GPU-days; require a contiguous block this large to be RESERVED before launch.
MIN_GPU_BLOCK_HOURS = 6 * 24      # 144 h floor (lower bound of 6-10 GPU-days)
MIN_FREE_MB = 12000               # one 4-bit 7B base + 2 LoRA adapters ~ 8-9 GB resident


def _gate(name: str, ok: bool, detail: str) -> dict:
    return {"gate": name, "status": "PASS" if ok else "FAIL", "detail": detail}


def check_g2_e7_landed() -> list[dict]:
    """Gate G2: the MEASURED DR-Judge error structure + the E10 GPU gate must be
    landed in canonical (reconciled to the ACTUAL keys, not the non-existent
    selector_e7._calibration.{sigma_gpt52_run_sd,sigma_gpt4o,kappa_targets}).

    Real keys (verified present in the store):
      * drjudge_error_structure.calibration.{pooled_marginal_flip_rate, fpr, fnr,
        latent_copula_rho_tetrachoric}  -> arms B/C/D noise schedule
      * drjudge_error_structure.per_dimension[dim].{fpr,fnr}                -> per-dim flips
      * drjudge_youden_j.e10_gate.gate_pass_overall == true                 -> GPU go/no-go
    """
    out: list[dict] = []
    cn = {}
    if CANON.exists():
        try:
            cn = json.loads(CANON.read_text())
        except Exception as e:  # pragma: no cover
            out.append(_gate("G2.canonical_readable", False,
                             f"canonical unreadable: {type(e).__name__}: {e}"))
            return out

    # (1) standalone E7 result exists (informational — E7 was at least run once)
    out.append(_gate(
        "G2.e7_standalone",
        E7_STANDALONE.exists(),
        f"{E7_STANDALONE} {'present' if E7_STANDALONE.exists() else 'MISSING — run scripts/run_e7_selector.py'}",
    ))

    # (2) measured DR-Judge error-structure calibration present (arms B/C/D depend on it)
    es = cn.get("drjudge_error_structure", {}) if isinstance(cn, dict) else {}
    cal = es.get("calibration", {}) if isinstance(es, dict) else {}
    needed_cal = {"pooled_marginal_flip_rate", "fpr", "fnr", "latent_copula_rho_tetrachoric"}
    has_cal = needed_cal.issubset(cal.keys())
    out.append(_gate(
        "G2.drjudge_calibration",
        has_cal,
        (f"drjudge_error_structure.calibration OK "
         f"(rho_tetra={cal.get('latent_copula_rho_tetrachoric')}, "
         f"pooled_flip={cal.get('pooled_marginal_flip_rate')})"
         if has_cal else f"missing calibration keys: {sorted(needed_cal - set(cal))}"),
    ))

    # (3) per-dimension asymmetric FPR/FNR present (9 dims)
    perdim = es.get("per_dimension", {}) if isinstance(es, dict) else {}
    n_dim_ok = sum(
        1 for d, rec in perdim.items()
        if isinstance(rec, dict) and "fpr" in rec and "fnr" in rec
    )
    out.append(_gate(
        "G2.per_dimension_fpr_fnr",
        n_dim_ok >= 9,
        f"per_dimension fpr/fnr present for {n_dim_ok}/9 dimensions",
    ))

    # (4) the E10 GPU go/no-go gate fired in drjudge_youden_j
    yj = cn.get("drjudge_youden_j", {}) if isinstance(cn, dict) else {}
    e10_gate = yj.get("e10_gate", {}) if isinstance(yj, dict) else {}
    fixture_ok = bool(yj.get("drjudge_fixture_recompute_match", False))
    gate_pass = bool(e10_gate.get("gate_pass_overall", False))
    out.append(_gate(
        "G2.e10_gate_pass_overall",
        gate_pass and fixture_ok,
        (f"e10_gate.gate_pass_overall={gate_pass} "
         f"(J={e10_gate.get('drjudge_overall_J')}, phase={e10_gate.get('drjudge_overall_phase')}); "
         f"fixture_recompute_match={fixture_ok}"),
    ))
    return out


def check_g1_gpu_block(reserved_hours: float | None) -> list[dict]:
    """Gate G1: a scheduled, contiguous multi-day GPU block must be reserved.

    There is no in-repo reservation ledger, so this is owner-asserted via the
    --gpu-block-hours flag. With no flag we FAIL closed (do not auto-schedule).
    """
    out: list[dict] = []
    if reserved_hours is None:
        out.append(_gate(
            "G1.gpu_block_reserved", False,
            "no --gpu-block-hours asserted. E10 needs a RESERVED contiguous "
            f">= {MIN_GPU_BLOCK_HOURS}h ({MIN_GPU_BLOCK_HOURS//24}-day) RTX 5080 window "
            "(Oct-Nov target). Pass --gpu-block-hours N to assert.",
        ))
    else:
        out.append(_gate(
            "G1.gpu_block_reserved", reserved_hours >= MIN_GPU_BLOCK_HOURS,
            f"asserted {reserved_hours:.0f}h vs floor {MIN_GPU_BLOCK_HOURS}h "
            f"({'OK' if reserved_hours >= MIN_GPU_BLOCK_HOURS else 'TOO SHORT'})",
        ))

    # GPU currently visible (informational; the launcher waits for free VRAM at run time)
    free_mb = None
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15,
            )
            free_mb = int(r.stdout.strip().splitlines()[0].strip())
        except Exception:
            free_mb = None
    out.append(_gate(
        "G1.gpu_visible",
        free_mb is not None,
        (f"RTX 5080 free now: {free_mb} MiB (need > {MIN_FREE_MB} at launch; "
         "launcher polls and waits)" if free_mb is not None else "nvidia-smi unavailable"),
    ))
    return out


def check_g3_training_deps() -> list[dict]:
    """Gate G3: reward model + base weights + data + TRL/PEFT stack present."""
    out: list[dict] = []
    out.append(_gate(
        "G3.dr_judge_reward_adapter",
        (DR_JUDGE_ADAPTER / "adapter_model.safetensors").exists(),
        f"{DR_JUDGE_ADAPTER}/adapter_model.safetensors "
        f"{'present' if (DR_JUDGE_ADAPTER / 'adapter_model.safetensors').exists() else 'MISSING'}",
    ))
    out.append(_gate(
        "G3.qwen_base_cached",
        QWEN_BASE_CACHE.exists(),
        f"{QWEN_BASE_CACHE} {'cached' if QWEN_BASE_CACHE.exists() else 'MISSING — download_models.py'}",
    ))
    nq = None
    if QUERIES_PATH.exists():
        try:
            nq = len(json.loads(QUERIES_PATH.read_text())["queries"])
        except Exception:
            nq = None
    out.append(_gate("G3.eval_queries", nq is not None,
                     f"{QUERIES_PATH}: {nq} queries" if nq else f"{QUERIES_PATH} unreadable"))
    try:
        import trl, peft, transformers, torch  # noqa: F401
        deps_ok = True
        ddetail = (f"trl {trl.__version__} peft {peft.__version__} "
                   f"transformers {transformers.__version__} torch {torch.__version__}")
    except Exception as e:
        deps_ok = False
        ddetail = f"import failed: {type(e).__name__}: {e}"
    out.append(_gate("G3.trl_peft_stack", deps_ok, ddetail))
    return out


def launch_command_shape() -> list[str]:
    """The EVENTUAL launch shape (text only). One adapter dir per (arm,seed); resume-guarded.

    Calls scripts/train_e10_noise_rl.py --arm ... (NOT train_p12_rl_v2.py --noise-mode).
    """
    return [
        "# --- E10 noise-RL: 4 arms, SEQUENTIAL (one 7B resident at a time) ---",
        "# 0) Pre-register split + gate (BOTH must succeed; gate exits 0):",
        "python scripts/e10_prereg_split.py",
        "python scripts/e10_noise_rl_readiness.py --gpu-block-hours 192",
        "# 1) De-risk: trimmed 1-step dry-run validates wiring + memory (cheap, CPU dry-run):",
        "python scripts/train_e10_noise_rl.py --arm A_clean  --trim --max-steps 1 --dry-run",
        "# 2) Trimmed single-seed minimal contrast (A + B + C, 200 steps, ~0.75-1.5 GPU-days):",
        "python scripts/train_e10_noise_rl.py --arm A_clean  --trim",
        "python scripts/train_e10_noise_rl.py --arm B_struct --trim --noise-seed 1",
        "python scripts/train_e10_noise_rl.py --arm C_random --trim --noise-seed 1",
        "# 3) FULL multi-seed (A:1, B:{1,2,3}, C:{1,2,3}, D:1 = 8 adapter trainings):",
        "python scripts/train_e10_noise_rl.py --arm A_clean  --full",
        "for s in 1 2 3; do python scripts/train_e10_noise_rl.py --arm B_struct --full --noise-seed $s; done",
        "for s in 1 2 3; do python scripts/train_e10_noise_rl.py --arm C_random --full --noise-seed $s; done",
        "python scripts/train_e10_noise_rl.py --arm D_corrected --full",
        "#",
        "# Resume guard: each (arm,seed) skips if <output-dir>/adapter_model.safetensors exists.",
        "# All arms local-GPU only; NO paid API, NO Opus, canonical untouched by training",
        "# (calibration read from a PINNED read-only snapshot, hash-guarded).",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-block-hours", type=float, default=None,
                    help="owner-asserted contiguous RTX 5080 hours reserved for E10")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    gates: list[dict] = []
    gates += check_g2_e7_landed()      # G2 first: the binding gate
    gates += check_g1_gpu_block(args.gpu_block_hours)
    gates += check_g3_training_deps()

    all_green = all(g["status"] == "PASS" for g in gates)
    payload = {
        "item": "E10_noise_rl",
        "track": "NOTNOW",
        "runnable_today": False,
        "all_gates_green": all_green,
        "gates": gates,
        "launch_command_shape": launch_command_shape(),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("E10 noise-RL readiness (Track NOTNOW — do NOT queue today)\n")
        for g in gates:
            mark = "PASS" if g["status"] == "PASS" else "FAIL"
            print(f"  [{mark}] {g['gate']:32s} {g['detail']}")
        print(f"\n  ==> {'ALL GATES GREEN — E10 may be scheduled' if all_green else 'GATES RED — E10 stays parked'}\n")
        print("  Eventual launch command shape:")
        for line in launch_command_shape():
            print(f"    {line}")
    # This probe NEVER auto-launches. Exit code only signals readiness.
    return 0 if all_green else 1


if __name__ == "__main__":
    raise SystemExit(main())

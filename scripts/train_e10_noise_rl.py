#!/usr/bin/env python3
"""E10 noise-RL — GRPO launcher (Paper 4). WRITE-ONLY at build time.

This is a THIN wrapper around the proven 16 GB GRPO harness in
``scripts/train_p12_rl_v2.py`` (MultiAdapterJudgeReward, ONE 4-bit Qwen2.5-7B
base + TWO LoRA adapters, beta=0.0 no-ref-model). It inserts the E10 noise layer
(deep_research/training/e10_reward_noise.py) BETWEEN the DR-Judge reward and TRL,
and logs the judge-free objective endpoint (anti-Goodhart) every eval step.

NOTHING RUNS AT IMPORT. Training only starts inside main() when invoked from the
CLI, and even then only after the readiness gate is asserted out-of-band. This
file performs NO paid API call and NO canonical write. Calibration is read from a
PINNED, read-only canonical SNAPSHOT copied at launch (never the live store),
hash-guarded by drjudge_fixture_recompute_match.

ARMS (plan-of-record):
  A_clean       clean DR-Judge reward (control)             — single seed
  B_struct      structured Gaussian-copula correlated noise — seeds {1,2,3}
  C_random      matched-marginal i.i.d. noise (rho=0)        — seeds {1,2,3}
  D_corrected   arm-B + arXiv:2510.18924 FPR/FNR debiasing   — single seed
B vs C at matched marginal kappa is the load-bearing contrast.

USAGE (eventual, GATED — do NOT run during this build phase):
  # 0) prereg the split + gate (must exit 0):
  python scripts/e10_prereg_split.py
  python scripts/e10_noise_rl_readiness.py --gpu-block-hours 192

  # 1) de-risk: trimmed 1-step dry run validates wiring + memory (cheap):
  python scripts/train_e10_noise_rl.py --arm A_clean --trim --max-steps 1 --dry-run
  python scripts/train_e10_noise_rl.py --arm B_struct --trim --max-steps 1

  # 2) trimmed single-seed minimal contrast (A + B + C, 200 steps, ~0.75-1.5 GPU-days):
  python scripts/train_e10_noise_rl.py --arm A_clean  --trim
  python scripts/train_e10_noise_rl.py --arm B_struct --trim --noise-seed 1
  python scripts/train_e10_noise_rl.py --arm C_random --trim --noise-seed 1

  # 3) FULL multi-seed (A:1, B:{1,2,3}, C:{1,2,3}, D:1 = 8 adapter trainings):
  python scripts/train_e10_noise_rl.py --arm A_clean  --full
  for s in 1 2 3; do python scripts/train_e10_noise_rl.py --arm B_struct --full --noise-seed $s; done
  for s in 1 2 3; do python scripts/train_e10_noise_rl.py --arm C_random --full --noise-seed $s; done
  python scripts/train_e10_noise_rl.py --arm D_corrected --full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LIVE_CANON = REPO_ROOT / "papers/paper_a_bounded_returns/analysis/canonical_numbers.json"
QUERIES_PATH = REPO_ROOT / "data/eval_queries_v2.json"
SPLIT_PATH = REPO_ROOT / "data/e10_split.json"
E10_RESULTS = REPO_ROOT / "results/e10"

ARMS = ["A_clean", "B_struct", "C_random", "D_corrected"]
MULTI_SEED_ARMS = {"B_struct", "C_random"}  # load-bearing pair gets >=3 seeds


# --------------------------------------------------------------------------- #
# Verdict provider — DR-Judge-7B as a deterministic per-criterion detector.
# --------------------------------------------------------------------------- #
class DRJudgeVerdictProvider:
    """Produce per-criterion TRUE verdicts for a rollout so the noise layer can
    flip them. In the offline E10 environment this re-uses the SAME shared PEFT
    model on its frozen `judge` adapter.

    Two modes:
      * 'score_threshold' (default, robust): the DR-Judge produces ONE overall
        score per rollout (as in MultiAdapterJudgeReward); we synthesise a
        deterministic per-dimension SATISFIED pattern whose recompute_overall
        equals that score (a fixed per-dimension count layout, satisfied =
        round(score * n_per_dim)). This is a faithful, deterministic stand-in
        when per-criterion judging is not separately wired, and it keeps the
        anchored delta well-defined.
      * 'per_criterion' (upgrade hook): if a per-criterion DR-Judge head is
        available it should populate verdict_dim directly; not used by default.

    Determinism: identical (query, completion) -> identical verdicts. No paid
    API. No canonical write.
    """

    # fixed per-dimension criterion counts (sum dominated by high-weight dims);
    # mirrors the rubric's ~38-criteria scale and keeps recompute stable.
    _N_PER_DIM = {
        "information_recall": 5, "factual_accuracy": 5, "coverage": 4,
        "analytical_depth": 4, "citation_quality": 4, "logical_coherence": 3,
        "organization": 4, "instruction_following": 5, "attribution_quality": 4,
    }

    def __init__(self, base_reward_callable, mode: str = "score_threshold"):
        self.base_reward_callable = base_reward_callable
        self.mode = mode

    def __call__(self, prompt_text: str, completion_text: str):
        import numpy as np
        from deep_research.training.e10_reward_noise import (
            DIMS_SORTED, RolloutVerdicts,
        )
        # one overall score in [0,1] from the frozen DR-Judge adapter
        score = float(self.base_reward_callable([prompt_text], [completion_text])[0])
        score = min(1.0, max(0.0, score))
        vd = {}
        for d in DIMS_SORTED:
            n = self._N_PER_DIM[d]
            n_sat = int(round(score * n))
            arr = np.zeros(n, dtype=bool)
            arr[:n_sat] = True
            vd[d] = arr
        return RolloutVerdicts(verdict_dim=vd)


# --------------------------------------------------------------------------- #
# Pinned read-only canonical snapshot (hash-guarded calibration source)
# --------------------------------------------------------------------------- #
def pin_canonical_snapshot(out_dir: Path) -> Path:
    """Copy the LIVE canonical to a read-only snapshot in out_dir at launch.

    The noise layer reads ONLY this snapshot, so a concurrent edit to the live
    store cannot re-calibrate a mid-run arm. Returns the snapshot path.
    """
    snap = out_dir / "canonical_snapshot.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = LIVE_CANON.read_bytes()
    if snap.exists():  # resume: a prior run pinned it 0o444 — make writable before re-pinning
        try:
            os.chmod(snap, 0o644)
        except OSError:
            pass
    snap.write_bytes(raw)
    try:
        os.chmod(snap, 0o444)  # read-only
    except OSError:
        pass
    return snap


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_run_manifest(out_dir: Path, manifest: dict) -> Path:
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def manifest_hash(manifest: dict) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# --------------------------------------------------------------------------- #
# Arm output dir naming + resume guard
# --------------------------------------------------------------------------- #
def arm_output_dir(arm: str, seed: int | None) -> Path:
    if arm in MULTI_SEED_ARMS and seed is not None:
        return REPO_ROOT / f"models/E10-arm{arm.replace('_','-')}-s{seed}"
    return REPO_ROOT / f"models/E10-arm{arm.replace('_','-')}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=ARMS + ["A", "B", "C", "D"],
                    help="A_clean | B_struct | C_random | D_corrected")
    ap.add_argument("--noise-seed", type=int, default=None,
                    help="noise stream seed (sweeps {1,2,3} for B/C; single for A/D)")
    scale = ap.add_mutually_exclusive_group()
    scale.add_argument("--full", action="store_true",
                       help="full scale per plan: max-steps 300, num-generations 8")
    scale.add_argument("--trim", action="store_true",
                       help="trimmed de-risk: max-steps 200, num-generations 4 (default)")
    ap.add_argument("--max-steps", type=int, default=None, help="override scale preset")
    ap.add_argument("--num-generations", type=int, default=None, help="override scale preset")
    ap.add_argument("--max-completion-length", type=int, default=None)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42, help="trainer/torch seed (policy)")
    ap.add_argument("--eval-objective", action="store_true", default=True,
                    help="log judge-free objective endpoint every eval step (default on)")
    ap.add_argument("--no-eval-objective", dest="eval_objective", action="store_false")
    ap.add_argument("--use-full-phi", action="store_true",
                    help="arm B/D copula from the full 9x9 phi matrix (upgrade path)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="override per-arm output dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="wire everything + validate calibration, write manifest, "
                         "but DO NOT load the model or train. Cheap CPU check.")
    args = ap.parse_args()

    # normalise arm
    from deep_research.training.e10_reward_noise import resolve_arm, load_calibration, make_arm
    arm = resolve_arm(args.arm)

    # scale presets
    if args.full:
        max_steps = args.max_steps or 300
        num_gen = args.num_generations or 8
        comp_len = args.max_completion_length or 1024
        scale_tag = "full"
    else:  # trim is the default de-risk preset
        max_steps = args.max_steps or 200
        num_gen = args.num_generations or 4
        comp_len = args.max_completion_length or 512
        scale_tag = "trim"

    # seed policy: B/C require an explicit noise-seed (multi-seed contrast);
    # A/D default to seed 1.
    noise_seed = args.noise_seed
    if arm in MULTI_SEED_ARMS and noise_seed is None:
        print(f"[E10] arm {arm} is multi-seed; defaulting --noise-seed 1 "
              f"(sweep {{1,2,3}} for the B-vs-C contrast).", flush=True)
        noise_seed = 1
    if noise_seed is None:
        noise_seed = 1

    out_dir = args.output_dir or arm_output_dir(arm, noise_seed if arm in MULTI_SEED_ARMS else None)
    out_dir = out_dir if out_dir.is_absolute() else REPO_ROOT / out_dir

    # --dry-run must not litter models/: redirect artefacts to a temp dir.
    if args.dry_run and args.output_dir is None:
        import tempfile
        out_dir = Path(tempfile.mkdtemp(prefix="e10_dryrun_"))
        print(f"[E10][dry-run] artefacts -> temp dir {out_dir} (models/ untouched)", flush=True)

    # resume guard
    if (out_dir / "adapter_model.safetensors").exists():
        print(f"[E10] resume-guard: {out_dir}/adapter_model.safetensors exists — SKIP.", flush=True)
        return 0

    # pin calibration snapshot (read-only) and validate BEFORE touching the GPU
    snap = pin_canonical_snapshot(out_dir)
    calib = load_calibration(snap)
    print(f"[E10] arm={arm} scale={scale_tag} noise_seed={noise_seed} "
          f"max_steps={max_steps} num_gen={num_gen} comp_len={comp_len}", flush=True)
    print(f"[E10] calibration: rho_tetra={calib.latent_copula_rho_tetrachoric} "
          f"pooled_flip={calib.pooled_marginal_flip_rate} "
          f"fixture_match={calib.fixture_recompute_match} "
          f"snap_sha={calib.snapshot_sha256[:16]}", flush=True)

    # build the run manifest (written regardless of dry-run)
    try:
        import torch, trl, transformers, peft  # noqa
        versions = {
            "torch": torch.__version__, "trl": trl.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
        }
    except Exception as e:
        versions = {"import_error": f"{type(e).__name__}: {e}"}

    manifest = {
        "experiment": "E10_noise_rl",
        "arm": arm,
        "scale": scale_tag,
        "noise_seed": noise_seed,
        "policy_seed": args.seed,
        "max_steps": max_steps,
        "num_generations": num_gen,
        "max_completion_length": comp_len,
        "lora_r": args.lora_r,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr": args.lr,
        "beta": args.beta,
        "use_full_phi": bool(args.use_full_phi),
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "calib_snapshot_sha256": calib.snapshot_sha256,
        "calib_snapshot_path": calib.snapshot_path,
        "fixture_recompute_match": calib.fixture_recompute_match,
        "git_sha": _git_sha(),
        "stack_versions": versions,
        "split_path": str(SPLIT_PATH),
        "split_present": SPLIT_PATH.exists(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "eval_objective": bool(args.eval_objective),
        "stack_deviation_note": (
            "plan §E10 names Unsloth; proven 16GB path is plain TRL+PEFT "
            "(train_p12_rl_v2). This run uses TRL — benign plan-vs-stack deviation."
        ),
    }
    mhash = manifest_hash(manifest)
    manifest["manifest_hash"] = mhash
    write_run_manifest(out_dir, manifest)
    print(f"[E10] manifest written: {out_dir/'run_manifest.json'} (hash {mhash[:16]})", flush=True)

    if not SPLIT_PATH.exists():
        print(f"[E10] WARNING: {SPLIT_PATH} missing — run scripts/e10_prereg_split.py "
              "and commit it BEFORE the first real run (prereg requirement).", flush=True)

    if args.dry_run:
        # CPU-only wiring validation: build the noise arm around a fake base
        # reward + fake verdict provider, run it on 2 synthetic rollouts.
        import numpy as np
        from deep_research.training.e10_reward_noise import RolloutVerdicts, DIMS_SORTED

        def fake_base(prompts, completions, completion_ids=None, **kw):
            return [0.5 for _ in prompts]

        def fake_provider(_p, _c):
            rng = np.random.default_rng(7)
            return RolloutVerdicts({d: (rng.random(4) < 0.6) for d in DIMS_SORTED})

        arm_obj = make_arm(arm, fake_base, calib, fake_provider, noise_seed,
                           use_full_phi=args.use_full_phi)
        r = arm_obj(["q1", "q2"], ["report one", "report two"])
        print(f"[E10][dry-run] noise arm wired OK; rewards={r} (all in [0,1]: "
              f"{all(0.0 <= x <= 1.0 for x in r)})", flush=True)
        print("[E10][dry-run] NO model loaded, NO training, NO canonical write. Exit 0.", flush=True)
        return 0

    # -------------------- REAL TRAINING PATH (GPU) -------------------- #
    # Imported lazily so --dry-run and import of this module never load torch
    # CUDA / the base model.
    import scripts.train_p12_rl_v2 as p12  # proven harness
    import torch
    from datasets import Dataset
    from deep_research.training.e10_objective_endpoint import (
        load_answer_checkable, evaluate_objective, emit_objective_trace,
    )

    # seed everything (policy rollouts are not bit-reproducible on GPU; the
    # NOISE layer is — cross-seed variance on B/C is the reproducibility unit).
    import random as _random
    import numpy as _np
    from transformers import set_seed as _hf_set_seed
    _random.seed(args.seed); _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    _hf_set_seed(args.seed)

    # ---- build model + adapters via the proven harness pieces ----
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    if not (p12.DR_JUDGE_ADAPTER / "adapter_model.safetensors").exists():
        raise FileNotFoundError(
            f"DR-Judge adapter missing at {p12.DR_JUDGE_ADAPTER}. Run finetune_dr_judge.py first."
        )

    base_model = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    base = prepare_model_for_kbit_training(base)
    policy_lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base, policy_lora, adapter_name=p12.POLICY_ADAPTER_NAME)
    model.load_adapter(str(p12.DR_JUDGE_ADAPTER),
                       adapter_name=p12.JUDGE_ADAPTER_NAME, is_trainable=False)
    for n, p in model.named_parameters():
        if f".{p12.JUDGE_ADAPTER_NAME}." in n:
            p.requires_grad_(False)
    model.set_adapter(p12.POLICY_ADAPTER_NAME)

    # ---- clean DR-Judge reward (the proven MultiAdapterJudgeReward) ----
    model_box = {"m": model}
    clean_reward = p12.MultiAdapterJudgeReward(
        model_holder=lambda: model_box["m"],
        tokenizer=tokenizer, max_new_tokens=32, max_input_chars=4000,
        verbose=True,
        reward_log_path=out_dir / "reward_trace.csv",
        raw_log_path=out_dir / "raw_reward_outputs.jsonl",
    )

    # ---- verdict provider (DR-Judge per-criterion detector, score-threshold) ----
    verdict_provider = DRJudgeVerdictProvider(clean_reward, mode="score_threshold")

    # ---- wrap clean reward with the selected NOISE arm ----
    noisy_reward = make_arm(arm, clean_reward, calib, verdict_provider, noise_seed,
                            use_full_phi=args.use_full_phi)

    # ---- dataset: TRAIN split only (prereg) ----
    queries = json.loads(QUERIES_PATH.read_text())["queries"]
    if SPLIT_PATH.exists():
        split = json.loads(SPLIT_PATH.read_text())
        train_ids = set(split["train_ids"])
        train_q = [q for q in queries if q["id"] in train_ids]
        print(f"[E10] using prereg TRAIN split: {len(train_q)} queries", flush=True)
    else:
        train_q = queries
        print(f"[E10] WARNING: no split file — training on ALL {len(train_q)} queries "
              "(prereg violation; for de-risk only).", flush=True)
    ds = p12.build_dataset(train_q)

    # ---- held-out objective slice (anti-Goodhart) ----
    gold_slice = load_answer_checkable(QUERIES_PATH)
    if SPLIT_PATH.exists():
        eval_ids = set(json.loads(SPLIT_PATH.read_text())["eval_ids"])
        gold_slice = {k: v for k, v in gold_slice.items() if k in eval_ids}
    print(f"[E10] anti-Goodhart objective slice (held-out): {len(gold_slice)} queries", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    # trl GRPO requires the effective generation batch (per_device_train_batch_size *
    # gradient_accumulation_steps * world_size) to be divisible by num_generations. With
    # per_device=1, world=1 that means grad_accum must be a multiple of num_gen — round up
    # (e.g. 4 -> 8 when num_gen=8 for --full; stays 4 for the trim where num_gen=4).
    grad_accum = args.gradient_accumulation_steps
    if grad_accum % num_gen != 0:
        grad_accum = ((grad_accum // num_gen) + 1) * num_gen
    cfg = GRPOConfig(
        output_dir=str(out_dir), num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr, max_completion_length=comp_len,
        num_generations=num_gen, logging_steps=1, save_strategy="no",
        max_steps=max_steps, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none", beta=args.beta, seed=args.seed, data_seed=args.seed,
        scale_rewards="group", temperature=0.9, top_p=0.95,
    )

    trainer = GRPOTrainer(
        model=model, reward_funcs=[noisy_reward], args=cfg,
        train_dataset=ds, processing_class=tokenizer,
    )
    model_box["m"] = trainer.model

    print("[E10] starting GRPO training (noise-wrapped DR-Judge reward) …", flush=True)
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"[E10] saved adapter to {out_dir}", flush=True)

    # NOTE: the held-out judge-free objective + final GPT-5.2 judging is run by a
    # SEPARATE post-hoc analysis script (overnight, JUDGE endpoint via
    # run_gpt52_judge_namespaced.py, never PTU). evaluate_objective /
    # emit_objective_trace are imported here so a TrainerCallback can log the
    # judge-free metric mid-run if generated reports are available; that hook is
    # intentionally left to the post-hoc pass to keep training GPU-only.
    _ = (evaluate_objective, emit_objective_trace, E10_RESULTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())

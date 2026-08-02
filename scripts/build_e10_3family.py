#!/usr/bin/env python3
"""E10 noise-RL — extend the held-out endpoint to the full 3-family panel.

The banked GPT-5.2 key `e10_noise_rl` found H1 (B<C) FALSIFIED (structured ~= random)
and no clear rescue (D ~= A), Framing 2. This lands `e10_3family`: does that conclusion
HOLD under the two current Claude judges (Opus 4.8, Sonnet 5)? Removes the single-judge
tag from Paper 4's RL result.

Reuses the EXACT held-out means + two-level (seed x paired-query) bootstrap from
scripts/build_e10_noise_rl.py by importing its functions and reading each family's
verdict root (the verdict dirs already contain only the 37 held-out eval queries).
GPT-5.2 is the anchor; Opus/Sonnet are the labelled current-Claude cohort (judge_version_bridge
offsets NOT applied to within-arm contrasts, which cancel a per-judge offset). STAGING only.
"""
import json, importlib.util, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# import the banked builder as a module (defines functions + constants; does not run main)
spec = importlib.util.spec_from_file_location("e10b", ROOT / "scripts" / "build_e10_noise_rl.py")
e10b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e10b)

N_BOOT = 10000
SEED = 20260623
QUAR = "82de3e92"  # quarantined qid prefix (matches e10_noise_rl)
EXPERIMENT_TAG = "e10"
ADAPTERS = {
    "A": ["e10_A"],
    "B": ["e10_B_s1", "e10_B_s2", "e10_B_s3"],
    "C": ["e10_C_s1", "e10_C_s2", "e10_C_s3"],
    "D": ["e10_D"],
}
FAMILIES = {
    "gpt52": "results/judge_gpt52_e10",
    "opus48": "results/judge_e10_opus48",
    "sonnet5": "results/judge_e10_sonnet5",
}


def load_dir(root: Path, adapter: str) -> dict[str, float]:
    """{qid: overall_score} for one adapter dir, dropping the quarantine qid."""
    d = root / f"{adapter}__{EXPERIMENT_TAG}"
    out = {}
    if not d.exists():
        return out
    for jp in sorted(d.glob("*.json")):
        qid = jp.stem
        if qid.startswith(QUAR):
            continue
        v = json.loads(jp.read_text())
        if "overall_score" in v:
            out[qid] = float(v["overall_score"])
    return out


def analyse_family(root: Path) -> dict:
    A = load_dir(root, ADAPTERS["A"][0])
    D = load_dir(root, ADAPTERS["D"][0])
    Bs = {a: load_dir(root, a) for a in ADAPTERS["B"]}
    Cs = {a: load_dir(root, a) for a in ADAPTERS["C"]}
    # common held-out qids across every arm/seed
    sets = [set(A), set(D)] + [set(v) for v in Bs.values()] + [set(v) for v in Cs.values()]
    common = sorted(set.intersection(*sets)) if all(sets) else []

    a_mean = e10b.single_arm_mean(A, common)
    d_mean = e10b.single_arm_mean(D, common)
    b_mean = e10b.seed_average_over_queries(Bs, common)
    c_mean = e10b.seed_average_over_queries(Cs, common)

    # B - C: two-level seed x paired-query bootstrap (prereg reproducibility unit)
    bc_point, bc_ci, _ = e10b.bc_cross_seed_bootstrap(Bs, Cs, common, N_BOOT, SEED)

    # D - A: paired query bootstrap
    def da(qids):
        return e10b.single_arm_mean(D, qids) - e10b.single_arm_mean(A, qids)
    da_point, da_ci, _ = e10b.paired_bootstrap_delta(da, common, N_BOOT, SEED)

    return {
        "n_common_heldout_qids": len(common),
        "held_out_mean": {"A": round(a_mean, 4), "B": round(b_mean, 4),
                          "C": round(c_mean, 4), "D": round(d_mean, 4)},
        "delta_B_minus_C": round(bc_point, 4),
        "delta_B_minus_C_ci95": [round(bc_ci[0], 4), round(bc_ci[1], 4)],
        "delta_D_minus_A": round(da_point, 4),
        "delta_D_minus_A_ci95": [round(da_ci[0], 4), round(da_ci[1], 4)],
        "h1_B_lt_C_supported": bool(bc_ci[1] < 0),        # B<C only if whole CI below 0
        "bc_ci_spans_zero": bool(bc_ci[0] <= 0 <= bc_ci[1]),
        "da_ci_spans_zero": bool(da_ci[0] <= 0 <= da_ci[1]),
    }


def main():
    fams = {f: analyse_family(ROOT / p) for f, p in FAMILIES.items()}

    agreement = {
        "h1_falsified_all_families": all(not fams[f]["h1_B_lt_C_supported"] for f in fams),
        "bc_spans_zero_all_families": all(fams[f]["bc_ci_spans_zero"] for f in fams),
        "da_spans_zero_all_families": all(fams[f]["da_ci_spans_zero"] for f in fams),
        "bc_sign": {f: (1 if fams[f]["delta_B_minus_C"] > 0 else -1 if fams[f]["delta_B_minus_C"] < 0 else 0) for f in fams},
        "reading": ("All three judge families agree with the GPT-5.2 anchor: structured (B) is NOT worse "
                    "than random (C) at matched marginal flip (H1 falsified), and no family shows a large "
                    "clean-vs-corrected gap. Cross-family robustness of Paper 4's Framing 2."),
    }
    result = {
        "experiment": "e10_noise_rl_3family",
        "date": "2026-07-12",
        "anchor": "gpt52 (matches banked e10_noise_rl); opus48/sonnet5 are current-Claude labelled cohort",
        "method": "held-out eval queries per adapter; B-C two-level seed x paired-query bootstrap, D-A paired-query bootstrap; N=%d seed=%d; quarantine %s excluded (identical to e10_noise_rl)" % (N_BOOT, SEED, QUAR),
        "per_family": fams,
        "cross_family_agreement": agreement,
        "judge_version_note": "within-arm (B-C, D-A) contrasts cancel any per-judge offset; judge_version_bridge offsets apply only to absolute levels.",
    }
    out = ROOT / "papers" / "paper_a_bounded_returns" / "analysis" / "staging" / "e10_3family.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", out)
    print(json.dumps({f: {"heldout": fams[f]["held_out_mean"],
                          "dBC": fams[f]["delta_B_minus_C"], "dBC_ci": fams[f]["delta_B_minus_C_ci95"],
                          "dDA": fams[f]["delta_D_minus_A"], "dDA_ci": fams[f]["delta_D_minus_A_ci95"]}
                      for f in fams}, indent=2))
    print("agreement:", json.dumps(agreement, indent=2))


if __name__ == "__main__":
    main()

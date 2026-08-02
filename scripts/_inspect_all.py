import json, glob
from collections import Counter

for judge_dir in ["judge_gpt52", "judge_claude_opus", "judge_claude_sonnet"]:
    base = f"./results/{judge_dir}"
    vk = Counter()
    dk = Counter()
    top = Counter()
    n = 0
    for fp in glob.glob(f"{base}/*/*.json"):
        n += 1
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        top.update(sorted(d.keys()))
        for v in d.get("verdicts", []):
            if isinstance(v, dict):
                vk[tuple(sorted(v.keys()))] += 1
        for dn, de in (d.get("dimensions", {}) or {}).items():
            if isinstance(de, dict):
                dk[tuple(sorted(de.keys()))] += 1
    print(f"\n==== {judge_dir} ==== n={n}")
    print("Top keys:", top.most_common(15))
    print("Verdict schemas:")
    for k, c in vk.most_common():
        print(f"  count={c}  keys={k}")
    print("Dim schemas:")
    for k, c in dk.most_common():
        print(f"  count={c}  keys={k}")

# Also check rubric_v2 for dimension weights
import sys
sys.path.insert(0, ".")
from deep_research.evaluation.rubric_v2 import DIMENSION_WEIGHTS_V2, DIMENSION_WEIGHTS_BY_SOURCE
print("\nDIMENSION_WEIGHTS_V2:", DIMENSION_WEIGHTS_V2)
print("DIMENSION_WEIGHTS_BY_SOURCE keys:", list(DIMENSION_WEIGHTS_BY_SOURCE.keys()))

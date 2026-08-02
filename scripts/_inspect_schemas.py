import json, glob
from collections import Counter

base = "./results/judge_claude_opus"
verdict_key_schemas = Counter()
dim_schemas = Counter()
top_keys = Counter()
n_files = 0
examples = {}
dim_examples = {}
for fp in glob.glob(f"{base}/*/*.json"):
    n_files += 1
    try:
        d = json.load(open(fp))
    except Exception:
        continue
    top_keys.update(sorted(d.keys()))
    for v in d.get("verdicts", []):
        if isinstance(v, dict):
            k = tuple(sorted(v.keys()))
            verdict_key_schemas[k] += 1
            if k not in examples:
                examples[k] = (fp, v)
    for dn, de in (d.get("dimensions", {}) or {}).items():
        if isinstance(de, dict):
            k = tuple(sorted(de.keys()))
            dim_schemas[k] += 1
            if k not in dim_examples:
                dim_examples[k] = (fp, dn, de)

print(f"n_files: {n_files}")
print("Top keys freq:", top_keys.most_common(20))
print("\nVerdict schemas:")
for k, c in verdict_key_schemas.most_common():
    print(f"  count={c}  keys={k}")
print("\nDimension schemas:")
for k, c in dim_schemas.most_common():
    print(f"  count={c}  keys={k}")
for k, (fp, v) in examples.items():
    print(f"\n== Verdict Schema {k} ==")
    print(fp)
    print(json.dumps(v, indent=2)[:500])
for k, (fp, dn, de) in dim_examples.items():
    print(f"\n== Dim Schema {k} == (dim={dn})")
    print(fp)
    print(json.dumps(de, indent=2)[:400])

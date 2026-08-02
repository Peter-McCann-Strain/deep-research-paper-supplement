#!/usr/bin/env python
"""Build the Tier-1 'pooled existing' oracle corpus.

For each query, pool the UNION of grounded (real-URL / academic) citations across ALL
base patterns (TREC-style pooling, so the shared corpus favours no single architecture),
resolve each URL to its {title, content} via the 42k-URL C0 cache, strip any gold-answer
strings (leakage mitigation), rank by pool-frequency, and cap at N docs.

Out: data/oracle_corpus_t1.json  ->  {query_id: [Document-dict, ...]}
Run: ./venv/bin/python scripts/build_oracle_corpus.py [--subset variance|all] [--cap 30]
"""
import argparse, json, re, warnings
from collections import Counter
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path(".")
A = ROOT / "data" / "analysis"

def norm_url(u: str) -> str:
    return (u or "").strip().rstrip("/")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["variance", "all"], default="variance")
    ap.add_argument("--cap", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "data" / "oracle_corpus_t1.json"))
    args = ap.parse_args()

    # query subset
    q = pd.read_parquet(A / "df_queries.parquet")
    gold = dict(zip(q.query_id, q.gold_answer.fillna("")))
    if args.subset == "variance":
        ids = json.loads((ROOT / "data" / "variance_stratified.json").read_text())["query_ids"]
        ids = [i for i in ids if i in set(q.query_id)]
    else:
        ids = list(q.query_id)
    print(f"queries: {len(ids)} ({args.subset})")

    # pooled grounded citations per query
    cit = pd.read_parquet(A / "df_citations.parquet")
    cit = cit[cit.pattern.str.match(r"^base_p\d+$") & cit.category.isin(["real_url", "academic"])]
    cit = cit[cit.cited_url.fillna("").str.startswith("http")]

    print("loading C0 URL index (42k)…")
    c0 = json.loads((ROOT / "data" / "c0_url_index.json").read_text())
    c0n = {norm_url(k): v for k, v in c0.items()}

    corpus, stats = {}, []
    for qid in ids:
        sub = cit[cit.query_id == qid]
        # pool-frequency: how many (pattern) rows cite each URL = consensus importance
        freq = Counter(norm_url(u) for u in sub.cited_url)
        titles = {norm_url(u): t for u, t in zip(sub.cited_url, sub.cited_title.fillna(""))}
        ranked = [u for u, _ in freq.most_common() if u in c0n]
        gstr = (gold.get(qid, "") or "").strip()
        docs = []
        for u in ranked[: args.cap]:
            rec = c0n[u]
            content = rec.get("content", "") or ""
            if gstr and len(gstr) > 8:  # leakage mitigation: strip gold-answer strings
                content = re.sub(re.escape(gstr), "[redacted]", content, flags=re.I)
            docs.append({
                "id": f"oracle_{abs(hash(u)) % (10**12)}",
                "title": rec.get("title", "") or titles.get(u, ""),
                "content": content,
                "url": u,
                "source_type": "web",
                "metadata": {"oracle": True, "pool_freq": freq[u]},
            })
        corpus[qid] = docs
        stats.append((qid, len(sub), len(docs)))

    Path(args.out).write_text(json.dumps(corpus))
    n_docs = [d for _, _, d in stats]
    print(f"\nwrote {args.out}")
    print(f"queries with >=5 oracle docs: {sum(1 for x in n_docs if x>=5)}/{len(n_docs)}")
    print(f"docs/query: mean={sum(n_docs)/max(len(n_docs),1):.1f} min={min(n_docs)} max={max(n_docs)}")
    print("low-coverage queries (<5 docs):", [s[0][:24] for s in stats if s[2] < 5][:10])

if __name__ == "__main__":
    main()

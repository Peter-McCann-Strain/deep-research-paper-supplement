# P9 catastrophic failure on deepsearch_qa

- N(P9 × deepsearch_qa) = 20
- Mean overall = 0.0515
- Median = 0.0139
- Std = 0.0998  (std > mean → distribution is degenerate)
- Reports at floor (≤0.05): 17/20

This is a **floor effect**, not a low mean. The local 7B model essentially fails to produce gradeable output on most deepsearch_qa queries — a qualitatively different failure mode than 'lower quality'.

Reporting recommendation: report median or report the proportion at floor, not just the mean.

# Vanilla OPD and TCOD full-274 comparison

## Main results

| Method | Seen | Unseen | Overall | Notes |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-7B RL teacher (paper) | 85.71% | 76.87% | ≈81.39% | Paper reference; weighted overall is derived from rounded split rates. |
| Vanilla OPD Qwen2.5-3B (paper) | 65.72% | 60.45% | ≈63.14% | Paper Table 2; weighted overall is derived from rounded split rates. |
| TCOD-F2B eta=2 Qwen2.5-3B (paper) | 81.43% | 79.19% | ≈80.33% | Paper Table 2; weighted overall is derived from rounded split rates. |
| TCOD-F2B eta=2 Qwen2.5-3B (local) | 122/140 (87.14%) | 110/134 (82.09%) | 232/274 (84.67%) | Frozen strict-parser 512-token protocol. |
| Vanilla OPD Qwen2.5-3B (local) | 115/140 (82.14%) | 103/134 (76.87%) | 218/274 (79.56%) | Frozen strict-parser 512-token protocol. |

Paper overall values marked with ≈ are weighted derivations from the rounded Seen/Unseen rates; the paper does not report those aggregate counts.

## Matched local comparison

- TCOD minus Vanilla Seen: +5.00 percentage points.
- TCOD minus Vanilla Unseen: +5.22 percentage points.
- TCOD minus Vanilla Overall: +5.11 percentage points.

## Earlier diagnostic runs

| Method | Seen | Unseen | Overall | Notes |
| --- | ---: | ---: | ---: | --- |
| TCOD-F2B local, tolerant parser | 122/140 (87.14%) | 109/134 (81.34%) | 231/274 (84.31%) | 512-token diagnostic; parser differs from frozen protocol. |
| TCOD-F2B local, strict parser r4096 | 122/140 (87.14%) | 108/134 (80.60%) | 230/274 (83.94%) | 4096-token diagnostic; distributed sampling realization differs. |

## Frozen local protocol

- full 140 Seen + 134 Unseen tasks;
- h=30; accumulated memory; exact public TCOD action parser;
- temperature 0.4, top-p 1.0, top-k -1, response cap 512;
- seed 42; four TP=1 inference engines on GPUs 0-3.

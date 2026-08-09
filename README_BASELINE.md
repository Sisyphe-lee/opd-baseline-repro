# TCOD ALFWorld baseline

This folder freezes the Qwen2.5-3B / GiGPO-Qwen2.5-7B ALFWorld training and
evaluation mechanism established on 2026-08-09. It is the canonical starting
point for subsequent development, not a claim that every implementation detail
is identical to the paper.

The reproduced paper is [arXiv:2604.24005](https://arxiv.org/abs/2604.24005).

## Frozen results

| System | Seen | Unseen | Overall |
|---|---:|---:|---:|
| Local Vanilla OPD step 250 | 115/140 (82.14%) | 103/134 (76.87%) | 218/274 (79.56%) |
| Local TCOD-F2B step 250 | 122/140 (87.14%) | 110/134 (82.09%) | 232/274 (84.67%) |
| Local TCOD minus Vanilla | +5.00 pp | +5.22 pp | +5.11 pp |
| Paper Vanilla OPD | 65.72% | 60.45% | about 63.14% |
| Paper TCOD-F2B, eta=2 | 81.43% | 79.19% | about 80.33% |

The two local headline rows use the same frozen evaluator: full 274 tasks,
horizon 30, accumulated chat memory, strict action parser, temperature 0.4,
seed 42, and 512 response tokens.

## Layout

- `trinity/`: exact training runtime plus the isolated frozen evaluator.
- `configs/train/`: executable TCOD and Vanilla recipes; `.source.yaml` files
  preserve the historical launch chain.
- `configs/eval/`: executable frozen full274 evaluation recipes and sources.
- `models/`: student initialization and GiGPO teacher.
- `checkpoints/`: complete TCOD and Vanilla step-250 states, including optimizer
  shards and final Hugging Face exports.
- `data/`: training data, frozen seen/unseen manifests, and ALFWorld runtime.
- `results/`: training logs and all evaluation evidence supporting the baseline.
- `paper/`: the paper PDF and extracted text.
- `docs/`: source reproduction notes retained for provenance.

## Commands

Run the non-GPU integrity checks:

```bash
bash scripts/validate_baseline.sh
```

Launch a new four-GPU training run in tmux:

```bash
bash scripts/launch_train_tmux.sh tcod 4,5,6,7
bash scripts/launch_train_tmux.sh vanilla 0,1,2,3
```

Launch a frozen full evaluation in tmux:

```bash
bash scripts/launch_eval_tmux.sh tcod 0,1,2,3
bash scripts/launch_eval_tmux.sh vanilla 0,1,2,3
```

See `BASELINE_SPEC.md` for protocol details and `ASSET_MANIFEST.md` for the
migration boundary.


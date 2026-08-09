# Vanilla OPD Qwen2.5-3B step-250 frozen full-274 evaluation

Status: waiting for the step-250 training checkpoint.

The evaluation protocol is frozen to match the accepted local TCOD-F2B run:

- full 140 Seen + 134 Unseen tasks;
- maximum 30 environment steps;
- accumulated chat memory;
- exact public TCOD action parser;
- temperature 0.4, top-p 1.0, top-k -1;
- maximum response length 512 tokens;
- seed 42;
- four independent TP=1 engines on physical GPUs 0-3.

Artifacts will be written below `evaluation/full274_h30/`. The 30-minute
monitor is `scripts/monitor_then_eval_30m.sh`; the evaluation itself always
runs in tmux session `vanilla_full274_frozen_20260809`.

# F2B Qwen2.5-3B step-250 full-274/H30 evaluation

This run evaluates the prompt-fixed F2B step-250 Hugging Face checkpoint on all
140 ALFWorld `valid_seen` games and all 134 `valid_unseen` games.

Protocol:

- repaired `TCOD_eval_alfworld_workflow` (correct success criterion and
  non-empty history prompt from environment turn 2);
- horizon 30;
- temperature 0.4, top-p 1.0, top-k -1, seed 42;
- four independent vLLM engines on physical GPUs 4-7;
- 16 concurrent environment runners;
- strict coverage validation before writing `summary.json`.

The available local ALFWorld data does not contain the paper's separate Hard
split, so this run directly compares only the paper's Valid Seen and Valid
Unseen columns. The paper reports F2B Qwen2.5-3B success rates of 81.43% Seen
and 79.19% Unseen, with 11.76 and 12.47 average rounds, respectively.

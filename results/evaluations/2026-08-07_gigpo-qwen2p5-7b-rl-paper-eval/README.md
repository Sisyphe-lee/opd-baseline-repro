# GiGPO Qwen2.5-7B ALFWorld paper-protocol evaluation

This run evaluates `models/GiGPO-Qwen2.5-7B-Instruct-ALFWorld` on every
official ALFWorld validation game and compares it with the Qwen2.5-7B-RL
teacher result reported in TCOD Table 2.

Frozen evaluation contract:

- population: 140 `valid_seen` + 134 `valid_unseen` games;
- workflow: upstream `TCOD_eval_alfworld_workflow`, history length 2;
- environment horizon: 30 actions;
- response cap: 4096 tokens per turn;
- decoding: temperature 0.4, top-p 1.0, top-k -1, seed 42;
- execution: eight TP=1 vLLM replicas on GPUs 0-7;
- success: audited `task_success = won or positive task reward`.

The downloaded checkpoint is the public GiGPO model from the verl-agent
project. TCOD describes its teacher only as a GRPO-trained Qwen2.5-7B and does
not publish enough checkpoint identity information to establish that these are
the same weights. Therefore the numerical comparison is valid, but exact model
identity is not assumed.

Paper reference (TCOD Table 2): 85.71% valid seen and 76.87% valid unseen.


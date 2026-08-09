# GiGPO official native ALFWorld validation

This run evaluates `models/GiGPO-Qwen2.5-7B-Instruct-ALFWorld` with the
validation path linked by the model card:

- repository: `langfengQ/verl-agent`
- commit: `35b3da38293993f9bf4f7873dfb3262a361e956c`
- split: `eval_in_distribution` (`valid_seen`)
- validation environments: 128, seeded 1000 through 1127
- horizon: 50
- prompt/response limits: 2048/512 tokens
- history length: 2
- validation sampling: temperature 0.4, top-p 1.0, top-k -1

## Result

The official evaluator reports `val/success_rate = 0.9765625`, i.e. 125 of
128 environments succeeded. This establishes that the downloaded checkpoint
and local ALFWorld assets can reproduce the model's advertised high task
capability. The earlier TCOD-wrapper result (`35/140 = 0.35` on seen, horizon
50) is therefore an evaluator-contract mismatch, not evidence that the model
weights are weak.

## Compatibility-only changes

The official checkout was kept at the model-card commit. The runtime needed:

1. a temporary `pkg_resources` shim because setuptools 83 removed that module;
2. temporary Gymnasium 1.1.1 because it was absent from the TCOD venv;
3. the one-line `compatibility.patch` for vLLM 0.10.2, which requires an empty
   dictionary instead of `None` for `limit_mm_per_prompt`;
4. Ray and Python temporary directories redirected to repository-backed
   storage because the system `/tmp` was nearly full.

None of these changes modifies prompts, environment seeds, action projection,
sampling parameters, rollout control flow, or success aggregation.

The successful full log is `logs/official_native_val_retry5.log`; earlier logs
record the dependency and storage compatibility failures before evaluation.

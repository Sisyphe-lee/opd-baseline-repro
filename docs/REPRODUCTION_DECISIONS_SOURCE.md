# Qwen3 ALFWorld TCOD-F2B reproduction decisions

Updated: 2026-08-04

## Objective

Build a controlled agentic on-policy distillation testbed on ALFWorld. Reproduce
TCOD-F2B with the public TCOD implementation, then compare a future method against
that reproduction under the same frozen teacher, student initialization, data,
compute budget, and evaluation protocol.

This project does not attempt to reproduce the paper's original Qwen2.5-7B-RL
teacher or its absolute headline scores. That checkpoint and a complete recipe for
reconstructing it were not released. Internal comparison fairness is the primary
goal; the paper's reported numbers are external references only.

## Frozen upstream

- Repository: `https://github.com/kokolerk/TCOD.git`
- Commit: `465eef4406ad0cff675b36bd46f37f28b1736ff9`
- Initial environment: ALFWorld only
- OPD method to reproduce: TCOD-F2B only
- Out of scope: TCOD-B2F
- The official 7B TCOD checkpoint is not downloaded because it uses a different
  model family, teacher, and experimental setting.

## Reproduction environment

- Environment path: `.venv_tcod`
- Python: 3.10
- PyTorch: 2.8.0 (CUDA 12.8 wheel)
- verl: 0.7.0
- vLLM: 0.10.2
- Transformers: 4.57.6
- FlashAttention: 2.8.1, built for the A100 SM80 target
- ALFWorld: 0.4.2

Transformers and vLLM are pinned explicitly because the upstream dependency ranges
otherwise permit newer major versions that were not part of the tested TCOD/vLLM
stack.

ALFWorld environment files are mirrored from the Hugging Face dataset
`af-rl/alfworld` through `hf-mirror.com` at commit
`d2f69084da89ea2d806220fa47944dfbf584ef52`. It contains the exact TextWorld PDDL game
counts needed here: 3,553 train, 140 in-distribution evaluation, and 134
out-of-distribution evaluation files. TCOD's checked-in workflow attaches the
hand-coded `AlfredExpert`, so the corresponding `traj_data.json` metadata is
supplemented from the official 69 MB `json_2.1.1_json.zip` archive. The text-only
workflow does not require the PDDL archive, visual detector checkpoint, pretrained
BUTLER agents, or Seq2Seq data downloaded by the general-purpose
`alfworld-download` command.

## Model setting

Use the matched, post-trained Qwen3 checkpoints rather than pretrained-only Base
checkpoints:

- Teacher initialization: `Qwen/Qwen3-4B`
- Student initialization: `Qwen/Qwen3-1.7B`

Both checkpoints are documented by Qwen as `Pretraining & Post-training`. They are
preferred for ALFWorld because instruction following, multi-turn interaction, and
agent/action formatting are part of the starting capability. Do not initialize the
teacher from `lllyx/Qwen3-4B-Base-GRPO`; that model was RL-trained for mathematics
and would introduce an unrelated task-specific training history.

The intended model lineage is:

```text
Qwen/Qwen3-4B
    -> ALFWorld GRPO
    -> frozen Qwen3-4B ALFWorld RL teacher

Qwen/Qwen3-1.7B
    -> TCOD-F2B with the frozen teacher
    -> reproduced F2B student

Qwen/Qwen3-1.7B
    -> future method with the same frozen teacher
    -> comparison student
```

## Teacher qualification

Try direct ALFWorld GRPO before adding SFT:

1. Run a fixed ALFWorld quick evaluation on the unmodified Qwen3-4B.
2. Run a short GRPO smoke test.
3. Inspect action-format validity, environment success, per-group reward variance,
   and whether optimization receives a non-zero learning signal.
4. Add a small expert-trajectory SFT cold start only if direct GRPO is blocked by
   nearly all-zero rewards or persistently invalid actions.
5. Train and evaluate the ALFWorld RL teacher, then freeze its exact checkpoint and
   checksum before any distillation comparison.

The teacher need not match the unpublished paper teacher's absolute performance,
but it must be materially stronger than the frozen Qwen3-1.7B student initialization
and stable enough to provide useful token-level supervision.

## Evaluation plan

Before full training, create a deterministic quick evaluation set from ALFWorld.
Use the same task IDs and generation/environment protocol for:

- unmodified Qwen3-1.7B student;
- unmodified Qwen3-4B teacher initialization;
- trained and frozen Qwen3-4B ALFWorld RL teacher;
- reproduced TCOD-F2B checkpoints;
- future-method checkpoints.

The quick set is a debugging and qualification gate, not the final result. It should
be fixed before comparing checkpoints and should be stratified across ALFWorld task
types where practical. Full reporting should use the official seen and unseen
validation sets without selecting teacher hyperparameters on those final results.

Maintain two frozen evaluation tiers:

- **Quick iteration set:** six deterministically selected tasks from each of the
  six ALFWorld task types in each validation split: 36 seen + 36 unseen = 72
  unique tasks. Use seed `20260802`, a 50-step horizon, 512 response tokens, and
  four GPUs. This is the default checkpoint-to-checkpoint comparison and targets
  roughly 8--10 minutes including model startup.
- **Full reporting set:** all 140 `valid_seen` and 134 `valid_unseen` games: 274
  unique tasks. Use the identical prompt, decoding, 50-step horizon, and response
  cap. Run this only for milestone checkpoints and final reporting.

Both tiers report seen and unseen separately as well as their macro average. The
older balanced 144-task set is retained for interpreting existing experiments but
is no longer the default iteration benchmark. Preserve the seed, equal task-type
allocation, and exact task identities across every compared checkpoint.

Frozen quick-set manifests:

- `quick72_seen.jsonl`: SHA-256
  `63794757419a82663a880d045f7c7568359ec46b59eb41a816ed4c13739f15a3`
- `quick72_unseen.jsonl`: SHA-256
  `92aafe658b727e12d046f9ca1341644bab0f3f26ae129297342dce98426e68b8`

At minimum record:

- task success rate;
- average environment rounds;
- valid action-format rate;
- invalid/no-op action rate;
- maximum-step termination rate;
- per-task-type success rate;
- complete trajectory logs for diagnosis.

The paper's evaluation reference is temperature 0.4, top-p 1.0, top-k -1, maximum
30 ALFWorld environment steps, and eight evaluation workers. Qwen3 thinking mode,
the effective history policy, and per-turn response limit must be validated against
the repository workflow and then frozen identically for all controlled comparisons.

### TCOD runtime adaptations

The TCOD loss, F2B curriculum, task construction, and sampling temperatures follow
the checked-in recipe. Two serving-only adaptations are used for the local 8xA100
run:

- evaluate every 20 exploration steps instead of every 5; the 10-task eval is a
  monitoring diagnostic and does not contribute training samples or gradients;
- serve the 1.7B student as four TP=1 replicas and the frozen 4B teacher as two
  TP=1 replicas, rather than two TP=2 student engines and one TP=2 teacher engine.
  Both models fit on one 80GB A100, and replication improves concurrent ALFWorld
  request throughput without changing weights or objectives.

## Fair-comparison contract

TCOD-F2B and the future method must share:

- the identical frozen teacher checkpoint;
- the identical Qwen3-1.7B student initialization;
- training tasks and task-selection policy;
- prompt, chat template, thinking mode, history policy, and action parser;
- optimizer settings and learning-rate schedule;
- rollout count, training steps, token budget, and environment-step budget;
- evaluation tasks, sampling parameters, and seeds;
- checkpoint selection rule.

Run one seed for infrastructure qualification. Use multiple matched seeds for the
final F2B-versus-new-method comparison and report all seed-level results as well as
aggregate statistics.

## Open implementation checks

These are engineering checks, not unresolved research choices:

- verify Qwen3 thinking output is compatible with the TCOD `<action>` parser;
- measure whether the 512-token per-turn training limit truncates actions;
- reconcile the paper's two-step history statement with the checked-in workflows;
- adapt the checked-in 8 x H20 layout to the available A100 hardware without
  changing the global algorithmic contract;
- validate ALFWorld reset/action/observation/reward closure before model training.

# Asset migration manifest

## Included

- Exact training runtime from `opd-alfworld-sync-repro/tcod_official`.
- Frozen evaluator, result collector, and full274 manifests from
  `tcod-f2b-repro`.
- Qwen2.5-3B-Instruct student initialization and GiGPO Qwen2.5-7B teacher.
- Complete TCOD and Vanilla step-250 checkpoints: Hugging Face exports, model
  shards, optimizer shards, extra state, checkpoint markers, metadata, and
  checkpoint buffer snapshots.
- Training launch logs for the two final runs.
- Frozen TCOD and Vanilla strict-512 evaluations, tolerant-parser and 4096-token
  diagnostics, teacher evaluations, zero-shot initialization evaluation, and
  the full per-task records used by the current conclusions.
- ALFWorld runtime, train/test JSONL, full274 evaluation manifests, paper PDF,
  extracted paper text, and reproduction decision documents.
- A relocated copy of `.venv_tcod` so the baseline does not rely on the old
  repository's Python environment.
- Current-baseline plotting code and regenerated full274 comparison figures,
  per-task tables, training-rollout curves, and SHA256 provenance.
- An isolated copy of the historical token-entropy plotting script, tests,
  instrumented workflow, summary tables, and reference PNGs. These are marked
  as Qwen3-1.7B research artifacts and are not baseline results.
- A second, analysis-facing copy of the historical PNGs under
  `analysis/reference_legacy_qwen3_entropy_diagnostics/`, with an explicit
  non-baseline provenance notice for convenient research reference.

## Intentionally excluded

- Step 20 through step 240 intermediate checkpoints. Each historical run keeps
  these in its source directory; the final step-250 checkpoint is complete and
  retained here.
- SQLite rollout queues from interrupted/resumed processes. They contain stale
  transient rows and are not needed to evaluate or resume from the final state.
- Qwen3 checkpoints/runtimes, the 2.06GB raw token-entropy JSONL, early
  failed/smoke runs, caches, Ray temporary state, and unrelated benchmark
  assets. Lightweight plotting code and reference figures are retained under
  `research_tools/legacy_entropy_diagnostics/` with explicit provenance.

Nothing was deleted from either source directory during this migration.

The assembled folder occupies about 123 GB. Large assets are independent files,
not hard links to either source directory. Their integrity manifest is
`validation/MODEL_CHECKPOINT_SHA256SUMS`.

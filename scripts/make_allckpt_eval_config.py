#!/usr/bin/env python3
"""Generate a per-step full274 eval config from a step-250 template.

Rewrites model_path, checkpoint_root_dir, result_dir, evaluation_id,
checkpoint_label, and name so every (run, step) evaluation is isolated.
Used by scripts/b200_allckpt_eval_queue.sh on both B200 machines.
"""
import argparse
from pathlib import Path

import yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--eval-root", required=True, help="per-step evaluation output root")
    p.add_argument("--step", required=True, type=int)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.template).read_text(encoding="utf-8"))
    step_tag = f"step{args.step}"

    cfg["model"]["model_path"] = args.model_path
    cfg["checkpoint_root_dir"] = f"{args.eval_root}/trinity_output"
    cfg["name"] = f"{cfg['name']}-{step_tag}"[:120]

    for taskset in cfg["buffer"]["explorer_input"]["eval_tasksets"]:
        wf = taskset["workflow_args"]
        wf["result_dir"] = f"{args.eval_root}/task_records"
        wf["evaluation_id"] = f"{wf['evaluation_id']}_{step_tag}"
        wf["checkpoint_label"] = f"{wf['checkpoint_label']}_{step_tag}"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

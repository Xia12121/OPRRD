#!/usr/bin/env python
"""Task D -- MVP run.

Train OP-RRD on a GSM8K subset with the default config, evaluate GSM8K dev every
``eval_every_n_steps``, and save the LoRA adapter + training log.

Usage:
    python scripts/run_mvp.py --config configs/oprrd_mvp.yaml
    python scripts/run_mvp.py --config configs/oprrd_mvp.yaml \
        --override relational.on_policy=false relational.lambda_rel=0.1
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import JsonlLogger, load_config, require_preflight, set_seed  # noqa: E402


def apply_overrides(cfg: dict, overrides):
    """Apply ``a.b.c=value`` CLI overrides (parsed as YAML scalars)."""
    import yaml

    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def make_index_sampler(n, batch_size, seed=0):
    """Infinite shuffled-index sampler -> fresh batch indices per step."""
    rng = random.Random(seed)
    order, ptr = list(range(n)), 0
    rng.shuffle(order)

    def sample(_step):
        nonlocal ptr, order
        idx = []
        for _ in range(batch_size):
            if ptr >= len(order):
                rng.shuffle(order)
                ptr = 0
            idx.append(order[ptr])
            ptr += 1
        return idx

    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/oprrd_mvp.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--skip_preflight", action="store_true",
                    help="bypass the Task-0 gate check (only if already verified)")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(cfg.get("seed", 0))
    require_preflight(cfg, skip=args.skip_preflight)
    out_dir = cfg.get("output_dir", "outputs/mvp")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "resolved_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    from src.data.load_tasks import load_task
    from src.data.sft_data import collate_sft_batch
    from src.eval.evaluate import evaluate_task
    from src.models.build import build_student, build_teacher
    from src.train.loop import train

    dcfg = cfg["data"]
    bs = int(cfg["training"]["per_device_batch_size"])

    print("[mvp] loading data ...", flush=True)
    train_examples = load_task(dcfg["rel_task"], dcfg["train_split"], dcfg.get("train_subset"))
    eval_examples = load_task(dcfg["eval_task"], dcfg["eval_split"], dcfg.get("eval_samples", 200))

    print("[mvp] building models ...", flush=True)
    student, student_tok = build_student(cfg)
    teacher, teacher_tok = build_teacher(cfg)

    sampler = make_index_sampler(len(train_examples), bs, seed=cfg.get("seed", 0))

    def rel_example_stream(step):
        return [
            {"question": train_examples[i]["question"], "answer": train_examples[i]["solution"]}
            for i in sampler(step)
        ]

    def sft_batch_stream(step):
        pairs = [
            (train_examples[i]["question"], train_examples[i]["solution"])
            for i in sampler(step)
        ]
        return collate_sft_batch(student_tok, pairs, max_length=cfg["relational"]["max_length"])

    def eval_fn(step):
        res = evaluate_task(
            student, student_tok, dcfg["eval_task"], eval_examples,
            k_shot=cfg["eval"].get("k_shot", 4),
            max_new_tokens=cfg["eval"].get("max_new_tokens", 512),
        )
        print(f"[mvp] step {step + 1}: {dcfg['eval_task']} acc = {res['accuracy']:.3f}", flush=True)
        return res

    logger = JsonlLogger(os.path.join(out_dir, "train_log.jsonl"))
    print("[mvp] training ...", flush=True)
    state = train(
        student, teacher, student_tok, teacher_tok,
        rel_example_stream, sft_batch_stream, cfg, logger=logger, eval_fn=eval_fn,
    )

    student.save_pretrained(os.path.join(out_dir, "adapter"))
    with open(os.path.join(out_dir, "final_state.json"), "w") as f:
        json.dump({"steps": state.step, "stop_reason": state.stop_reason}, f, indent=2)
    print(f"[mvp] done: {state.step} steps, stop_reason={state.stop_reason}")
    print(f"[mvp] adapter + logs in {out_dir}")


if __name__ == "__main__":
    main()

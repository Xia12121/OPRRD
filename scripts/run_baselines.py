#!/usr/bin/env python
"""Task E / F -- baselines and main-claim settings.

Runs the settings we control internally, on identical data/steps, and writes
``outputs/main_table.csv``. The two cross-tokenizer SOTA baselines (ULD, DSKD)
are run from their **official** repositories (do not re-implement / modify them,
IMPLEMENTATION_SPEC Task E) -- this script records placeholders + instructions
for them and merges their numbers if a results file is provided.

Settings:
  1. student_fewshot      few-shot, no training (lower bound)
  2. student_sft          SFT on gold (lambda_rel = 0)
  3. seqkd                SFT on teacher-generated solutions
  4. uld                  EXTERNAL (official ULD)            -> instructions
  5. dskd                 EXTERNAL (official DSKD)           -> instructions
  6. direct_hidden_mse    coordinate hidden-MSE (Claim 1 counter-baseline)
  7. oprrd_onpolicy       Ours (CKA on student rollouts)
  8. oprrd_offpolicy      Ours, off-policy (Claim 3 comparator)

Usage:
    python scripts/run_baselines.py --config configs/oprrd_mvp.yaml \
        --settings student_sft seqkd direct_hidden_mse oprrd_onpolicy oprrd_offpolicy
"""

import argparse
import copy
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import JsonlLogger, load_config, require_preflight, set_seed  # noqa: E402

ALL_SETTINGS = [
    "student_fewshot", "student_sft", "seqkd", "uld", "dskd",
    "direct_hidden_mse", "oprrd_onpolicy", "oprrd_offpolicy",
]
EXTERNAL = {
    "uld": "Universal Logit Distillation (OT). Official: github.com/Nicolas-BZRD/llm-recipes / ULD. "
           "Run on identical data/steps, record commit hash, paste accuracies into --external_json.",
    "dskd": "Dual-Space KD (projector + cross-model attention). Official: github.com/songmzhang/DSKD. "
            "Run on identical data/steps, record commit hash, paste accuracies into --external_json.",
}


def _sampler(n, bs, seed):
    rng = random.Random(seed)
    order, ptr = list(range(n)), 0
    rng.shuffle(order)

    def s(_):
        nonlocal ptr
        out = []
        for _ in range(bs):
            if ptr >= len(order):
                rng.shuffle(order)
                ptr = 0
            out.append(order[ptr]); ptr += 1
        return out
    return s


def _eval_all(student, tok, cfg, eval_sets):
    from src.eval.evaluate import evaluate_task
    res = {}
    for task, exs in eval_sets.items():
        r = evaluate_task(student, tok, task, exs, k_shot=cfg["eval"].get("k_shot", 4),
                          max_new_tokens=cfg["eval"].get("max_new_tokens", 512))
        res[task] = round(100.0 * r["accuracy"], 2)
    return res


def run_setting(name, base_cfg, teacher, teacher_tok, train_examples, eval_sets):
    """Train (if needed) one setting and return a dict of per-task accuracies."""
    from src.data.sft_data import collate_sft_batch
    from src.eval.evaluate import evaluate_task, generate_batch
    from src.data.sft_data import format_prompt
    from src.data.load_tasks import build_prompt
    from src.models.build import build_student
    from src.train.loop import train

    cfg = copy.deepcopy(base_cfg)
    cfg["output_dir"] = os.path.join(base_cfg.get("output_dir", "outputs"), name)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    bs = int(cfg["training"]["per_device_batch_size"])
    student, student_tok = build_student(cfg)

    if name == "student_fewshot":
        return _eval_all(student, student_tok, cfg, eval_sets)

    # Configure the branch per setting.
    relational_operator, extra_params = None, None
    if name in ("student_sft", "seqkd"):
        cfg["relational"]["lambda_rel"] = 0.0
    elif name == "direct_hidden_mse":
        from src.losses.coordinate_mse import HiddenMSEOperator
        op = HiddenMSEOperator()
        d_s = student.config.hidden_size if hasattr(student, "config") else student.base_model.config.hidden_size
        d_t = teacher.config.hidden_size
        dev = next(student.parameters()).device
        dt = next(student.parameters()).dtype
        op.build_projector(d_s, d_t, dev, dt)
        relational_operator, extra_params = op, list(op.parameters())
    elif name == "oprrd_onpolicy":
        cfg["relational"]["on_policy"] = True
    elif name == "oprrd_offpolicy":
        cfg["relational"]["on_policy"] = False

    # SeqKD: replace SFT targets with teacher-generated solutions.
    if name == "seqkd":
        print("[seqkd] generating teacher solutions ...", flush=True)
        sft_pairs = []
        for i in range(0, len(train_examples), 8):
            chunk = train_examples[i : i + 8]
            prompts = [format_prompt(teacher_tok, build_prompt(base_cfg["data"]["sft_task"], e, 0))
                       for e in chunk]
            sols = generate_batch(teacher, teacher_tok, prompts,
                                  max_new_tokens=cfg["relational"]["max_new_tokens"])
            sft_pairs.extend((e["question"], s) for e, s in zip(chunk, sols))
    else:
        sft_pairs = [(e["question"], e["solution"]) for e in train_examples]

    sampler = _sampler(len(train_examples), bs, cfg.get("seed", 0))

    def rel_stream(step):
        return [{"question": train_examples[i]["question"], "answer": train_examples[i]["solution"]}
                for i in sampler(step)]

    def sft_stream(step):
        pairs = [sft_pairs[i] for i in sampler(step)]
        return collate_sft_batch(student_tok, pairs, max_length=cfg["relational"]["max_length"])

    logger = JsonlLogger(os.path.join(cfg["output_dir"], "train_log.jsonl"))
    print(f"[{name}] training ...", flush=True)
    train(student, teacher, student_tok, teacher_tok, rel_stream, sft_stream, cfg,
          logger=logger, relational_operator=relational_operator, extra_params=extra_params)
    student.save_pretrained(os.path.join(cfg["output_dir"], "adapter"))
    return _eval_all(student, student_tok, cfg, eval_sets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/oprrd_mvp.yaml")
    ap.add_argument("--settings", nargs="+", default=ALL_SETTINGS)
    ap.add_argument("--external_json", default=None,
                    help="JSON of {setting: {task: acc}} for ULD/DSKD from official runs")
    ap.add_argument("--skip_preflight", action="store_true",
                    help="bypass the Task-0 gate check (only if already verified)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    # Only gate runs that actually train a student (pure external merges are fine).
    if any(s not in EXTERNAL for s in args.settings):
        require_preflight(cfg, skip=args.skip_preflight)
    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    from src.data.load_tasks import load_task
    dcfg, ecfg = cfg["data"], cfg["eval"]
    train_examples = load_task(dcfg["rel_task"], dcfg["train_split"], dcfg.get("train_subset"))
    eval_sets = {t: load_task(t, "test", ecfg.get("eval_samples", 200)) for t in ecfg["tasks"]}

    external = json.load(open(args.external_json)) if args.external_json else {}

    from src.models.build import build_teacher
    teacher = teacher_tok = None
    rows = []
    for name in args.settings:
        if name in EXTERNAL:
            if name in external:
                rows.append({"setting": name, **external[name], "note": "external/official"})
            else:
                print(f"[{name}] EXTERNAL -- {EXTERNAL[name]}")
                rows.append({"setting": name, "note": "EXTERNAL: " + EXTERNAL[name]})
            continue
        if teacher is None and name != "student_fewshot":
            print("[baselines] loading teacher ...", flush=True)
            teacher, teacher_tok = build_teacher(cfg)
        accs = run_setting(name, cfg, teacher, teacher_tok, train_examples, eval_sets)
        rows.append({"setting": name, **accs})
        print(f"[{name}] {accs}", flush=True)

    # Write the main table.
    tasks = ecfg["tasks"]
    table_path = os.path.join(out_dir, "main_table.csv")
    with open(table_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["setting"] + tasks + ["note"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ["setting"] + tasks + ["note"]})
    print(f"[baselines] wrote {table_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Task 0 -- Pre-flight gate.

Measure teacher vs student few-shot accuracy on GSM8K (and MATH) and STOP the
whole pipeline unless the teacher is clearly stronger.

Gate (IMPLEMENTATION_SPEC Task 0 / method-doc §7, non-negotiable):
    teacher GSM8K accuracy must exceed student by >= 10 percentage points,
    else there is no relational signal to distil -- abort and swap teacher/task.

Usage:
    python scripts/preflight_gate.py --config configs/oprrd_mvp.yaml --n 500
Exit code 0 = PASS, 1 = FAIL (so it can gate a shell pipeline).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.evaluate import evaluate_task  # noqa: E402
from src.data.load_tasks import load_task  # noqa: E402
from src.utils.config import load_config, set_seed  # noqa: E402

GSM8K_GATE_POINTS = 10.0  # teacher must beat student by >= 10 points on GSM8K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/oprrd_mvp.yaml")
    ap.add_argument("--n", type=int, default=500, help="eval samples per task")
    ap.add_argument("--tasks", nargs="+", default=["gsm8k", "math"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--k_shot", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    out_path = args.out or os.path.join(cfg.get("output_dir", "outputs"), "preflight.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Build models (heavy import deferred so --help stays light).
    from src.models.build import build_student, build_teacher

    print("[preflight] loading student ...", flush=True)
    student, student_tok = build_student(cfg)
    print("[preflight] loading teacher ...", flush=True)
    teacher, teacher_tok = build_teacher(cfg)

    report = {"gate_points": GSM8K_GATE_POINTS, "n_per_task": args.n, "tasks": {}}
    for task in args.tasks:
        examples = load_task(task, split="test", n=args.n)
        print(f"[preflight] evaluating student on {task} ({len(examples)}) ...", flush=True)
        s = evaluate_task(student, student_tok, task, examples, k_shot=args.k_shot)
        print(f"[preflight] evaluating teacher on {task} ...", flush=True)
        t = evaluate_task(teacher, teacher_tok, task, examples, k_shot=args.k_shot)
        gap = 100.0 * (t["accuracy"] - s["accuracy"])
        report["tasks"][task] = {
            "student_acc": s["accuracy"],
            "teacher_acc": t["accuracy"],
            "gap_points": gap,
            "n": s["n"],
        }
        print(
            f"[preflight] {task}: student={s['accuracy']:.3f} "
            f"teacher={t['accuracy']:.3f} gap={gap:+.1f} pts",
            flush=True,
        )

    gsm_gap = report["tasks"].get("gsm8k", {}).get("gap_points", None)
    passed = gsm_gap is not None and gsm_gap >= GSM8K_GATE_POINTS
    report["gsm8k_gap_points"] = gsm_gap
    report["PASS"] = passed

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    if passed:
        print(f"PASS: teacher beats student on GSM8K by {gsm_gap:+.1f} pts "
              f"(>= {GSM8K_GATE_POINTS}). Proceed to Task A/B/C.")
        print(f"[preflight] wrote {out_path}")
        sys.exit(0)
    else:
        print(f"FAIL: GSM8K gap = {gsm_gap} pts (< {GSM8K_GATE_POINTS}). STOP.")
        print("Swap in a stronger teacher or a different task; do NOT continue.")
        print(f"[preflight] wrote {out_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()

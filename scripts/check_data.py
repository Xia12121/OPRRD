#!/usr/bin/env python
"""Probe that each task dataset loads with the expected fields, and that the
loader + scorer agree on a couple of examples. Run once on the GPU/dev node to
confirm data availability before launching the pipeline.

    python scripts/check_data.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def probe_raw():
    from datasets import load_dataset

    candidates = {
        "gsm8k": [("openai/gsm8k", "main", "test")],
        "math": [
            ("hendrycks/competition_math", None, "test"),
            ("EleutherAI/hendrycks_math", None, "test"),
            ("lighteval/MATH", "all", "test"),
            ("nlile/hendrycks-MATH-benchmark", None, "test"),
            ("HuggingFaceH4/MATH-500", None, "test"),
        ],
        "arc_challenge": [("allenai/ai2_arc", "ARC-Challenge", "test")],
        "boolq": [("google/boolq", None, "validation")],
    }
    found = {}
    for task, cands in candidates.items():
        for path, name, split in cands:
            for sp in (split, "train"):
                try:
                    ds = load_dataset(path, name, split=f"{sp}[:2]") if name else load_dataset(path, split=f"{sp}[:2]")
                    print(f"OK   {task:14s} {path} ({name}) split={sp} fields={list(ds[0].keys())}")
                    found[task] = (path, name, sp)
                    break
                except Exception as e:
                    print(f"FAIL {task:14s} {path} ({name}) split={sp} {type(e).__name__}: {str(e)[:90]}")
            if task in found:
                break
    return found


def probe_loader():
    from src.data.load_tasks import load_task, score, build_prompt

    for task in ["gsm8k", "math", "arc_challenge", "boolq"]:
        try:
            ex = load_task(task, split="test", n=2)
            e = ex[0]
            ok_self = score(task, e["solution"], e)  # gold solution should score correct
            print(f"loader {task}: gold={e['gold']!r} self_score={ok_self} prompt_head={build_prompt(task, e)[:60]!r}")
        except Exception as ex_e:
            print(f"loader {task}: ERROR {type(ex_e).__name__}: {ex_e}")


if __name__ == "__main__":
    print("=== raw dataset availability ===")
    found = probe_raw()
    print("=== loader + scorer self-check ===")
    probe_loader()

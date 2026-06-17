"""Config loading and small training utilities."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config (e.g. ``configs/oprrd_mvp.yaml``)."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def require_preflight(cfg: Dict[str, Any], skip: bool = False) -> None:
    """Enforce the Task-0 gate before training (IMPLEMENTATION_SPEC §4/§5).

    Reads ``<output_dir>/preflight.json`` and aborts unless ``PASS`` is True.
    ``skip=True`` bypasses the check (only for runs already known to pass).
    """
    if skip:
        print("[preflight] gate check SKIPPED by request.")
        return
    path = os.path.join(cfg.get("output_dir", "outputs"), "preflight.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"[preflight] gate report not found at {path}. Run "
            f"`python scripts/preflight_gate.py --config <cfg>` first "
            f"(or pass --skip_preflight if already verified)."
        )
    with open(path) as f:
        report = json.load(f)
    if not report.get("PASS", False):
        raise SystemExit(
            f"[preflight] gate FAILED (GSM8K gap = {report.get('gsm8k_gap_points')} pts "
            f"< required). STOP: swap in a stronger teacher or a different task."
        )
    print(f"[preflight] gate PASS (GSM8K gap = {report.get('gsm8k_gap_points')} pts).")


class JsonlLogger:
    """Append-only JSONL logger for training/eval records."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # Truncate any stale log so a fresh run starts clean.
        open(self.path, "w").close()
        self.records: List[Dict[str, Any]] = []

    def log(self, record: Dict[str, Any]) -> None:
        self.records.append(record)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

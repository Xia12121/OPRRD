"""Task evaluation: batched generation + accuracy scoring.

Used by the pre-flight gate (Task 0), MVP periodic eval (Task D), and the main
table (Task E). Greedy decoding by default for reproducible accuracy numbers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from src.data.load_tasks import build_prompt, score
from src.data.sft_data import format_prompt


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> List[str]:
    """Generate completions for chat-formatted ``prompts`` (returns text only)."""
    device = next(model.parameters()).device
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, padding=True, truncation=True, max_length=2048,
                    return_tensors="pt").to(device)
    was_training = model.training
    model.eval()
    gen = model.generate(
        **enc,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if was_training:
        model.train()
    tokenizer.padding_side = prev_side
    input_len = enc["input_ids"].shape[1]
    return [tokenizer.decode(g[input_len:], skip_special_tokens=True) for g in gen]


@torch.no_grad()
def evaluate_task(
    model,
    tokenizer,
    task: str,
    examples: List[Dict[str, Any]],
    k_shot: int = 4,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    temperature: float = 0.0,
    keep_predictions: bool = False,
) -> Dict[str, Any]:
    """Evaluate ``model`` on ``examples`` for ``task``; return accuracy + counts."""
    correct = 0
    n = len(examples)
    predictions: List[Dict[str, Any]] = []

    for i in range(0, n, batch_size):
        chunk = examples[i : i + batch_size]
        prompts = [
            format_prompt(tokenizer, build_prompt(task, ex, k_shot)) for ex in chunk
        ]
        completions = generate_batch(model, tokenizer, prompts, max_new_tokens, temperature)
        for ex, comp in zip(chunk, completions):
            ok = score(task, comp, ex)
            correct += int(ok)
            if keep_predictions:
                predictions.append(
                    {"question": ex["question"][:200], "gold": ex["gold"],
                     "completion": comp[:400], "correct": ok}
                )

    result = {"task": task, "accuracy": correct / n if n else 0.0, "n": n, "correct": correct}
    if keep_predictions:
        result["predictions"] = predictions
    return result

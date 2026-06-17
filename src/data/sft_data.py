"""SFT data for the CE branch (next-token cross-entropy on gold solutions).

The CE branch is deliberately trained on *independent, high-quality* data
(question + gold solution), never on the student rollout, so it cannot reinforce
bad answers (method-doc D4 / §4.3).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def format_prompt(tokenizer, question: str) -> str:
    """Chat-formatted prompt string (kept consistent with the training loop)."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return question + "\n"


def build_sft_example(
    tokenizer, question: str, answer: str, max_length: int = 1024
) -> Dict[str, List[int]]:
    """One SFT example: full input_ids with prompt tokens masked (-100) in labels."""
    prompt = format_prompt(tokenizer, question)
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt + answer, add_special_tokens=True)["input_ids"]
    full_ids = full_ids[:max_length]

    labels = list(full_ids)
    n_prompt = min(len(prompt_ids), len(full_ids))
    for i in range(n_prompt):
        labels[i] = -100  # do not compute CE on the prompt
    return {"input_ids": full_ids, "labels": labels}


def collate_sft_batch(
    tokenizer, pairs: List[Tuple[str, str]], max_length: int = 1024
) -> Dict[str, torch.Tensor]:
    """Right-pad a list of (question, answer) pairs into a CE batch."""
    examples = [build_sft_example(tokenizer, q, a, max_length) for q, a in pairs]
    T = max(len(e["input_ids"]) for e in examples)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_ids, attention_mask, labels = [], [], []
    for e in examples:
        ids = e["input_ids"]
        n = len(ids)
        input_ids.append(ids + [pad_id] * (T - n))
        attention_mask.append([1] * n + [0] * (T - n))
        labels.append(e["labels"] + [-100] * (T - n))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

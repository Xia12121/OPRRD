"""OP-RRD training loop (Task C) -- Algorithm 1.

Two branches per step:
  * relational branch  -- student rollout (on-policy) -> CKA on per-sample,
    block-diagonal span Gram matrices across middle-layer pairs.
  * CE branch          -- standard next-token cross-entropy on independent SFT
    data (keeps language/task ability, never trains on the rollout -> D4).

Non-negotiable constraints (IMPLEMENTATION_SPEC §5), enforced here:
  1. teacher forward runs under ``no_grad`` and is detached -> no gradient to it.
  2. relational Gram is built **within each sample** (block-diagonal); spans are
     never mixed across samples.
  3. default layer pairs are middle layers (``{0.5, 0.667}``), not final-only.
  4. ``L_total = L_CE + lambda_rel * L_rel`` with ``L_rel`` divided by ``B*|P|``.
  5. kill criteria are checked every step.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.align.span_align import mean_pool_by_charspan, select_response_spans
from src.losses.cka_loss import get_relational_operator


# --------------------------------------------------------------------------- #
# Layer pairing (method-doc §4.5)                                              #
# --------------------------------------------------------------------------- #
def compute_layer_pairs(
    num_student_layers: int, num_teacher_layers: int, depths: Sequence[float]
) -> List[Tuple[int, int]]:
    """Map relative depths to (student_layer, teacher_layer) hidden-state indices.

    ``m(l) = round((l / L_S) * L_T)``. Indices are into the HF
    ``output_hidden_states`` tuple (length ``L+1``; index 0 = embeddings), so a
    student layer index ``l in [1, L_S]`` selects ``hidden_states[l]``.
    """
    pairs: List[Tuple[int, int]] = []
    for depth in depths:
        s = int(round(depth * num_student_layers))
        s = max(1, min(num_student_layers, s))
        t = int(round((s / num_student_layers) * num_teacher_layers))
        t = max(1, min(num_teacher_layers, t))
        pairs.append((s, t))
    return pairs


def _num_hidden_layers(model) -> int:
    """Number of transformer blocks, robust to PEFT / wrapper nesting."""
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "num_hidden_layers"):
        return cfg.num_hidden_layers
    base = getattr(model, "base_model", None)
    if base is not None:
        return _num_hidden_layers(base)
    inner = getattr(model, "model", None)
    if inner is not None:
        return _num_hidden_layers(inner)
    raise AttributeError("could not determine num_hidden_layers for model")


# --------------------------------------------------------------------------- #
# Tokenisation helpers                                                         #
# --------------------------------------------------------------------------- #
def format_prompt(tokenizer, question: str) -> str:
    """Render a question into the model's chat prompt (string, no tokens)."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return question + "\n"


def tokenize_for_forward(tokenizer, texts: List[str], max_length: int):
    """Right-padded batch encoding plus per-token char offsets and real lengths.

    Right padding keeps the real tokens at positions ``[0:len_b]`` so they align
    with ``offset_mapping[b][:len_b]`` and ``hidden[b, :len_b]``.
    """
    prev_pad, prev_trunc = tokenizer.padding_side, tokenizer.truncation_side
    tokenizer.padding_side = "right"
    # Left-truncate: the response (where last_k/uniform spans live) is at the END
    # of Z, so when Z is too long we must keep the tail, not the prompt prefix.
    tokenizer.truncation_side = "left"
    enc = tokenizer(
        texts,
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_tensors="pt",
    )
    tokenizer.padding_side, tokenizer.truncation_side = prev_pad, prev_trunc
    offsets = enc.pop("offset_mapping")  # [B, T, 2]
    lengths = enc["attention_mask"].sum(dim=1)  # [B]
    return enc, offsets, lengths


# --------------------------------------------------------------------------- #
# Rollout (on-policy) / off-policy text source for the relational branch        #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_relational_texts(
    student,
    student_tok,
    examples: List[Dict[str, str]],
    on_policy: bool,
    rollout_temperature: float = 0.7,
    max_new_tokens: int = 256,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Return per-sample ``Z`` text and response char-bounds ``(resp_start, end)``.

    on_policy : ``Z = prompt + student.generate(prompt)`` (the student's own
                trajectory -- D3).
    off_policy: ``Z = prompt + gold_answer`` (the off-policy ablation, Claim 3).
    The student forward in the relational branch later re-tokenises ``Z`` so the
    char offsets line up exactly with this string.
    """
    prompt_texts = [format_prompt(student_tok, ex["question"]) for ex in examples]
    z_texts: List[str] = []
    bounds: List[Tuple[int, int]] = []

    if on_policy:
        device = next(student.parameters()).device
        prev_side = student_tok.padding_side
        student_tok.padding_side = "left"  # left-pad so generation continues the prompt
        enc = student_tok(prompt_texts, padding=True, return_tensors="pt").to(device)
        student.eval()
        gen = student.generate(
            **enc,
            do_sample=rollout_temperature > 0,
            temperature=max(rollout_temperature, 1e-5),
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=student_tok.pad_token_id,
        )
        student.train()
        student_tok.padding_side = prev_side
        input_len = enc["input_ids"].shape[1]
        for b, ptext in enumerate(prompt_texts):
            completion = student_tok.decode(gen[b, input_len:], skip_special_tokens=True)
            z = ptext + completion
            z_texts.append(z)
            bounds.append((len(ptext), len(z)))
    else:
        for ex, ptext in zip(examples, prompt_texts):
            z = ptext + ex["answer"]
            z_texts.append(z)
            bounds.append((len(ptext), len(z)))

    return z_texts, bounds


# --------------------------------------------------------------------------- #
# Relational loss over a batch (block-diagonal, sample-internal)               #
# --------------------------------------------------------------------------- #
@dataclass
class RelStats:
    rel_loss: float = 0.0
    mean_cka: Optional[float] = None  # true CKA mean; None for non-CKA operators
    n_terms: int = 0  # (sample, layer-pair) terms actually computed
    n_skipped: int = 0  # terms skipped for too-few valid spans


def relational_branch(
    student,
    teacher,
    student_tok,
    teacher_tok,
    z_texts: List[str],
    bounds: List[Tuple[int, int]],
    layer_pairs: List[Tuple[int, int]],
    operator_name: str = "cka",
    span_strategy: str = "last_k",
    K_spans: int = 16,
    K_min: int = 4,
    max_length: int = 1024,
    eps: float = 1e-6,
    operator=None,
) -> Tuple[torch.Tensor, RelStats]:
    """Compute ``L_rel`` and stats for one rollout batch.

    Returns ``(L_rel, stats)`` with ``L_rel`` already divided by ``B*|P|``
    (method-doc §4.2). Each sample's Gram is built independently; there is no
    cross-sample term anywhere in this function (block-diagonal by construction).

    ``operator`` overrides the name-based lookup (e.g. a stateful
    ``HiddenMSEOperator`` for the Claim-1 baseline).
    """
    # mean_cka is only a true CKA value when the actual CKA operator is used;
    # for the cosine-Frobenius ablation and the hidden-MSE baseline, ``1 - loss``
    # is NOT a CKA, so we report mean_cka = None for them (the kill criterion and
    # the logged curve then fall back to the operator-agnostic rel_loss trend).
    is_cka = operator is None and operator_name == "cka"
    operator = operator if operator is not None else get_relational_operator(operator_name)
    B = len(z_texts)
    P = len(layer_pairs)
    device = next(student.parameters()).device

    # Per-sample char spans over the response region only.
    spans_per_sample = [
        select_response_spans(z, rs, re_, strategy=span_strategy, K=K_spans)
        for z, (rs, re_) in zip(z_texts, bounds)
    ]

    # --- student forward (WITH gradient); keep only the needed layers ---
    enc_s, off_s, len_s = tokenize_for_forward(student_tok, z_texts, max_length)
    enc_s = {k: v.to(device) for k, v in enc_s.items()}
    out_s = student(
        input_ids=enc_s["input_ids"],
        attention_mask=enc_s["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
    )
    s_layers = sorted({s for s, _ in layer_pairs})
    student_hidden = {l: out_s.hidden_states[l] for l in s_layers}  # [B, Ts, d_s]
    del out_s  # drop the full hidden-states container (frees the unused layers)

    # --- teacher forward (NO gradient, detached); keep only the needed layers ---
    teacher_device = next(teacher.parameters()).device
    enc_t, off_t, len_t = tokenize_for_forward(teacher_tok, z_texts, max_length)
    enc_t = {k: v.to(teacher_device) for k, v in enc_t.items()}
    with torch.no_grad():
        out_t = teacher(
            input_ids=enc_t["input_ids"],
            attention_mask=enc_t["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
    t_layers = sorted({t for _, t in layer_pairs})
    teacher_hidden = {l: out_t.hidden_states[l].detach() for l in t_layers}
    del out_t

    total = student_hidden[s_layers[0]].new_zeros(())
    stats = RelStats()
    cka_sum = 0.0

    for b in range(B):
        spans_b = spans_per_sample[b]
        if len(spans_b) < K_min:
            stats.n_skipped += P
            continue
        ls = int(len_s[b].item())
        lt = int(len_t[b].item())
        off_s_b = off_s[b][:ls].tolist()
        off_t_b = off_t[b][:lt].tolist()
        for (s, t) in layer_pairs:
            H_S, valid_s = mean_pool_by_charspan(student_hidden[s][b, :ls], off_s_b, spans_b)
            H_T, valid_t = mean_pool_by_charspan(
                teacher_hidden[t][b, :lt].to(device), off_t_b, spans_b
            )
            valid = valid_s & valid_t
            if int(valid.sum().item()) < K_min:
                stats.n_skipped += 1
                continue
            loss_pair = operator(H_S[valid], H_T[valid], eps)
            total = total + loss_pair
            if is_cka:
                cka_sum += float(1.0 - loss_pair.detach().item())
            stats.n_terms += 1

    L_rel = total / (B * P)  # nominal B*|P| normalisation (skipped terms = 0)
    stats.rel_loss = float(L_rel.detach().item())
    stats.mean_cka = (cka_sum / stats.n_terms) if (is_cka and stats.n_terms) else None
    return L_rel, stats


# --------------------------------------------------------------------------- #
# CE branch                                                                    #
# --------------------------------------------------------------------------- #
def ce_branch(student, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Standard next-token CE with prompt tokens masked to -100 in ``labels``."""
    device = next(student.parameters()).device
    out = student(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        use_cache=False,
    )
    logits = out.logits[:, :-1, :].contiguous()
    labels = batch["labels"].to(device)[:, 1:].contiguous()
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
    )


# --------------------------------------------------------------------------- #
# Kill criteria (IMPLEMENTATION_SPEC §4)                                       #
# --------------------------------------------------------------------------- #
class KillCriteria:
    """Automated stop conditions, checked every step."""

    def __init__(
        self,
        max_total_loss: float = 50.0,
        rel_window: int = 150,
        max_runtime_s: float = 24 * 3600,
        lambda_rel: float = 0.05,
    ):
        self.max_total_loss = max_total_loss
        self.rel_window = rel_window
        self.max_runtime_s = max_runtime_s
        self.lambda_rel = lambda_rel
        self._cka_hist: List[float] = []
        self._rel_hist: List[float] = []

    def check(
        self,
        total_loss: float,
        rel_loss: float,
        mean_cka: Optional[float],
        elapsed_s: float,
    ) -> Optional[str]:
        # loss_nan: NaN/Inf or runaway magnitude.
        if not torch.isfinite(torch.tensor(total_loss)) or total_loss > self.max_total_loss:
            return "loss_nan"

        # rel_not_learning (spec §4): over rel_window steps the relational branch
        # is not improving -- rel_loss has not fallen AND (when a true CKA is
        # available, i.e. the CKA operator) mean_cka has not risen. For non-CKA
        # operators mean_cka is None, so we rely on the rel_loss trend alone.
        self._cka_hist.append(mean_cka)
        self._rel_hist.append(rel_loss)
        if self.lambda_rel > 0 and len(self._rel_hist) > self.rel_window:
            rel_then = self._rel_hist[-self.rel_window - 1]
            rel_now = self._rel_hist[-1]
            rel_stalled = rel_now >= rel_then - 1e-4
            cka_then = self._cka_hist[-self.rel_window - 1]
            cka_now = self._cka_hist[-1]
            if cka_then is not None and cka_now is not None:
                cka_stalled = cka_now <= cka_then + 1e-4
                if rel_stalled and cka_stalled:
                    return "rel_not_learning"
            elif rel_stalled:
                return "rel_not_learning"

        # runtime: a single training run exceeding the budget.
        if elapsed_s > self.max_runtime_s:
            return "runtime"
        return None


# --------------------------------------------------------------------------- #
# Trainer                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class TrainState:
    step: int = 0
    history: List[dict] = field(default_factory=list)
    stop_reason: Optional[str] = None


def train(
    student,
    teacher,
    student_tok,
    teacher_tok,
    rel_example_stream: Callable[[int], List[Dict[str, str]]],
    sft_batch_stream: Callable[[int], Dict[str, torch.Tensor]],
    cfg: dict,
    logger=None,
    eval_fn: Optional[Callable[[int], dict]] = None,
    relational_operator=None,
    extra_params=None,
) -> TrainState:
    """Run the OP-RRD training loop.

    ``rel_example_stream(step)`` -> list of ``{"question","answer"}`` for the
    relational branch; ``sft_batch_stream(step)`` -> a collated CE batch. Both
    are callables so the caller controls sampling/shuffling.

    ``relational_operator`` / ``extra_params`` let a baseline plug in a stateful
    operator (e.g. the coordinate hidden-MSE projectors) whose parameters must be
    optimised alongside the student.
    """
    rcfg = cfg["relational"]
    tcfg = cfg["training"]
    L_S = _num_hidden_layers(student)
    L_T = _num_hidden_layers(teacher)
    layer_pairs = compute_layer_pairs(L_S, L_T, rcfg["layer_pairs_rel_depth"])
    lambda_rel = float(rcfg["lambda_rel"])

    params = [p for p in student.parameters() if p.requires_grad]
    if extra_params:
        params = params + list(extra_params)
    opt = torch.optim.AdamW(params, lr=float(tcfg["learning_rate"]))
    max_steps = int(tcfg["max_steps"])
    accum = int(tcfg.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))

    # Scheduler counts optimiser steps (one per `accum` micro-steps).
    total_opt = max(1, max_steps // accum)
    warmup_opt = int(tcfg.get("warmup_ratio", 0.0) * total_opt)
    use_cosine = tcfg.get("scheduler", "cosine") == "cosine"

    def lr_lambda(opt_step: int) -> float:
        if opt_step < warmup_opt:
            return (opt_step + 1) / max(1, warmup_opt)
        if use_cosine:
            progress = (opt_step - warmup_opt) / max(1, total_opt - warmup_opt)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    kill = KillCriteria(
        rel_window=int(cfg.get("kill", {}).get("rel_window", 150)),
        max_runtime_s=float(cfg.get("kill", {}).get("max_runtime_s", 24 * 3600)),
        lambda_rel=lambda_rel,
    )
    state = TrainState()
    start = time.time()
    student.train()
    opt.zero_grad()

    for step in range(max_steps):
        # --- relational branch ---
        rel_loss_t = torch.zeros((), device=next(student.parameters()).device)
        rel_stats = RelStats()
        if lambda_rel > 0:
            examples = rel_example_stream(step)
            z_texts, bounds = build_relational_texts(
                student,
                student_tok,
                examples,
                on_policy=bool(rcfg.get("on_policy", True)),
                rollout_temperature=float(rcfg.get("rollout_temperature", 0.7)),
                max_new_tokens=int(rcfg.get("max_new_tokens", 256)),
            )
            rel_loss_t, rel_stats = relational_branch(
                student,
                teacher,
                student_tok,
                teacher_tok,
                z_texts,
                bounds,
                layer_pairs,
                operator_name=rcfg.get("operator", "cka"),
                span_strategy=rcfg.get("span_strategy", "last_k"),
                K_spans=int(rcfg.get("K_spans", 16)),
                K_min=int(rcfg.get("K_min", 4)),
                max_length=int(rcfg.get("max_length", 1024)),
                operator=relational_operator,
            )

        # --- CE branch (independent SFT data) ---
        ce_loss_t = ce_branch(student, sft_batch_stream(step))

        # --- combine ---
        total_t = ce_loss_t + lambda_rel * rel_loss_t
        (total_t / accum).backward()
        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad()

        rec = {
            "step": step,
            "ce_loss": float(ce_loss_t.detach().item()),
            "rel_loss": rel_stats.rel_loss,
            "total_loss": float(total_t.detach().item()),
            "mean_cka": rel_stats.mean_cka,
            "n_rel_terms": rel_stats.n_terms,
            "n_rel_skipped": rel_stats.n_skipped,
            "lr": sched.get_last_lr()[0],
        }
        state.history.append(rec)
        if logger is not None:
            logger.log(rec)

        reason = kill.check(
            rec["total_loss"], rec["rel_loss"], rec["mean_cka"], time.time() - start
        )
        if reason is not None:
            state.stop_reason = reason
            _dump_kill(cfg, reason, step, rec)
            break

        if eval_fn is not None and (step + 1) % int(tcfg.get("eval_every_n_steps", 100)) == 0:
            ev = eval_fn(step)
            if logger is not None:
                logger.log({"step": step, "eval": ev})

        state.step = step + 1

    # Flush a trailing partial accumulation window so the last micro-steps'
    # gradients are applied (else the final ``max_steps % accum`` backward()s are
    # discarded on return).
    if state.stop_reason is None and (max_steps % accum != 0):
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad()

    return state


def _dump_kill(cfg: dict, reason: str, step: int, rec: dict) -> None:
    """Persist a small diagnostic when a kill criterion fires."""
    import json
    import os

    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "kill_dump.json"), "w") as f:
        json.dump({"reason": reason, "step": step, "record": rec}, f, indent=2)

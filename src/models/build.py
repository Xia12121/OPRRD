"""Model construction (Task: models/build.py).

Production path:
  * student  -- ``Qwen2.5-1.5B-Instruct`` + LoRA (trainable).
  * teacher  -- ``Llama-3.1-8B-Instruct`` loaded in 8-bit, fully frozen.

Test path:
  * :func:`build_toy_student_teacher` builds a tiny randomly-initialised
    Qwen2 / Llama pair with *different* tokenizers and *different* hidden sizes,
    so the training loop can be exercised end-to-end on CPU.

Non-negotiable: the teacher receives no gradient -- every teacher parameter has
``requires_grad = False`` and the teacher always runs under ``no_grad`` in the
loop (IMPLEMENTATION_SPEC §5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def _prepare_tokenizer(tok):
    """Ensure a pad token exists and padding is right-sided (forward branch)."""
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def build_student(cfg: dict, device: Optional[str] = None):
    """Load the student and wrap it with LoRA (trainable adapter)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    mcfg = cfg["model"]
    name = mcfg["student"]
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tok = _prepare_tokenizer(AutoTokenizer.from_pretrained(name))
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)

    lora = LoraConfig(
        r=mcfg.get("lora_r", 16),
        lora_alpha=mcfg.get("lora_alpha", 32),
        target_modules=mcfg.get(
            "lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        lora_dropout=mcfg.get("lora_dropout", 0.0),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    if device:
        model.to(device)
    return model, tok


def build_teacher(cfg: dict, device: Optional[str] = None):
    """Load the teacher (8-bit if requested + CUDA available) and freeze it."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mcfg = cfg["model"]
    name = mcfg["teacher"]
    tok = _prepare_tokenizer(AutoTokenizer.from_pretrained(name))

    load_8bit = bool(mcfg.get("teacher_load_in_8bit", True)) and torch.cuda.is_available()
    if load_8bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=quant, device_map="auto"
        )
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
        if device:
            model.to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


@dataclass
class ToyPair:
    student: "torch.nn.Module"
    teacher: "torch.nn.Module"
    student_tok: object
    teacher_tok: object


def build_toy_student_teacher(
    student_tok_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    teacher_tok_name: str = "gpt2",
    d_student: int = 64,
    d_teacher: int = 96,
    n_student_layers: int = 4,
    n_teacher_layers: int = 6,
    lora: bool = True,
    seed: int = 0,
) -> ToyPair:
    """Tiny random Qwen2(student)/Llama(teacher) pair for CPU loop tests.

    Uses two *real, different* tokenizers (so offsets/vocab differ) but tiny,
    randomly-initialised weights. The student is LoRA-wrapped; the teacher is
    frozen. Hidden sizes differ on purpose to exercise the d_s != d_t path.
    """
    from transformers import (
        AutoTokenizer,
        LlamaConfig,
        LlamaForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    torch.manual_seed(seed)
    student_tok = _prepare_tokenizer(AutoTokenizer.from_pretrained(student_tok_name))
    teacher_tok = _prepare_tokenizer(AutoTokenizer.from_pretrained(teacher_tok_name))

    s_cfg = Qwen2Config(
        vocab_size=len(student_tok),
        hidden_size=d_student,
        intermediate_size=2 * d_student,
        num_hidden_layers=n_student_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
    )
    t_cfg = LlamaConfig(
        vocab_size=len(teacher_tok),
        hidden_size=d_teacher,
        intermediate_size=2 * d_teacher,
        num_hidden_layers=n_teacher_layers,
        num_attention_heads=6,
        num_key_value_heads=3,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
    )
    student = Qwen2ForCausalLM(s_cfg)
    teacher = LlamaForCausalLM(t_cfg)

    if lora:
        from peft import LoraConfig, get_peft_model

        student = get_peft_model(
            student,
            LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            ),
        )

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return ToyPair(student, teacher, student_tok, teacher_tok)

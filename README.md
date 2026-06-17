# OP-RRD: On-Policy Relational Representation Distillation (cross-architecture)

Reference implementation for the method in `../OP-RRD_method_design_cn.md` and
`../IMPLEMENTATION_SPEC.md`.

**One line.** Across architectures we do *not* align hidden-state coordinates.
Instead, on the student's own rollouts, we align the **pairwise relational
geometry** of teacher and student representations with **linear CKA** (invariant
to per-side orthogonal transforms and scaling, so it handles `d_s != d_t`), and
we test the claim that coordinate alignment is *actively harmful* in this regime.

Teacher: `Llama-3.1-8B-Instruct` (frozen, 8-bit). Student: `Qwen2.5-1.5B-Instruct`
(LoRA). Different tokenizers, hidden sizes, depths, and bases throughout.

## Novelty chain (what the experiments must show)
1. **Coordinate alignment is harmful** — `direct_hidden_mse` ≤ `student_sft`.
2. **Relational geometry beats cross-tokenizer SOTA** — Ours > ULD, DSKD on
   GSM8K/MATH.
3. **On-policy helps** — Ours on-policy > Ours off-policy.
4. **Negative control** — on BoolQ (single-step) Ours should *not* beat SFT much.

## Layout
```
src/align/span_align.py    Task A  char-offset span pooling + span selection
src/losses/cka_loss.py     Task B  linear-CKA loss (+ cosine-Frobenius ablation)
src/losses/coordinate_mse.py       coordinate hidden-MSE (Claim-1 baseline)
src/train/loop.py          Task C  Algorithm 1: rollout + CKA + CE, kill criteria
src/data/load_tasks.py             GSM8K/MATH/ARC/BoolQ load + scoring
src/data/sft_data.py               CE-branch SFT collator (prompt masked)
src/eval/evaluate.py               batched generation + accuracy
src/models/build.py                student LoRA + frozen 8-bit teacher (+ toy pair)
scripts/preflight_gate.py  Task 0  teacher-vs-student gap gate (>= 10 pts on GSM8K)
scripts/run_mvp.py         Task D  MVP training run
scripts/run_baselines.py   Task E/F baselines + main claims -> outputs/main_table.csv
tests/                             CPU unit tests (run without a GPU)
configs/oprrd_mvp.yaml             default + ablation hyper-parameters
```

## Non-negotiable constraints (enforced in code)
- Teacher is detached and runs under `no_grad`; **no gradient reaches it.**
- Relational Gram is built **within each sample** (block-diagonal); never across
  samples.
- Default layer pairs are **middle layers** `{0.5, 0.667}`, not final-only.
- `L_total = L_CE + lambda_rel * L_rel`, with `L_rel` divided by `B*|P|`.
- ULD / DSKD use their **official** implementations (record commit hashes).

## Environment
CPU dev / unit tests:
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q                         # 36 tests, no GPU needed
```
GPU run (after a CUDA torch is installed): also `pip install -r requirements-gpu.txt`.
Gated teacher: `huggingface-cli login` with access to `meta-llama/Llama-3.1-8B-Instruct`.
Point caches at a large volume, e.g. `export HF_HOME=/data/hf`.

## Run order (matches the dependency graph)
```bash
# Task 0 -- gate; STOPS (exit 1) unless teacher beats student by >= 10 pts on GSM8K
python scripts/preflight_gate.py --config configs/oprrd_mvp.yaml --n 500

# Task D -- MVP
python scripts/run_mvp.py --config configs/oprrd_mvp.yaml

# Task E/F -- baselines + main claims
python scripts/run_baselines.py --config configs/oprrd_mvp.yaml \
  --settings student_fewshot student_sft seqkd direct_hidden_mse \
             oprrd_onpolicy oprrd_offpolicy
# ULD / DSKD: run their official repos, then merge:
#   --settings uld dskd --external_json external_results.json
```

## Ablations (Task G)
Override config keys, e.g.:
```bash
python scripts/run_mvp.py --override relational.operator=cosine_frobenius   # CKA vs cosine-F (centering)
python scripts/run_mvp.py --override relational.layer_pairs_rel_depth=[1.0] # middle vs final
python scripts/run_mvp.py --override relational.span_strategy=uniform       # span strategy
python scripts/run_mvp.py --override relational.lambda_rel=0.1              # lambda sweep
```

## Kill criteria (auto, in the loop)
`loss_nan` (NaN/Inf or total > 50), `rel_not_learning` (mean_cka flat & rel_loss
flat for `rel_window` steps), `runtime` (> 24h). A fired criterion writes
`outputs/.../kill_dump.json` and stops the run.

## Test status
`pytest -q` → all green (CKA invariances, cross-tokenizer pooling, loop
detach/block-diagonal/on-off-policy, kill criteria, scoring). The math-critical
tests (`test_cka_loss.py`, `test_span_align.py`) run on CPU and are the
correctness backbone of the method.
```

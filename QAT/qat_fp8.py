"""
QAT FP8 (E4M3, W8A8): Quantization-Aware Training

핵심 설계:
  Phase 1 (QAT) 레시피 - dynamic activation:
    weights:            dynamic=false  (init 시 weight 텐서에서 스케일 직접 계산 → OK)
    input_activations:  dynamic=true   (배치마다 스케일 재계산 → 사전 캘리브레이션 불필요)
    → fake-quant가 정확한 FP8 노이즈를 주입 → 가중치가 양자화에 강인해짐

  Phase 2 (PTQ) 레시피 - static activation:
    → dynamic=false 정적 스케일 + 캘리브레이션 데이터로 최종 스케일 확정 후 압축 저장
    → vLLM / compressed-tensors 호환 포맷

사용법:
  python QAT/qat_fp8.py \
      --model models/EXAONE-4.0-1.2B-VocabPruned \
      --dataset dataset/dataset_N/qat_train \
      --epochs 1
"""

import os
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 헤드리스 서버에서 디스플레이 없이 저장
import matplotlib.pyplot as plt

import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from llmcompressor import active_session, callbacks as LC, oneshot, reset_session


# ╔══════════════════════════════════════════════════════════════╗
# ║  HYPERPARAMETERS — 이 섹션만 수정하면 됩니다                  ║
# ╚══════════════════════════════════════════════════════════════╝

QAT_LR            = 2e-5
QAT_EPOCHS        = 1        # 데이터 충분 시 1에폭도 수만 스텝
QAT_BATCH_SIZE    = 4
QAT_GRAD_ACCUM    = 8
NUM_TRAIN_SAMPLES = 50000    # 데이터 충분 가정 (--samples로 override)
MAX_SEQ_LEN       = 1024
PTQ_CALIB_SAMPLES = 512

DATASET_ID    = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"


# ╔══════════════════════════════════════════════════════════════╗
# ║  레시피                                                       ║
# ╚══════════════════════════════════════════════════════════════╝

# Phase 1: QAT용 — activation dynamic=true (사전 캘리브레이션 없이 정확한 fake-quant)
QAT_RECIPE = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: true
                        symmetric: true
                    targets: ["Linear"]
"""

# Phase 2: PTQ용 — static activation으로 최종 압축 (vLLM 호환)
PTQ_RECIPE = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
"""


def plot_loss(log_history, model_name, quant_type, timestamp):
    """학습 Loss 곡선을 QAT_Loss 폴더에 PNG로 저장"""
    entries = [e for e in log_history if "loss" in e]
    if not entries:
        print("  [경고] 학습 로그가 없어 Loss 그래프를 건너뜁니다.")
        return

    steps  = [e["step"] for e in entries]
    losses = [e["loss"] for e in entries]

    os.makedirs("QAT_Loss", exist_ok=True)
    filename  = f"{model_name}-{quant_type}_{timestamp}.png"
    save_path = os.path.join("QAT_Loss", filename)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses, color="steelblue", linewidth=1.2, label="train loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name}  QAT-{quant_type}  Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Loss 그래프 저장: {save_path}")


class QATTrainer(Trainer):
    """llmcompressor lifecycle callbacks를 HuggingFace Trainer에 통합"""

    def training_step(self, model, inputs, num_items_in_batch=None):
        LC.batch_start(batch_data=inputs)
        loss = super().training_step(model, inputs, num_items_in_batch)
        LC.batch_end()
        return loss


def load_train_dataset(tokenizer, dataset_path, num_samples, max_seq_len, seed):
    if dataset_path:
        print(f"  로컬 데이터셋 사용: {dataset_path}")
        train_ds = load_from_disk(dataset_path)
        if len(train_ds) > num_samples:
            train_ds = train_ds.select(range(num_samples))
    else:
        print(f"  기본 데이터셋 사용: {DATASET_ID}")
        raw_ds = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)
        raw_ds = raw_ds.shuffle(seed=seed, buffer_size=10000)
        from datasets import Dataset
        train_ds = Dataset.from_list(list(raw_ds.take(num_samples)))

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=True,
                tokenize=False,
            )
        }

    train_ds = train_ds.map(preprocess, num_proc=4)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )

    train_ds = train_ds.map(
        tokenize_fn,
        remove_columns=train_ds.column_names,
        num_proc=4,
    )
    # 너무 짧은 샘플 제거 (최소 32 토큰)
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) >= 32)
    print(f"  학습 샘플 수: {len(train_ds)}")
    return train_ds


def main():
    parser = argparse.ArgumentParser(description="FP8 E4M3 W8A8 Quantization-Aware Training")
    parser.add_argument("--model", "-m", type=str,
                        default="models/EXAONE-4.0-1.2B-VocabPruned")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--dataset", "-d", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=QAT_EPOCHS)
    parser.add_argument("--lr", type=float, default=QAT_LR)
    parser.add_argument("--batch-size", type=int, default=QAT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=QAT_GRAD_ACCUM)
    parser.add_argument("--samples", type=int, default=NUM_TRAIN_SAMPLES)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--calib-samples", type=int, default=PTQ_CALIB_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_name  = os.path.basename(args.model.rstrip("/\\"))
    output_path = args.output or os.path.join("models", f"{model_name}-QAT-FP8")
    trainer_tmp = "/tmp/qat_fp8_trainer"
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("FP8 QAT (W8A8, E4M3) — Phase 1: dynamic-act QAT / Phase 2: static PTQ")
    print("=" * 60)
    print(f"  모델:              {args.model}")
    print(f"  출력:              {output_path}")
    print(f"  [Phase 1] 에폭:    {args.epochs}")
    print(f"  [Phase 1] LR:      {args.lr} (cosine decay, warmup 5%)")
    print(f"  [Phase 1] 배치:    {args.batch_size} × {args.grad_accum} grad_accum")
    print(f"  [Phase 1] 샘플:    {args.samples}")
    print(f"  [Phase 1] 최대길이: {args.max_seq_len}")
    print(f"  [Phase 2] 캘리브:  {args.calib_samples}개")
    print()

    # ── Step 1: 모델 로드 ──────────────────────────────────────
    print("Step 1: 모델 + 토크나이저 로드")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    print(f"  파라미터: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # ── Step 2: 데이터셋 ──────────────────────────────────────
    print("Step 2: 학습 데이터 준비")
    train_ds = load_train_dataset(
        tokenizer, args.dataset, args.samples, args.max_seq_len, args.seed
    )
    print()

    # ── Step 3: Phase 1 — QAT ─────────────────────────────────
    print("=" * 60)
    print("Phase 1: QAT 학습 (dynamic-activation fake-quant)")
    print("  weight:     dynamic=false (init 시 텐서값으로 스케일 계산)")
    print("  activation: dynamic=true  (배치마다 스케일 재계산 → 정확한 FP8 노이즈)")
    print("=" * 60)

    session = active_session()
    session.initialize(recipe=QAT_RECIPE, model=model)
    print("  fake-quant 노드 삽입 완료\n")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 전체 스텝 수 계산 (로깅용)
    steps_per_epoch = len(train_ds) // (args.batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    print(f"  총 학습 스텝: {total_steps:,} ({steps_per_epoch:,} × {args.epochs} epochs)\n")

    training_args = TrainingArguments(
        output_dir=trainer_tmp,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=50,
        logging_first_step=True,
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=args.seed,
    )

    trainer = QATTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )
    trainer.train()
    plot_loss(trainer.state.log_history, model_name, "FP8", run_timestamp)

    session.finalize()
    model.config.use_cache = True
    shutil.rmtree(trainer_tmp, ignore_errors=True)
    print("\n  Phase 1 완료: QAT-최적화된 BF16 가중치\n")

    # ── Step 4: Phase 2 — PTQ 캘리브레이션 + 압축 저장 ─────────
    print("=" * 60)
    print("Phase 2: PTQ 캘리브레이션 + FP8 static 압축 저장")
    print("  학습 데이터 앞부분을 캘리브레이션 데이터로 재사용")
    print("=" * 60)

    reset_session()

    calib_size = min(args.calib_samples, len(train_ds))
    calib_ds   = train_ds.select(range(calib_size))
    print(f"  캘리브레이션 샘플: {calib_size}개\n")

    os.makedirs(output_path, exist_ok=True)
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=PTQ_RECIPE,
        save_compressed=True,
        output_dir=output_path,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=calib_size,
    )

    # ── Step 5: 저장 ──────────────────────────────────────────
    tokenizer.save_pretrained(output_path)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "quant_format": "FP8-E4M3-W8A8",
        "pipeline": "QAT(dynamic-act fake-quant) + PTQ(static calibration)",
        "model": args.model,
        "output": output_path,
        "phase1": {
            "recipe": "dynamic activation fake-quant",
            "epochs": args.epochs,
            "lr": args.lr,
            "scheduler": "cosine",
            "warmup_ratio": 0.05,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_seq_len": args.max_seq_len,
            "num_samples": len(train_ds),
        },
        "phase2": {
            "recipe": "static activation PTQ",
            "calib_samples": calib_size,
            "max_seq_len": args.max_seq_len,
        },
        "dataset": args.dataset or DATASET_ID,
    }
    with open(os.path.join(output_path, "qat_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    orig = list(Path(args.model).glob("*.safetensors"))
    qat  = list(Path(output_path).glob("*.safetensors"))
    if orig and qat:
        o = sum(f.stat().st_size for f in orig)
        q = sum(f.stat().st_size for f in qat)
        print(f"\n  원본: {o/1e9:.2f} GB  →  QAT-FP8: {q/1e9:.2f} GB  (압축률 {o/q:.2f}x)")

    print()
    print("=" * 60)
    print(f"완료! FP8 QAT 모델: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

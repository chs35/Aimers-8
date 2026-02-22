"""
QAT INT4 (W4A16, 가중치만): Quantization-Aware Training

핵심 설계:
  Phase 1 (QAT) — weight-only fake-quant:
    weights: dynamic=false (init 시 group 단위로 weight 텐서에서 스케일 계산 → OK)
    activation: 없음 (W4A16이므로 activation 양자화 불필요)
    → group-wise fake-quant로 가중치가 INT4 오차에 강인해짐

  Phase 2 (GPTQ PTQ) — Hessian 기반 정밀 INT4 압축:
    → QAT로 최적화된 BF16 가중치에 GPTQ 적용
    → 단순 round-to-nearest보다 훨씬 낮은 양자화 오차

사용법:
  python QAT/qat_int4.py \
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
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from llmcompressor import active_session, callbacks as LC, oneshot, reset_session


def _import_peft():
    """KD LoRA 모드에서만 peft를 import (선택적 의존성)"""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        import subprocess
        print("peft 패키지가 없습니다. 설치 중...")
        subprocess.check_call(["pip", "install", "peft", "-q"])
        from peft import LoraConfig, get_peft_model, TaskType
    return LoraConfig, get_peft_model, TaskType


# ╔══════════════════════════════════════════════════════════════╗
# ║  HYPERPARAMETERS — 이 섹션만 수정하면 됩니다                  ║
# ╚══════════════════════════════════════════════════════════════╝

QAT_LR            = 2e-5
QAT_EPOCHS        = 1
QAT_BATCH_SIZE    = 4
QAT_GRAD_ACCUM    = 8
NUM_TRAIN_SAMPLES = 50000
MAX_SEQ_LEN       = 1024
PTQ_CALIB_SAMPLES = -1   # -1 = 전체 학습 데이터 사용

KD_ALPHA       = 0.5   # KD loss 비중 (0=CE만, 1=KD만)
KD_TEMPERATURE = 2.0   # logit softening 온도

KD_LORA_RANK    = 16   # KD LoRA rank (0이면 전체 파라미터 학습)
KD_LORA_ALPHA   = 32   # LoRA alpha (통상 rank × 2)
KD_LORA_DROPOUT = 0.05
KD_LORA_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

DATASET_ID    = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"


# ╔══════════════════════════════════════════════════════════════╗
# ║  레시피                                                       ║
# ╚══════════════════════════════════════════════════════════════╝

# Phase 1: QAT용 — group-wise fake-quant (사전 캘리브레이션 없이 weight 텐서에서 스케일 계산)
QAT_RECIPE = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 4
                        type: int
                        strategy: group
                        group_size: 128
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
"""

# Phase 2: GPTQ PTQ용 — Hessian 기반 최적화 (더 정확한 INT4)
PTQ_RECIPE = """
quant_stage:
    quant_modifiers:
        GPTQModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 4
                        type: int
                        strategy: group
                        group_size: 128
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
    ax.plot(steps, losses, color="darkorange", linewidth=1.2, label="train loss")
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


class KDQATTrainer(QATTrainer):
    """QATTrainer + KL divergence Knowledge Distillation"""

    def __init__(self, teacher_model, kd_alpha=0.5, kd_temp=2.0, **kwargs):
        super().__init__(**kwargs)
        self.teacher   = teacher_model
        self.kd_alpha  = kd_alpha
        self.kd_temp   = kd_temp

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs  = model(**inputs)
        ce_loss  = outputs.loss

        with torch.no_grad():
            teacher_out = self.teacher(**inputs)

        T = self.kd_temp
        kd_loss = F.kl_div(
            F.log_softmax(outputs.logits / T, dim=-1),
            F.softmax(teacher_out.logits / T, dim=-1),
            reduction="batchmean",
        ) * (T ** 2)

        loss = (1 - self.kd_alpha) * ce_loss + self.kd_alpha * kd_loss
        return (loss, outputs) if return_outputs else loss


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
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) >= 32)
    print(f"  학습 샘플 수: {len(train_ds)}")
    return train_ds


def main():
    parser = argparse.ArgumentParser(description="INT4 W4A16 Weight-Only Quantization-Aware Training")
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
    parser.add_argument(
        "--calib-samples", type=int, default=PTQ_CALIB_SAMPLES,
        help="캘리브레이션 샘플 수 (-1이면 전체 학습 데이터 사용, 기본: -1)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kd-alpha", type=float, default=KD_ALPHA,
                        help="KD loss 비중 (0.0~1.0, 기본: 0.5)")
    parser.add_argument("--kd-temp", type=float, default=KD_TEMPERATURE,
                        help="KD softening 온도 (기본: 2.0)")
    parser.add_argument("--no-kd", action="store_true",
                        help="KD 비활성화 (CE loss만 사용)")
    parser.add_argument("--kd-lora-rank", type=int, default=KD_LORA_RANK,
                        help="KD LoRA rank (0이면 LoRA 없이 전체 파라미터 학습, 기본: 16)")
    args = parser.parse_args()

    model_name  = os.path.basename(args.model.rstrip("/\\"))
    output_path = args.output or os.path.join("models", f"{model_name}-QAT-INT4")
    trainer_tmp = "/tmp/qat_int4_trainer"
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("INT4 QAT (W4A16) — Phase 1: group fake-quant / Phase 2: GPTQ")
    print("=" * 60)
    print(f"  모델:              {args.model}")
    print(f"  출력:              {output_path}")
    print(f"  [Phase 1] 에폭:    {args.epochs}")
    print(f"  [Phase 1] LR:      {args.lr} (cosine decay, warmup 5%)")
    print(f"  [Phase 1] 배치:    {args.batch_size} × {args.grad_accum} grad_accum")
    print(f"  [Phase 1] 샘플:    {args.samples}")
    print(f"  [Phase 1] 최대길이: {args.max_seq_len}")
    print(f"  [Phase 2] GPTQ 캘리브: {args.calib_samples}개")
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
    print("Phase 1: QAT 학습 (group-wise INT4 fake-quant, W4A16)")
    print("  weight: dynamic=false, group_size=128 (init 시 텐서값으로 스케일 계산)")
    print("  activation: 미적용 (W4A16, BF16 유지)")
    print("=" * 60)

    # Teacher 로드 (session.initialize 전 원본 BF16 상태)
    if not args.no_kd:
        print("  Teacher 모델 로드 (원본 BF16, 동결)")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        print(f"  KD alpha={args.kd_alpha}, temperature={args.kd_temp}\n")
    else:
        teacher_model = None

    # LoRA 적용 (KD 활성 + rank > 0: fake-quant 삽입 전에 어댑터 부착)
    kd_lora_enabled = teacher_model is not None and args.kd_lora_rank > 0
    if kd_lora_enabled:
        LoraConfig, get_peft_model, TaskType = _import_peft()
        print(f"  KD LoRA 적용 (rank={args.kd_lora_rank}, alpha={args.kd_lora_rank * 2})")
        lora_cfg = LoraConfig(
            r=args.kd_lora_rank,
            lora_alpha=args.kd_lora_rank * 2,
            lora_dropout=KD_LORA_DROPOUT,
            target_modules=KD_LORA_MODULES,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        # gradient_checkpointing + LoRA 호환성
        model.enable_input_require_grads()
        print()

    session = active_session()
    session.initialize(recipe=QAT_RECIPE, model=model)
    print("  fake-quant 노드 삽입 완료\n")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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

    TrainerClass = KDQATTrainer if teacher_model is not None else QATTrainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )
    if teacher_model is not None:
        trainer_kwargs.update(
            teacher_model=teacher_model,
            kd_alpha=args.kd_alpha,
            kd_temp=args.kd_temp,
        )
    trainer = TrainerClass(**trainer_kwargs)
    trainer.train()
    plot_loss(trainer.state.log_history, model_name, "INT4", run_timestamp)

    session.finalize()

    # LoRA 병합 → QAT-최적화된 가중치에 LoRA 흡수
    if kd_lora_enabled:
        print("  LoRA 가중치 병합 (merge_and_unload)...")
        model = model.merge_and_unload()

    model.config.use_cache = True
    shutil.rmtree(trainer_tmp, ignore_errors=True)

    # Teacher 삭제 → Phase 2 PTQ 전 VRAM 확보
    if teacher_model is not None:
        del teacher_model
        torch.cuda.empty_cache()

    print("\n  Phase 1 완료: QAT-최적화된 BF16 가중치\n")

    # ── Step 4: Phase 2 — GPTQ 캘리브레이션 + 압축 저장 ─────────
    print("=" * 60)
    print("Phase 2: GPTQ 캘리브레이션 + INT4 압축 저장")
    print("  Hessian 기반 round 최적화 → QAT 가중치에 맞는 정밀 스케일 계산")
    print("=" * 60)

    reset_session()

    if args.calib_samples < 0:
        # -1: 전체 학습 데이터 사용 (셔플 후)
        calib_ds = train_ds.shuffle(seed=args.seed)
        calib_size = len(calib_ds)
    else:
        # 양수: 셔플 후 지정 수만큼 선택 (기존의 range() 편향 버그 수정)
        calib_size = min(args.calib_samples, len(train_ds))
        calib_ds   = train_ds.shuffle(seed=args.seed).select(range(calib_size))
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
        "quant_format": "INT4-W4A16",
        "pipeline": "QAT(group fake-quant) + PTQ(GPTQ calibration)",
        "model": args.model,
        "output": output_path,
        "phase1": {
            "recipe": "INT4 group fake-quant, group_size=128",
            "epochs": args.epochs,
            "lr": args.lr,
            "scheduler": "cosine",
            "warmup_ratio": 0.05,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_seq_len": args.max_seq_len,
            "num_samples": len(train_ds),
            "kd": not args.no_kd,
            "kd_alpha": args.kd_alpha if not args.no_kd else None,
            "kd_temp": args.kd_temp if not args.no_kd else None,
            "kd_lora": kd_lora_enabled,
            "kd_lora_rank": args.kd_lora_rank if kd_lora_enabled else None,
        },
        "phase2": {
            "recipe": "GPTQ",
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
        print(f"\n  원본: {o/1e9:.2f} GB  →  QAT-INT4: {q/1e9:.2f} GB  (압축률 {o/q:.2f}x)")

    print()
    print("=" * 60)
    print(f"완료! INT4 QAT 모델: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

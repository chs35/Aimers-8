"""
QAT FP8 (E4M3, W8A8): Quantization-Aware Training

2단계 파이프라인:
  Phase 1 — QAT: fake-quant 노드 삽입 후 CE loss 학습
    → 가중치가 양자화 오차에 강인해짐 (BF16 상태 유지)
  Phase 2 — PTQ 캘리브레이션: 학습된 BF16 가중치에 정적 스케일 보정 후 압축 저장
    → oneshot()이 save_pretrained를 래핑하여 compressed-tensors 포맷으로 저장

평가 서버 요구사항 (README.md):
  - compressed-tensors==0.13.0  (llmcompressor 없음)
  - vllm==0.14.1
  - 모델 포맷: compressed-tensors (HuggingFace 호환)

사용법:
  python QAT/qat_fp8.py \
      --model models/EXAONE-4.0-1.2B-VocabPruned \
      --dataset dataset/dataset_N/qat_train \
      --epochs 3
"""

import os
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

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

QAT_LR = 2e-5
QAT_EPOCHS = 3
QAT_BATCH_SIZE = 4
QAT_GRAD_ACCUM = 8
NUM_TRAIN_SAMPLES = 2000
MAX_SEQ_LEN = 512

# PTQ 캘리브레이션 샘플 수 (Phase 2)
# 학습 데이터에서 재사용하므로 별도 데이터셋 불필요
PTQ_CALIB_SAMPLES = 512

# ── 기본 데이터셋 (--dataset 미지정 시) ──
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"


# ╔══════════════════════════════════════════════════════════════╗
# ║  FP8 E4M3 W8A8 레시피                                        ║
# ╚══════════════════════════════════════════════════════════════╝

FP8_RECIPE = """
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


class QATTrainer(Trainer):
    """llmcompressor lifecycle callbacks를 HuggingFace Trainer에 통합"""

    def training_step(self, model, inputs, num_items_in_batch=None):
        LC.batch_start(batch_data=inputs)
        loss = super().training_step(model, inputs, num_items_in_batch)
        LC.batch_end()
        return loss


def load_train_dataset(tokenizer, dataset_path, num_samples, max_seq_len, seed):
    """학습 데이터셋 로드 + 전처리"""
    if dataset_path:
        print(f"  로컬 데이터셋 사용: {dataset_path}")
        train_ds = load_from_disk(dataset_path)
    else:
        print(f"  기본 데이터셋 사용: {DATASET_ID}")
        raw_ds = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)
        raw_ds = raw_ds.shuffle(seed=seed, buffer_size=5000)
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

    train_ds = train_ds.map(preprocess)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )

    train_ds = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
    print(f"  학습 샘플 수: {len(train_ds)}")
    return train_ds


def main():
    parser = argparse.ArgumentParser(description="FP8 E4M3 W8A8 Quantization-Aware Training")
    parser.add_argument("--model", "-m", type=str,
                        default="models/EXAONE-4.0-1.2B-VocabPruned",
                        help="입력 모델 경로 (Vocab-Pruned 모델)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="출력 경로 (기본: models/<name>-QAT-FP8)")
    parser.add_argument("--dataset", "-d", type=str, default=None,
                        help="로컬 데이터셋 경로 (prepare_dataset.py의 qat_train)")
    parser.add_argument("--epochs", type=int, default=QAT_EPOCHS,
                        help=f"학습 에폭 수 (기본값: {QAT_EPOCHS})")
    parser.add_argument("--lr", type=float, default=QAT_LR,
                        help=f"학습률 (기본값: {QAT_LR})")
    parser.add_argument("--batch-size", type=int, default=QAT_BATCH_SIZE,
                        help=f"배치 크기 (기본값: {QAT_BATCH_SIZE})")
    parser.add_argument("--grad-accum", type=int, default=QAT_GRAD_ACCUM,
                        help=f"그래디언트 누적 스텝 (기본값: {QAT_GRAD_ACCUM})")
    parser.add_argument("--samples", type=int, default=NUM_TRAIN_SAMPLES,
                        help=f"기본 데이터셋 사용 시 학습 샘플 수 (기본값: {NUM_TRAIN_SAMPLES})")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN,
                        help=f"최대 시퀀스 길이 (기본값: {MAX_SEQ_LEN})")
    parser.add_argument("--calib-samples", type=int, default=PTQ_CALIB_SAMPLES,
                        help=f"PTQ 캘리브레이션 샘플 수 (기본값: {PTQ_CALIB_SAMPLES})")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드 (기본값: 42)")
    args = parser.parse_args()

    model_name = os.path.basename(args.model.rstrip("/\\"))
    output_path = args.output or os.path.join("models", f"{model_name}-QAT-FP8")
    trainer_tmp_dir = "/tmp/qat_fp8_trainer"

    print("=" * 60)
    print("FP8 QAT (W8A8, E4M3) — 2단계 파이프라인")
    print("=" * 60)
    print(f"  모델:              {args.model}")
    print(f"  출력:              {output_path}")
    print(f"  [Phase 1] 에폭:    {args.epochs}")
    print(f"  [Phase 1] LR:      {args.lr}")
    print(f"  [Phase 1] 배치:    {args.batch_size} × {args.grad_accum} grad_accum")
    print(f"  [Phase 2] 캘리브:  {args.calib_samples}개")
    print(f"  최대 길이:         {args.max_seq_len}")
    print(f"  시드:              {args.seed}")
    print()

    # ===========================================================
    # Step 1: 모델 + 토크나이저 로드
    # ===========================================================
    print("=" * 60)
    print("Step 1: 모델 + 토크나이저 로드")
    print("=" * 60)

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

    # ===========================================================
    # Step 2: 데이터셋 로드 + 전처리
    # ===========================================================
    print("=" * 60)
    print("Step 2: 학습 데이터 준비")
    print("=" * 60)

    train_ds = load_train_dataset(
        tokenizer=tokenizer,
        dataset_path=args.dataset,
        num_samples=args.samples,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    print()

    # ===========================================================
    # Step 3: Phase 1 — QAT 학습 (fake-quant, BF16 유지)
    # ===========================================================
    print("=" * 60)
    print("Step 3: Phase 1 — QAT 학습 (fake-quant 삽입 후 CE loss)")
    print("=" * 60)

    # 3-a: fake-quant 노드 삽입
    print("  [3-a] session.initialize() — fake-quant 노드 삽입...")
    session = active_session()
    session.initialize(recipe=FP8_RECIPE, model=model)
    print("  완료")

    # 3-b: CE loss 학습
    print("  [3-b] QATTrainer 학습 시작...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=trainer_tmp_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = QATTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )
    trainer.train()

    # 3-c: fake-quant 노드 제거 (가중치는 QAT-최적화된 BF16)
    print("  [3-c] session.finalize() — fake-quant 노드 제거...")
    session.finalize()
    model.config.use_cache = True  # 추론 모드 복원

    # 임시 파일 정리
    shutil.rmtree(trainer_tmp_dir, ignore_errors=True)
    print()

    # ===========================================================
    # Step 4: Phase 2 — PTQ 캘리브레이션 + 압축 저장
    #
    # session.finalize() 후 모델은 QAT-최적화 BF16 상태.
    # oneshot()이 실제 FP8 양자화(static scale 계산 + 가중치 압축)를 수행하고,
    # save_pretrained를 래핑하여 compressed-tensors 포맷으로 저장.
    # → 평가 서버의 compressed-tensors==0.13.0 + vLLM==0.14.1에서 로드 가능.
    # ===========================================================
    print("=" * 60)
    print("Step 4: Phase 2 — PTQ 캘리브레이션 + FP8 압축 저장")
    print("=" * 60)

    reset_session()

    calib_size = min(args.calib_samples, len(train_ds))
    calib_ds = train_ds.select(range(calib_size))
    print(f"  캘리브레이션 샘플: {calib_size}개 (학습 데이터에서 재사용)")

    os.makedirs(output_path, exist_ok=True)
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=FP8_RECIPE,
        save_compressed=True,
        output_dir=output_path,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=calib_size,
    )

    # ===========================================================
    # Step 5: 토크나이저 + 메타데이터 저장
    # ===========================================================
    print("=" * 60)
    print("Step 5: 저장")
    print("=" * 60)

    tokenizer.save_pretrained(output_path)
    print(f"  토크나이저 저장: {output_path}")

    metadata = {
        "created_at": datetime.now().isoformat(),
        "quant_format": "FP8-E4M3-W8A8",
        "pipeline": "QAT(fake-quant) + PTQ(calibration)",
        "model": args.model,
        "output": output_path,
        "phase1_hyperparameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
            "num_train_samples": len(train_ds),
        },
        "phase2_hyperparameters": {
            "calib_samples": calib_size,
            "max_seq_len": args.max_seq_len,
        },
        "dataset": args.dataset or DATASET_ID,
    }
    meta_path = os.path.join(output_path, "qat_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  메타데이터 저장: {meta_path}")

    orig_safetensors = list(Path(args.model).glob("*.safetensors"))
    qat_safetensors = list(Path(output_path).glob("*.safetensors"))
    if orig_safetensors and qat_safetensors:
        original_size = sum(f.stat().st_size for f in orig_safetensors)
        qat_size = sum(f.stat().st_size for f in qat_safetensors)
        print(f"  원본 크기: {original_size / 1e9:.2f} GB")
        print(f"  QAT 크기:  {qat_size / 1e9:.2f} GB")
        print(f"  압축률:    {original_size / qat_size:.2f}x")

    print()
    print("=" * 60)
    print(f"완료! FP8 QAT 모델 저장 위치: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

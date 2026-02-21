"""
QAT INT4 (W4A16, 가중치만): Quantization-Aware Training

llmcompressor의 train()을 사용해 fake-quant 노드를 삽입한 상태로
CE loss 학습 후 실제 INT4 포맷으로 저장한다.
activation은 양자화하지 않음 (W4A16: 가중치 4bit, activation BF16).

파이프라인:
  Step 1: Vocab-Pruned 모델 + 토크나이저 로드 (bf16)
  Step 2: qat_train 데이터셋 로드 + 전처리
  Step 3: llmcompressor.train() 호출 (recipe 적용 + CE 학습)
  Step 4: tokenizer 저장 + qat_metadata.json 저장

사용법:
  python QAT/qat_int4.py \
      --model models/EXAONE-4.0-1.2B-VocabPruned \
      --dataset dataset/dataset_N/qat_train \
      --epochs 3
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from llmcompressor import train


# ╔══════════════════════════════════════════════════════════════╗
# ║  HYPERPARAMETERS — 이 섹션만 수정하면 됩니다                  ║
# ╚══════════════════════════════════════════════════════════════╝

QAT_LR = 2e-5
QAT_EPOCHS = 3
QAT_BATCH_SIZE = 4
QAT_GRAD_ACCUM = 8
NUM_TRAIN_SAMPLES = 2000
MAX_SEQ_LEN = 512

# ── 기본 데이터셋 (--dataset 미지정 시) ──
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"


# ╔══════════════════════════════════════════════════════════════╗
# ║  INT4 W4A16 레시피 (가중치만 양자화)                          ║
# ╚══════════════════════════════════════════════════════════════╝

INT4_QAT_RECIPE = """
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
                        symmetric: true
                    targets: ["Linear"]
"""


def load_train_dataset(tokenizer, dataset_path, num_samples, max_seq_len, seed):
    """학습 데이터셋 로드 + 전처리 (KD_Recovery.py 패턴 재사용)"""
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
    parser = argparse.ArgumentParser(description="INT4 W4A16 Weight-Only Quantization-Aware Training")
    parser.add_argument("--model", "-m", type=str,
                        default="models/EXAONE-4.0-1.2B-VocabPruned",
                        help="입력 모델 경로 (Vocab-Pruned 모델)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="출력 경로 (기본: models/<name>-QAT-INT4)")
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
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드 (기본값: 42)")
    args = parser.parse_args()

    # 출력 경로 결정
    model_name = os.path.basename(args.model.rstrip("/\\"))
    output_path = args.output or os.path.join("models", f"{model_name}-QAT-INT4")

    print("=" * 60)
    print("INT4 QAT (W4A16, 가중치만)")
    print("=" * 60)
    print(f"  모델:        {args.model}")
    print(f"  출력:        {output_path}")
    print(f"  에폭:        {args.epochs}")
    print(f"  학습률:      {args.lr}")
    print(f"  배치 크기:   {args.batch_size}")
    print(f"  그래디언트:  {args.grad_accum} 스텝 누적")
    print(f"  최대 길이:   {args.max_seq_len}")
    print(f"  시드:        {args.seed}")
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  파라미터: {total_params:,}")
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

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ===========================================================
    # Step 3: QAT 학습 (llmcompressor.train)
    # ===========================================================
    print("=" * 60)
    print("Step 3: INT4 QAT 학습")
    print("=" * 60)

    model.config.use_cache = False

    os.makedirs(output_path, exist_ok=True)

    train(
        model=model,
        tokenizer=tokenizer,
        dataset=train_ds,
        recipe=INT4_QAT_RECIPE,
        data_collator=data_collator,
        output_dir=output_path,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        save_compressed=True,
    )

    # ===========================================================
    # Step 4: 토크나이저 + 메타데이터 저장
    # ===========================================================
    print("=" * 60)
    print("Step 4: 저장")
    print("=" * 60)

    tokenizer.save_pretrained(output_path)
    print(f"  토크나이저 저장: {output_path}")

    metadata = {
        "created_at": datetime.now().isoformat(),
        "quant_format": "INT4-W4A16",
        "model": args.model,
        "output": output_path,
        "hyperparameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
        },
        "dataset": args.dataset or DATASET_ID,
        "num_train_samples": len(train_ds),
    }
    meta_path = os.path.join(output_path, "qat_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  메타데이터 저장: {meta_path}")

    # 모델 크기 비교
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
    print(f"완료! INT4 QAT 모델 저장 위치: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

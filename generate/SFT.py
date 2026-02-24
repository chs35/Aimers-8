"""
SFT: 단순 LoRA Supervised Fine-Tuning

파이프라인:
  1. 로컬 Arrow 포맷 데이터셋 로드 (prepare_dataset.py 출력 형식)
  2. chat template 적용 → tokenize
  3. LoRA 적용 (get_peft_model)
  4. HF Trainer로 학습
  5. LoRA 병합 (merge_and_unload) → 저장

사용법:
  python generate/SFT.py --dataset dataset/dataset_1/kd_train --rank 16

기본 출력 이름:
  models/<모델명>-SFT

GPU 호환: L4(sm_89) 이상 모든 GPU에서 동작
필요 패키지: pip install peft  (Dockerfile에 미포함)
"""

import os
import json
import shutil
import argparse
import subprocess
from datetime import datetime
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    print("peft 패키지가 없습니다. 설치 중...")
    subprocess.check_call(["pip", "install", "peft", "-q"])
    from peft import LoraConfig, get_peft_model, TaskType

# ╔══════════════════════════════════════════════════════════════╗
# ║  HYPERPARAMETERS — 이 섹션만 수정하면 됩니다                  ║
# ╚══════════════════════════════════════════════════════════════╝

# ── 모델 ──
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"

# ── LoRA ──
LORA_RANK    = 16
LORA_ALPHA   = 32          # rank * 2 (관례)
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── 학습 ──
LR          = 2e-4
EPOCHS      = 1
BATCH_SIZE  = 4
GRAD_ACCUM  = 8
MAX_SEQ_LEN = 512
SEED        = 42

# ── 환경 ──
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE                                              ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":

    # ===========================================================
    # CLI Arguments
    # ===========================================================
    parser = argparse.ArgumentParser(description="LoRA SFT (Supervised Fine-Tuning)")
    parser.add_argument("--dataset", "-d", type=str, required=True,
                        help="학습 데이터 폴더 (HF Arrow 포맷, prepare_dataset.py 출력)")
    parser.add_argument("--model", "-m", type=str, default=MODEL_ID,
                        help=f"모델 경로 (HuggingFace ID 또는 로컬 경로, 기본값: {MODEL_ID})")
    parser.add_argument("--rank", type=int, default=LORA_RANK,
                        help=f"LoRA rank (기본값: {LORA_RANK})")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="저장 경로 (기본값: models/<모델명>-SFT)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"학습 epoch 수 (기본값: {EPOCHS})")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"학습률 (기본값: {LR})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"per-device 배치 크기 (기본값: {BATCH_SIZE})")
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM,
                        help=f"gradient accumulation steps (기본값: {GRAD_ACCUM})")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN,
                        help=f"최대 시퀀스 길이 (기본값: {MAX_SEQ_LEN})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"랜덤 시드 (기본값: {SEED})")
    args = parser.parse_args()

    # 모델 경로 해석
    model_id = args.model
    model_name = os.path.basename(model_id.rstrip("/\\"))
    if os.path.isdir(model_id):
        model_path = model_id
    else:
        model_path = os.path.join("models", model_name)

    output_dir = args.output or os.path.join("models", f"{model_name}-SFT")

    # ===========================================================
    # Step 1: 모델 및 토크나이저 로드
    # ===========================================================
    print("=" * 60)
    print("Step 1: 모델 및 토크나이저 로드")
    print("=" * 60)

    if os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, "config.json")):
        print(f"  로컬 모델 사용: {model_path}")
    else:
        os.makedirs(model_path, exist_ok=True)
        print(f"  모델 다운로드: {model_id} → {model_path}")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, local_dir=model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  모델: {model_id}")
    print(f"  총 파라미터: {total_params:,}")
    print()

    # ===========================================================
    # Step 2: 데이터셋 로드 및 전처리
    # ===========================================================
    print("=" * 60)
    print("Step 2: 데이터셋 로드 및 전처리")
    print("=" * 60)
    print(f"  데이터셋 경로: {args.dataset}")

    ds = load_from_disk(args.dataset)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=True,
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
    print(f"  학습 샘플 수: {len(ds)}")
    print()

    # ===========================================================
    # Step 3: LoRA 적용
    # ===========================================================
    print("=" * 60)
    print("Step 3: LoRA 적용")
    print("=" * 60)

    lora_alpha = args.rank * 2
    print(f"  rank={args.rank}, alpha={lora_alpha}, modules={LORA_TARGET_MODULES}")

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.config.use_cache = False
    model.enable_input_require_grads()
    print()

    # ===========================================================
    # Step 4: 학습
    # ===========================================================
    print("=" * 60)
    print("Step 4: LoRA SFT 학습")
    print("=" * 60)
    print(f"  epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}, max_seq_len={args.max_seq_len}")

    training_args = TrainingArguments(
        output_dir="./sft_lora_tmp",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    print("  학습 완료!")
    print()

    # ===========================================================
    # Step 5: LoRA 병합 및 저장
    # ===========================================================
    print("=" * 60)
    print("Step 5: LoRA 병합 및 저장")
    print("=" * 60)

    print("  LoRA 가중치 병합 (merge_and_unload)...")
    model = model.merge_and_unload()
    model.config.use_cache = True

    shutil.rmtree("./sft_lora_tmp", ignore_errors=True)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "method": "LoRA SFT (Supervised Fine-Tuning)",
        "model": model_id,
        "dataset": args.dataset,
        "lora_rank": args.rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "train_samples": len(ds),
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  모델 저장 완료: {output_dir}/")
    print(f"  메타데이터: {meta_path}")

    # ===========================================================
    # 최종 요약
    # ===========================================================
    print()
    print("=" * 60)
    print("최종 요약")
    print("=" * 60)
    print(f"  모델:      {model_id}")
    print(f"  데이터셋:  {args.dataset} ({len(ds)}개 샘플)")
    print(f"  LoRA:      rank={args.rank}, alpha={lora_alpha}")
    print(f"  학습:      {args.epochs} epochs, lr={args.lr}")
    print(f"  저장 위치: {output_dir}/")

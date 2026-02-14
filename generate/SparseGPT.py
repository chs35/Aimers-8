#!/usr/bin/env python3
"""
SparseGPT Pruning Script - Optimized for Current Environment

SparseGPT 기법의 핵심:
- Layer-wise unstructured pruning
- Calibration data를 사용한 Hessian 기반 weight 제거
- 모델 성능 유지하면서 파라미터 감소
"""

import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier

# ============================================================
# 1. 설정값 정의
# ============================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
LOCAL_MODEL_DIR = os.path.join("models", MODEL_ID.split("/")[-1])
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# Calibration 설정 (성능에 큰 영향)
NUM_CALIBRATION_SAMPLES = 512  # 256 -> 512로 증가
MAX_SEQUENCE_LENGTH = 2048     # 512 -> 2048로 증가 (더 긴 컨텍스트)

# Pruning 설정
SPARSITY = 0.5  # 50% unstructured sparsity
SAVE_DIR = "models/EXAONE-4.0-1.2B-SparseGPT-v2"

# 환경 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=" * 60)
print("SparseGPT Pruning - Optimized Version")
print("=" * 60)
print(f"Model: {MODEL_ID}")
print(f"Target Sparsity: {SPARSITY * 100}%")
print(f"Calibration Samples: {NUM_CALIBRATION_SAMPLES}")
print(f"Max Sequence Length: {MAX_SEQUENCE_LENGTH}")
print("=" * 60)

# ============================================================
# 2. 모델 및 토크나이저 불러오기
# ============================================================
print("\n[1/5] 모델 및 토크나이저 불러오기...")
if os.path.isdir(LOCAL_MODEL_DIR) and os.path.isfile(os.path.join(LOCAL_MODEL_DIR, "config.json")):
    model_path = LOCAL_MODEL_DIR
    print(f"  로컬 모델 사용: {LOCAL_MODEL_DIR}")
else:
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    print(f"  모델 다운로드: {MODEL_ID} → {LOCAL_MODEL_DIR}")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_MODEL_DIR)
    model_path = LOCAL_MODEL_DIR
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

# Tokenizer에 pad_token 설정 (없으면 eos_token 사용)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✓ 모델 로딩 완료")
print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================
# 3. Calibration 데이터셋 준비
# ============================================================
print(f"\n[2/5] Calibration 데이터셋 준비...")
print(f"  - Dataset: {DATASET_ID}")
print(f"  - Loading {NUM_CALIBRATION_SAMPLES} samples...")

# 스트리밍 방식으로 데이터 로드
raw_ds_iter = load_dataset(
    DATASET_ID,
    split=DATASET_SPLIT,
    streaming=True
)
sampled_data = list(raw_ds_iter.take(NUM_CALIBRATION_SAMPLES))
ds = Dataset.from_list(sampled_data)

print(f"✓ {len(sampled_data)} 샘플 로드 완료")

# 데이터 전처리 함수
def preprocess(example):
    """
    대화를 텍스트로 변환
    - 채팅 템플릿 적용
    - add_generation_prompt=False (calibration에는 불필요)
    """
    text = tokenizer.apply_chat_template(
        example["conversations"],
        add_generation_prompt=False,  # Calibration에는 generation prompt 불필요
        tokenize=False
    )
    return {"text": text}

# 전처리 적용
print("  - Preprocessing samples...")
ds = ds.map(preprocess, remove_columns=ds.column_names)

# 샘플 통계 확인
sample_lengths = []
for sample in ds.select(range(min(10, len(ds)))):
    tokens = tokenizer(sample["text"], truncation=False)
    sample_lengths.append(len(tokens["input_ids"]))

print(f"✓ 전처리 완료")
print(f"  - Average length (first 10 samples): {sum(sample_lengths) / len(sample_lengths):.0f} tokens")
print(f"  - Max length will be: {MAX_SEQUENCE_LENGTH} tokens")

# ============================================================
# 4. SparseGPT Recipe 설정
# ============================================================
print(f"\n[3/5] SparseGPT Recipe 설정...")

recipe = [
    SparseGPTModifier(
        sparsity=SPARSITY,
        targets="Linear",  # 모든 Linear 레이어 대상
        ignore=["lm_head"],  # lm_head는 제외 (출력 레이어)
        sequential_update=True,  # Layer-wise sequential pruning
        dampening_frac=0.01,  # Hessian 안정화
        block_size=128,  # 블록 단위 처리
    )
]

print(f"✓ Recipe 설정 완료")
print(f"  - Sparsity: {SPARSITY * 100}%")
print(f"  - Targets: All Linear layers (except lm_head)")
print(f"  - Sequential update: True (layer-wise)")
print(f"  - Dampening: 0.01")
print(f"  - Block size: 128")

# ============================================================
# 5. Pruning 실행
# ============================================================
print(f"\n[4/5] SparseGPT Pruning 실행...")
print("  (이 작업은 수 분이 소요될 수 있습니다)")

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("✓ Pruning 완료")

# Pruning 후 통계
total_params = sum(p.numel() for p in model.parameters())
zero_params = sum((p == 0).sum().item() for p in model.parameters())
actual_sparsity = zero_params / total_params

print(f"  - Total parameters: {total_params:,}")
print(f"  - Zero parameters: {zero_params:,}")
print(f"  - Actual sparsity: {actual_sparsity * 100:.2f}%")

# ============================================================
# 6. 모델 저장
# ============================================================
print(f"\n[5/5] 모델 저장...")

# 모델 저장 (vLLM 호환 형식)
model.save_pretrained(SAVE_DIR, save_compressed=False)
tokenizer.save_pretrained(SAVE_DIR)

# vLLM 호환성: quantization_config 제거
config_path = os.path.join(SAVE_DIR, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

if "quantization_config" in config:
    print("  - Removing quantization_config for vLLM compatibility...")
    del config["quantization_config"]
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

# 메타데이터 저장
metadata = {
    "model_id": MODEL_ID,
    "method": "SparseGPT",
    "target_sparsity": SPARSITY,
    "actual_sparsity": actual_sparsity,
    "calibration_samples": NUM_CALIBRATION_SAMPLES,
    "max_seq_length": MAX_SEQUENCE_LENGTH,
    "dataset": DATASET_ID,
    "total_parameters": total_params,
    "zero_parameters": zero_params,
}

with open(os.path.join(SAVE_DIR, "pruning_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✓ 모델 저장 완료: {SAVE_DIR}")
print(f"  - Model files saved")
print(f"  - Metadata saved to pruning_metadata.json")

print("\n" + "=" * 60)
print("✅ SparseGPT Pruning 완료!")
print("=" * 60)
print(f"저장 위치: {SAVE_DIR}")
print(f"실제 Sparsity: {actual_sparsity * 100:.2f}%")
print(f"파라미터 감소: {zero_params:,} / {total_params:,}")
print("=" * 60)

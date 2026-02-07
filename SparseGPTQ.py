"""
SparseGPTQ: Unstructured Pruning (SparseGPT) + Mixed-Precision Quantization (GPTQ)

전략:
  SparseGPT로 불필요한 가중치를 0으로 만든 뒤, GPTQ가 나머지 가중치를 더 정밀하게 양자화.
  → 프루닝이 양자화 품질을 개선하는 시너지 효과 (OPQ 논문 참고)

GPU 호환성:
  - 2:4 structured sparsity + quantization → sm_90 (Hopper/H100) 이상 필요
  - 본 스크립트는 unstructured sparsity 사용 → L4(sm_89) 등 모든 GPU에서 동작
  - 저장 시 sparsity 메타데이터를 제거하여 vLLM이 순수 GPTQ 모델로 인식

참고 논문:
  - SparseGPT (arXiv 2301.00774): One-shot unstructured pruning
  - GPTQ (arXiv 2210.17323): One-shot post-training quantization
  - OPQ (arXiv 2205.11141): Simultaneous pruning + quantization in one shot
"""

import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
from llmcompressor.modifiers.quantization import GPTQModifier

# ============================================================
# 1. 설정값 정의
# ============================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# 프루닝 설정
# 비구조적 프루닝: 가중치의 SPARSITY%를 0으로 설정
# 2:4 구조(mask_structure="2:4")는 sm_90+ GPU에서만 vLLM 가속 가능하므로 사용하지 않음
SPARSITY = 0.5  # 50% unstructured sparsity

# 환경 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# 2. 모델 및 토크나이저 불러오기
# ============================================================
print("--- 1. 모델 및 토크나이저 불러오기 ---")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# ============================================================
# 3. 데이터셋 준비 (메모리 최적화 및 객체 복구)
# ============================================================
print("--- 2. 데이터셋 준비 (메모리 최적화 및 객체 복구) ---")
raw_ds_iter = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)
sampled_data = list(raw_ds_iter.take(NUM_CALIBRATION_SAMPLES))

ds = Dataset.from_list(sampled_data)

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False,
        )
    }

ds = ds.map(preprocess)

# ============================================================
# 4. 레시피 설정: SparseGPT (Unstructured Pruning) + GPTQ (Mixed-Precision)
# ============================================================
print(f"--- 3. 레시피 설정: SparseGPT ({SPARSITY*100:.0f}% unstructured) + GPTQ (Layer-wise Mixed Precision) ---")

# EXAONE-4.0-1.2B: 30 레이어 (0~29)
layer_0_4 = [f"re:model.layers.{i}\\..*_proj$" for i in range(0, 5)]
layer_5_24 = [f"re:model.layers.{i}\\..*_proj$" for i in range(5, 25)]
layer_25_29 = [f"re:model.layers.{i}\\..*_proj$" for i in range(25, 30)]

recipe = [
    # ── Step 1: Unstructured Pruning via SparseGPT ──
    # 불필요한 가중치를 0으로 만들어 GPTQ 양자화 품질 개선
    # mask_structure 미지정 → unstructured (모든 GPU에서 vLLM 호환)
    SparseGPTModifier(
        sparsity=SPARSITY,
        targets="Linear",
        ignore=["lm_head"],
    ),

    # ── Step 2: Layer-wise Mixed Precision GPTQ Quantization ──
    # 초반/후반 레이어(민감): W8A16, 중간 레이어(중복 많음): W4A16

    # 초반 레이어 (0-4): W8A16
    GPTQModifier(
        targets=layer_0_4,
        scheme="W8A16",
        ignore=["embed_tokens", "lm_head"],
    ),

    # 중간 레이어 (5-24): W4A16
    GPTQModifier(
        targets=layer_5_24,
        scheme="W4A16",
        ignore=["embed_tokens", "lm_head"],
    ),

    # 후반 레이어 (25-29): W8A16
    GPTQModifier(
        targets=layer_25_29,
        scheme="W8A16",
        ignore=["embed_tokens", "lm_head"],
    ),
]

# ============================================================
# 5. 압축 실행 (Oneshot)
# ============================================================
print("--- 4. 압축 실행 (SparseGPT Pruning + GPTQ Quantization Oneshot) ---")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# ============================================================
# 6. 결과 저장
# ============================================================
print("--- 5. 결과 저장 ---")
SAVE_DIR = "EXAONE-4.0-1.2B-Sparse-GPTQ"

# .gitignore에 SAVE_DIR 추가 (없는 경우)
GITIGNORE_PATH = ".gitignore"
if os.path.exists(GITIGNORE_PATH):
    with open(GITIGNORE_PATH, "r") as f:
        gitignore_content = f.read()

    if SAVE_DIR not in gitignore_content:
        with open(GITIGNORE_PATH, "a") as f:
            f.write(f"\n{SAVE_DIR}\n")
        print(f"'{SAVE_DIR}'을(를) .gitignore에 추가했습니다.")
    else:
        print(f"'{SAVE_DIR}'은(는) 이미 .gitignore에 있습니다.")
else:
    with open(GITIGNORE_PATH, "w") as f:
        f.write(f"{SAVE_DIR}\n")
    print(f".gitignore 파일을 생성하고 '{SAVE_DIR}'을(를) 추가했습니다.")

model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

# ============================================================
# 7. vLLM 호환성을 위해 config.json에서 sparsity 정보 제거
# ============================================================
# compressed-tensors 포맷에 sparsity_config가 포함되면
# vLLM이 sparse+quantized 커널(sm_90+)을 사용하려 하여 L4 등에서 에러 발생
# → sparsity_config를 제거하면 vLLM이 순수 GPTQ 모델로 인식
print("--- 6. vLLM 호환성을 위해 config.json에서 sparsity 정보 제거 ---")
config_path = os.path.join(SAVE_DIR, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

if "quantization_config" in config:
    qconfig = config["quantization_config"]

    # sparsity_config 제거
    if "sparsity_config" in qconfig:
        del qconfig["sparsity_config"]
        print("  sparsity_config 제거 완료")

    # 중첩된 quantization_config 내부도 확인
    if "quantization_config" in qconfig and isinstance(qconfig["quantization_config"], dict):
        inner = qconfig["quantization_config"]
        if "sparsity_config" in inner:
            del inner["sparsity_config"]
            print("  inner sparsity_config 제거 완료")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print("  config.json 업데이트 완료 (순수 GPTQ 모델로 인식됨)")

print(f"\n완료! Sparse + Quantized 모델이 '{SAVE_DIR}'에 저장되었습니다.")
print(f"  - Pruning: SparseGPT {SPARSITY*100:.0f}% unstructured (양자화 품질 개선용)")
print(f"  - Quantization: Layer 0-4 W8A16, Layer 5-24 W4A16, Layer 25-29 W8A16")
print(f"  - vLLM 호환: sparsity 메타 제거 → 순수 GPTQ로 로드 (L4/A100 등 모든 GPU)")

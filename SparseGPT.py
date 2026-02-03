import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier

# 1. 설정값 정의
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# 프루닝 설정
SPARSITY = 0.5  # 50% unstructured sparsity

# 환경 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("--- 1. 모델 및 토크나이저 불러오기 ---")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print("--- 2. 데이터셋 준비 (메모리 최적화 및 객체 복구) ---")
# 스트리밍으로 데이터를 가져와서 리스트에 담습니다.
raw_ds_iter = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)
sampled_data = list(raw_ds_iter.take(NUM_CALIBRATION_SAMPLES))

# 리스트를 다시 HuggingFace Dataset 객체로 변환하여 column_names 속성을 살려줍니다.
ds = Dataset.from_list(sampled_data)

def preprocess(example):
    # 채팅 템플릿 적용
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False
        )
    }

# 전처리 적용
ds = ds.map(preprocess)

print(f"--- 3. 프루닝(SparseGPT) 레시피 설정 (Sparsity: {SPARSITY}) ---")

recipe = [
    SparseGPTModifier(
        sparsity=SPARSITY,
        targets="Linear",
        ignore=["lm_head"],
    )
]

print("--- 4. 프루닝 실행 (Oneshot) ---")
# SparseGPT 알고리즘 가동
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("--- 5. 결과 저장 ---")
# 프루닝된 모델 저장
SAVE_DIR = "EXAONE-4.0-1.2B-SparseGPT"

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

model.save_pretrained(SAVE_DIR, save_compressed=False)
tokenizer.save_pretrained(SAVE_DIR)

# vLLM 호환성을 위해 quantization_config 제거
config_path = os.path.join(SAVE_DIR, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

if "quantization_config" in config:
    print("vLLM 호환성을 위해 quantization_config를 제거합니다...")
    del config["quantization_config"]
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("quantization_config가 제거되었습니다.")

print(f"축하합니다! 프루닝 모델이 '{SAVE_DIR}'에 저장되었습니다.")
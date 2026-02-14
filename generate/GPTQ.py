import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# 1. 설정값 정의
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
LOCAL_MODEL_DIR = os.path.join("models", MODEL_ID.split("/")[-1])
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# 환경 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("--- 1. 모델 및 토크나이저 불러오기 ---")
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
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

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

print("--- 3. 양자화(GPTQ) 레시피 설정 (Layer-wise Mixed Precision) ---")

# 레이어 인덱스를 문자열 정규표현식으로 변환하여 리스트 생성
# Linear projection 레이어만 타겟팅 (gate_proj, up_proj, down_proj, q_proj, k_proj, v_proj, o_proj)
layer_0_4 = [f"re:model.layers.{i}\..*_proj$" for i in range(0, 5)]
layer_5_24 = [f"re:model.layers.{i}\..*_proj$" for i in range(5, 25)]
layer_25_29 = [f"re:model.layers.{i}\..*_proj$" for i in range(25, 30)]

recipe = [
    # 1. 초반 레이어 (0-4): W8A16
    GPTQModifier(
        targets=layer_0_4,
        scheme="W8A16",
        ignore=["embed_tokens", "lm_head"],
    ),

    # 2. 중간 레이어 (5-24): W4A16
    GPTQModifier(
        targets=layer_5_24,
        scheme="W4A16",
        ignore=["embed_tokens", "lm_head"],
    ),

    # 3. 후반 레이어 (25-29): W8A16
    GPTQModifier(
        targets=layer_25_29,
        scheme="W8A16",
        ignore=["embed_tokens", "lm_head"],
    )
]

print("--- 4. 양자화 실행 (Oneshot) ---")
# GPTQ 알고리즘 가동
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("--- 5. 결과 저장 ---")
# 압축된 상태로 저장
SAVE_DIR = "EXAONE-4.0-1.2B-GPTQ-RECIPE1"

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

print(f"축하합니다! 경량화 모델이 '{SAVE_DIR}'에 저장되었습니다.")
print(f"참고: GPTQ 모델은 quantization_config가 포함되어 저장됩니다.")
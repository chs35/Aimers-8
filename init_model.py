import os

# hf_transfer 활성화 (빠른 다운로드)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams

MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
MODELS_DIR = "models"
MODEL_NAME = MODEL_ID.split("/")[-1]
LOCAL_DIR = os.path.join(MODELS_DIR, MODEL_NAME)

# 모델 다운로드
if not os.path.exists(LOCAL_DIR):
    print(f"Downloading {MODEL_ID} to {LOCAL_DIR}...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"Download complete: {LOCAL_DIR}")
else:
    print(f"Model already exists: {LOCAL_DIR}")
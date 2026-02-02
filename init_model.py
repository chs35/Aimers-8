import os

# hf_transfer 활성화 (빠른 다운로드)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams

MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
LOCAL_DIR = "/workspace/EXAONE-4.0-1.2B"

# 1. 모델 다운로드
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

# 2. 모델 로드
print("Loading model with vLLM...")
llm = LLM(
    model=LOCAL_DIR,
    dtype="auto",
    trust_remote_code=True,
    enforce_eager=True,  # torch.compile 비활성화 (segfault 방지)
    max_model_len=4096,  # 메모리 절약
)

# 3. 테스트 추론
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

prompts = [
    "대한민국의 수도는",
    "What is machine learning?",
    "1 + 1 = ",
]

print("\n" + "=" * 50)
print("Test Inference")
print("=" * 50)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    print(f"\nPrompt: {prompt}")
    print(f"Output: {generated}")

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_ROOT = PROJECT_ROOT / "models"
DEFAULT_PROMPT_CONFIG = PROJECT_ROOT / "src" / "kolongrag" / "config" / "prompt.yaml"

# README 고정 정책 (수정 불가)
DATASET_NAME = "LGAI-EXAONE/Ko-LongRAG"
DATASET_SPLIT = "test"
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.85
VLLM_BATCH_SIZE = "auto"
VLLM_MAX_GEN_TOKS = 16384
VLLM_APPLY_CHAT_TEMPLATE = True
TOKENIZER_LOCAL_FILES_ONLY = True

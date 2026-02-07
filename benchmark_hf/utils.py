"""vLLM 기본 설정 상수 + 텍스트 유틸리티"""

import re


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 대회 평가 서버와 동일한 vLLM 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VLLM_DEFAULTS = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.85,
    "trust_remote_code": True,
    "max_gen_toks": 16384,
}


def strip_thinking(text: str) -> str:
    """<think>...</think> 블록 제거 (enable_thinking 사용 시)"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

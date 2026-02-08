from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from src.kolongrag.config.options import (
    TOKENIZER_LOCAL_FILES_ONLY,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_MAX_GEN_TOKS,
    VLLM_TENSOR_PARALLEL_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass
class RuntimeComponents:
    tokenizer: Any
    llm: Any
    sampling_params: Any


def load_runtime(
    model_path: str,
    temperature: float,
    top_p: float,
) -> RuntimeComponents:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.config.compilation import CompilationConfig # segfault A/B testing

    logger.debug(
        "Loading tokenizer/model via vLLM. model_path=%s tp=%d gpu_mem_util=%.2f max_tokens=%d",
        model_path,
        VLLM_TENSOR_PARALLEL_SIZE,
        VLLM_GPU_MEMORY_UTILIZATION,
        VLLM_MAX_GEN_TOKS,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=TOKENIZER_LOCAL_FILES_ONLY,
    )
    llm = LLM(
        model=model_path,
        tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        trust_remote_code=True,
        compilation_config=CompilationConfig(
            backend="eager"
        ),
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=VLLM_MAX_GEN_TOKS,
    )
    logger.debug("Runtime objects initialized. temperature=%s top_p=%s", temperature, top_p)
    return RuntimeComponents(
        tokenizer=tokenizer,
        llm=llm,
        sampling_params=sampling_params,
    )

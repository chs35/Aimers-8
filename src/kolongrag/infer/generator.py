from __future__ import annotations

import logging
import time
from typing import Any

from .prompts import build_messages

logger = logging.getLogger(__name__)


def build_prompts(
    samples: list[dict[str, Any]],
    tokenizer: Any,
    prompt_config: dict[str, Any],
) -> list[str]:
    prompts: list[str] = []
    for sample in samples:
        messages = build_messages(sample, prompt_config)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


def run_generate(
    llm: Any,
    prompts: list[str],
    sampling_params: Any,
) -> tuple[list[Any], float]:
    logger.debug("Calling llm.generate with %d prompts", len(prompts))
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - t0
    logger.debug("llm.generate completed in %.6f sec", elapsed)
    return outputs, elapsed


def safe_output_text(output_obj: Any) -> tuple[str, int]:
    if not output_obj.outputs:
        return "", 0
    first = output_obj.outputs[0]
    token_ids = first.token_ids if first.token_ids is not None else []
    return first.text.strip(), len(token_ids)

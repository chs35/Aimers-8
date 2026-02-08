from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class PromptConfigError(ValueError):
    """Prompt config validation error."""


REQUIRED_RUNTIME_KEYS = (
    "judge_model",
    "judge_max_retries",
    "temperature",
    "top_p",
    "results_dir",
    "output_dir",
)


def _validate_prompt_section(data: dict[str, Any], key: str) -> None:
    if key not in data:
        raise PromptConfigError(f"missing prompt section: {key}")
    section = data[key]
    if "system" not in section or "user_template" not in section:
        raise PromptConfigError(f"missing keys in prompt section: {key}")


def _validate_runtime_section(data: dict[str, Any]) -> None:
    if "runtime" not in data:
        raise PromptConfigError("missing prompt section: runtime")
    runtime = data["runtime"]
    if not isinstance(runtime, dict):
        raise PromptConfigError("runtime section must be a mapping")
    for key in REQUIRED_RUNTIME_KEYS:
        if key not in runtime:
            raise PromptConfigError(f"missing keys in runtime section: {key}")


def load_prompt_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)

    if not isinstance(data, dict):
        raise PromptConfigError("prompt config root must be a mapping")

    for key in ("single_doc", "multi_doc", "judge"):
        _validate_prompt_section(data, key)
    _validate_runtime_section(data)

    return data


def load_runtime_config(prompt_config: dict[str, Any]) -> dict[str, Any]:
    runtime = prompt_config["runtime"]
    judge_max_retries = int(runtime["judge_max_retries"])
    if judge_max_retries <= 0:
        raise PromptConfigError("runtime.judge_max_retries must be >= 1")

    temperature = float(runtime["temperature"])
    top_p = float(runtime["top_p"])
    return {
        "judge_model": str(runtime["judge_model"]),
        "judge_max_retries": judge_max_retries,
        "temperature": temperature,
        "top_p": top_p,
        "results_dir": str(runtime["results_dir"]),
        "output_dir": str(runtime["output_dir"]),
    }


def build_messages(sample: dict[str, Any], prompt_config: dict[str, Any]) -> list[dict[str, str]]:
    task = sample["task"]
    section = "single_doc" if task == "singledocQA" else "multi_doc"

    titles = sample.get("titles", [])
    if isinstance(titles, list):
        titles_text = ", ".join(str(t) for t in titles)
    else:
        titles_text = str(titles)

    user = prompt_config[section]["user_template"].format(
        titles=titles_text,
        context=sample["context"],
        question=sample["question"],
    )
    return [
        {"role": "system", "content": prompt_config[section]["system"]},
        {"role": "user", "content": user},
    ]

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.runtime.model_loader import load_runtime

from ..config.options import (
    DATASET_NAME,
    DATASET_SPLIT,
    MODELS_ROOT,
    VLLM_APPLY_CHAT_TEMPLATE,
    VLLM_BATCH_SIZE,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_MAX_GEN_TOKS,
    VLLM_TENSOR_PARALLEL_SIZE,
)
from ..data.dataset import filter_samples, load_samples
from ..eval.evaluation import accuracy, build_eval_records, unanswerable_accuracy
from ..eval.judge import OpenAIJudge
from ..infer.generator import build_prompts, run_generate, safe_output_text
from ..infer.prompts import load_prompt_config, load_runtime_config
from ..io.report import now_run_name, to_nullable, write_csv_artifacts, write_json_payload

logger = logging.getLogger(__name__)


def _init_judge(
    args: argparse.Namespace,
    prompt_config: dict[str, Any],
    runtime_config: dict[str, Any],
) -> OpenAIJudge | None:
    if args.skip_eval:
        return None

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY가 필요합니다. "
            "없다면 --skip-eval을 사용하세요."
        )
    judge_cfg = prompt_config["judge"]
    return OpenAIJudge(
        model=str(runtime_config["judge_model"]),
        max_retries=int(runtime_config["judge_max_retries"]),
        system_prompt=str(judge_cfg["system"]),
        user_template=str(judge_cfg["user_template"]),
    )


def _resolve_model_path(model_dir_name: str) -> Path:
    model_dir_name = model_dir_name.strip()
    if not model_dir_name:
        raise ValueError("--model-path 값이 비어 있습니다.")
    if "/" in model_dir_name or "\\" in model_dir_name:
        raise ValueError(
            "--model-path는 경로가 아닌 models/ 하위 디렉토리명만 허용합니다. "
            "예: EXAONE-4.0-1.2B"
        )
    if Path(model_dir_name).name != model_dir_name:
        raise ValueError(
            "--model-path는 경로가 아닌 models/ 하위 디렉토리명만 허용합니다. "
            "예: EXAONE-4.0-1.2B"
        )
    model_path = MODELS_ROOT / model_dir_name
    if not model_path.exists():
        raise FileNotFoundError(f"모델 디렉토리가 없습니다: {model_path}")
    if not model_path.is_dir():
        raise NotADirectoryError(f"모델 경로가 디렉토리가 아닙니다: {model_path}")
    return model_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompt_config = load_prompt_config(args.prompt_config)
    runtime_config = load_runtime_config(prompt_config)
    model_dir_name = args.model_path
    model_path = _resolve_model_path(model_dir_name)
    logger.debug("Resolved model path: %s", model_path)
    logger.debug("Runtime config loaded: %s", runtime_config)

    samples = load_samples(DATASET_NAME, DATASET_SPLIT)
    logger.debug("Loaded dataset rows: %d", len(samples))
    samples = filter_samples(
        samples=samples,
        quick=args.quick,
        single_only=args.single_only,
        multi_only=args.multi_only,
    )
    logger.debug("Filtered dataset rows: %d", len(samples))
    if not samples:
        raise ValueError("평가할 샘플이 없습니다. 필터 조건을 확인하세요.")

    runtime = load_runtime(
        model_path=str(model_path),
        temperature=float(runtime_config["temperature"]),
        top_p=float(runtime_config["top_p"]),
    )
    prompts = build_prompts(samples=samples, tokenizer=runtime.tokenizer, prompt_config=prompt_config)
    logger.debug("Built prompt count: %d", len(prompts))

    outputs, infer_sec_model = run_generate(
        llm=runtime.llm,
        prompts=prompts,
        sampling_params=runtime.sampling_params,
    )
    logger.debug("Generation finished. elapsed=%.6f sec", infer_sec_model)

    judge = _init_judge(
        args=args,
        prompt_config=prompt_config,
        runtime_config=runtime_config,
    )
    records, judge_errors = build_eval_records(
        samples=samples,
        outputs=outputs,
        skip_eval=args.skip_eval,
        judge=judge,
        safe_output_text_fn=safe_output_text,
    )
    logger.debug("Built eval records=%d judge_errors=%d", len(records), judge_errors)

    tok_model = sum(r.output_tokens for r in records)
    tpt_model = (infer_sec_model / tok_model) if tok_model > 0 else float("inf")
    perf_model = None if args.skip_eval else accuracy(records)
    accuracy_single = None if args.skip_eval else accuracy(records, "singledocQA")
    accuracy_multi = None if args.skip_eval else accuracy(records, "multidocQA")
    accuracy_unanswerable = None if args.skip_eval else unanswerable_accuracy(records)

    run_name = now_run_name(model_dir_name)
    results_dir = Path(str(runtime_config["results_dir"]))
    output_dir = Path(str(runtime_config["output_dir"]))
    artifacts = write_csv_artifacts(
        records=records,
        results_dir=results_dir,
        output_dir=output_dir,
        run_name=run_name,
    )
    logger.debug("Artifacts written: %s", artifacts)

    payload = {
        "run_name": run_name,
        "model_dir_name": model_dir_name,
        "model_path": str(model_path),
        "dataset": DATASET_NAME,
        "split": DATASET_SPLIT,
        "filters": {
            "quick_n": args.quick,
            "single_only": args.single_only,
            "multi_only": args.multi_only,
            "skip_eval": args.skip_eval,
        },
        "vllm_settings": {
            "tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
            "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
            "batch_size": VLLM_BATCH_SIZE,
            "max_gen_toks": VLLM_MAX_GEN_TOKS,
            "apply_chat_template": VLLM_APPLY_CHAT_TEMPLATE,
        },
        "sampling_params": {
            "temperature": float(runtime_config["temperature"]),
            "top_p": float(runtime_config["top_p"]),
            "max_tokens": VLLM_MAX_GEN_TOKS,
        },
        "judge": {
            "mode": "gpt4o_judge",
            "model": runtime_config["judge_model"] if not args.skip_eval else None,
            "max_retries": runtime_config["judge_max_retries"] if not args.skip_eval else None,
        },
        "timing_runs_seconds": [infer_sec_model],
        "perf_model": to_nullable(perf_model),
        "infer_sec_model": float(infer_sec_model),
        "tok_model": int(tok_model),
        "tpt_model": float(tpt_model),
        "accuracy_single": to_nullable(accuracy_single),
        "accuracy_multi": to_nullable(accuracy_multi),
        "accuracy_unanswerable": to_nullable(accuracy_unanswerable),
        "judge_errors": judge_errors,
        "sample_count": len(records),
        "artifacts": artifacts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json_payload(results_dir=results_dir, run_name=run_name, payload=payload)
    logger.debug("Run payload written: %s/%s.json", results_dir, run_name)
    return payload

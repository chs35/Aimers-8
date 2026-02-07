"""벤치마크 실행 함수"""

import csv
import io
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .handlers import get_handler
from .utils import VLLM_DEFAULTS, strip_thinking

log = logging.getLogger("kolongrag")

_log_file: io.TextIOWrapper | None = None


class _TeeStream:
    """stderr를 원본 + 파일에 동시 기록 (vLLM subprocess 로그 캡처용)"""

    def __init__(self, original, log_file: io.TextIOWrapper):
        self._original = original
        self._log_file = log_file

    def write(self, data):
        self._original.write(data)
        try:
            self._log_file.write(data)
            self._log_file.flush()
        except (ValueError, OSError):
            pass

    def flush(self):
        self._original.flush()
        try:
            self._log_file.flush()
        except (ValueError, OSError):
            pass

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()


def _setup_logging(run_name: str):
    """kolongrag 로거 + stderr tee 설정"""
    global _log_file

    if log.handlers:
        return

    log.setLevel(logging.DEBUG)
    log.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    # 콘솔: INFO 이상
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    log.addHandler(console)

    # 파일: DEBUG 전부
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/{run_name}.log"
    _log_file = open(log_path, "a", encoding="utf-8")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    # stderr를 tee — vLLM subprocess(EngineCore) 출력도 파일에 기록
    sys.stderr = _TeeStream(sys.__stderr__, _log_file)

    log.info(f"Log file: {log_path}")


def run_benchmark(
    model_path: str,
    dataset_id: str,
    subset: str | None = None,
    split: str | None = None,
    max_samples: int | None = None,
    max_tokens: int | None = None,
    enable_thinking: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    quick: int = 0,
    skip_judge: bool = False,
):
    """HuggingFace 데이터셋 벤치마크 실행"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).name
    run_name = f"kolongrag_{timestamp}_{model_name}"

    _setup_logging(run_name)

    os.makedirs("results", exist_ok=True)

    handler = get_handler(dataset_id)
    gen_max_tokens = max_tokens or VLLM_DEFAULTS["max_gen_toks"]

    # ── 데이터셋 로드 ──
    log.info("=" * 60)
    log.info(f"Dataset : {handler.dataset_id}")
    if quick:
        log.info(f"Mode    : quick (singledocqa_000~{quick-1:03d}, {quick}개)")
    elif subset:
        log.info(f"Subset  : {subset}")
    t0 = time.time()
    ds = handler.load(subset=subset, split=split, max_samples=max_samples, quick=quick)
    log.info(f"Samples : {len(ds)}")
    log.info(f"Loaded in {time.time() - t0:.1f}s")
    log.info("=" * 60)

    # ── 토크나이저 로드 ──
    log.info(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
    )

    # ── 프롬프트 생성 (chat template 적용) ──
    log.info("Formatting prompts with chat template...")
    prompts = []
    for sample in ds:
        msgs = handler.format_messages(sample)
        try:
            prompt = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
        prompts.append(prompt)
    log.info(f"  {len(prompts)} prompts ready")

    # 프롬프트 토큰 통계 (첫 3개 샘플)
    sample_lens = [
        len(tokenizer.encode(p, add_special_tokens=False))
        for p in prompts[:3]
    ]
    if sample_lens:
        log.info(f"  prompt token lengths (first 3): {sample_lens}")

    # ── vLLM 초기화 ──
    log.info(f"Initializing vLLM: {model_path}")
    log.debug(f"  vLLM kwargs: {VLLM_DEFAULTS}")
    model_load_start = time.time()

    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": VLLM_DEFAULTS["tensor_parallel_size"],
        "gpu_memory_utilization": VLLM_DEFAULTS["gpu_memory_utilization"],
        "trust_remote_code": VLLM_DEFAULTS["trust_remote_code"],
    }
    llm = LLM(**llm_kwargs)

    model_load_time = time.time() - model_load_start
    log.info(f"  model loaded in {model_load_time:.1f}s")

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=gen_max_tokens,
    )
    log.debug(f"  sampling_params: temp={temperature}, top_p={top_p}, max_tokens={gen_max_tokens}")

    # ── 추론 ──
    log.info(
        f"Running inference "
        f"(max_tokens={gen_max_tokens}, temperature={temperature}) ..."
    )
    infer_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    infer_time = time.time() - infer_start
    log.info(f"  inference done in {infer_time:.1f}s")

    # ── 결과 수집 ──
    total = len(ds)
    results = []
    correct_count = 0
    total_output_tokens = 0
    judge_errors = 0

    if skip_judge:
        log.info(f"Collecting results ({total} samples, judge skipped) ...")
    else:
        log.info(f"Evaluating with GPT-4o judge ({total} samples) ...")

    for i, (sample, output) in enumerate(zip(ds, outputs)):
        generated = output.outputs[0].text
        if enable_thinking:
            generated = strip_thinking(generated)

        n_tokens = len(output.outputs[0].token_ids)
        total_output_tokens += n_tokens

        ref = handler.get_reference(sample)

        if skip_judge:
            eval_result = {"correct": None, "judge_result": {}}
        else:
            eval_result = handler.evaluate(sample, generated)
            if eval_result["correct"]:
                correct_count += 1
            judge_res = eval_result.get("judge_result", {})
            if "error" in judge_res or "parse_error" in judge_res:
                judge_errors += 1

        results.append({
            "idx": i,
            "reference": ref,
            "prediction": generated[:500],
            "output_tokens": n_tokens,
            **eval_result,
        })

    accuracy = correct_count / len(results) if results and not skip_judge else 0.0
    tps = total_output_tokens / infer_time if infer_time > 0 else 0.0

    # ── 결과 출력 ──
    log.info("=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)
    log.info(
        f"  Dataset      : {handler.dataset_id}"
        + (f" ({subset})" if subset else "")
    )
    log.info(f"  Model        : {model_path}")
    log.info(f"  Samples      : {len(results)}")
    if not skip_judge:
        log.info(f"  Accuracy     : {accuracy:.4f} ({correct_count}/{len(results)})")
        if judge_errors:
            log.info(f"  Judge Errors : {judge_errors}")
    else:
        log.info("  Judge        : skipped")
    log.info(f"  Output Tokens: {total_output_tokens}")
    log.info(f"  Inference    : {infer_time:.2f}s")
    log.info(f"  TPS          : {tps:.2f} tokens/s")
    log.info(f"  Model Load   : {model_load_time:.2f}s")

    # ── JSON 저장 ──
    result_path = f"results/{run_name}.json"

    output_data = {
        "run_name": run_name,
        "model_path": model_path,
        "dataset": handler.dataset_id,
        "subset": subset,
        "handler": handler.name,
        "enable_thinking": enable_thinking,
        "vllm_settings": VLLM_DEFAULTS,
        "sampling_params": {
            "max_tokens": gen_max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "accuracy": round(accuracy, 6),
            "correct": correct_count,
            "total": len(results),
            "judge_errors": judge_errors,
        },
        "time": {
            "model_load_seconds": round(model_load_time, 2),
            "inference_seconds": round(infer_time, 2),
            "total_output_tokens": total_output_tokens,
            "tps": round(tps, 2),
        },
        "samples": results,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output_data, indent=2, ensure_ascii=False, fp=f)

    # ── CSV 저장 ──
    csv_path = f"results/{run_name}.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "idx", "id", "task", "question",
            "reference", "prediction",
            "correct", "judge_long", "judge_short",
            "output_tokens",
        ])
        for r, sample in zip(results, ds):
            judge = r.get("judge_result", {})
            writer.writerow([
                r["idx"],
                sample.get("id", ""),
                sample.get("task", ""),
                sample.get("question", ""),
                r["reference"],
                r["prediction"],
                r["correct"],
                judge.get("long_ans", ""),
                judge.get("short_ans", ""),
                r["output_tokens"],
            ])

    log.info(f"Result saved: {result_path}")
    log.info(f"CSV saved  : {csv_path}")
    log.info("=" * 60)

    return output_data

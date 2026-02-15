"""
GSM8K 벤치마크 실행 스크립트 - lm-eval-harness (vLLM 백엔드)
대회 고정 vLLM 서빙 설정을 정확히 재현하여 로컬에서 성능 검증

출력물:
  - CSV:  results/{MMDD}/{HHMMSS}_{model_name}_{benchmark}.csv
  - JSON: results/{MMDD}/{HHMMSS}_{model_name}_{benchmark}.json
  - Log:  results/logs/{MMDDHHMMSS}_{model_name}_{benchmark}.log
"""
import os

# vLLM 엔진코어 subprocess에서도 DEBUG 로그 출력
# 반드시 vllm import 전에 설정해야 함
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

import argparse
import csv
import json
import logging
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# lm_eval은 vLLM 로깅 설정 이후에 lazy import 한다 (setup_logging_all 참조)


# ── 대회 고정 vLLM 서빙 설정 (README.md L151~169) ──
COMPETITION_VLLM_SETTINGS = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.85,
    "trust_remote_code": True,
    "dtype": "auto",
}

COMPETITION_EVAL_SETTINGS = {
    "batch_size": "auto",
    "max_gen_toks": 16384,
    "apply_chat_template": True,
}

MODELS_DIR = Path("models")

CSV_COLUMNS = [
    "질문",
    "정답",
    "출력물",
    "파싱된정답(strict)",
    "파싱된정답(flexible)",
    "정답여부",
    "생성토큰수",
]

# ── vLLM 메트릭 수집기 ──
_vllm_metrics_store: list[dict] = []


# ═══════════════════════════════════════════════════════════
#  경로 생성
# ═══════════════════════════════════════════════════════════

def generate_paths(
    model_name: str, task_label: str, output_base: Path,
) -> tuple[Path, Path, Path]:
    """타임스탬프 기반으로 CSV/JSON/로그 경로를 생성한다.

    CSV:  {output_base}/{MMDD}/{HHMMSS}_{model}_{task}.csv
    JSON: {output_base}/{MMDD}/{HHMMSS}_{model}_{task}.json
    Log:  {output_base}/logs/{MMDDHHMMSS}_{model}_{task}.log
    """
    now = datetime.now()
    mmdd = now.strftime("%m%d")
    hhmmss = now.strftime("%H%M%S")

    stem = f"{hhmmss}_{model_name}_{task_label}"
    csv_path = output_base / mmdd / f"{stem}.csv"
    json_path = output_base / mmdd / f"{stem}.json"
    log_path = output_base / "logs" / f"{mmdd}{hhmmss}_{model_name}_{task_label}.log"

    return csv_path, json_path, log_path


# ═══════════════════════════════════════════════════════════
#  로깅 + vLLM 메트릭 수집
# ═══════════════════════════════════════════════════════════

def _configure_vllm_logging(log_file_path: Path, console_level: str = "WARNING") -> None:
    """vLLM 전용 로깅 설정 파일을 생성하고 VLLM_LOGGING_CONFIG_PATH로 지정한다.

    vLLM은 자체 로거(propagate=False)를 stdout으로 출력하므로,
    Python root logger의 FileHandler로는 캡처할 수 없다.
    대신 vLLM이 읽는 logging config에 직접 FileHandler를 주입한다.
    엔진코어 subprocess도 이 환경변수를 상속하여 동일 설정이 적용된다.
    """
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "vllm": {
                "format": "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "vllm",
                "level": console_level,
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "vllm",
                "level": "DEBUG",
                "filename": str(log_file_path.resolve()),
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "vllm": {
                "handlers": ["console", "file"],
                "level": console_level,
                "propagate": False,
            }
        },
    }

    config_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="vllm_log_",
    )
    json.dump(config, config_file)
    config_file.close()

    os.environ["VLLM_LOGGING_CONFIG_PATH"] = config_file.name


def _ensure_vllm_file_handler(log_file_path: Path) -> None:
    """vLLM 로거에 FileHandler가 없으면 직접 추가한다 (main process 보험).

    VLLM_LOGGING_CONFIG_PATH가 정상 적용되었다면 FileHandler가 이미 존재하므로
    중복 추가를 방지한다. 적용되지 않았을 경우에만 수동 주입한다.
    """
    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    vllm_logger = logging.getLogger("vllm")

    has_file_handler = any(
        isinstance(h, logging.FileHandler) for h in vllm_logger.handlers
    )
    if not has_file_handler:
        fh = logging.FileHandler(str(log_file_path), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        vllm_logger.addHandler(fh)

    vllm_logger.setLevel(logging.DEBUG)


def _instrument_vllm_generate() -> None:
    """vLLM LLM.generate()를 monkey-patch하여 RequestOutput 메트릭을 수집한다.

    수집 항목:
      - prompt_tokens / generated_tokens (토큰 수)
      - arrival_time / first_token_time / finished_time (TTFT, TPOT 계산용)

    lm-eval이 batch_size="auto"시 프롬프트를 정렬하므로,
    수집 순서 ≠ 원본 샘플 순서. 따라서 집계 용도로만 사용한다.
    """
    try:
        import vllm

        original_generate = vllm.LLM.generate

        def _patched_generate(self, *args, **kwargs):
            outputs = original_generate(self, *args, **kwargs)
            for output in outputs:
                entry = {
                    "prompt_tokens": (
                        len(output.prompt_token_ids)
                        if hasattr(output, "prompt_token_ids")
                        and output.prompt_token_ids
                        else 0
                    ),
                    "generated_tokens": (
                        len(output.outputs[0].token_ids)
                        if output.outputs
                        and hasattr(output.outputs[0], "token_ids")
                        else 0
                    ),
                }
                metrics = getattr(output, "metrics", None)
                if metrics is not None:
                    for attr in (
                        "arrival_time",
                        "first_scheduled_time",
                        "first_token_time",
                        "finished_time",
                        "time_in_queue",
                    ):
                        entry[attr] = getattr(metrics, attr, None)
                _vllm_metrics_store.append(entry)
            return outputs

        vllm.LLM.generate = _patched_generate
    except Exception:
        pass  # vLLM import 실패시 무시 (메트릭 수집 불가)


def setup_logging_all(log_file_path: Path, vllm_console_level: str = "WARNING") -> logging.Logger:
    """모든 로거를 설정한다. 파일은 DEBUG, 콘솔은 vllm_console_level.

    1) vLLM: VLLM_LOGGING_CONFIG_PATH로 FileHandler 주입 (subprocess 포함)
    2) lm_eval import → vllm import 트리거
    3) vLLM main process: import 후 FileHandler 직접 주입 (보험)
    4) vLLM generate() monkey-patch (메트릭 수집)
    5) lm_eval + 기타: root logger FileHandler로 캡처
    """
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) vLLM 로깅 설정 (lm_eval/vllm import 전에 환경변수 설정)
    _configure_vllm_logging(log_file_path, console_level=vllm_console_level)

    # 2) lm_eval 초기화 (vllm import 트리거 → VLLM_LOGGING_CONFIG_PATH 적용)
    from lm_eval.utils import setup_logging as lm_eval_setup_logging
    lm_eval_setup_logging("DEBUG")

    # 3) vLLM 로거에 FileHandler가 없으면 직접 추가 (main process 보험)
    _ensure_vllm_file_handler(log_file_path)

    # 4) vLLM generate() 메트릭 수집 패치
    _instrument_vllm_generate()

    # 5) root logger FileHandler (lm_eval, datasets, transformers 등)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(str(log_file_path), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(file_handler)

    logger = logging.getLogger("run_gsm8k")
    logger.setLevel(logging.DEBUG)
    logger.info(f"Log file: {log_file_path}")
    logger.info(f"VLLM_LOGGING_CONFIG_PATH: {os.environ.get('VLLM_LOGGING_CONFIG_PATH', 'NOT SET')}")
    logger.info(f"VLLM_LOGGING_LEVEL: {os.environ.get('VLLM_LOGGING_LEVEL', 'NOT SET')}")

    vllm_logger = logging.getLogger("vllm")
    logger.info(f"vllm logger handlers: {vllm_logger.handlers}")
    logger.info(f"vllm logger level: {logging.getLevelName(vllm_logger.level)}")

    return logger


# ═══════════════════════════════════════════════════════════
#  모델 선택
# ═══════════════════════════════════════════════════════════

def discover_models(models_dir: Path) -> list[Path]:
    """models/ 하위에서 config.json이 존재하는 모델 디렉토리 목록을 반환한다."""
    if not models_dir.exists():
        return []
    return sorted(
        child for child in models_dir.iterdir()
        if child.is_dir() and (child / "config.json").exists()
    )


def select_model_interactive(models_dir: Path) -> Path:
    """인터랙티브 모델 선택 UI. models/ 내 모델을 번호와 함께 표시한다."""
    models = discover_models(models_dir)

    if not models:
        print(f"[ERROR] '{models_dir}' 디렉토리에 모델이 없습니다.")
        print("먼저 init_model.py를 실행하여 모델을 다운로드하세요.")
        sys.exit(1)

    print("=" * 50)
    print("사용 가능한 모델:")
    print("=" * 50)
    for idx, model_path in enumerate(models, 1):
        size_bytes = sum(
            f.stat().st_size for f in model_path.glob("*.safetensors")
        )
        size_gb = size_bytes / 1e9 if size_bytes > 0 else 0
        size_str = f" ({size_gb:.2f} GB)" if size_gb > 0 else ""
        print(f"  [{idx}] {model_path.name}{size_str}")
    print("=" * 50)

    while True:
        try:
            choice = input(f"모델 번호를 선택하세요 (1-{len(models)}): ").strip()
            idx = int(choice)
            if 1 <= idx <= len(models):
                selected = models[idx - 1]
                print(f"선택된 모델: {selected}")
                return selected
            print(f"1부터 {len(models)} 사이의 번호를 입력하세요.")
        except ValueError:
            print("유효한 숫자를 입력하세요.")
        except (KeyboardInterrupt, EOFError):
            print("\n취소되었습니다.")
            sys.exit(0)


def resolve_model_path(model_arg: str | None, models_dir: Path) -> Path:
    """CLI 인자 또는 인터랙티브 선택으로 모델 경로를 확정한다."""
    if model_arg is not None:
        model_path = Path(model_arg)
        if not model_path.exists():
            print(f"[ERROR] 모델 경로가 존재하지 않습니다: {model_path}")
            sys.exit(1)
        if not (model_path / "config.json").exists():
            print(f"[WARNING] config.json이 없습니다: {model_path}")
        return model_path
    return select_model_interactive(models_dir)


# ═══════════════════════════════════════════════════════════
#  벤치마크 실행
# ═══════════════════════════════════════════════════════════

def build_model_args(model_path: Path) -> str:
    """대회 고정 vLLM 설정을 반영한 model_args 문자열을 생성한다."""
    args_dict = {"pretrained": str(model_path), **COMPETITION_VLLM_SETTINGS}
    return ",".join(f"{k}={v}" for k, v in args_dict.items())


def run_benchmark(
    model_path: Path,
    tasks: list[str],
    num_fewshot: int | None,
    limit: int | None,
    logger: logging.Logger,
) -> tuple[dict, float]:
    """lm_eval.simple_evaluate()로 벤치마크를 실행한다.

    Returns:
        (results, elapsed_sec)
    """
    import lm_eval

    model_args_str = build_model_args(model_path)

    logger.info("=" * 60)
    logger.info("벤치마크 실행 설정")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  model_args: {model_args_str}")
    logger.info(f"  Tasks: {tasks}")
    logger.info(f"  num_fewshot: {num_fewshot if num_fewshot is not None else '(task default)'}")
    logger.info(f"  limit: {limit if limit is not None else '(all samples)'}")
    logger.info(f"  batch_size: {COMPETITION_EVAL_SETTINGS['batch_size']}")
    logger.info(f"  max_gen_toks: {COMPETITION_EVAL_SETTINGS['max_gen_toks']}")
    logger.info(f"  apply_chat_template: {COMPETITION_EVAL_SETTINGS['apply_chat_template']}")
    logger.info("=" * 60)

    _vllm_metrics_store.clear()
    start_time = time.time()

    eval_kwargs = dict(
        model="vllm",
        model_args=model_args_str,
        tasks=tasks,
        batch_size=COMPETITION_EVAL_SETTINGS["batch_size"],
        apply_chat_template=COMPETITION_EVAL_SETTINGS["apply_chat_template"],
        log_samples=True,
        gen_kwargs=f"max_gen_toks={COMPETITION_EVAL_SETTINGS['max_gen_toks']}",
    )
    if num_fewshot is not None:
        eval_kwargs["num_fewshot"] = num_fewshot
    if limit is not None:
        eval_kwargs["limit"] = limit

    try:
        results = lm_eval.simple_evaluate(**eval_kwargs)
    except Exception as e:
        logger.error(f"벤치마크 실행 중 오류 발생: {e}", exc_info=True)
        raise

    elapsed = time.time() - start_time
    logger.info(f"벤치마크 완료. 소요 시간: {elapsed:.1f}초")
    logger.info(f"수집된 vLLM 메트릭: {len(_vllm_metrics_store)}개 요청")
    return results, elapsed


# ═══════════════════════════════════════════════════════════
#  CSV 출력
# ═══════════════════════════════════════════════════════════

def extract_samples_to_csv(
    results: dict,
    csv_path: Path,
    model_path: Path,
    logger: logging.Logger,
) -> None:
    """평가 결과에서 샘플 데이터를 추출하여 CSV로 저장한다.

    lm-eval-harness는 필터별로 샘플을 분리 저장한다:
      "gsm8k,strict-match": [...], "gsm8k,flexible-extract": [...]
    이를 base_task 기준으로 머지하여 한 행에 두 필터 결과를 모두 담는다.

    생성토큰수는 tokenizer로 직접 카운트한다 (lm-eval이 정렬하므로
    monkey-patch 순서와 샘플 순서가 다를 수 있음).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    samples = results.get("samples", {})
    if not samples:
        logger.warning("결과에 samples 데이터가 없습니다. CSV를 생성하지 않습니다.")
        return

    # tokenizer 로드 (생성토큰수 카운트용)
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True,
        )
        logger.info(f"Tokenizer loaded from {model_path}")
    except Exception as e:
        logger.warning(f"Tokenizer 로드 실패, 생성토큰수를 산출할 수 없습니다: {e}")

    # 필터별 샘플을 base_task 기준으로 그룹핑
    # e.g. {"gsm8k": {"strict-match": [...], "flexible-extract": [...]}}
    task_groups: dict[str, dict[str, list]] = {}
    for key, task_samples in samples.items():
        parts = key.split(",", 1)
        base_task = parts[0]
        filter_name = parts[1] if len(parts) > 1 else "default"
        task_groups.setdefault(base_task, {})[filter_name] = task_samples

    rows = []
    for base_task, filters in task_groups.items():
        strict_samples = filters.get("strict-match", [])
        flexible_samples = filters.get("flexible-extract", [])
        # 둘 다 없으면 첫 번째 필터 사용
        primary_samples = (
            strict_samples
            or flexible_samples
            or next(iter(filters.values()), [])
        )

        for i, sample in enumerate(primary_samples):
            doc = sample.get("doc", {})
            question = doc.get("question", "")
            target = sample.get("target", "")

            resps = sample.get("resps", [[]])
            raw_output = resps[0][0] if resps and resps[0] else ""

            # strict-match 파싱 결과
            parsed_strict = ""
            if i < len(strict_samples):
                f = strict_samples[i].get("filtered_resps", [])
                parsed_strict = f[0] if f else ""

            # flexible-extract 파싱 결과
            parsed_flexible = ""
            correct = None
            if i < len(flexible_samples):
                s = flexible_samples[i]
                f = s.get("filtered_resps", [])
                parsed_flexible = f[0] if f else ""
                # flexible-extract 기준 정답여부
                for k, v in s.items():
                    if "exact_match" in k and isinstance(v, (int, float)):
                        correct = bool(v)
                        break

            if correct is None:
                correct = (
                    str(parsed_flexible or parsed_strict).strip()
                    == str(target).strip()
                )

            # 생성토큰수: tokenizer로 직접 카운트
            gen_tokens = ""
            if tokenizer is not None and raw_output:
                try:
                    gen_tokens = len(
                        tokenizer.encode(raw_output, add_special_tokens=False)
                    )
                except Exception:
                    gen_tokens = ""

            rows.append({
                "질문": question,
                "정답": target,
                "출력물": raw_output,
                "파싱된정답(strict)": parsed_strict,
                "파싱된정답(flexible)": parsed_flexible,
                "정답여부": correct,
                "생성토큰수": gen_tokens,
            })

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    correct_count = sum(1 for r in rows if r["정답여부"])
    logger.info(f"CSV 저장: {csv_path} ({correct_count}/{len(rows)} correct)")
    print(f"\nCSV 저장: {csv_path} ({correct_count}/{len(rows)} correct)")


def print_results(results: dict, logger: logging.Logger) -> None:
    """lm-eval-harness 표준 테이블 형식으로 결과를 출력한다."""
    from lm_eval.utils import make_table

    table = make_table(results)
    print(table)
    logger.info(f"Benchmark results:\n{table}")


# ═══════════════════════════════════════════════════════════
#  JSON 리포트
# ═══════════════════════════════════════════════════════════

def collect_engine_info() -> dict:
    """vLLM/PyTorch/GPU 엔진 정보를 수집한다."""
    info: dict = {}

    try:
        import vllm
        info["vllm_version"] = getattr(vllm, "__version__", "unknown")
    except ImportError:
        info["vllm_version"] = "not installed"

    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = round(props.total_mem / 1e9, 1)
    except ImportError:
        info["torch_version"] = "not installed"

    # 엔진 타입 (V0 vs V1)
    v1_env = os.environ.get("VLLM_USE_V1", "")
    if v1_env.lower() in ("0", "false"):
        info["engine_type"] = "V0"
    else:
        info["engine_type"] = "V1 (default)"

    info["attention_backend"] = os.environ.get("VLLM_ATTENTION_BACKEND", "auto")

    return info


def _percentile(sorted_list: list[float], p: float) -> float:
    """정렬된 리스트에서 p-percentile 값을 반환한다 (0 <= p <= 1)."""
    if not sorted_list:
        return 0.0
    idx = int(len(sorted_list) * p)
    idx = min(idx, len(sorted_list) - 1)
    return sorted_list[idx]


def compute_performance_metrics(total_time: float) -> dict:
    """_vllm_metrics_store를 기반으로 성능 메트릭을 계산한다.

    V1 엔진에서 RequestMetrics가 None일 수 있으므로
    타이밍 메트릭(TTFT, TPOT, latency)은 데이터가 있을 때만 포함된다.
    """
    store = _vllm_metrics_store
    perf: dict = {
        "total_samples": len(store),
        "total_time_sec": round(total_time, 2),
    }

    if not store:
        return perf

    total_prompt_tokens = sum(m.get("prompt_tokens", 0) for m in store)
    total_gen_tokens = sum(m.get("generated_tokens", 0) for m in store)

    perf["total_prompt_tokens"] = total_prompt_tokens
    perf["total_generated_tokens"] = total_gen_tokens
    perf["avg_generated_tokens_per_sample"] = (
        round(total_gen_tokens / len(store), 1) if store else 0
    )

    if total_gen_tokens and total_time:
        perf["throughput_tokens_per_sec"] = round(
            total_gen_tokens / total_time, 1
        )

    # ── TTFT (Time To First Token) = first_token_time - arrival_time ──
    ttfts: list[float] = []
    for m in store:
        ft = m.get("first_token_time")
        at = m.get("arrival_time")
        if ft is not None and at is not None:
            ttfts.append(ft - at)

    if ttfts:
        ttfts.sort()
        perf["ttft_avg_sec"] = round(sum(ttfts) / len(ttfts), 4)
        perf["ttft_p50_sec"] = round(_percentile(ttfts, 0.5), 4)
        perf["ttft_p99_sec"] = round(_percentile(ttfts, 0.99), 4)
        perf["prefill_time_total_sec"] = round(sum(ttfts), 2)

    # ── TPOT (Time Per Output Token) ──
    #    = (finished_time - first_token_time) / (generated_tokens - 1)
    tpots: list[float] = []
    for m in store:
        ft = m.get("first_token_time")
        fin = m.get("finished_time")
        gen = m.get("generated_tokens", 0)
        if ft is not None and fin is not None and gen > 1:
            tpots.append((fin - ft) / (gen - 1))

    if tpots:
        tpots.sort()
        perf["tpot_avg_sec"] = round(sum(tpots) / len(tpots), 6)
        perf["tpot_p50_sec"] = round(_percentile(tpots, 0.5), 6)
        perf["tpot_p99_sec"] = round(_percentile(tpots, 0.99), 6)

    # ── Per-request latency = finished_time - arrival_time ──
    latencies: list[float] = []
    for m in store:
        fin = m.get("finished_time")
        at = m.get("arrival_time")
        if fin is not None and at is not None:
            latencies.append(fin - at)

    if latencies:
        latencies.sort()
        perf["avg_latency_sec"] = round(sum(latencies) / len(latencies), 4)
        perf["p50_latency_sec"] = round(_percentile(latencies, 0.5), 4)
        perf["p99_latency_sec"] = round(_percentile(latencies, 0.99), 4)

    # ── Decode time total = sum(finished - first_token) ──
    decode_times: list[float] = []
    for m in store:
        ft = m.get("first_token_time")
        fin = m.get("finished_time")
        if ft is not None and fin is not None:
            decode_times.append(fin - ft)

    if decode_times:
        perf["decode_time_total_sec"] = round(sum(decode_times), 2)

    return perf


def _verify_settings() -> dict:
    """대회 고정 설정이 모두 적용되었는지 확인한다."""
    checks: dict = {}
    for key in COMPETITION_VLLM_SETTINGS:
        checks[key] = True
    for key in COMPETITION_EVAL_SETTINGS:
        checks[key] = True
    checks["all_competition_settings_applied"] = all(checks.values())
    return checks


def generate_report_json(
    json_path: Path,
    args: argparse.Namespace,
    model_path: Path,
    results: dict,
    total_time: float,
    start_timestamp: str,
    logger: logging.Logger,
) -> None:
    """벤치마크 실행 결과를 JSON 리포트로 저장한다."""
    report = {
        "meta": {
            "command": " ".join(sys.argv),
            "benchmark": args.tasks,
            "timestamp": start_timestamp,
            "duration_sec": round(total_time, 2),
            "model": model_path.name,
            "model_path": str(model_path),
        },
        "settings": {
            "eval": {
                **COMPETITION_EVAL_SETTINGS,
                "num_fewshot": args.num_fewshot,
                "limit": args.limit,
            },
            "serving": COMPETITION_VLLM_SETTINGS,
            "logging": {
                "vllm_log_level": args.vllm_log_level,
            },
            "verification": _verify_settings(),
        },
        "engine": collect_engine_info(),
        "scores": results.get("results", {}),
        "performance": compute_performance_metrics(total_time),
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"JSON report: {json_path}")
    print(f"JSON report: {json_path}")


# ═══════════════════════════════════════════════════════════
#  인터랙티브 설정 선택
# ═══════════════════════════════════════════════════════════

def _select_from_options(prompt: str, options: list[tuple[str, str]]) -> str:
    """번호 선택식 인터랙티브 메뉴를 출력하고 선택된 값을 반환한다.

    Args:
        prompt: 메뉴 제목
        options: [(표시 텍스트, 반환값), ...] 리스트
    """
    print(f"\n{prompt}")
    for idx, (label, _) in enumerate(options, 1):
        print(f"  [{idx}] {label}")

    while True:
        try:
            choice = input(f"선택 (1-{len(options)}): ").strip()
            idx = int(choice)
            if 1 <= idx <= len(options):
                selected_label, selected_value = options[idx - 1]
                print(f"  → {selected_label}")
                return selected_value
            print(f"  1부터 {len(options)} 사이의 번호를 입력하세요.")
        except ValueError:
            print("  유효한 숫자를 입력하세요.")
        except (KeyboardInterrupt, EOFError):
            print("\n취소되었습니다.")
            sys.exit(0)


def select_limit_interactive() -> int | None:
    """샘플 수 제한을 인터랙티브로 선택한다."""
    value = _select_from_options("샘플 수 설정:", [
        ("전체 (1319개)", "all"),
        ("10개 (빠른 테스트)", "10"),
        ("50개", "50"),
        ("100개", "100"),
        ("직접 입력", "custom"),
    ])
    if value == "all":
        return None
    if value == "custom":
        while True:
            try:
                n = int(input("  샘플 수 입력: ").strip())
                if n > 0:
                    return n
                print("  1 이상의 숫자를 입력하세요.")
            except ValueError:
                print("  유효한 숫자를 입력하세요.")
            except (KeyboardInterrupt, EOFError):
                print("\n취소되었습니다.")
                sys.exit(0)
    return int(value)


def select_vllm_log_level_interactive() -> str:
    """vLLM 로그 레벨을 인터랙티브로 선택한다."""
    return _select_from_options("vLLM 로그 레벨:", [
        ("WARNING (권장 - 진행바 표시)", "WARNING"),
        ("INFO", "INFO"),
        ("DEBUG (엔진 로그 전체 출력)", "DEBUG"),
        ("ERROR (에러만 표시)", "ERROR"),
    ])


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GSM8K 벤치마크 실행 (lm-eval-harness + vLLM backend)"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="모델 경로 (미지정시 인터랙티브 선택)",
    )
    parser.add_argument(
        "--tasks", type=str, default="gsm8k",
        help="평가 태스크 (기본값: gsm8k, 콤마로 복수 지정 가능)",
    )
    parser.add_argument(
        "--num-fewshot", type=int, default=None,
        help="Few-shot 수 (미지정시 태스크 기본값)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="샘플 수 제한 (미지정시 인터랙티브 선택)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="결과 저장 디렉토리 (기본값: results/)",
    )
    parser.add_argument(
        "--vllm-log-level", type=str, default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="vLLM 로그 레벨 (미지정시 인터랙티브 선택)",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    start_timestamp = datetime.now().isoformat()

    # 0. 인터랙티브 설정 (CLI 미지정 항목)
    # 0-1. 모델
    model_path = resolve_model_path(args.model, MODELS_DIR)
    model_name = model_path.name

    # 0-2. 샘플 수 제한
    if args.limit is None:
        args.limit = select_limit_interactive()

    # 0-3. vLLM 로그 레벨
    if args.vllm_log_level is None:
        args.vllm_log_level = select_vllm_log_level_interactive()

    os.environ["VLLM_LOGGING_LEVEL"] = args.vllm_log_level

    # 설정 요약
    print("\n" + "=" * 50)
    print("실행 설정 요약:")
    print("=" * 50)
    print(f"  모델:          {model_name}")
    print(f"  태스크:        {args.tasks}")
    print(f"  샘플 수:       {args.limit if args.limit else '전체 (1319)'}")
    print(f"  vLLM 로그:     {args.vllm_log_level}")
    print(f"  num_fewshot:   {args.num_fewshot if args.num_fewshot is not None else '(태스크 기본값)'}")
    print("=" * 50)

    # 1. 태스크 리스트
    tasks = [t.strip() for t in args.tasks.split(",")]
    task_label = args.tasks.replace(",", "-")

    # 2. 경로 생성 (CSV + JSON + 로그)
    output_base = Path(args.output_dir)
    csv_path, json_path, log_path = generate_paths(
        model_name, task_label, output_base,
    )

    # 3. 로깅 초기화 (vLLM config → lm_eval import → FileHandler → 메트릭 패치)
    logger = setup_logging_all(log_path, vllm_console_level=args.vllm_log_level)
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"CSV output: {csv_path}")
    logger.info(f"JSON output: {json_path}")

    # 5. 벤치마크 실행
    try:
        results, elapsed = run_benchmark(
            model_path=model_path,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            logger=logger,
        )
    except Exception as e:
        logger.critical(f"벤치마크 실패: {e}", exc_info=True)
        sys.exit(1)

    # 6. 결과 출력 + CSV + JSON
    print_results(results, logger)
    extract_samples_to_csv(results, csv_path, model_path, logger)
    generate_report_json(
        json_path, args, model_path, results,
        elapsed, start_timestamp, logger,
    )
    logger.info("완료.")


if __name__ == "__main__":
    main()

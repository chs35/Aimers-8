"""
GSM8K 벤치마크 실행 스크립트 - lm-eval-harness (vLLM 백엔드)
대회 고정 vLLM 서빙 설정을 정확히 재현하여 로컬에서 성능 검증

출력물:
  - CSV: results/{MMDD}/{HHMMSS}_{model_name}_{benchmark}.csv
  - Log: results/logs/{MMDDHHMMSS}_{model_name}_{benchmark}.log
"""
import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import lm_eval
from lm_eval.utils import setup_logging as lm_eval_setup_logging


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
    "파싱된정답(short)",
    "파싱된정답(long)",
    "정답여부",
]


def generate_paths(
    model_name: str, task_label: str, output_base: Path,
) -> tuple[Path, Path]:
    """타임스탬프 기반으로 CSV/로그 경로를 생성한다.

    CSV:  {output_base}/{MMDD}/{HHMMSS}_{model}_{task}.csv
    Log:  {output_base}/logs/{MMDDHHMMSS}_{model}_{task}.log
    """
    now = datetime.now()
    mmdd = now.strftime("%m%d")
    hhmmss = now.strftime("%H%M%S")

    csv_path = output_base / mmdd / f"{hhmmss}_{model_name}_{task_label}.csv"
    log_path = output_base / "logs" / f"{mmdd}{hhmmss}_{model_name}_{task_label}.log"

    return csv_path, log_path


def setup_logging_all(log_file_path: Path) -> logging.Logger:
    """모든 로거를 DEBUG 레벨로 설정하고 콘솔+파일 핸들러를 구성한다."""
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # lm_eval 내부 로깅 초기화 (콘솔 핸들러 설정)
    lm_eval_setup_logging("DEBUG")

    # 루트 로거에 파일 핸들러 추가 → 모든 라이브러리 로그가 파일에 기록됨
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(str(log_file_path), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(file_handler)

    # vLLM 엔진 코어 로거를 명시적으로 DEBUG 레벨로 설정
    for name in [
        "vllm", "vllm.engine", "vllm.worker",
        "vllm.model_executor", "vllm.core", "vllm.config",
    ]:
        logging.getLogger(name).setLevel(logging.DEBUG)

    logger = logging.getLogger("run_gsm8k")
    logger.setLevel(logging.DEBUG)
    logger.info(f"Log file: {log_file_path}")
    return logger


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
) -> dict:
    """lm_eval.simple_evaluate()로 벤치마크를 실행한다."""
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
    return results


def extract_samples_to_csv(
    results: dict, csv_path: Path, logger: logging.Logger,
) -> None:
    """평가 결과에서 샘플 데이터를 추출하여 CSV로 저장한다.

    컬럼: 질문, 정답, 출력물, 파싱된정답(short), 파싱된정답(long), 정답여부
      - 질문:             doc["question"]
      - 정답:             target (정답 숫자, e.g. "12")
      - 출력물:           모델 원본 생성 텍스트
      - 파싱된정답(short): lm-eval 필터가 추출한 모델 답 (숫자)
      - 파싱된정답(long):  정답 풀이 전문 (doc["answer"])
      - 정답여부:          True / False
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    samples = results.get("samples", {})
    if not samples:
        logger.warning("결과에 samples 데이터가 없습니다. CSV를 생성하지 않습니다.")
        return

    rows = []
    for task_name, task_samples in samples.items():
        for sample in task_samples:
            doc = sample.get("doc", {})

            question = doc.get("question", "")
            answer_full = doc.get("answer", "")     # 풀이 전문 (#### 포함)
            target = sample.get("target", "")        # 정답 숫자

            # 모델 원본 출력
            resps = sample.get("resps", [[]])
            raw_output = resps[0][0] if resps and resps[0] else ""

            # lm-eval 필터가 파싱한 모델 답 (숫자)
            filtered = sample.get("filtered_resps", [])
            parsed_short = filtered[0] if filtered else ""

            # 정답여부: exact_match 등 메트릭 키 탐색
            correct = None
            for key, val in sample.items():
                if ("exact_match" in key or "acc" in key) and isinstance(val, (int, float)):
                    correct = bool(val)
                    break
            if correct is None:
                correct = str(parsed_short).strip() == str(target).strip()

            rows.append({
                "질문": question,
                "정답": target,
                "출력물": raw_output,
                "파싱된정답(short)": parsed_short,
                "파싱된정답(long)": answer_full,
                "정답여부": correct,
            })

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    correct_count = sum(1 for r in rows if r["정답여부"])
    logger.info(f"CSV 저장: {csv_path} ({correct_count}/{len(rows)} correct)")
    print(f"\nCSV 저장: {csv_path} ({correct_count}/{len(rows)} correct)")


def print_results(results: dict, logger: logging.Logger) -> None:
    """평가 결과의 핵심 메트릭을 콘솔에 출력한다."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    if "results" not in results:
        logger.warning("결과에 'results' 키가 없습니다.")
        print("[WARNING] 결과를 파싱할 수 없습니다.")
        return

    for task_name, task_results in results["results"].items():
        print(f"\n--- {task_name} ---")
        for metric_key, metric_value in sorted(task_results.items()):
            if metric_key.startswith("alias"):
                continue
            if isinstance(metric_value, float):
                print(f"  {metric_key}: {metric_value:.4f}")
            else:
                print(f"  {metric_key}: {metric_value}")

    print("=" * 60)

    try:
        log_results = {k: v for k, v in results.items() if k != "samples"}
        logger.info(
            f"Full results (without samples):\n"
            f"{json.dumps(log_results, indent=2, default=str)}"
        )
    except Exception:
        logger.warning("결과 JSON 직렬화 실패", exc_info=True)


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
        help="샘플 수 제한 (디버깅용)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="결과 저장 디렉토리 (기본값: results/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. 모델 경로 확정
    model_path = resolve_model_path(args.model, MODELS_DIR)
    model_name = model_path.name

    # 2. 태스크 리스트
    tasks = [t.strip() for t in args.tasks.split(",")]
    task_label = args.tasks.replace(",", "-")

    # 3. 경로 생성 (CSV + 로그)
    output_base = Path(args.output_dir)
    csv_path, log_path = generate_paths(model_name, task_label, output_base)

    # 4. 로깅 초기화
    logger = setup_logging_all(log_path)
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"CSV output: {csv_path}")

    # 5. 벤치마크 실행
    try:
        results = run_benchmark(
            model_path=model_path,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            logger=logger,
        )
    except Exception as e:
        logger.critical(f"벤치마크 실패: {e}", exc_info=True)
        sys.exit(1)

    # 6. CSV 저장 + 결과 출력
    extract_samples_to_csv(results, csv_path, logger)
    print_results(results, logger)
    logger.info("완료.")


if __name__ == "__main__":
    main()

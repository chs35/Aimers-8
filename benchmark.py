"""
EXAONE-4.0-1.2B 벤치마크 테스트
평가 서버와 동일한 vLLM 설정 사용
"""
import argparse
import time
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# vLLM 설정
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"  # suppress_stdout 비활성화 (Logger 호환성)

import yaml
import lm_eval
from lm_eval.models.vllm_causallms import VLLM

# 벤치마크 목록 로드
SCRIPT_DIR = Path(__file__).parent
BENCHMARKS_FILE = SCRIPT_DIR / "config" / "benchmark.yaml"

def load_benchmarks() -> dict:
    """benchmarks.yaml에서 벤치마크 목록 로드"""
    with open(BENCHMARKS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_reasoning_tasks(benchmarks: dict) -> set:
    """reasoning이 필요한 태스크 목록 반환"""
    return set(benchmarks.get("reasoning_tasks", []))


def get_categories(benchmarks: dict) -> dict:
    """카테고리 목록 반환"""
    return benchmarks.get("categories", {})


def get_all_tasks(benchmarks: dict, category: str = None) -> list:
    """카테고리별 또는 전체 태스크 목록 반환"""
    categories = get_categories(benchmarks)

    if category and category in categories:
        return categories[category]

    # 전체 태스크
    tasks = []
    for task_list in categories.values():
        tasks.extend(task_list)
    return tasks


class Logger:
    """로그를 파일과 콘솔에 동시 출력"""
    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def fileno(self):
        """vLLM 호환을 위해 terminal의 fileno 반환"""
        return self.terminal.fileno()

    def close(self):
        self.log_file.close()


def get_run_name(model_path: str) -> str:
    """실행 이름 생성: 날짜시간_모델명"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).name
    return f"{timestamp}_{model_name}"


def run_benchmark(model_path: str, tasks: list[str], use_reasoning: bool = False):
    """lm_eval로 벤치마크 실행 (시간 측정 포함)"""

    # 디렉토리 생성
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 실행 이름 및 파일 경로
    suffix = "_reasoning" if use_reasoning else ""
    run_name = get_run_name(model_path) + suffix
    log_path = f"logs/{run_name}.log"
    result_path = f"results/{run_name}.json"

    # 로거 설정
    logger = Logger(log_path)
    sys.stdout = logger

    try:
        tasks_str = ",".join(tasks)

        print(f"\n{'='*60}")
        print(f"Run Name: {run_name}")
        print(f"Model: {model_path}")
        print(f"Tasks: {tasks_str}")
        print(f"Use Reasoning: {use_reasoning}")
        print(f"Log: {log_path}")
        print(f"Result: {result_path}")
        print(f"{'='*60}\n")

        # 전체 시간 측정 시작 (모델 로드 포함)
        total_start = time.time()

        # 모델 로드 설정
        model_kwargs = {
            "pretrained": model_path,
            "trust_remote_code": True,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.85,
            "max_model_len": 16384,
            "batch_size": "auto",
            "enforce_eager": True,  # torch.compile segfault 방지
        }

        # reasoning 모드면 think_end_token 추가
        if use_reasoning:
            model_kwargs["think_end_token"] = "</think>"

        print("Loading model...")
        load_start = time.time()
        model = VLLM(**model_kwargs)
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f}s\n")

        # 각 태스크별 시간 측정
        task_times = {}
        all_results = {}
        total_bench_time = 0

        for task in tasks:
            print(f"\n{'='*60}")
            print(f"Running task: {task} (reasoning={use_reasoning})")
            print(f"{'='*60}")

            task_start = time.time()
            results = lm_eval.simple_evaluate(
                model=model,
                tasks=[task],
                batch_size="auto",
                apply_chat_template=True,
            )
            task_time = time.time() - task_start
            task_times[task] = task_time
            total_bench_time += task_time

            # 결과 저장
            if task in results["results"]:
                all_results[task] = results["results"][task]

            print(f"Task '{task}' completed in {task_time:.2f}s")

            # 태스크 결과 출력
            if task in results["results"]:
                for metric, value in results["results"][task].items():
                    if isinstance(value, (int, float)) and not metric.endswith("_stderr"):
                        if isinstance(value, float):
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: {value}")

        total_time = time.time() - total_start

        # 최종 결과 출력
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")

        for task_name, task_result in all_results.items():
            print(f"\n[{task_name}]")
            for metric, value in task_result.items():
                if isinstance(value, (int, float)) and not metric.endswith("_stderr"):
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")

        # 시간 측정 결과
        print(f"\n{'='*60}")
        print("TIME MEASUREMENT")
        print(f"{'='*60}")
        print(f"\n[Per Task]")
        for task, t in task_times.items():
            print(f"  {task}: {t:.2f}s")

        print(f"\n[Summary]")
        print(f"  Model load time:      {load_time:.2f}s")
        print(f"  Total benchmark time: {total_bench_time:.2f}s")
        print(f"  Total time:           {total_time:.2f}s")

        # JSON 결과 저장
        output_data = {
            "run_name": run_name,
            "model_path": model_path,
            "tasks": tasks,
            "use_reasoning": use_reasoning,
            "timestamp": datetime.now().isoformat(),
            "time": {
                "model_load_seconds": load_time,
                "per_task_seconds": task_times,
                "total_benchmark_seconds": total_bench_time,
                "total_seconds": total_time,
            },
            "results": all_results,
        }

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"Log saved: {log_path}")
        print(f"Result saved: {result_path}")
        print(f"{'='*60}")

    finally:
        sys.stdout = logger.terminal
        logger.close()


def main():
    parser = argparse.ArgumentParser(description="EXAONE Benchmark (평가 서버 동일 설정)")
    parser.add_argument("model_path", help="모델 경로 (예: /workspace/EXAONE-4.0-1.2B)")
    parser.add_argument("tasks", help="벤치마크 태스크 (카테고리명 또는 태스크명, 쉼표 구분)")
    parser.add_argument("--reasoning", "-r", action="store_true",
                        help="reasoning 모드 강제 활성화 (think_end_token 적용)")
    parser.add_argument("--no-reasoning", "-nr", action="store_true",
                        help="reasoning 모드 강제 비활성화")

    args = parser.parse_args()

    benchmarks = load_benchmarks()
    reasoning_tasks = get_reasoning_tasks(benchmarks)

    categories = get_categories(benchmarks)

    # 태스크 파싱
    if args.tasks == "all":
        tasks = get_all_tasks(benchmarks)
        # all인 경우 reasoning/non-reasoning 분리 실행
        reasoning_list = [t for t in tasks if t in reasoning_tasks]
        non_reasoning_list = [t for t in tasks if t not in reasoning_tasks]

        if non_reasoning_list:
            print(f"\n[Non-reasoning tasks: {', '.join(non_reasoning_list)}]")
            run_benchmark(args.model_path, non_reasoning_list, use_reasoning=False)

        if reasoning_list:
            print(f"\n[Reasoning tasks: {', '.join(reasoning_list)}]")
            run_benchmark(args.model_path, reasoning_list, use_reasoning=True)
    else:
        # 카테고리 또는 개별 태스크
        if args.tasks in categories:
            tasks = categories[args.tasks]
            # 카테고리 내 태스크 중 reasoning 태스크가 있는지 확인
            use_reasoning = any(t in reasoning_tasks for t in tasks)
        else:
            tasks = args.tasks.split(",")
            # 개별 태스크는 reasoning_tasks에 있는지 확인
            use_reasoning = any(t in reasoning_tasks for t in tasks)

        # 명령줄 옵션으로 오버라이드
        if args.reasoning:
            use_reasoning = True
        elif args.no_reasoning:
            use_reasoning = False

        run_benchmark(args.model_path, tasks, use_reasoning=use_reasoning)


if __name__ == "__main__":
    benchmarks = load_benchmarks()
    categories = get_categories(benchmarks)
    reasoning_tasks = get_reasoning_tasks(benchmarks)

    print("Available categories:")
    for cat, tasks in categories.items():
        task_marks = []
        for t in tasks:
            mark = f"{t}*" if t in reasoning_tasks else t
            task_marks.append(mark)
        print(f"  {cat}: {', '.join(task_marks)}")

    print("\n  (* = reasoning mode, think_end_token 적용)")
    print(f"\nReasoning tasks: {', '.join(reasoning_tasks)}")
    print()
    main()

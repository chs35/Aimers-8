"""
EXAONE-4.0-1.2B 벤치마크 테스트
평가 서버와 동일한 방식: vLLM 서버 + OpenAI API 호환
"""
import argparse
import subprocess
import time
import json
import os
import sys
import signal
import requests
from datetime import datetime
from pathlib import Path

import yaml
import lm_eval

# 벤치마크 목록 로드
SCRIPT_DIR = Path(__file__).parent
BENCHMARKS_FILE = SCRIPT_DIR / "config" / "benchmark.yaml"

# 기본 설정
DEFAULT_PORT = 8000
DEFAULT_HOST = "localhost"


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
        return self.terminal.fileno()

    def close(self):
        self.log_file.close()


def get_run_name(model_path: str, use_reasoning: bool) -> str:
    """실행 이름 생성: 날짜시간_모델명"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).name
    suffix = "_reasoning" if use_reasoning else ""
    return f"{timestamp}_{model_name}{suffix}"


def wait_for_server(base_url: str, timeout: int = 300) -> bool:
    """vLLM 서버가 준비될 때까지 대기"""
    start_time = time.time()
    health_url = f"{base_url}/health"

    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)

    return False


def start_vllm_server(
    model_path: str,
    port: int = DEFAULT_PORT,
    enable_thinking: bool = False,
) -> subprocess.Popen:
    """vLLM 서버 시작"""

    cmd = [
        "vllm", "serve", model_path,
        "--trust-remote-code",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "16384",
        "--enforce-eager",
        "--port", str(port),
    ]

    # enable_thinking 설정
    chat_template_kwargs = json.dumps({"enable_thinking": enable_thinking})
    cmd.extend(["--chat-template-kwargs", chat_template_kwargs])

    print(f"Starting vLLM server...")
    print(f"Command: {' '.join(cmd)}")

    # 서버 프로세스 시작
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,  # 프로세스 그룹 생성
    )

    return process


def stop_vllm_server(process: subprocess.Popen):
    """vLLM 서버 종료"""
    if process and process.poll() is None:
        print("Stopping vLLM server...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        print("Server stopped.")


def run_benchmark(
    model_path: str,
    tasks: list[str],
    use_reasoning: bool = False,
    port: int = DEFAULT_PORT,
):
    """lm_eval로 벤치마크 실행 (vLLM 서버 방식)"""

    # 디렉토리 생성
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 실행 이름 및 파일 경로
    run_name = get_run_name(model_path, use_reasoning)
    log_path = f"logs/{run_name}.log"
    result_path = f"results/{run_name}.json"

    # 로거 설정
    logger = Logger(log_path)
    sys.stdout = logger

    server_process = None
    base_url = f"http://{DEFAULT_HOST}:{port}/v1"

    try:
        tasks_str = ",".join(tasks)
        model_name = Path(model_path).name

        print(f"\n{'='*60}")
        print(f"Run Name: {run_name}")
        print(f"Model: {model_path}")
        print(f"Tasks: {tasks_str}")
        print(f"Use Reasoning: {use_reasoning}")
        print(f"Enable Thinking: {use_reasoning}")
        print(f"Server URL: {base_url}")
        print(f"Log: {log_path}")
        print(f"Result: {result_path}")
        print(f"{'='*60}\n")

        # 전체 시간 측정 시작
        total_start = time.time()

        # vLLM 서버 시작
        print("=" * 60)
        print("STARTING VLLM SERVER")
        print("=" * 60)

        server_start = time.time()
        server_process = start_vllm_server(
            model_path=model_path,
            port=port,
            enable_thinking=use_reasoning,
        )

        # 서버 준비 대기
        print("Waiting for server to be ready...")
        if not wait_for_server(f"http://{DEFAULT_HOST}:{port}"):
            raise RuntimeError("vLLM server failed to start within timeout")

        server_load_time = time.time() - server_start
        print(f"Server ready in {server_load_time:.2f}s\n")

        # model_args 설정 (local-chat-completions용)
        model_args = f"model={model_name},base_url={base_url},num_concurrent=1,max_retries=3,timeout=600"

        # think_end_token 설정 (reasoning 모드일 때)
        if use_reasoning:
            model_args += ",think_end_token=</think>"

        print(f"model_args: {model_args}\n")

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
                model="local-chat-completions",
                model_args=model_args,
                tasks=[task],
                batch_size=1,  # API 호출은 batch_size=1 권장
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
        print(f"  Server load time:     {server_load_time:.2f}s")
        print(f"  Total benchmark time: {total_bench_time:.2f}s")
        print(f"  Total time:           {total_time:.2f}s")

        # JSON 결과 저장
        output_data = {
            "run_name": run_name,
            "model_path": model_path,
            "model_name": model_name,
            "tasks": tasks,
            "use_reasoning": use_reasoning,
            "enable_thinking": use_reasoning,
            "think_end_token": "</think>" if use_reasoning else None,
            "server_url": base_url,
            "timestamp": datetime.now().isoformat(),
            "time": {
                "server_load_seconds": server_load_time,
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
        # 서버 종료
        stop_vllm_server(server_process)
        sys.stdout = logger.terminal
        logger.close()


def main():
    parser = argparse.ArgumentParser(
        description="EXAONE Benchmark (vLLM Server + OpenAI API 호환)"
    )
    parser.add_argument("model_path", help="모델 경로 (예: /workspace/EXAONE-4.0-1.2B)")
    parser.add_argument("tasks", help="벤치마크 태스크 (카테고리명 또는 태스크명, 쉼표 구분)")
    parser.add_argument("--reasoning", "-r", action="store_true",
                        help="reasoning 모드 강제 활성화 (enable_thinking=True)")
    parser.add_argument("--no-reasoning", "-nr", action="store_true",
                        help="reasoning 모드 강제 비활성화 (enable_thinking=False)")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT,
                        help=f"vLLM 서버 포트 (기본값: {DEFAULT_PORT})")

    args = parser.parse_args()

    benchmarks = load_benchmarks()
    reasoning_tasks = get_reasoning_tasks(benchmarks)
    categories = get_categories(benchmarks)

    # 태스크 파싱
    if args.tasks == "all":
        tasks = get_all_tasks(benchmarks)
    elif args.tasks in categories:
        tasks = categories[args.tasks]
    else:
        tasks = args.tasks.split(",")

    # 명령줄 옵션으로 강제 지정된 경우
    if args.reasoning:
        run_benchmark(args.model_path, tasks, use_reasoning=True, port=args.port)
    elif args.no_reasoning:
        run_benchmark(args.model_path, tasks, use_reasoning=False, port=args.port)
    else:
        # 자동 분리: reasoning/non-reasoning 태스크 구분
        reasoning_list = [t for t in tasks if t in reasoning_tasks]
        non_reasoning_list = [t for t in tasks if t not in reasoning_tasks]

        if non_reasoning_list and reasoning_list:
            print(f"\n[Auto-split: reasoning과 non-reasoning 태스크 분리 실행]")
            print(f"  Non-reasoning: {', '.join(non_reasoning_list)}")
            print(f"  Reasoning: {', '.join(reasoning_list)}")

            run_benchmark(args.model_path, non_reasoning_list, use_reasoning=False, port=args.port)
            run_benchmark(args.model_path, reasoning_list, use_reasoning=True, port=args.port)
        elif reasoning_list:
            run_benchmark(args.model_path, reasoning_list, use_reasoning=True, port=args.port)
        else:
            run_benchmark(args.model_path, non_reasoning_list, use_reasoning=False, port=args.port)


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

    print("\n  (* = reasoning mode, enable_thinking=True)")
    print(f"\nReasoning tasks: {', '.join(reasoning_tasks)}")
    print()
    main()

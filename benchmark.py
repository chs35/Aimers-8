"""
vLLM 직접 로딩 벤치마크 테스트
lm_eval의 vllm 모델 타입 사용

대회 평가 서버와 동일한 vLLM Serving 옵션:
- tensor_parallel_size = 1
- gpu_memory_utilization = 0.85
- batch_size = auto
- max_gen_toks = 16384
- apply_chat_template = true
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
import lm_eval
from transformers import AutoTokenizer

# 벤치마크 설정 파일
SCRIPT_DIR = Path(__file__).parent
BENCHMARKS_FILE = SCRIPT_DIR / "config" / "benchmark.yaml"

# 대회 평가 서버와 동일한 vLLM Serving 옵션
VLLM_SETTINGS = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.85,
    "max_gen_toks": 16384,
    "trust_remote_code": True,
    # 추가 설정 (대회 미명시, 로컬 테스트용)
    "max_model_len": 16384,
    "enforce_eager": True,  # torch.compile segfault 방지
}


def load_benchmarks() -> dict:
    """benchmark.yaml에서 벤치마크 목록 로드"""
    with open(BENCHMARKS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    """stdout/stderr를 파일과 콘솔에 동시 출력"""
    def __init__(self, log_path: str, stream_type: str = "stdout"):
        self.stream_type = stream_type
        if stream_type == "stdout":
            self.terminal = sys.stdout
        else:
            self.terminal = sys.stderr
        self.log_file = open(log_path, "a", encoding="utf-8")

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


def get_run_name(model_name: str) -> str:
    """실행 이름 생성: 날짜시간_모델명"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = Path(model_name).name
    return f"{timestamp}_{name}"


def build_model_args(model_path: str, enable_thinking: bool = False) -> str:
    """vLLM model_args 문자열 생성 (대회 설정과 동일)"""
    args = [
        f"pretrained={model_path}",
        f"tensor_parallel_size={VLLM_SETTINGS['tensor_parallel_size']}",
        f"gpu_memory_utilization={VLLM_SETTINGS['gpu_memory_utilization']}",
        f"max_gen_toks={VLLM_SETTINGS['max_gen_toks']}",
        f"trust_remote_code={str(VLLM_SETTINGS['trust_remote_code']).lower()}",
        f"max_model_len={VLLM_SETTINGS['max_model_len']}",
        f"enforce_eager={str(VLLM_SETTINGS['enforce_eager']).lower()}",
        # enable_thinking: chat template에 전달 (False면 thinking 비활성화)
        f"enable_thinking={str(enable_thinking).lower()}",
    ]
    return ",".join(args)


def count_tokens(tokenizer, text: str) -> int:
    """텍스트의 토큰 수 계산"""
    if not text or text == "N/A":
        return 0
    try:
        return len(tokenizer.encode(text))
    except:
        return 0


def run_benchmark(
    model_path: str,
    tasks: list[str],
    enable_thinking: bool = False,
):
    """lm_eval로 vLLM 직접 로딩 벤치마크 실행 (대회 설정과 동일)"""

    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    run_name = get_run_name(model_path)
    log_path = f"logs/{run_name}.log"
    result_path = f"results/{run_name}.json"

    # 로그 파일 초기화
    open(log_path, "w").close()

    # stdout, stderr 모두 캡처
    stdout_logger = Logger(log_path, "stdout")
    stderr_logger = Logger(log_path, "stderr")
    sys.stdout = stdout_logger
    sys.stderr = stderr_logger

    try:
        tasks_str = ",".join(tasks)
        model_args = build_model_args(model_path, enable_thinking=enable_thinking)

        print(f"\n{'='*60}")
        print(f"Run Name: {run_name}")
        print(f"Model: {model_path}")
        print(f"Tasks: {tasks_str}")
        print(f"enable_thinking: {enable_thinking}")
        print(f"vLLM Settings: {VLLM_SETTINGS}")
        print(f"model_args: {model_args}")
        print(f"Log: {log_path}")
        print(f"Result: {result_path}")
        print(f"{'='*60}\n")

        # 토크나이저 로드 (토큰 수 계산용)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        total_start = time.time()

        # 모델 로드 시간 측정 (첫 번째 태스크에서 측정)
        print(f"Loading model...")
        model_load_start = time.time()

        # 태스크별 개별 실행 (시간 측정을 위해)
        all_results = {}
        all_samples = {}
        task_times = {}
        task_tokens = {}
        task_time_per_token = {}

        for task_idx, task in enumerate(tasks):
            print(f"\n{'='*60}")
            print(f"Running task [{task_idx+1}/{len(tasks)}]: {task}")
            print(f"{'='*60}")

            task_start = time.time()

            # think_end_token이 model_args에 설정되어 있으면
            # 모델이 <think>...</think>를 출력해도 </think> 이후 내용만 평가에 사용됨
            results = lm_eval.simple_evaluate(
                model="vllm",
                model_args=model_args,
                tasks=[task],
                batch_size="auto",
                apply_chat_template=True,
                log_samples=True,
            )

            task_time = time.time() - task_start

            # 첫 번째 태스크에서 모델 로드 시간 계산
            if task_idx == 0:
                model_load_time = task_time * 0.1  # 대략적인 추정 (실제로는 lm_eval 내부에서 측정 필요)

            # 결과 수집
            if task in results.get("results", {}):
                all_results[task] = results["results"][task]
            if task in results.get("samples", {}):
                all_samples[task] = results["samples"][task]

            # 출력 토큰 수 계산 (doc_id로 중복 제거)
            # lm_eval은 여러 메트릭(strict-match, flexible-extract 등)에 대해
            # 동일한 샘플을 중복 기록하므로 doc_id로 중복 제거 필요
            total_output_tokens = 0
            samples = results.get("samples", {}).get(task, [])
            seen_doc_ids = set()
            for sample in samples:
                doc_id = sample.get("doc_id")
                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                resps = sample.get("resps", [])
                if resps and len(resps) > 0:
                    resp = resps[0][0] if isinstance(resps[0], (list, tuple)) else resps[0]
                    total_output_tokens += count_tokens(tokenizer, resp)

            task_times[task] = task_time
            task_tokens[task] = total_output_tokens

            # 토큰당 추론 시간 계산 (ms/token)
            if total_output_tokens > 0:
                time_per_token = (task_time * 1000) / total_output_tokens  # ms per token
                task_time_per_token[task] = time_per_token
            else:
                task_time_per_token[task] = 0

            print(f"\nTask '{task}' completed:")
            print(f"  Time: {task_time:.2f}s")
            print(f"  Output tokens: {total_output_tokens}")
            print(f"  Time per token: {task_time_per_token[task]:.2f} ms/token")

        total_time = time.time() - total_start
        total_tokens = sum(task_tokens.values())
        total_inference_time = sum(task_times.values())

        # 전체 토큰당 추론 시간
        if total_tokens > 0:
            overall_time_per_token = (total_inference_time * 1000) / total_tokens
        else:
            overall_time_per_token = 0

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

        print(f"\n{'='*60}")
        print("TIME MEASUREMENT")
        print(f"{'='*60}")
        print(f"\n[Per Task]")
        for task in tasks:
            print(f"  {task}:")
            print(f"    Time: {task_times.get(task, 0):.2f}s")
            print(f"    Tokens: {task_tokens.get(task, 0)}")
            print(f"    Time/Token: {task_time_per_token.get(task, 0):.2f} ms/token")

        print(f"\n[Summary]")
        print(f"  Total inference time: {total_inference_time:.2f}s")
        print(f"  Total output tokens: {total_tokens}")
        print(f"  Overall time per token: {overall_time_per_token:.2f} ms/token")
        print(f"  Total time (with overhead): {total_time:.2f}s")

        # 샘플별 상세 로그 출력 (doc_id로 중복 제거)
        print(f"\n{'='*60}")
        print("SAMPLE DETAILS")
        print(f"{'='*60}")

        for task_name, samples in all_samples.items():
            # doc_id로 중복 제거
            unique_samples = {}
            for sample in samples:
                doc_id = sample.get("doc_id")
                if doc_id not in unique_samples:
                    unique_samples[doc_id] = sample

            print(f"\n[{task_name}] - {len(unique_samples)} samples (unique)")
            print("-" * 60)

            for i, (doc_id, sample) in enumerate(unique_samples.items(), start=1):
                doc = sample.get("doc", {})

                # 문제/질문 추출 (태스크마다 필드명이 다름)
                question = doc.get("question") or doc.get("query") or doc.get("problem") or str(doc)

                # 모델 응답 추출
                resps = sample.get("resps", [])
                if resps and len(resps) > 0:
                    model_response = resps[0][0] if isinstance(resps[0], (list, tuple)) else resps[0]
                else:
                    model_response = "N/A"

                # 추출된 답변
                filtered = sample.get("filtered_resps", ["N/A"])

                # 정답 추출 (태스크마다 필드명이 다름)
                target = sample.get("target") or doc.get("answer") or doc.get("label") or "N/A"

                print(f"\n--- Sample {i} ---")
                print(f"Question: {question}")
                print(f"Model Response: {model_response}")
                print(f"Extracted Answer: {filtered}")
                print(f"Correct Answer: {target}")

        # JSON 결과 저장
        output_data = {
            "run_name": run_name,
            "model_path": model_path,
            "tasks": tasks,
            "enable_thinking": enable_thinking,
            "vllm_settings": VLLM_SETTINGS,
            "timestamp": datetime.now().isoformat(),
            "time": {
                "total_seconds": total_time,
                "total_inference_seconds": total_inference_time,
                "total_output_tokens": total_tokens,
                "overall_ms_per_token": overall_time_per_token,
                "per_task": {
                    task: {
                        "seconds": task_times.get(task, 0),
                        "output_tokens": task_tokens.get(task, 0),
                        "ms_per_token": task_time_per_token.get(task, 0),
                    }
                    for task in tasks
                },
            },
            "results": {},
        }

        for task_name, task_result in all_results.items():
            # doc_id로 중복 제거 후 정답/오답 번호 그룹화 (1부터 시작, 로그와 일치)
            samples = all_samples.get(task_name, [])
            unique_samples = {}
            for sample in samples:
                doc_id = sample.get("doc_id")
                if doc_id not in unique_samples:
                    unique_samples[doc_id] = sample

            correct_ids = []
            incorrect_ids = []
            for i, (doc_id, sample) in enumerate(unique_samples.items(), start=1):
                # 정답 여부 확인 (exact_match 또는 acc 메트릭 기준)
                is_correct = sample.get("exact_match") or sample.get("acc") or False
                if is_correct:
                    correct_ids.append(i)
                else:
                    incorrect_ids.append(i)

            output_data["results"][task_name] = {
                "enable_thinking": enable_thinking,
                "metrics": task_result,
                "num_samples": len(unique_samples),
                "correct": correct_ids,
                "incorrect": incorrect_ids,
            }

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"Log saved: {log_path}")
        print(f"Result saved: {result_path}")
        print(f"{'='*60}")

    finally:
        sys.stdout = stdout_logger.terminal
        sys.stderr = stderr_logger.terminal
        stdout_logger.close()
        stderr_logger.close()


def main():
    parser = argparse.ArgumentParser(description="vLLM 벤치마크 (대회 설정과 동일)")
    parser.add_argument("tasks", help="벤치마크 태스크 (카테고리명 또는 태스크명, 쉼표 구분)")
    parser.add_argument("--model", "-m", required=True,
                        help="모델 경로 (예: models/EXAONE-4.0-1.2B)")
    parser.add_argument("--thinking", "-t", action="store_true", default=False,
                        help="enable_thinking 활성화 (기본값: False, 대회 설정과 동일)")
    args = parser.parse_args()

    benchmarks = load_benchmarks()
    categories = get_categories(benchmarks)

    # 태스크 파싱
    if args.tasks == "all":
        tasks = get_all_tasks(benchmarks)
    elif args.tasks in categories:
        tasks = categories[args.tasks]
    else:
        tasks = args.tasks.split(",")

    run_benchmark(
        args.model,
        tasks,
        enable_thinking=args.thinking,
    )


if __name__ == "__main__":
    benchmarks = load_benchmarks()
    categories = get_categories(benchmarks)

    print("Available categories:")
    for cat, tasks in categories.items():
        print(f"  {cat}: {', '.join(tasks)}")
    print()
    main()

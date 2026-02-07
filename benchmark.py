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
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# NOTE: vLLM(0.14.1) + torch(2.9.x) 환경에서 torchdynamo/torch.compile 경로가
# logprobs(=multiple_choice/loglikelihood) 요청 시 segfault를 유발하는 사례가 있습니다.
# 로컬 벤치마크가 죽지 않도록 기본적으로 torchdynamo를 비활성화합니다.
# (필요 시 실행 전에 `TORCHDYNAMO_DISABLE=0`으로 override 가능)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import yaml
import lm_eval
from lm_eval.models import get_model
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
    "enforce_eager": False,  # torch.compile segfault 방지
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

    def log_only(self, message):
        """로그 파일에만 기록 (터미널 출력 없음)"""
        self.log_file.write(message + "\n")
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
        f"enforce_eager={str(VLLM_SETTINGS['enforce_eager']).lower()}",
        # enable_thinking: chat template에 전달 (False면 thinking 비활성화)
        f"enable_thinking={str(enable_thinking).lower()}",
    ]
    if VLLM_SETTINGS.get("kv_cache_dtype"):
        args.append(f"kv_cache_dtype={VLLM_SETTINGS['kv_cache_dtype']}")
    return ",".join(args)


def count_tokens(tokenizer, text: str) -> int:
    """텍스트의 토큰 수 계산"""
    if not text or text == "N/A":
        return 0
    try:
        return len(tokenizer.encode(text))
    except:
        return 0


def parse_vllm_info(log_path: str) -> dict:
    """로그 파일에서 vLLM 설정 정보 파싱"""
    vllm_info = {
        "max_num_batched_tokens": None,
        "gpu_kv_cache_tokens": None,
        "max_concurrency": None,
    }

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        # max_num_batched_tokens 파싱
        match = re.search(r"max_num_batched_tokens=(\d+)", content)
        if match:
            vllm_info["max_num_batched_tokens"] = int(match.group(1))

        # GPU KV cache size 파싱
        match = re.search(r"GPU KV cache size: ([\d,]+) tokens", content)
        if match:
            vllm_info["gpu_kv_cache_tokens"] = int(match.group(1).replace(",", ""))

        # Maximum concurrency 파싱
        match = re.search(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", content)
        if match:
            vllm_info["max_concurrency"] = float(match.group(1))

    except Exception:
        pass

    return vllm_info


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

        # 모델 로드 시간 측정 (추론 시간과 분리)
        print(f"Loading model...")
        model_load_start = time.time()
        lm = get_model("vllm", model_args)
        model_load_time = time.time() - model_load_start
        print(f"Model loaded in {model_load_time:.2f}s")

        # 태스크별 개별 실행 (시간 측정을 위해)
        all_results = {}
        all_samples = {}
        task_times = {}
        task_tokens = {}
        task_tps = {}  # Tokens Per Second (throughput)
        vllm_info = {}  # vLLM 내부 설정 (KV cache 등)

        for task_idx, task in enumerate(tasks):
            print(f"\n{'='*60}")
            print(f"Running task [{task_idx+1}/{len(tasks)}]: {task}")
            print(f"{'='*60}")

            task_start = time.time()

            # 이미 로드된 모델 재사용 (모델 로드 시간 제외)
            # think_end_token이 model_args에 설정되어 있으면
            # 모델이 <think>...</think>를 출력해도 </think> 이후 내용만 평가에 사용됨
            results = lm_eval.simple_evaluate(
                model=lm,
                tasks=[task],
                batch_size="auto",
                apply_chat_template=True,
                log_samples=True,
            )

            task_time = time.time() - task_start

            # 첫 번째 태스크 완료 후 vLLM 정보 파싱 (모델 로드 시 설정된 값)
            if task_idx == 0:
                vllm_info = parse_vllm_info(log_path)
                if vllm_info.get("max_num_batched_tokens"):
                    print(f"\n[vLLM Info]")
                    print(f"  max_num_batched_tokens: {vllm_info['max_num_batched_tokens']}")
                    print(f"  gpu_kv_cache_tokens: {vllm_info['gpu_kv_cache_tokens']}")
                    print(f"  max_concurrency: {vllm_info['max_concurrency']}x")

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

            # TPS (Tokens Per Second) 계산 - batch=auto 환경에서 throughput 측정
            if task_time > 0:
                tps = total_output_tokens / task_time  # tokens per second
                task_tps[task] = tps
            else:
                task_tps[task] = 0

            print(f"\nTask '{task}' completed:")
            print(f"  Time: {task_time:.2f}s")
            print(f"  Output tokens: {total_output_tokens}")
            print(f"  TPS: {task_tps[task]:.2f} tokens/s")

        total_time = time.time() - total_start
        total_tokens = sum(task_tokens.values())
        total_inference_time = sum(task_times.values())

        # 전체 TPS (Tokens Per Second)
        if total_inference_time > 0:
            overall_tps = total_tokens / total_inference_time
        else:
            overall_tps = 0

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
            print(f"    TPS: {task_tps.get(task, 0):.2f} tokens/s")

        print(f"\n[Summary]")
        print(f"  Model load time: {model_load_time:.2f}s")
        print(f"  Total inference time: {total_inference_time:.2f}s")
        print(f"  Total output tokens: {total_tokens}")
        print(f"  Overall TPS: {overall_tps:.2f} tokens/s")
        print(f"  Total time (with overhead): {total_time:.2f}s")

        # 샘플별 상세 로그 출력 (로그 파일에만 기록)
        stdout_logger.log_only(f"\n{'='*60}")
        stdout_logger.log_only("SAMPLE DETAILS")
        stdout_logger.log_only(f"{'='*60}")

        for task_name, samples in all_samples.items():
            # doc_id별로 모든 샘플 그룹화 (strict/flexible 분리)
            doc_samples_log = {}
            for sample in samples:
                doc_id = sample.get("doc_id")
                if doc_id is None:
                    continue
                if doc_id not in doc_samples_log:
                    doc_samples_log[doc_id] = []
                doc_samples_log[doc_id].append(sample)

            # 샘플 상세 정보는 로그 파일에만 기록 (터미널 출력 없음)
            stdout_logger.log_only(f"\n[{task_name}] - {len(doc_samples_log)} samples (unique)")
            stdout_logger.log_only("-" * 60)

            for i, doc_id in enumerate(sorted(doc_samples_log.keys()), start=1):
                samples_list = doc_samples_log[doc_id]
                sample = samples_list[0]  # 첫 번째 샘플에서 기본 정보 추출
                doc = sample.get("doc", {})

                # 문제/질문 추출 (태스크마다 필드명이 다름)
                question = doc.get("question") or doc.get("query") or doc.get("problem") or str(doc)

                # 모델 응답 추출
                resps = sample.get("resps", [])
                if resps and len(resps) > 0:
                    model_response = resps[0][0] if isinstance(resps[0], (list, tuple)) else resps[0]
                else:
                    model_response = "N/A"

                # strict/flexible 추출된 답변 분리
                if len(samples_list) >= 2:
                    strict_filtered = samples_list[0].get("filtered_resps", ["N/A"])
                    flexible_filtered = samples_list[1].get("filtered_resps", ["N/A"])
                else:
                    strict_filtered = sample.get("filtered_resps", ["N/A"])
                    flexible_filtered = strict_filtered

                # 정답 추출 (태스크마다 필드명이 다름)
                target = sample.get("target") or doc.get("answer") or doc.get("label") or "N/A"

                stdout_logger.log_only(f"\n--- Sample {i} ---")
                stdout_logger.log_only(f"Question: {question}")
                stdout_logger.log_only(f"Model Response: {model_response}")
                stdout_logger.log_only(f"Extracted (strict): {strict_filtered}")
                stdout_logger.log_only(f"Extracted (flexible): {flexible_filtered}")
                stdout_logger.log_only(f"Correct Answer: {target}")

        # JSON 결과 저장
        output_data = {
            "run_name": run_name,
            "model_path": model_path,
            "tasks": tasks,
            "enable_thinking": enable_thinking,
            "vllm_settings": VLLM_SETTINGS,
            "vllm_info": vllm_info,  # KV cache, max_num_batched_tokens 등
            "model_args": model_args,
            "timestamp": datetime.now().isoformat(),
            "time": {
                "total_seconds": total_time,
                "model_load_seconds": model_load_time,
                "total_inference_seconds": total_inference_time,
                "total_output_tokens": total_tokens,
                "overall_tps": overall_tps,
                "per_task": {
                    task: {
                        "seconds": task_times.get(task, 0),
                        "output_tokens": task_tokens.get(task, 0),
                        "tps": task_tps.get(task, 0),
                    }
                    for task in tasks
                },
            },
            "results": {},
        }

        for task_name, task_result in all_results.items():
            # doc_id별로 샘플 그룹화 (strict-match, flexible-extract 분리)
            samples = all_samples.get(task_name, [])
            doc_samples = {}
            for sample in samples:
                doc_id = sample.get("doc_id")
                if doc_id is None:
                    continue
                if doc_id not in doc_samples:
                    doc_samples[doc_id] = []
                doc_samples[doc_id].append(sample)

            # strict-match, flexible-extract 각각 정답/오답 추적
            strict_correct = []
            strict_incorrect = []
            flexible_correct = []
            flexible_incorrect = []

            sorted_doc_ids = sorted(doc_samples.keys())
            for i, doc_id in enumerate(sorted_doc_ids, start=1):
                samples_list = doc_samples[doc_id]

                if len(samples_list) >= 2:
                    # gsm8k 등: 첫 번째가 strict-match, 두 번째가 flexible-extract
                    strict_result = bool(samples_list[0].get("exact_match") or samples_list[0].get("acc"))
                    flexible_result = bool(samples_list[1].get("exact_match") or samples_list[1].get("acc"))
                else:
                    # 단일 메트릭 태스크 (hellaswag, mmlu 등)
                    result = bool(samples_list[0].get("exact_match") or samples_list[0].get("acc"))
                    strict_result = result
                    flexible_result = result

                if strict_result:
                    strict_correct.append(i)
                else:
                    strict_incorrect.append(i)

                if flexible_result:
                    flexible_correct.append(i)
                else:
                    flexible_incorrect.append(i)

            output_data["results"][task_name] = {
                "enable_thinking": enable_thinking,
                "metrics": task_result,
                "num_samples": len(doc_samples),
                "strict_match": {
                    "correct": strict_correct,
                    "incorrect": strict_incorrect,
                },
                "flexible_extract": {
                    "correct": flexible_correct,
                    "incorrect": flexible_incorrect,
                },
            }

        # JSON 저장 (correct/incorrect 배열은 한 줄로 압축)
        json_str = json.dumps(output_data, indent=2, ensure_ascii=False)

        def compact_int_array(match):
            """숫자 배열을 한 줄로 압축"""
            return re.sub(r'\s+', ' ', match.group(0))

        json_str = re.sub(r'"correct": \[\s*[\d,\s]*\]', compact_int_array, json_str)
        json_str = re.sub(r'"incorrect": \[\s*[\d,\s]*\]', compact_int_array, json_str)

        with open(result_path, "w", encoding="utf-8") as f:
            f.write(json_str)

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
    parser.add_argument(
        "--enforce-eager",
        dest="enforce_eager",
        action="store_true",
        help="vLLM enforce_eager=True (CUDA graph/compile 비활성화, 안정성 우선)",
    )
    parser.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="vLLM enforce_eager=False (대회 기본 설정에 더 근접, compile/cudagraph 사용)",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default=None,
        help="KV cache dtype (예: fp8_e5m2). 미지정 시 vLLM 기본값(auto) 사용",
    )
    parser.set_defaults(enforce_eager=VLLM_SETTINGS["enforce_eager"])
    args = parser.parse_args()

    # CLI 옵션으로 vLLM 설정 override
    VLLM_SETTINGS["enforce_eager"] = bool(args.enforce_eager)
    if args.kv_cache_dtype:
        VLLM_SETTINGS["kv_cache_dtype"] = args.kv_cache_dtype

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

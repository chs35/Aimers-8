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

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

PRIMARY_METRIC_PRIORITY = [
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "acc_norm",
    "acc",
    "exact_match",
]


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
        # 특수 토큰(BOS/EOS 등) 제외하고 응답 텍스트 토큰 수만 계산
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        # 일부 토크나이저 구현체는 add_special_tokens 인자를 받지 않을 수 있음
        return len(tokenizer.encode(text))
    except:
        return 0


def disable_tokenizer_truncation(tokenizer) -> None:
    """
    tokenizer.json에 저장된 truncation 설정이 토큰 카운팅을 왜곡하지 않도록 비활성화.
    (모델 추론 파이프라인이 아니라, 여기서는 로컬 카운팅 tokenizer에만 적용)
    """
    try:
        backend = getattr(tokenizer, "backend_tokenizer", None) or getattr(tokenizer, "_tokenizer", None)
        if backend is not None and hasattr(backend, "no_truncation"):
            backend.no_truncation()
    except Exception:
        pass


def parse_vllm_info(log_path: str) -> dict:
    """로그 파일에서 vLLM 설정 정보 파싱"""
    vllm_info = {
        "max_num_batched_tokens": None,
        "gpu_kv_cache_tokens": None,
        "max_concurrency": None,
        "model_load_seconds": None,
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

        # 모델 로드 시간 파싱 (예: "Model loading took 2.39 GiB memory and 7.664517 seconds")
        match = re.search(r"Model loading took .+ and ([\d.]+) seconds", content)
        if match:
            vllm_info["model_load_seconds"] = float(match.group(1))

    except Exception:
        pass

    return vllm_info


def strip_ansi(text: str) -> str:
    """ANSI escape 시퀀스 제거"""
    return ANSI_ESCAPE_RE.sub("", text)


def read_log_delta(log_path: str, start_offset: int) -> str:
    """start_offset 이후 로그만 읽기"""
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(start_offset)
            return f.read()
    except Exception:
        return ""


def parse_generate_metrics_from_log(log_text: str) -> dict:
    """
    lm_eval 진행 로그에서 generate 단계 메트릭 추출.
    - generate_seconds: Processed prompts 진행바의 최종 경과시간(mm:ss)
    - first_progress_seconds: 첫 번째 non-zero 진행까지의 시간(초)
    - est_output_tps: 진행바 마지막 output toks/s 추정치
    - est_input_tps: 진행바 마지막 input toks/s 추정치
    """
    metrics = {
        "generate_seconds": None,
        "first_progress_seconds": None,
        "est_output_tps": None,
        "est_input_tps": None,
    }

    if not log_text:
        return metrics

    cleaned = strip_ansi(log_text)
    lines = [line for line in re.split(r"[\r\n]+", cleaned) if line]

    first_nonzero_seconds = None
    final_seconds = -1
    final_output_tps = None
    final_input_tps = None

    for line in lines:
        if "Processed prompts:" not in line:
            continue

        match_time = re.search(r"\[(\d+):(\d+)<", line)
        if not match_time:
            continue

        seconds = int(match_time.group(1)) * 60 + int(match_time.group(2))
        if first_nonzero_seconds is None and seconds > 0:
            first_nonzero_seconds = seconds

        if seconds >= final_seconds:
            match_output = re.search(r"output:\s*([0-9]+(?:\.[0-9]+)?)", line)
            match_input = re.search(r"input:\s*([0-9]+(?:\.[0-9]+)?)", line)
            final_seconds = seconds
            if match_output:
                final_output_tps = float(match_output.group(1))
            if match_input:
                final_input_tps = float(match_input.group(1))

    if final_seconds >= 0:
        metrics["generate_seconds"] = float(final_seconds)
    if first_nonzero_seconds is not None:
        metrics["first_progress_seconds"] = float(first_nonzero_seconds)
    metrics["est_output_tps"] = final_output_tps
    metrics["est_input_tps"] = final_input_tps
    return metrics


def pick_primary_metric(metrics: dict) -> tuple[str | None, float | None]:
    """태스크별 대표 metric 선택"""
    for key in PRIMARY_METRIC_PRIORITY:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)

    for key, value in metrics.items():
        if key == "alias" or key.endswith("_stderr"):
            continue
        if isinstance(value, (int, float)):
            return key, float(value)

    return None, None


def summarize_perf(task_metrics: dict[str, dict]) -> dict:
    """
    태스크별 대표 metric + 평균 성능 요약.
    task_metrics: {"gsm8k": {...metric dict...}, ...}
    """
    per_task = {}
    values = []

    for task, metrics in task_metrics.items():
        metric_name, metric_value = pick_primary_metric(metrics)
        if metric_name is None or metric_value is None:
            continue
        per_task[task] = {
            "primary_metric": metric_name,
            "value": metric_value,
        }
        values.append(metric_value)

    mean_value = (sum(values) / len(values)) if values else None
    return {
        "per_task": per_task,
        "mean": mean_value,
    }


def load_baseline_summary(result_path: str | None) -> dict | None:
    """기존 결과 JSON에서 대회 산식 추정에 필요한 기준값 로드"""
    if not result_path:
        return None

    path = Path(result_path)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    baseline_task_metrics = {}
    for task_name, task_info in data.get("results", {}).items():
        if isinstance(task_info, dict):
            metrics = task_info.get("metrics")
            if isinstance(metrics, dict):
                baseline_task_metrics[task_name] = metrics

    time_info = data.get("time", {})
    generate_seconds = time_info.get("total_generate_seconds")
    if generate_seconds is None:
        generate_seconds = time_info.get("total_inference_seconds")

    tokens = time_info.get("total_output_tokens")
    if tokens is None:
        tokens = time_info.get("total_output_tokens_estimated")

    return {
        "path": str(path),
        "tasks": data.get("tasks", []),
        "perf": summarize_perf(baseline_task_metrics),
        "generate_seconds": generate_seconds,
        "tokens": tokens,
    }


def run_benchmark(
    model_path: str,
    tasks: list[str],
    enable_thinking: bool = False,
    baseline_result_path: str | None = None,
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
        disable_tokenizer_truncation(tokenizer)

        total_start = time.time()
        baseline_summary = load_baseline_summary(baseline_result_path)

        # 태스크별 개별 실행 (시간 측정을 위해)
        # 모델 로드 시간은 첫 번째 태스크 완료 후 로그에서 파싱
        all_results = {}
        all_samples = {}
        task_times = {}
        task_tokens = {}
        task_tps = {}  # wall-clock 기준 Tokens Per Second
        task_generate_seconds = {}  # generate phase만 추정 시간
        task_generate_tps = {}  # generate-only 기준 Tokens Per Second
        task_first_progress_seconds = {}  # 첫 토큰 도달 지연 근사
        task_est_output_tps = {}  # lm_eval 진행바 output toks/s
        task_est_input_tps = {}  # lm_eval 진행바 input toks/s
        vllm_info = {}  # vLLM 내부 설정 (KV cache 등)
        model_load_time = 0.0  # 모델 로드 시간 (첫 태스크에서 분리)

        for task_idx, task in enumerate(tasks):
            print(f"\n{'='*60}")
            print(f"Running task [{task_idx+1}/{len(tasks)}]: {task}")
            print(f"{'='*60}")

            task_start = time.time()
            log_offset_before_task = os.path.getsize(log_path)

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

            # 첫 번째 태스크 완료 후 vLLM 정보 파싱 (모델 로드 시 설정된 값)
            if task_idx == 0:
                vllm_info = parse_vllm_info(log_path)
                # 모델 로드 시간을 첫 번째 태스크 시간에서 분리
                if vllm_info.get("model_load_seconds"):
                    model_load_time = vllm_info["model_load_seconds"]
                    task_time = task_time - model_load_time  # 순수 추론 시간
                    print(f"\n[Model Load Time: {model_load_time:.2f}s separated from first task]")
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

            # generate 구간 전용 메트릭 파싱 (대회 산식 정렬용)
            task_log_chunk = read_log_delta(log_path, log_offset_before_task)
            parsed_generate = parse_generate_metrics_from_log(task_log_chunk)

            generate_seconds = parsed_generate.get("generate_seconds")
            if generate_seconds is None or generate_seconds <= 0:
                generate_seconds = task_time
            task_generate_seconds[task] = generate_seconds

            if generate_seconds and generate_seconds > 0:
                task_generate_tps[task] = total_output_tokens / generate_seconds
            else:
                task_generate_tps[task] = 0

            task_first_progress_seconds[task] = parsed_generate.get("first_progress_seconds")
            task_est_output_tps[task] = parsed_generate.get("est_output_tps")
            task_est_input_tps[task] = parsed_generate.get("est_input_tps")

            print(f"\nTask '{task}' completed:")
            print(f"  Wall time: {task_time:.2f}s")
            print(f"  Generate-only time: {generate_seconds:.2f}s")
            print(f"  Output tokens: {total_output_tokens}")
            print(f"  Wall TPS: {task_tps[task]:.2f} tokens/s")
            print(f"  Generate TPS: {task_generate_tps[task]:.2f} tokens/s")
            if task_est_output_tps[task] is not None:
                print(f"  vLLM est output TPS: {task_est_output_tps[task]:.2f} tokens/s")
            if task_first_progress_seconds[task] is not None:
                print(f"  First progress latency: {task_first_progress_seconds[task]:.2f}s")

        total_time = time.time() - total_start
        total_tokens = sum(task_tokens.values())
        total_inference_time = sum(task_times.values())
        total_generate_time = sum(task_generate_seconds.values())

        # 전체 TPS (Tokens Per Second)
        if total_inference_time > 0:
            overall_tps = total_tokens / total_inference_time
        else:
            overall_tps = 0

        if total_generate_time > 0:
            overall_generate_tps = total_tokens / total_generate_time
        else:
            overall_generate_tps = 0

        if total_tokens > 0:
            seconds_per_token_wall = total_inference_time / total_tokens
            seconds_per_token_generate = total_generate_time / total_tokens
        else:
            seconds_per_token_wall = None
            seconds_per_token_generate = None

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
            print(f"    Wall time: {task_times.get(task, 0):.2f}s")
            print(f"    Generate-only time: {task_generate_seconds.get(task, 0):.2f}s")
            print(f"    Tokens: {task_tokens.get(task, 0)}")
            print(f"    Wall TPS: {task_tps.get(task, 0):.2f} tokens/s")
            print(f"    Generate TPS: {task_generate_tps.get(task, 0):.2f} tokens/s")
            if task_est_output_tps.get(task) is not None:
                print(f"    vLLM est output TPS: {task_est_output_tps[task]:.2f} tokens/s")
            if task_first_progress_seconds.get(task) is not None:
                print(f"    First progress latency: {task_first_progress_seconds[task]:.2f}s")

        print(f"\n[Summary]")
        print(f"  Model load time: {model_load_time:.2f}s")
        print(f"  Total wall inference time: {total_inference_time:.2f}s")
        print(f"  Total generate-only time: {total_generate_time:.2f}s")
        print(f"  Total output tokens: {total_tokens}")
        print(f"  Overall wall TPS: {overall_tps:.2f} tokens/s")
        print(f"  Overall generate TPS: {overall_generate_tps:.2f} tokens/s")
        if seconds_per_token_wall is not None:
            print(f"  Wall seconds/token: {seconds_per_token_wall:.6f}")
        if seconds_per_token_generate is not None:
            print(f"  Generate-only seconds/token: {seconds_per_token_generate:.6f}")
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
                "total_inference_seconds": total_inference_time,  # wall-clock
                "total_generate_seconds": total_generate_time,  # generate phase only
                "total_output_tokens": total_tokens,
                "overall_tps": overall_tps,  # wall-clock
                "overall_generate_tps": overall_generate_tps,
                "seconds_per_output_token_wall": seconds_per_token_wall,
                "seconds_per_output_token_generate_only": seconds_per_token_generate,
                "per_task": {
                    task: {
                        "seconds": task_times.get(task, 0),  # wall-clock
                        "generate_seconds": task_generate_seconds.get(task, 0),
                        "first_progress_seconds": task_first_progress_seconds.get(task),
                        "output_tokens": task_tokens.get(task, 0),
                        "tps": task_tps.get(task, 0),  # wall-clock
                        "generate_tps": task_generate_tps.get(task, 0),
                        "est_output_tps": task_est_output_tps.get(task),
                        "est_input_tps": task_est_input_tps.get(task),
                    }
                    for task in tasks
                },
            },
            "performance": summarize_perf(all_results),
            "contest_estimate": {},
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

        # 대회 산식 추정치 계산 (기준 결과 파일이 있을 때만)
        contest_estimate = {
            "enabled": bool(baseline_summary),
            "baseline_result_path": baseline_summary["path"] if baseline_summary else None,
            "perf_norm": None,
            "speed_norm": None,
            "score": None,
            "perf_model": None,
            "perf_base": None,
            "seconds_per_token_model": seconds_per_token_generate,
            "seconds_per_token_base": None,
            "speed_source": "generate_only",
            "note": (
                "SpeedNorm/Score는 로컬 추정치이며, 평가서버 비공개 데이터/고정 script.py와 차이가 날 수 있음."
            ),
        }

        if baseline_summary:
            current_perf = output_data["performance"]
            baseline_perf = baseline_summary.get("perf", {})

            current_perf_tasks = current_perf.get("per_task", {})
            baseline_perf_tasks = baseline_perf.get("per_task", {})
            common_tasks = sorted(set(current_perf_tasks.keys()) & set(baseline_perf_tasks.keys()))

            if common_tasks:
                model_perf_values = [current_perf_tasks[t]["value"] for t in common_tasks]
                base_perf_values = [baseline_perf_tasks[t]["value"] for t in common_tasks]
                perf_model = sum(model_perf_values) / len(model_perf_values)
                perf_base = sum(base_perf_values) / len(base_perf_values)
            else:
                perf_model = current_perf.get("mean")
                perf_base = baseline_perf.get("mean")

            contest_estimate["perf_model"] = perf_model
            contest_estimate["perf_base"] = perf_base
            if isinstance(perf_model, (int, float)) and isinstance(perf_base, (int, float)) and perf_base > 0:
                contest_estimate["perf_norm"] = perf_model / perf_base

            base_generate_seconds = baseline_summary.get("generate_seconds")
            base_tokens = baseline_summary.get("tokens")
            if isinstance(base_generate_seconds, (int, float)) and isinstance(base_tokens, (int, float)) and base_tokens > 0:
                base_seconds_per_token = base_generate_seconds / base_tokens
                contest_estimate["seconds_per_token_base"] = base_seconds_per_token

                if (
                    isinstance(seconds_per_token_generate, (int, float))
                    and seconds_per_token_generate > 0
                    and base_seconds_per_token > 0
                ):
                    contest_estimate["speed_norm"] = 1 - (seconds_per_token_generate / base_seconds_per_token)

            if (
                isinstance(contest_estimate.get("perf_norm"), (int, float))
                and isinstance(contest_estimate.get("speed_norm"), (int, float))
            ):
                contest_estimate["score"] = max(
                    0.5 * contest_estimate["perf_norm"] + 0.5 * contest_estimate["speed_norm"],
                    0.0,
                )

            print(f"\n{'='*60}")
            print("CONTEST SCORE ESTIMATE")
            print(f"{'='*60}")
            print(f"  Baseline result: {baseline_summary['path']}")
            if contest_estimate["perf_norm"] is not None:
                print(f"  PerfNorm: {contest_estimate['perf_norm']:.6f}")
            else:
                print("  PerfNorm: N/A")
            if contest_estimate["speed_norm"] is not None:
                print(f"  SpeedNorm: {contest_estimate['speed_norm']:.6f}")
            else:
                print("  SpeedNorm: N/A")
            if contest_estimate["score"] is not None:
                print(f"  Score (estimated): {contest_estimate['score']:.6f}")
            else:
                print("  Score (estimated): N/A")

        output_data["contest_estimate"] = contest_estimate

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
    parser.add_argument(
        "--baseline-result",
        type=str,
        default=None,
        help=(
            "기준(base) 결과 JSON 경로. 지정 시 대회 산식 "
            "(PerfNorm/SpeedNorm/Score) 로컬 추정치를 함께 계산"
        ),
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
        baseline_result_path=args.baseline_result,
    )


if __name__ == "__main__":
    benchmarks = load_benchmarks()
    categories = get_categories(benchmarks)

    print("Available categories:")
    for cat, tasks in categories.items():
        print(f"  {cat}: {', '.join(tasks)}")
    print()
    main()

# Benchmark 사용법

EXAONE-4.0 모델 벤치마크 테스트 도구

## 기본 사용법

```bash
python benchmark.py <모델경로> <태스크>
```

## 예시

### 카테고리별 실행

```bash
# 수학 벤치마크 (reasoning 자동 적용)
python benchmark.py /workspace/EXAONE-4.0-1.2B math

# 영어 벤치마크
python benchmark.py /workspace/EXAONE-4.0-1.2B english

# 한국어 벤치마크
python benchmark.py /workspace/EXAONE-4.0-1.2B korean

# 코드 벤치마크 (reasoning 자동 적용)
python benchmark.py /workspace/EXAONE-4.0-1.2B code
```

### 개별 태스크 실행

```bash
# 단일 태스크
python benchmark.py /workspace/EXAONE-4.0-1.2B gsm8k

# 여러 태스크 (쉼표 구분)
python benchmark.py /workspace/EXAONE-4.0-1.2B gsm8k,hellaswag,arc_easy
```

### 전체 벤치마크 실행

```bash
# reasoning/non-reasoning 자동 분리 실행
python benchmark.py /workspace/EXAONE-4.0-1.2B all
```

### Reasoning 모드 강제 설정

```bash
# reasoning 강제 활성화
python benchmark.py /workspace/EXAONE-4.0-1.2B hellaswag --reasoning
python benchmark.py /workspace/EXAONE-4.0-1.2B hellaswag -r

# reasoning 강제 비활성화
python benchmark.py /workspace/EXAONE-4.0-1.2B gsm8k --no-reasoning
python benchmark.py /workspace/EXAONE-4.0-1.2B gsm8k -nr
```

## 설정 파일

`config/benchmark.yaml`에서 카테고리와 reasoning 태스크를 설정합니다.

```yaml
# 카테고리 정의
categories:
  korean:
    - klue_nli
    - klue_sts
  math:
    - gsm8k
    - math
  # ...

# reasoning이 필요한 태스크 목록
reasoning_tasks:
  - gsm8k
  - math
  - humaneval
  - mbpp
```

## Reasoning 모드란?

EXAONE-4.0은 `<think>...</think>` 태그로 추론 과정을 출력합니다.

- **Reasoning 모드 활성화**: `think_end_token="</think>"` 적용하여 추론 과정을 제거하고 최종 답만 추출
- **Reasoning 모드 비활성화**: 출력 그대로 평가

### Reasoning이 필요한 태스크

| 태스크 | Reasoning | 이유 |
|--------|-----------|------|
| gsm8k, math | O | 수학 문제, CoT 추론 필요 |
| humaneval, mbpp | O | 코드 생성, 추론 필요 |
| mmlu, hellaswag | X | 객관식/단답형 |
| arc, piqa, winogrande | X | 객관식/단답형 |

## 출력 파일

실행 후 다음 파일이 생성됩니다:

```
logs/
  └── 20250203_120000_EXAONE-4.0-1.2B.log          # 실행 로그
  └── 20250203_120000_EXAONE-4.0-1.2B_reasoning.log  # reasoning 모드 로그

results/
  └── 20250203_120000_EXAONE-4.0-1.2B.json          # 결과 JSON
  └── 20250203_120000_EXAONE-4.0-1.2B_reasoning.json # reasoning 모드 결과
```

### JSON 결과 구조

```json
{
  "run_name": "20250203_120000_EXAONE-4.0-1.2B",
  "model_path": "/workspace/EXAONE-4.0-1.2B",
  "tasks": ["gsm8k"],
  "use_reasoning": true,
  "timestamp": "2025-02-03T12:00:00",
  "time": {
    "model_load_seconds": 12.34,
    "per_task_seconds": {"gsm8k": 294.88},
    "total_benchmark_seconds": 294.88,
    "total_seconds": 307.22
  },
  "results": {
    "gsm8k": {
      "exact_match,strict-match": 0.65,
      "exact_match,flexible-extract": 0.68
    }
  }
}
```

## 평가 서버 설정

평가 서버와 동일한 vLLM 설정을 사용합니다:

```python
tensor_parallel_size = 1
gpu_memory_utilization = 0.85
max_model_len = 16384
batch_size = "auto"
apply_chat_template = True
```

## 문제 해결

### Segfault 발생 시

L4 GPU에서 torch.compile 관련 segfault가 발생할 수 있습니다.
코드에 `enforce_eager=True`가 설정되어 있어 자동으로 우회됩니다.

### 낮은 점수가 나올 때

1. Reasoning 모드가 올바르게 설정되었는지 확인
2. `--reasoning` 또는 `--no-reasoning` 옵션으로 테스트
3. `config/benchmark.yaml`의 `reasoning_tasks` 확인

# Benchmark 사용법

EXAONE-4.0 모델 벤치마크 테스트 도구 (대회 평가와 동일한 vLLM 설정)

## 사용법

```bash
# 카테고리별 실행
python benchmark.py math --model models/EXAONE-4.0-1.2B
python benchmark.py korean --model models/EXAONE-4.0-1.2B
python benchmark.py all --model models/EXAONE-4.0-1.2B

# 개별 태스크
python benchmark.py gsm8k --model models/EXAONE-4.0-1.2B
python benchmark.py gsm8k,hellaswag,arc_easy --model models/EXAONE-4.0-1.2B
```

## 옵션

| 옵션 | 단축 | 기본값 | 설명 |
|-----|------|-------|------|
| `--model` | `-m` | (필수) | 모델 경로 |
| `--thinking` | `-t` | False | enable_thinking 활성화 |

### enable_thinking 옵션

EXAONE-4.0은 reasoning 모드(`enable_thinking`)를 지원합니다.

```bash
# thinking 비활성화 (기본값, 대회 설정과 동일)
python benchmark.py gsm8k --model models/EXAONE-4.0-1.2B

# thinking 활성화
python benchmark.py gsm8k --model models/EXAONE-4.0-1.2B --thinking
```

- `enable_thinking=False` (기본값): 모델이 `<think>` 블록 없이 바로 답변 출력
- `enable_thinking=True`: 모델이 `<think>...</think>` 블록으로 추론 과정 출력 후 답변

> 대회 평가 서버는 enable_thinking을 명시하지 않으므로 기본값(False)이 적용됩니다.

## vLLM 설정 (대회와 동일)

대회 평가 서버와 동일한 vLLM Serving 옵션:

```
tensor_parallel_size = 1
gpu_memory_utilization = 0.85
batch_size = auto
max_gen_toks = 16384
apply_chat_template = true
```

추가 로컬 테스트 설정 (대회 미명시):
```
enforce_eager = true  # torch.compile segfault 방지
trust_remote_code = true
```

## 카테고리 목록

`config/benchmark.yaml`에서 설정:

| 카테고리 | 태스크 |
|---------|--------|
| korean | klue_nli, klue_sts, kobest_hellaswag, kobest_copa |
| english | hellaswag, arc_easy, arc_challenge, winogrande, piqa |
| math | gsm8k, math |
| reasoning | mmlu, truthfulqa_mc2 |
| code | humaneval, mbpp |

## 출력 파일

```
logs/
  └── 20250203_120000_EXAONE-4.0-1.2B.log    # 상세 로그 (모델 응답 포함)

results/
  └── 20250203_120000_EXAONE-4.0-1.2B.json   # 결과 요약
```

### 로그 파일 내용

각 샘플별 상세 정보가 기록됩니다:
```
--- Sample 1 ---
Question: Janet's ducks lay 16 eggs per day...
Model Response: First, Janet's ducks lay 16 eggs...
Extracted Answer: ['18']
Correct Answer: #### 18
```

### JSON 결과 구조

```json
{
  "run_name": "20250203_120000_EXAONE-4.0-1.2B",
  "model_path": "models/EXAONE-4.0-1.2B",
  "tasks": ["gsm8k"],
  "enable_thinking": false,
  "vllm_settings": {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.85,
    "max_gen_toks": 16384,
    "trust_remote_code": true,
    "enforce_eager": true
  },
  "timestamp": "2025-02-03T12:00:00",
  "time": {
    "total_seconds": 620.0,
    "total_inference_seconds": 580.0,
    "total_output_tokens": 125000,
    "overall_ms_per_token": 4.64,
    "per_task": {
      "gsm8k": {
        "seconds": 580.0,
        "output_tokens": 125000,
        "ms_per_token": 4.64
      }
    }
  },
  "results": {
    "gsm8k": {
      "enable_thinking": false,
      "metrics": {
        "exact_match,strict-match": 0.65,
        "exact_match,strict-match_stderr": 0.012,
        "exact_match,flexible-extract": 0.68,
        "exact_match,flexible-extract_stderr": 0.011
      },
      "num_samples": 1319,
      "strict_match": {
        "correct": [2, 3, 6, 7, ...],
        "incorrect": [1, 4, 5, 8, ...]
      },
      "flexible_extract": {
        "correct": [1, 2, 3, 6, 7, ...],
        "incorrect": [4, 5, 8, ...]
      }
    }
  }
}
```

- `enable_thinking`: thinking 모드 활성화 여부 (전체 및 태스크별)
- `num_samples`: 실제 평가 샘플 수 (중복 제거)
- `strict_match`: 엄격한 정답 매칭 결과
  - `correct`: 정답 샘플 번호 목록 (1부터 시작)
  - `incorrect`: 오답 샘플 번호 목록
- `flexible_extract`: 유연한 정답 추출 결과
  - `correct`: 정답 샘플 번호 목록 (1부터 시작)
  - `incorrect`: 오답 샘플 번호 목록
- `ms_per_token`: 토큰당 추론 시간 (대회 점수 산정에 사용)

## 문제 해결

### Segfault 발생 시

L4 GPU에서 torch.compile 관련 segfault가 발생할 수 있습니다.
`enforce_eager=True` 설정이 기본으로 적용되어 자동으로 우회됩니다.

## 대화 모드 (init_chat.py)

벤치마크가 아닌 대화용:

```bash
python init_chat.py --model models/EXAONE-4.0-1.2B
```

- `quit` / `q` - 종료
- `clear` - 대화 초기화

# DACON 평가 서버 재현 기술 조사 보고서 (Ko-LongRAG 우선, v2)
기준일: 2026-02-08

## 1. 결론 요약
1. 대회 서버 재현의 핵심은 `고정 Docker 환경 + 고정 vLLM 설정 + 오프라인 배치 추론`이다. (근거: `README.md:164`, `README.md:165`, `README.md:167`, `README.md:168`, `Dockerfile:4`, `Dockerfile:58`, `Dockerfile:87`)
2. Ko-LongRAG는 로컬 점수 예측용 1순위 벤치마크로 적합하며, `singledocQA/multidocQA/unanswerable`를 분리 측정해야 예측력이 올라간다. (근거: `docs/kolongrag.md:43`, `docs/kolongrag.md:44`, `docs/kolongrag.md:45`, `docs/kolongrag.md:48`)
3. Ko-LongRAG 공식 프로토콜은 GPT-4o Judge를 쓰지만, 대회 서버는 인터넷 불가이므로 서버 실평가는 자동 메트릭 기반일 가능성이 높다. GPT-4o Judge는 로컬 캘리브레이션 용도로 쓰는 것이 현실적이다. (근거: `docs/kolongrag_examples.md:52`, `docs/kolongrag_examples.md:55`, `README.md:177`)
4. 현재 저장소의 로컬 로그 수치는 정확한 base 점수 근거로 확정할 수 없으므로, 수치 자체가 아니라 파이프라인 구조 검증 근거로만 사용한다.
5. 본 계획의 실행 출력은 평가산식 재현이 아니라 `perf_model`, `infer_sec_model`, `tok_model`, `tpt_model` 4개 지표로 고정한다.

## 2. 근거 요약 맵
| 범주 | 핵심 사실 | 근거 |
|---|---|---|
| 산식(참고) | `Score = max(0.5*PerfNorm + 0.5*SpeedNorm, 0)` | `README.md:35`, `README.md:39`, `README.md:43` |
| 제출/서버 | `submit.zip`에 `model/`만 제출, 서버가 `script.py`, `data/`, `output/submission.csv` 추가 | `README.md:69`, `README.md:71`, `README.md:74` |
| 오프라인 제약 | 평가 서버 인터넷 불가 | `README.md:177` |
| vLLM 고정 | `tensor_parallel_size=1`, `gpu_memory_utilization=0.85`, `max_gen_toks=16384`, `apply_chat_template=true` | `README.md:164`, `README.md:165`, `README.md:167`, `README.md:168` |
| 모델 스펙 | 1.07B, 30 layers, GQA 32/8, vocab 102,400, context 65,536 | `docs/model_card.md:43`~`docs/model_card.md:47` |
| Korean 권장 | 한국어 대화 시 `temperature=0.1` 권장 | `docs/model_card.md:1135` |
| Ko-LongRAG | 600개, single/multi 각 300, unanswerable 16.6% | `docs/kolongrag.md:43`~`docs/kolongrag.md:48` |
| Judge | long/short 둘 다 평가하는 GPT-4o Judge 프롬프트 | `docs/kolongrag_examples.md:52`, `docs/kolongrag_examples.md:55`, `docs/kolongrag_examples.md:78`, `docs/kolongrag_examples.md:81`, `docs/kolongrag_examples.md:82` |
| Docker 동일성 | CUDA 12.8, torch 2.9.0+cu128, vllm 0.14.1 | `Dockerfile:4`, `Dockerfile:58`, `Dockerfile:87` |
| 로컬 실측 | Ko-LongRAG quick 실행 로그로 vLLM kwargs/sampling/시간 집계 경로를 확인(수치값은 참고용) | |

## 3. `script.py` 역설계 (Phase 1~5)

### Phase 1. 모델/토크나이저 로딩
- 서버 고정 호출 형태는 HF 호환 (`AutoTokenizer`, `AutoModelForCausalLM`)이다.
- 추론 엔진은 vLLM 0.14.1, TP=1, GPU 메모리 유틸 0.85로 고정된다.

### Phase 2. 데이터 로딩/프롬프트 구성
- 서버 내 `data/`의 비공개 벤치셋을 로딩한다.
- Ko-LongRAG 스타일의 `singledocQA/multidocQA` 분기 템플릿을 적용한다.
- `apply_chat_template=true` 고정이므로 message list를 chat prompt로 렌더링 후 추론한다.

### Phase 3. vLLM 오프라인 배치 추론
- `llm.generate(prompts, sampling_params)` 형태의 offline batch.
- 추론 시간은 generate 구간 wall-clock으로 측정하고, 생성 토큰은 output token 합으로 집계한다.

### Phase 4. 결과 파싱/메트릭
- Ko-LongRAG 로컬 평가는 GPT-4o Judge 기반 정확도 계산.
- 서버 실평가 쪽은 인터넷 불가 제약 때문에 Judge API 방식은 부적합하므로, 자동 메트릭(Exact Match/ROUGE 계열 등) 가능성이 높다.
- 설치 패키지(`math-verify`, `rouge_score`, `sacrebleu`, `langdetect`)는 복수 태스크형 자동 채점 파이프라인과 정합적이다.

### Phase 5. 점수 산출/CSV 저장
- 본 실행 계획에서는 `PerfNorm`, `SpeedNorm`, `Score`를 계산하지 않는다.
- 대신 `perf_model`, `infer_sec_model`, `tok_model`, `tpt_model`를 표준 출력/JSON에 기록한다.
- 서버는 `output/submission.csv`를 요구 경로로 사용.
- 로컬은 재현성 분석을 위해 상세 JSON/CSV를 병행 저장하는 것이 적절하다.

## 4. Ko-LongRAG 로컬 평가 구현안 (핵심)

### 4.1 데이터 준비
- 로딩:
```python
from datasets import load_dataset
ds = load_dataset("LGAI-EXAONE/Ko-LongRAG", split="test")
```
- 분리:
  - `task == "singledocQA"`
  - `task == "multidocQA"`
- unanswerable 정책:
  - 정답 문자열이 `unanswerable` 또는 `주어진 정보로 답할 수 없다` 계열이면 unanswerable 그룹으로 태깅.
  - 최종 리포트에 `accuracy_unanswerable` 별도 집계.

### 4.1.1 프롬프트 YAML 관리
- 프롬프트 소스: `docs/kolongrag.md`, `docs/kolongrag_examples.md`
- 저장 파일: `configs/prompts/kolongrag.yaml`
- 구조:
```yaml
single_doc:
  system: "당신은 도움이 되는 어시스턴트입니다."
  user_template: |
    ... {{titles}} ... {{context}} ... {{question}} ...
multi_doc:
  system: "당신은 도움이 되는 어시스턴트입니다."
  user_template: |
    ... {{titles}} ... {{context}} ... {{question}} ...
judge:
  system: |
    You are an expert evaluator of text answers in Korean.
    ...
  user_template: |
    Question: {{question}}
    Answer: {{answer}}
    long_ans: {{long_ans}}
    short_ans: {{short_ans}}
```
- 구현 규칙:
  - 코드 하드코딩 금지, YAML 로드 후 템플릿 렌더링만 허용.
  - 템플릿 변수 누락 시 즉시 예외 처리(fail-fast).

### 4.2 추론 파이프라인
- 모델:
  - `vllm.LLM(model=..., tensor_parallel_size=1, gpu_memory_utilization=0.85, trust_remote_code=True, max_gen_toks=16384)`
- 프롬프트:
  - `singledocQA`/`multidocQA` 공식 템플릿 분기 사용.
  - `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` 적용.
- 긴 컨텍스트 처리:
  - 기본값은 truncate 없이 full context.
  - 안전장치로 prompt token 길이 로깅.
  - OOM 발생 시에만 `max_num_batched_tokens`/batch pressure 완화.
- 샘플링 권장(.yaml로 관리):
  - 기본: `temperature=0.0`, `top_p=1.0`, `max_tokens=16384` (평가 재현성 우선, max_tokens는 고정)
  - 보조 실험: `temperature=0.1` (한국어 code-switch 억제 가이드 반영)
- 시간/토큰:
  - `inference_seconds = t_end - t_start`
  - `total_output_tokens = sum(len(out.outputs[0].token_ids))`

### 4.3 GPT-4o Judge 구현
- 입력:
  - `long_ans = 모델 원문 응답`
  - `short_ans = 휴리스틱 추출`
- `short_ans` 기본 휴리스틱:
  1. 접두어 제거: `답변은 다음과 같습니다:`, `정답:`, `A:` 등
  2. 첫 문장 또는 첫 줄 추출
  3. 비어 있으면 `long_ans` 대입
- Judge 호출:
  - 모델: `gpt-4o`
  - 온도: `0` (채점 일관성)
  - 출력 파싱: `ast.literal_eval` + 키/값 검증
- 재시도:
  - 네트워크/파싱 실패 시 exponential backoff로 최대 3회.
  - 최종 실패 시 `judge_error` 카운트 + 샘플 기록.
- 비용 최적화:
  - `sha256(question|answer|long_ans|short_ans)` 캐시 키로 중복 호출 방지.
  - 동시성은 rate-limit 고려하여 2~6 worker.
- 단가 입력형 비용식:
```python
N = 600
cost_usd = N * ((avg_input_tokens/1_000_000) * input_price
              + (avg_output_tokens/1_000_000) * output_price) * (1 + retry_rate)
```

### 4.4 메트릭 계산
- 샘플 정답 판정:
  - `is_correct = (judge_long == "correct") or (judge_short == "correct")`
- 집계:
  - `accuracy_total`
  - `accuracy_single`
  - `accuracy_multi`
  - `accuracy_unanswerable`
- Perf 반영:
  - Ko-LongRAG 단독 단계에서는 `Perf_model = accuracy_total`
  - 확장 단계에서는 다중 벤치 weighted aggregation 적용.

## 5. 추론 속도 측정 (tpt_model)
1. `infer_sec_model`은 모델 로딩 제외, `generate` 구간만 측정한다.
2. `tok_model`은 output token 총합으로 측정한다.
3. `tpt_model`은 `infer_sec_model / tok_model`로 계산한다.
4. warm-up 1회 후 실측 3회 median을 권장한다.

## 6. 로컬 출력 로직 (필수 4지표)
```python
def build_local_metrics(perf_model, infer_sec_model, tok_model):
    tpt_model = infer_sec_model / tok_model if tok_model > 0 else float("inf")
    return {
        "perf_model": perf_model,
        "infer_sec_model": infer_sec_model,
        "tok_model": tok_model,
        "tpt_model": tpt_model,
    }
```

### `calculate_performance` 내부 추정
- 대안 A: 벤치마크별 단순 평균.
- 대안 B: 샘플 수 가중 평균.
- 권장 기본값: 대안 B(샘플 수 가중). 이유는 분할/태스크 크기 차이가 큰 경우 안정적이다.
- 불확실성: 실제 서버 가중치가 비공개이므로 `perf_model`의 리더보드 상관은 제출-회수 캘리브레이션으로 보정 필요.

## 7. 비공개 벤치마크 추정

### 7.1 패키지 시그널 기반
| 패키지 | 시사 태스크 | 해석 |
|---|---|---|
| `math-verify`, `sympy`, `antlr4-python3-runtime` | 수학 추론(AIME/GSM8K/MATH류) | 수식 동치 판정 자동채점 가능성 높음 |
| `rouge_score` | 요약/long-context QA | Ko-LongRAG/LongBench 계열과 정합 |
| `sacrebleu` | 번역 | 다국어 평가 항목 가능성 |
| `langdetect` | 언어 검증 | ko/en/es 응답 품질 제약 가능성 |
| `evaluate` | 통합 메트릭 런처 | 다중 벤치 자동 집계 가능성 |

### 7.2 모델카드 교차
- Long-context: HELMET/RULER/LongBench/Ko-LongBench.
- Math/Coding: AIME/HMMT/LiveCodeBench.
- Multilingual: KMMLU/KSM/MMMLU/MATH500(ES).
- 결론:
  - Ko-LongRAG 포함 가능성은 높지만, 모델카드 명시는 Ko-LongBench라서 동일 데이터셋이라고 단정할 수는 없다.
  - 실전 예측은 Ko-LongRAG를 코어로 두고 수학/다국어 보조 지표를 추가하는 다축 평가가 안전하다.

## 8. “인터넷 불가 vs GPT-4o Judge” 모순 분석
1. 사실 A: Ko-LongRAG 공식 문서/예시는 GPT-4o Judge를 사용한다. (`docs/kolongrag_examples.md:52`)
2. 사실 B: 대회 서버는 인터넷 불가다. (`README.md:177`)
3. 추론:
   - 서버 내부 평가에 외부 OpenAI API를 직접 호출하는 구조는 실현 불가.
   - 따라서 서버 실평가는 자동 메트릭(문자열/정규화/ROUGE/수학검증) 기반일 가능성이 높다.
4. 운영 전략:
   - 로컬: GPT-4o Judge로 의미정답 보정.
   - 제출 예측: Judge 점수와 자동 메트릭 점수를 동시 기록해 리더보드와의 상관이 높은 축을 선택.

## 9. 캘리브레이션 루프
1. 로컬에서는 우선 `perf_model`, `tpt_model`을 일관된 설정으로 반복 측정한다.
2. 제출 가능한 체크포인트를 리더보드로 보내 대응점을 수집한다.
3. `local_metrics(perf_model, tpt_model)`와 `leaderboard_score` 사이의 보정식(선형/등위 기반)을 피팅한다.
4. 이후 후보 모델은 절대 점수 대신 `보정된 추정 순위/추정 점수`로 해석한다.
5. 새 제출 결과가 추가될 때마다 보정식을 재학습한다.

## 10. 위험/한계/완화책
| 위험 | 영향 | 완화 |
|---|---|---|
| 비공개 벤치/가중치 | 로컬-리더보드 편차 | 캘리브레이션 루프 운영 |
| GPT-4o Judge 비용 | 실험 반복 비용 증가 | 캐시, 샘플링 평가, 단가 입력형 예산관리 |
| Judge 비결정성 | 동일 입력 변동 | `temperature=0`, 재시도 정책 고정 |
| `short_ans` 추출 오차 | 과소/과대평가 | 휴리스틱 + fallback(`short_ans=long_ans`) |
| 인터넷 불가 서버 | Judge 직접 재현 불가 | 자동 메트릭 병행 및 상관 분석 |
| 버전 차이 | 점수 변동 | Docker 베이스 고정, 라이브러리 핀 유지 |

## 11. 공개 인터페이스/타입 명세 (구현 기준)
```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class EvalSample:
    id: str
    task: Literal["singledocQA", "multidocQA"]
    titles: list[str]
    context: str
    question: str
    answer: str

@dataclass
class GenerationRecord:
    id: str
    prompt: str
    prediction: str
    output_tokens: int
    latency_ms: float

@dataclass
class JudgeResult:
    long_ans: str
    short_ans: str
    judge_long: Literal["correct", "incorrect"]
    judge_short: Literal["correct", "incorrect"]
    is_correct: bool

@dataclass
class RunMetrics:
    perf_model: float
    infer_sec_model: float
    tok_model: int
    tpt_model: float
```

### 실행 진입점(Entrypoint)
- 권장 진입점: `python -m src.kolongrag_eval`
- 대체 진입점: `python src/kolongrag_eval.py`
- OpenAI 키 로딩 위치: 프로젝트 루트 `.env.example` (`OPENAI_API_KEY`)

### CLI 플래그 스펙
| 플래그 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--model-path` | str | 없음(필수) | 평가할 모델 경로. `models/` 하위 폴더를 사용 (`models/EXAONE-4.0-1.2B`, `models/EXAONE-4.0-1.2B-FP8`) |
| `--quick [N]` | int (optional) | `100`(숫자 생략 시) | `singledocqa_000`부터 N개만 실행. 예: `--quick` = `singledocqa_000~099`, `--quick 32` = `singledocqa_000~031` |
| `--single-only` | bool | `False` | `singledocQA`만 평가 |
| `--multi-only` | bool | `False` | `multidocQA`만 평가 |
| `--skip-eval` | bool | `False` | Judge 평가 단계 스킵. 생성/속도 지표만 산출 |

### 플래그 조합 규칙
| 규칙 | 동작 |
|---|---|
| `--single-only` + `--multi-only` 동시 지정 | 오류 처리 후 종료 (mutually exclusive) |
| `--quick` + `--multi-only` | 오류 처리 후 종료 (`--quick`은 singledoc 전용) |
| `--quick` + `--single-only` | 허용 (동일 의미, `--quick` 우선 적용) |
| `--skip-eval` | `perf_model`을 `null`로 기록하고 Judge 컬럼은 빈값 처리 |

### CLI 예시
```bash
# full 평가 (single + multi)
python -m src.kolongrag_eval --model-path models/EXAONE-4.0-1.2B

# quick 기본 100개 (singledocqa_000~099)
python -m src.kolongrag_eval --model-path models/EXAONE-4.0-1.2B --quick

# quick 32개 (singledocqa_000~031)
python -m src.kolongrag_eval --model-path models/EXAONE-4.0-1.2B-FP8 --quick 32

# single만 평가
python -m src.kolongrag_eval --model-path models/EXAONE-4.0-1.2B --single-only

# multi만 평가 + 평가 스킵(생성/속도만)
python -m src.kolongrag_eval --model-path models/EXAONE-4.0-1.2B-FP8 --multi-only --skip-eval
```

### 출력 파일 스키마
1. `results/<run>.json`
| 필드 | 타입 | 설명 |
|---|---|---|
| `run_name` | str | 실행 식별자 (`YYYYMMDD_HHMMSS_<model>`) |
| `model_path` | str | 입력 모델 경로 |
| `dataset` | str | 고정: `LGAI-EXAONE/Ko-LongRAG` |
| `filters` | object | `quick_n`, `single_only`, `multi_only`, `skip_eval` |
| `vllm_settings` | object | README 고정 옵션 반영값 |
| `sampling_params` | object | `temperature`, `top_p`, `max_tokens` |
| `perf_model` | float \| null | Judge 평가 정확도(스킵 시 `null`) |
| `infer_sec_model` | float | 모델 로딩 제외 추론 시간(초) |
| `tok_model` | int | 생성 토큰 총합 |
| `tpt_model` | float | `infer_sec_model / tok_model` |
| `judge_errors` | int | Judge 파싱/호출 실패 수 |
| `sample_count` | int | 실제 평가 샘플 수 |
| `artifacts` | object | `result_csv`, `submission_csv`, `canonical_submission_csv` 경로 |
| `timestamp` | str | ISO8601 |

2. `results/<run>.csv`
| 컬럼 | 타입 | 설명 |
|---|---|---|
| `idx` | int | 내부 순번 |
| `id` | str | Ko-LongRAG 샘플 ID |
| `task` | str | `singledocQA` 또는 `multidocQA` |
| `question` | str | 질문 |
| `reference` | str | 정답 |
| `prediction` | str | 모델 응답(long_ans 원문) |
| `short_ans` | str | 휴리스틱 추출 short answer |
| `judge_long` | str \| empty | `correct`/`incorrect` (skip 시 빈값) |
| `judge_short` | str \| empty | `correct`/`incorrect` (skip 시 빈값) |
| `is_correct` | bool \| empty | 샘플 정오답 (skip 시 빈값) |
| `output_tokens` | int | 샘플 생성 토큰 수 |

3. `output/<run>_submission.csv` + `output/submission.csv`
| 컬럼 | 타입 | 설명 |
|---|---|---|
| `id` | str | 샘플 ID |
| `prediction` | str | 제출 포맷용 예측 문자열 |

- 매핑 규칙:
  - `output/<run>_submission.csv`: 실행 히스토리 보관용
  - `output/submission.csv`: 대회 제출 포맷과 동일한 canonical 경로

## 12. 검증 시나리오
1. Prompt 렌더링: task별 템플릿 변수 누락 검사.
2. Long-context 안정성: 길이 상위 multidoc 샘플 OOM/정확도 점검.
3. Unanswerable: unanswerable 그룹 정확도 별도 리포팅.
4. Judge 파서: 정상 dict/코드블록/깨진 출력에 대한 복구 테스트.
5. 속도: 모델 로딩 포함/제외 시간 분리 검증.
6. 재현성: 동일 조건 2회 이상 반복 편차 확인.
7. 출력 sanity: 각 실행에서 `tpt_model == infer_sec_model / tok_model`가 일치하는지 확인.
8. `--quick` 기본값 검증: `--quick` 단독 실행 시 정확히 `singledocqa_000~099`만 처리되는지 확인.
9. `--quick 32` 검증: 정확히 `singledocqa_000~031`만 처리되는지 확인.
10. 플래그 충돌 검증: `--single-only --multi-only`, `--quick --multi-only`가 실패 종료되는지 확인.
11. `--skip-eval` 검증: `perf_model=null`, Judge 컬럼 빈값, `infer_sec_model/tok_model/tpt_model`은 정상 기록되는지 확인.

## 13. 구현 가능한 최소 코드 예시 (핵심 루프)
```python
import time, ast, re, json, hashlib
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from openai import OpenAI

UNANSWERABLE_HINTS = ["unanswerable", "주어진 정보로 답할 수 없다", "주어진 문서내에서 답할 수 있는 정보가 충분하지 않습니다"]

def build_prompt(sample):
    if sample["task"] == "singledocQA":
        user = (
            "다음 문서를 살펴보고, 질문에 대한 답을 추출하세요.\n"
            "질문에 대한 답만 생성하세요. 답변은 매우 간결해야 합니다.\n"
            "답변을 문서에서 찾을 수 없는 경우, '주어진 정보로 답할 수 없다'로 응답하세요.\n"
            f"문서는 제목 목록: {sample['titles']}.\n"
            f"문서는 다음과 같습니다: {sample['context']}\n"
            f"질문: {sample['question']}"
        )
    else:
        user = (
            "다음 문서를 검토하고 질문에 답하세요.\n"
            f"문서 제목 목록: {sample['titles']}\n"
            f"문서: {sample['context']}\n"
            f"질문: {sample['question']}\n"
            "질문에 대한 답만 생성하고 다른 어떤 것도 생성하지 마세요.\n"
            "답을 찾을 수 없으면 '주어진 정보로 답할 수 없다'로 응답하세요."
        )
    return [{"role": "system", "content": "당신은 도움이 되는 어시스턴트입니다."},
            {"role": "user", "content": user}]

def extract_short_ans(text):
    s = text.strip()
    s = re.sub(r"^(답변은\\s*다음과\\s*같습니다\\s*[:：]\\s*)", "", s)
    s = re.sub(r"^(정답\\s*[:：]\\s*)", "", s)
    s = s.split("\n")[0].strip()
    s = re.split(r"[.?!]\\s+", s)[0].strip()
    return s if s else text.strip()

def parse_judge_dict(raw):
    t = raw.strip()
    t = re.sub(r"^```(?:python)?\\s*", "", t)
    t = re.sub(r"\\s*```$", "", t)
    obj = ast.literal_eval(t)
    if not isinstance(obj, dict): raise ValueError("not dict")
    if obj.get("long_ans") not in {"correct", "incorrect"}: raise ValueError("bad long")
    if obj.get("short_ans") not in {"correct", "incorrect"}: raise ValueError("bad short")
    return obj

def run_eval(model_path, openai_api_key):
    ds = load_dataset("LGAI-EXAONE/Ko-LongRAG", split="test")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.85, trust_remote_code=True, max_gen_toks=16384)

    prompts = [tok.apply_chat_template(build_prompt(s), tokenize=False, add_generation_prompt=True) for s in ds]
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=16384)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    infer_sec = time.perf_counter() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)

    client = OpenAI(api_key=openai_api_key)
    cache, correct = {}, 0
    for s, o in zip(ds, outs):
        long_ans = o.outputs[0].text.strip()
        short_ans = extract_short_ans(long_ans)
        key = hashlib.sha256(f"{s['question']}|{s['answer']}|{long_ans}|{short_ans}".encode()).hexdigest()
        if key not in cache:
            # 실제 구현에서는 공식 Judge system/user prompt 전문 사용
            resp = client.chat.completions.create(
                model="gpt-4o", temperature=0,
                messages=[{"role": "system", "content": "Return only python dict."},
                          {"role": "user", "content": f"Question:{s['question']}\nAnswer:{s['answer']}\nlong_ans:{long_ans}\nshort_ans:{short_ans}"}]
            )
            cache[key] = parse_judge_dict(resp.choices[0].message.content)
        j = cache[key]
        is_correct = (j["long_ans"] == "correct") or (j["short_ans"] == "correct")
        correct += int(is_correct)

    acc = correct / len(ds)
    return {
        "perf_model": acc,
        "infer_sec_model": infer_sec,
        "tok_model": total_tokens,
        "tpt_model": (infer_sec / total_tokens) if total_tokens > 0 else float("inf"),
    }
```

## 14. 실행 환경 원칙
- Dockerfile은 대회 동일본 유지.
- 로컬 Judge용 패키지는 컨테이너 실행 후 별도 설치:
```bash
docker build -t exaone-eval .
docker run --gpus all -it exaone-eval bash
pip install "openai>=1.0.0"
```

## 15. 명시적 가정/기본값
1. Ko-LongRAG를 로컬 점수 예측의 코어 벤치로 둔다.
2. GPT-4o Judge는 로컬 캘리브레이션 도구로 사용한다.
3. 비용은 단가 입력형으로 계산한다.
4. `short_ans`는 접두어 제거 후 첫 핵심 span 추출을 기본으로 한다.
5. 추론 파이프라인은 커스텀 스크립트 기준으로 작성한다.
6. 보고서는 한국어 중심, 기술 용어 영문 병기를 유지한다.
7. 본 보고서는 문서/로그 근거 기반이며, 서버 비공개 `script.py`는 추정(역설계) 결과임을 전제한다.
8. 실행 출력은 `perf_model`, `infer_sec_model`, `tok_model`, `tpt_model` 4개 지표로 고정하며, base 기준값 기반의 산식 계산은 수행하지 않는다.

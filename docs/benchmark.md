# Benchmark 가이드 (2026-02 리뉴얼)

이 문서는 `benchmark.py`의 **현행 동작** 기준입니다. 기존 legacy 설명은 폐기했습니다.

## 1. 목적
`benchmark.py`는 아래 2가지를 동시에 수행합니다.

1. `lm_eval` 기반 정확도 평가
2. 속도 측정 2종 분리
   - `wall` 기준: `simple_evaluate` 전체 구간(로컬 오버헤드 포함)
   - `generate-only` 기준: `Processed prompts` 진행 로그에서 추출한 생성 구간

대회 산식 추정은 `generate-only` 기준을 사용합니다.

## 2. 실행 방법

### 2.1 기본 실행
```bash
python benchmark.py gsm8k --model models/EXAONE-4.0-1.2B
python benchmark.py math --model models/EXAONE-4.0-1.2B
python benchmark.py all --model models/EXAONE-4.0-1.2B
```

### 2.2 FP8 모델 실행
```bash
python benchmark.py gsm8k --model models/EXAONE-4.0-1.2B-FP8
```

### 2.3 대회 산식 로컬 추정 (baseline 결과와 비교)
```bash
python benchmark.py gsm8k \
  --model models/EXAONE-4.0-1.2B-FP8 \
  --baseline-result results/gsm8k-base.json
```

## 3. CLI 옵션
- `tasks` (positional): 카테고리명 또는 태스크명(쉼표 구분)
- `--model, -m`: 모델 경로
- `--thinking, -t`: `enable_thinking=True`
- `--enforce-eager`: `enforce_eager=True`
- `--no-enforce-eager`: `enforce_eager=False`
- `--kv-cache-dtype`: 예: `fp8_e5m2`
- `--baseline-result`: 기준(base) 결과 JSON 경로. 지정 시 `PerfNorm/SpeedNorm/Score` 추정

## 4. 출력 파일
- 로그: `logs/<timestamp>_<model>.log`
- 결과: `results/<timestamp>_<model>.json`

## 5. 시간/속도 지표 정의

### 5.1 `wall` 지표
- `total_inference_seconds`
- `overall_tps`

특징: 로컬 파이프라인 오버헤드가 포함됩니다.

### 5.2 `generate-only` 지표
- `total_generate_seconds`
- `overall_generate_tps`
- `seconds_per_output_token_generate_only`

특징: 대회 산식 정렬용으로 사용합니다.

## 6. 대회 산식 로컬 추정
`--baseline-result`를 주면 `contest_estimate`를 계산합니다.

- `PerfNorm = Perf_model / Perf_base`
- `SpeedNorm = 1 - ((Time/Tokens)_model / (Time/Tokens)_base)`
- `Score = max(0.5 * PerfNorm + 0.5 * SpeedNorm, 0)`

주의:
- 로컬 추정치는 **비공개 평가셋 + 운영진 고정 script.py**와 다를 수 있습니다.
- 따라서 제출 점수와 1:1 일치를 보장하지 않습니다.

## 7. 결과 JSON 핵심 필드
- `time.total_inference_seconds`: wall 기준 총 시간
- `time.total_generate_seconds`: generate-only 총 시간
- `time.total_output_tokens`: 출력 토큰 수(응답 텍스트 기준)
- `time.seconds_per_output_token_generate_only`: 대회 산식 정렬용 핵심 값
- `performance`: 태스크별 대표 metric + 평균
- `contest_estimate`: baseline 기준 로컬 추정(`perf_norm`, `speed_norm`, `score`)

## 8. 재현성 권장 프로토콜
1. base/quantized를 **같은 스크립트/같은 옵션**으로 실행
2. 각 모델 3회 이상 반복
3. 비교는 `generate-only` 지표 우선
4. 최종 판단은 단일 태스크가 아니라 대회 유사 태스크 묶음으로 수행

## 9. 알려진 이슈
- `TORCHDYNAMO_DISABLE=1`이 기본 설정되어 있습니다.
- 일부 양자화 모델은 커널 경로에 따라 FP8이 항상 빠르지 않을 수 있습니다.
- `tokenizer.json`의 truncation 아티팩트는 재현성 리스크이므로, 모델 간 tokenizer 설정 차이를 점검하세요.

# 이슈 리포트: FP8 로컬 벤치마크와 평가서버 결과 불일치

작성일: 2026-02-07

## 1) 현상 요약
- 로컬 `benchmark.py` 결과에서는 FP8 모델이 원본 대비 느리게 측정됨.
- 평가서버 제출 결과에서는 FP8 모델 점수/시간이 원본보다 유리함.
  - 원본: `0.503449358` (13분 12초)
  - FP8: `0.6163913361` (9분 40초)

## 2) 로그/결과 기반 사실
- 로컬 결과(JSON)
  - 원본: `results/gsm8k-base.json` 기준 `overall_tps=602.64`, `time=340.98s`, `tokens=205,487`
  - FP8: `results/20260207_212100_EXAONE-4.0-1.2B-FP8.json` 기준 `overall_tps=556.49`, `time=364.88s`, `tokens=203,052`
- vLLM 내부 진행 로그(`Processed prompts`)에서도 FP8가 느리게 관측됨
  - 원본: `output ~675 toks/s` (`logs/gsm8k-base.log:694`)
  - FP8: `output ~623.95 toks/s` (`logs/20260207_212100_EXAONE-4.0-1.2B-FP8.log:675`)
  - generate 완료까지 걸린 시간: 원본 `05:06`, FP8 `05:27` (약 +21초)
  - 첫 샘플 도달 지연: 원본 `00:28`, FP8 `00:37` (약 +9초)
- `benchmark.py`는 첫 태스크에서 `model loading`만 분리하고(`benchmark.py:252-255`), 엔진 warmup/graph capture는 추론시간에 포함함
  - 원본 init engine: 8.53s (`logs/gsm8k-base.log:40`)
  - FP8 init engine: 10.46s (`logs/20260207_212100_EXAONE-4.0-1.2B-FP8.log:40`)

### 2-1) \"토큰 수가 비슷한데 FP8이 왜 더 느렸는가\" 정량 설명
- 로컬 결과 기준:
  - 원본 `Time/Tokens = 340.98 / 205,487 = 0.001659 s/token`
  - FP8 `Time/Tokens = 364.88 / 203,052 = 0.001797 s/token`
  - FP8이 `+8.29%` 더 느림
- vLLM 내부 진행률 로그의 최종 추정 속도:
  - 원본 output 약 `675 toks/s` (`logs/gsm8k-base.log:694`)
  - FP8 output 약 `623.95 toks/s` (`logs/20260207_212100_EXAONE-4.0-1.2B-FP8.log:675`)
  - FP8 output 처리율이 `-7.56%` 낮음
- 즉, **이번 GSM8K 로컬 워크로드에서는 FP8 커널 경로 자체 처리율이 낮게 나왔고**, 이 처리율 하락이 총 시간 증가를 설명함.
- 추가로 첫 응답 도달 지연(초기 구간)도 FP8 쪽이 더 길어 총 시간 차이를 키움.
- 샘플 응답 길이(로그 기반 문자/단어 통계)는 FP8이 오히려 약간 더 짧음(평균 chars: base 372.7 vs fp8 366.3). 따라서 \"FP8이 더 장문을 출력해서 느려졌다\"는 가설은 이번 로그와 맞지 않음.

## 3) 원인 분석
### 원인 A: 로컬 측정식과 대회 산식의 불일치
- `benchmark.py`는 분모를 "응답 텍스트를 다시 토크나이즈한 output tokens"로 계산함(`benchmark.py:268-290`).
- 분자는 `lm_eval.simple_evaluate(...)` 전체 벽시계 시간(`benchmark.py:233-247`)으로, 아래가 함께 포함됨.
  - 태스크 context 구축
  - evaluator 후처리
  - vLLM 첫 요청 warmup/graph capture 일부
- 즉, 로컬 `overall_tps`는 대회의 `Time/Tokens` 정의와 1:1로 대응하지 않음.

### 원인 B: 평가 데이터셋/워크로드 차이
- 로컬은 `gsm8k` 단일 태스크(1319개)로 측정.
- 대회는 비공개 전체 벤치셋 100%로 평가(`README.md`).
- 평가서버는 운영진 고정 `script.py`로 실행되며, 로컬 `lm_eval` 파이프라인과 동일하다고 보장되지 않음.
- FP8의 속도 이득/손해는 prefill/decode 비중, 시퀀스 길이 분포, 배치 동작에 따라 달라질 수 있어 단일 `gsm8k` 결과로 전체 평가 속도를 대체할 수 없음.

### 원인 C: FP8 모델 tokenizer 아티팩트(중요)
- 원본/FP8 토크나이저 비교 시, FP8 `tokenizer.json`에만 truncation 설정이 저장됨.
  - 원본: `truncation = null`
  - FP8: `truncation.max_length = 2048`
- 해당 차이는 `experiments/quantize_fp8.py`에서 calibration 시 `truncation=True` 사용(`experiments/quantize_fp8.py:81-86`) 후 같은 tokenizer를 `save_pretrained`(`experiments/quantize_fp8.py:147`) 하면서 유입된 것으로 보임.
- 이 설정은 평가 시 긴 입력이 존재할 경우 유효 입력 길이/토큰 수에 영향을 줄 수 있어, 속도 개선과 성능 변화를 동시에 유발할 수 있음.

### 원인 C-1: \"그렇다면 로컬에서도 2048 truncation이 적용되어야 하지 않나?\"에 대한 정리
- `benchmark.py`(및 평가 코드)가 \"truncation을 명시하지 않았다\"고 해서, **토크나이저 파일에 저장된 truncation이 항상 비활성**이라고 단정할 수는 없음.
  - HF Tokenizers는 truncation을 토크나이저 상태로 `enable_truncation(max_length=...)` 하여 켜고, `no_truncation()`으로 끄는 API를 제공함. (즉, tokenizer.json에 truncation이 저장되면 로드 시 기본 상태가 달라질 수 있음)
  - 실제로 Transformers/vLLM이 내부에서 이 상태를 매 호출마다 강제로 off 하는지 여부는 구현에 의존하므로, **길이가 2048 토큰을 넘는 고정 프롬프트로 모델별 입력 토큰 길이를 직접 비교**해 확인하는 것이 안전함.
- 또한 `gsm8k`는 입력 길이가 짧은 편이라, 설령 truncation이 켜져 있어도 실제로 잘리는 샘플 비율이 낮아 체감 영향이 작을 수 있음.
- 결론적으로, **이번 로컬에서 FP8이 느린 관측은 truncation 아티팩트와 모순되지 않음**. (해당 아티팩트는 \"특정 길이 분포의 데이터\"에서만 영향을 크게 줄 수 있는 재현성 리스크)

### 원인 D: 로컬 실행 조건의 미세 불일치
- `benchmark.py`는 기본으로 `TORCHDYNAMO_DISABLE=1`을 강제(`benchmark.py:21-25`)함.
- 평가서버 고정 스펙에는 해당 환경변수 강제가 명시되어 있지 않음(`README.md`).
- compile/dynamo 경로 차이가 양자화 모델에 더 민감하게 작동할 가능성이 있음.

## 4) 공식 문서 근거(@context7)
- vLLM는 prefill/decode 요청 토큰을 분리 추적하는 성능 지표 구조를 가짐.
  - 참고: https://docs.vllm.ai/en/latest/api/vllm/v1/metrics/perf
- vLLM compressed-tensors FP8 스킴은 최소 capability(예: 89, Lovelace+) 등 하드웨어 조건과 커널 경로 제약이 존재.
  - 참고: https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8
- vLLM prompt truncation 제어는 `truncate_prompt_tokens`로 하며 기본값은 `None`.
  - 참고: https://docs.vllm.ai/en/latest/serving/openai_compatible_server
- Transformers에서 truncation 기본 동작은 `False`(명시 시에만 truncation).
  - 참고: https://github.com/huggingface/transformers/blob/main/docs/source/en/pad_truncation.md
- Hugging Face Tokenizers는 truncation을 토크나이저 상태로 켜고(`enable_truncation`) 끌 수 있음(`no_truncation`).
  - 참고: https://context7.com/huggingface/tokenizers/llms.txt
- lm-eval task는 filter pipeline을 복수로 둘 수 있으며(샘플/필터 출력 구조 영향), post-process 기준 평가가 가능.
  - 참고: https://github.com/eleutherai/lm-evaluation-harness/blob/main/docs/task_guide.md

## 5) 결론
- 현재 불일치의 1차 원인은 **측정 방법/평가 데이터의 차이**이며, 
- 재현성 리스크의 핵심은 **FP8 모델에만 존재하는 tokenizer truncation(2048) 아티팩트**다.

## 5-1) \"FP8이 로컬에선 느리고 서버에선 빠른\" 현상에 대한 가설
아래는 현재 로그와 대회 규칙을 함께 봤을 때의 **검증 가능한 가설**이다.

### 가설 A (가장 유력): 워크로드 길이 분포 차이로 인한 커널 경로 역전
- 로컬 `gsm8k`는 짧은 문제/응답 위주라 FP8의 이점(메모리 대역 절감)이 충분히 발휘되지 않았을 수 있음.
- 실제 로컬 로그에서는 FP8 처리율이 낮게 관측됨:
  - base output ~675 toks/s
  - FP8 output ~623.95 toks/s
- 반면 평가서버는 비공개 전체 벤치셋(다양한 길이/태스크)이라, 긴 컨텍스트 구간에서 FP8이 유리해질 가능성이 큼.

### 가설 B: FP8은 KV 캐시 여유로 동시성 잠재력이 더 크지만, `gsm8k` 단일 태스크로는 포화되지 않음
- 로컬 로그상 FP8이 KV cache/concurrency는 더 좋게 나옴:
  - base max concurrency 4.07x
  - FP8 max concurrency 4.31x
- 즉 \"잠재 동시성\"은 높지만, 현재 단일 태스크/길이 분포에서는 실제 처리율 개선으로 연결되지 않았을 수 있음.

### 가설 C: 평가서버 표시 시간(전체 실행시간)과 로컬 측정 경계가 다름
- 대회 페이지 안내상 표시 시간은 \"모델 로드 + 추론 + 파싱\" 전체 시간.
- 로컬 `benchmark.py`는 `simple_evaluate` 구간 중심 측정이며, 평가서버 고정 `script.py`의 세부 구현과 다름.
- 따라서 로컬에서 느리게 보여도 평가서버 전체 실행시간이 더 짧아질 수 있음.

### 가설 D: `tokenizer truncation` 아티팩트는 \"항상 속도 개선\"이 아니라 \"조건부 리스크\"
- FP8 tokenizer에만 truncation 2048이 저장되어 있어 재현성 리스크는 맞음.
- 다만 `benchmark.py`/vLLM 기본 동작상 truncation이 강제 적용되지 않을 수 있고, `gsm8k`는 길이가 짧아 영향이 작을 수 있음.
- 그래서 이번 로컬 느림과 모순되지 않음.

## 6) 권장 조치
- 조치 1: FP8 모델의 `tokenizer.json`에서 truncation 블록 제거 후 재평가.
- 조치 2: `benchmark.py`에 "생성 구간만" 별도 측정(최소한 vLLM generate phase와 분리) 지표를 추가.
- 조치 3: 토큰 수를 `resps` 재토크나이즈 대신, 가능하면 엔진/요청 usage 기반(input/output 분리)으로 수집.
- 조치 4: 제출 전 최종 비교는 단일 `gsm8k`가 아니라 대회와 유사한 태스크 묶음으로 수행.

## 7) 빠른 검증 체크리스트
- `models/EXAONE-4.0-1.2B-FP8/tokenizer.json`의 `truncation` 제거 여부 확인.
- 제거 전/후 동일 seed, 동일 GPU(L4), 동일 옵션으로 최소 2회 반복 측정.
- `generate_until` 구간 시간과 output token 속도를 우선 비교하고, 그 다음 전체 벽시계 시간을 비교.
- 장문 입력(2048 토큰 초과) 1개를 고정으로 넣어 실제 입력 토큰 길이가 잘리는지(모델별 차이) 스모크 테스트 수행.

## 8) 로컬 측정식-대회 산식 정렬 해결안 (필수)
목표: 로컬에서 대회 산식과 동일한 정의로 `SpeedNorm`/`Score`를 계산할 수 있도록 측정 경계를 재설계.

### 8-1. 측정 경계 정의를 대회 산식에 맞게 고정
- `Time_model`: \"실제 추론(generate) 구간\" 시간만 사용.
- 제외 항목: 모델 로드, CUDA graph capture/warmup, 데이터 로딩, 후처리/정답 채점, 결과 직렬화.
- `Tokens_model`: 재토크나이즈 결과 대신, **엔진이 실제 생성한 토큰 수**를 사용.

### 8-2. 구현 방식 (권장)
- 정확도(Perf)는 기존 `lm_eval` 유지.
- 속도(Time/Tokens)는 별도 경량 측정기로 분리:
  - vLLM `LLM.generate(...)` 직접 호출.
  - 응답별 `len(output.outputs[0].token_ids)` 합계를 `Tokens_model`로 사용.
  - 타이머는 `generate` 직전/직후로 측정.
  - 첫 요청 warmup 1회는 버리고 본 측정 시작.

### 8-3. 산식 계산을 리포트에 명시
- `PerfNorm_model = Perf_model / Perf_base_model`
- `SpeedNorm_model = 1 - ( (Time_model / Tokens_model) / (Time_base / Tokens_base) )`
- `Score = max(0.5 * PerfNorm_model + 0.5 * SpeedNorm_model, 0)`

### 8-4. 재현성 프로토콜
- 각 모델별 최소 3회 반복 실행 후 평균/표준편차 기록.
- 실행 고정값: 동일 GPU, 동일 드라이버/CUDA, 동일 seed, 동일 태스크 집합, 동일 generation 파라미터.
- 비교 시에는 반드시 \"같은 측정기\"로 base/quantized를 함께 계산.

### 8-5. 문서/출력 포맷 권장
- 결과 JSON에 아래 필드 분리 저장:
  - `time_total_wallclock`
  - `time_generate_only`
  - `tokens_generated_engine`
  - `time_per_token_generate_only`
  - `speed_norm_vs_base`
  - `score_estimated`
- 이렇게 해야 로컬 수치와 대회 점수 차이의 원인을 즉시 추적 가능.

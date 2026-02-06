# TPOT 이슈 조사 보고서 (EXAONE-4.0-1.2B, GPTQ 4bit, vLLM 0.14.1, NVIDIA L4)

## 0) 범위/제약
- 로컬에서는 `README.md`만 확인했습니다(그 외 파일/코드/로그 전문 미열람).
- 근거는 `README.md` + 사용자가 제공한 로그 일부 + (Context7) vLLM/Transformers 공식 문서/코드 + (Tavily) 공개 웹 자료입니다.

---

## 1) 핵심 결론 (Executive Summary)
1. **“양자화로 메모리는 줄었는데 TPOT가 거의 안 줄어드는 현상”은 충분히 흔한 패턴**입니다. (Weight-only INT4는 주로 가중치 메모리/Linear 비용만 줄이고, TPOT를 지배하는 decode attention/KV 비용은 거의 그대로일 수 있음)
2. **가장 유력한 직접 원인(우선순위 1)은 `pre-SM90 GPU(L4) + Marlin + bf16` 조합**입니다. vLLM 자체가 *“SM90 이전 + bf16에서 Marlin 성능이 떨어질 수 있으니 fp16을 고려”* 하라고 안내하는 코드가 존재합니다. (`vLLM marlin_utils`)
3. 로그의 `fix_mistral_regex=True` 경고는 **Transformers 4.57.3에서 “로컬 디렉토리에 모델+토크나이저 파일이 함께 있을 때” 오탐으로 트리거될 수 있음**이 공식 이슈로 확인됩니다. 다만 **오탐 단정은 금물**이며, 실제 토큰화 mismatch면 토큰 수/종료시점/정확도에도 영향을 줄 수 있어 분리 검증이 필요합니다.
4. 대회 환경은 vLLM 런타임 옵션이 사실상 고정이므로, **TPOT 개선 레버는 `model/` 아티팩트(= config/tokenizer/weights/quant-format) 쪽**에 집중하는 것이 현실적입니다. (`README.md:19-21`, `README.md:151-169`)

---

## 2) 평가 산식/환경에서 “TPOT”가 갖는 의미
- 속도 점수는 기본적으로 **평균 토큰당 시간** 형태로 들어갑니다:  
  `SpeedNorm = 1 - (Time/Tokens)_model / (Time/Tokens)_base` (`README.md:41-43`)
- 즉, 목표는 **Time을 줄이거나(=더 빠르게), Token당 처리량을 올려서** `Time/Tokens`를 낮추는 것입니다.
- 평가 서버/라이브러리 고정: L4, `vllm==0.14.1`, `transformers==4.57.3` 등 (`README.md:83-125`)
- 서빙 옵션 고정 적용(중요): `tensor_parallel_size=1`, `gpu_memory_utilization=0.85`, `batch_size=auto`, `max_gen_toks=16384`, `apply_chat_template=true` (`README.md:163-169`)

---

## 3) 로그에서 “사실로 확인되는 점” (제공 로그 기반)
- vLLM 엔진 설정에 `dtype=torch.bfloat16`로 찍혀 있음.
- `quantization=gptq_marlin` 및 “Using gptq_marlin kernel.” → Marlin 커널 경로를 타고 있음.
- Attention backend는 `FLASH_ATTN`.
- 모델 최대 길이가 `65536`으로 잡혀 있고, chunked prefill이 켜져 있음.
- `fix_mistral_regex=True` 경고가 반복 출력됨.

---

## 4) 원인 가설 (우선순위/반증 방법 포함)

### H1. (최우선, 신뢰도 높음) **L4(pre-SM90)에서 Marlin+bf16 경로가 비효율**
- 근거(공식): vLLM `marlin_utils`에 아래 로직이 존재합니다.  
  *GPU capability major < 9(SM90 미만) AND dtype == bf16* 인 경우, **fp16 전환을 고려**하라고 `info_once`를 띄웁니다.  
  → 즉, “현재 조합이 느릴 수 있다”는 것을 vLLM이 직접 인정하는 형태입니다. (`vLLM marlin_utils`)
- 왜 TPOT에 영향?  
  GPTQ-INT4의 핵심 이득(Linear)을 Marlin이 담당하는데, 그 Marlin이 bf16에서 비효율이면 양자화 이득이 상쇄될 수 있습니다.
- 반증(추천 A/B): 같은 GPTQ 모델에서 **엔진 dtype만 fp16으로 바뀌게** 만들었을 때 TPOT가 유의미하게 내려가면 H1 지지.

### H2. (보조, 신뢰도 중간~높음) **decode TPOT는 attention/KV/scheduler가 지배 → weight-only quant 이득이 약함**
- 요지: 디코드 단계의 토큰당 시간은 (대략) **KV cache 읽기/attention 연산량/스케줄링 오버헤드** 비중이 커서, Linear만 줄여서는 TPOT가 크게 안 내려갈 수 있습니다.
- 특히 1.2B급은 “원래도 weights가 병목이 아닌” 케이스가 많아 메모리만 줄어도 TPOT가 그대로일 수 있습니다.
- 반증: vLLM 메트릭(또는 프로파일)에서 decode time 비중이 압도적이고, fp16 전환 후에도 큰 변화가 없으면 H2가 더 중요.

### H3. (사례 근거 있음, 신뢰도 중간) **GPTQ/AWQ는 ‘항상’ 빠르지 않다**
- vLLM 이슈 #4359: vLLM 자체 경고로 “GPTQ/AWQ는 최적화가 완전하지 않아 non-quantized보다 느릴 수 있다”는 사용자 보고가 있음.
- vLLM 이슈 #5316: 실제 서빙 벤치에서 **gptq가 TPOT를 악화**시키는 수치 예시가 존재(Mean TPOT 증가).  
  → “메모리 감소 = TPOT 감소”가 아니라는 외부 근거.

### H4. (주의, 신뢰도 중간) **`fix_mistral_regex` 경고는 오탐일 수 있으나, 진짜 mismatch면 TPOT/정확도에 영향**
- Transformers 이슈 #42591: `transformers==4.57.3`에서 **로컬 디렉토리(모델 파일 포함)로 tokenizer를 로드하면 non-Mistral tokenizer에도 경고가 잘못 뜨는 재현**이 보고됨.  
  → 경고 자체를 “원인”으로 단정하면 위험.
- 다만, 실제로 regex mismatch가 있으면 토큰 수/stop 패턴/정확도에 영향을 줄 수 있어 “무시”도 위험.  
- 반증: (1) 동일 텍스트 토큰화 결과가 기대와 크게 다르거나 (2) stop이 길어져 출력 토큰이 늘어나는 현상이 관찰되면 실제 이슈 가능성↑.

---

## 5) 해결/개선 방안 (대회 제약 내 “모델 아티팩트” 중심, 우선순위)

### A1. (우선순위 1) **모델 아티팩트에서 `torch_dtype=float16`로 고정되게 재패키징**
- 목표: vLLM이 `dtype=torch.float16`으로 엔진을 뜨게 만들어 **Marlin+bf16 비효율 경로를 회피**.
- 구현 아이디어(제출물 `model/` 내에서 가능한 범주):
  - `config.json`의 `"torch_dtype"`를 `"float16"`로 맞추는 방식이 가장 현실적(평가 서버는 dtype 옵션을 명시하지 않음 → auto 동작 가능성 높음).
  - unquantized로 남는 레이어(예: lm_head 등)가 있다면 그 가중치 dtype도 fp16로 맞추는 것을 고려.
- 기대효과: TPOT 개선 가능성이 가장 큼(H1 직접 타격).
- 리스크: 드물게 bf16 대비 수치 안정성/정확도에 미세 영향 가능 → A/B로 PerfNorm 확인.

### A2. (우선순위 2) **`fix_mistral_regex` 경고는 “오탐/진짜”를 분리 검증**
- 오탐 가능성 근거가 강하므로(#42591), 경고만 보고 tokenizer를 뜯어고치기 전에:
  - 짧은 샘플 텍스트(영문/한글/구두점 포함)의 **토큰 수/토큰 분해**가 상식적으로 이상한지 확인
  - `finish_reason`가 stop인지 length인지, 출력 길이가 베이스 대비 과도하게 늘었는지 확인
- 진짜 mismatch로 판단되면: 평가 서버는 `AutoTokenizer.from_pretrained(MODEL, ...)`에 fix flag를 못 주므로, **제출 tokenizer 파일(tokenizer.json)의 regex를 교정**하는 식으로 해결해야 함(단, 모델 학습 시 사용된 tokenizer와 어긋나면 정확도 리스크).

### A3. (우선순위 3) **동일 조건 A/B 측정으로 “TPOT 병목” 확정**
- 비교축(최소 세트):
  - `GPTQ + bf16(auto)` vs `GPTQ + fp16(유도)`  
  - 가능하면 `Base(BF16/FP16)`도 같이 측정(quant 자체가 느린지 분리)
- 판단 기준:
  - 1순위: `Time/Tokens`(TPOT 관점) 개선 폭
  - 2순위: 성능(PerfNorm) 유지
- 해석:
  - fp16 전환만으로 TPOT가 내려가면 H1 지배적.
  - 거의 변화 없으면 H2/H3(구조적/attention 지배) 쪽으로 우선순위 이동.

### A4. (차선책) **GPTQ 외 포맷 재탐색: L4에서 유리한 “FP8/INT8 계열” 가능성**
- L4는 FP8 텐서코어 지원(하드웨어 스펙 문서 다수) → 상황에 따라 INT4보다 **FP8/INT8 경로가 TPOT에 유리**할 수 있음.
- 대회 서버는 `compressed-tensors==0.13.0`가 고정 설치(`README.md:117`)되어 있어, vLLM의 compressed-tensors 계열 포맷도 현실적 후보.
- 단, “가능성”이므로 반드시 로컬 L4에서 벤치로 확인(포맷에 따라 오히려 느릴 수도 있음).

### A5. (장기/고효율) **TPOT를 진짜로 줄이려면 FLOPs/attention 비용 자체를 줄이는 구조적 경량화**
- 예: 레이어/헤드/hidden 축소, KV-head 축소(GQA/MQA화), 프루닝 + 재튜닝/증류 등.
- 효과: weight-only보다 TPOT에 직접적일 가능성 높음(H2 타격).
- 비용/리스크: 정확도 하락 가능성이 커 PerfNorm 방어를 위한 튜닝이 필요.

---

## 6) “이번 케이스”에 대한 권장 우선순위 (요약)
1) **`torch_dtype=float16` 유도(아티팩트 레벨)**로 먼저 TPOT가 내려가는지 확인  
2) `fix_mistral_regex`는 **Transformers 4.57.3 오탐 가능성(#42591)**을 전제로, 토큰화/종료시점 이상이 있을 때만 교정  
3) 그래도 TPOT가 안 내려가면: **(i) 포맷 재탐색(FP8/INT8/다른 4bit 방식) → (ii) 구조적 경량화** 순으로 확장

---

## Sources
- `README.md:41` (SpeedNorm 정의), `README.md:83-125` (평가 서버/패키지 고정), `README.md:163-169` (vLLM 옵션 고정)
- vLLM marlin bf16 경고 코드: `https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/utils/marlin_utils`
- vLLM GPTQModel/Marlin 설명: `https://docs.vllm.ai/en/latest/features/quantization/gptqmodel/`
- vLLM 이슈(quant 느릴 수 있음): `https://github.com/vllm-project/vllm/issues/4359`, `https://github.com/vllm-project/vllm/issues/5316`
- Transformers 4.57.3 오탐 이슈: `https://github.com/huggingface/transformers/issues/42591`
- Ada(=CC 8.9) 근거: `https://docs.nvidia.com/cuda/pdf/Ada_Tuning_Guide.pdf`
- L4가 Ada 계열임을 뒷받침(예: NVIDIA 드라이버 릴리즈 노트): `http://docs.nvidia.com/datacenter/tesla/tesla-release-notes-535-104-12/index.html`

원하시면, 지금 가진 정보(README + 제공 로그 일부) 기준으로 **“A1(fp16 유도)”를 실제 제출 아티팩트에서 어떻게 안전하게 적용할지(변경해야 할 파일/키, 주의점, 롤백 포인트)**까지 체크리스트 형태로 더 구체화해드릴게요.
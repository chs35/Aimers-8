# EXAONE-4.0-1.2B 새로운 양자화 플랜

## Context: 왜 새 플랜이 필요한가

기존 GPTQ 4bit 양자화 결과가 **속도도 느리고 정확도도 떨어지는** 최악의 결과를 보임:

| 모델 | ms/token | GSM8K 정확도(strict) | KV cache |
|------|----------|---------------------|----------|
| **Base (BF16)** | 1.598 | 67.25% | 229,808 |
| **GPTQ 4bit** | **1.967 (+23%)** | **61.03% (-6.2%p)** | 293,584 |

**근본 원인 (issue_repot.md 분석):**
1. L4(SM89, Ada Lovelace)에서 Marlin 커널 + bf16 조합이 비효율적
2. 1.2B 소형 모델에서 Weight-only INT4는 TPOT에 거의 도움이 안 됨 (attention/KV가 병목)
3. GPTQ 캘리브레이션으로 인한 정확도 하락

---

## 평가 산식 분석

```
Score = 0.5 × PerfNorm + 0.5 × SpeedNorm
PerfNorm = Perf_model / Perf_base  (정확도 비율)
SpeedNorm = 1 - (ms/token)_model / (ms/token)_base  (속도 개선 비율)
```

**핵심 인사이트:** 정확도 1%p 하락 = 속도 1%p 개선과 동일한 가치. 따라서 **정확도 손실이 적으면서 속도가 올라가는 방법**이 최선.

| 시나리오 | PerfNorm | SpeedNorm | Score |
|---------|----------|-----------|-------|
| 베이스 모델 (변화 없음) | 1.00 | 0.00 | **0.500** |
| FP8 W8A8 (99% 정확도, 25% 빠름) | 0.99 | 0.25 | **0.620** |
| AWQ 4bit (93% 정확도, 40% 빠름) | 0.93 | 0.40 | **0.665** |
| 기존 GPTQ 4bit (91% 정확도, -23% 느림) | 0.91 | -0.23 | **0.340** |

---

## 전략: 3단계 우선순위

### [우선순위 1] FP8 W8A8 Dynamic (가장 유망)

**왜 이게 최선인가:**
- L4는 Ada Lovelace(SM89)로 **FP8 텐서코어 하드웨어 네이티브 지원**
- W8A8 = Weight + Activation 모두 양자화 → **compute 자체를 줄여 TPOT 직접 감소**
- 정확도 손실 거의 없음 (near-lossless, PerfNorm ≈ 0.99+)
- **캘리브레이션 데이터 불필요** (Dynamic quantization)
- vLLM이 compressed-tensors 포맷 **자동 인식** (런타임 플래그 불필요)
- 평가 서버에 `compressed-tensors==0.13.0` 이미 설치됨

**구현 (llmcompressor 사용):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    trust_remote_code=True,
)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

oneshot(model=model, recipe=recipe)

model.save_pretrained("models/EXAONE-4.0-1.2B-fp8-w8a8", save_compressed=True)
tokenizer.save_pretrained("models/EXAONE-4.0-1.2B-fp8-w8a8")
```

**config.json에 자동 생성되는 설정:**
```json
{
  "quantization_config": {
    "quant_method": "compressed-tensors",
    ...
  }
}
```

**중요: config.json의 `torch_dtype`을 `"float16"`으로 설정**
- L4에서 bf16보다 fp16이 양자화 커널과 더 잘 호환됨

**기대 효과:**
- PerfNorm: ~0.97-1.00 (거의 무손실)
- SpeedNorm: ~0.15-0.30 (15-30% 속도 향상)
- Score: ~0.56-0.65

---

### [우선순위 2] AWQ 4bit + fp16 + Marlin (공격적 속도 전략)

**왜 AWQ인가 (GPTQ 대신):**
- Ada Lovelace에서 AWQ Marlin이 GPTQ Marlin보다 **~15% 빠름** (RTX 4090 벤치마크)
- AWQ가 GPTQ보다 정확도 보존이 더 좋음 (캘리브레이션 오버피팅 적음)
- 다만 AWQ는 **반드시 float16 필수** (bf16 미지원 → ValueError 발생)

**구현 (autoawq 사용):**
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    trust_remote_code=True,
)

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}

model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_dataset)
model.save_quantized("models/EXAONE-4.0-1.2B-awq-4bit")
tokenizer.save_pretrained("models/EXAONE-4.0-1.2B-awq-4bit")
```

**후처리 필수:** config.json의 `torch_dtype`을 `"float16"`으로 변경

**기대 효과:**
- PerfNorm: ~0.90-0.95
- SpeedNorm: ~0.30-0.45
- Score: ~0.60-0.70

**리스크:** autoawq가 평가 서버에 설치되어 있지 않으므로, AWQ 모델이 vLLM의 `awq_marlin` 경로를 자동으로 타는지 확인 필요. vLLM은 config.json의 `quant_method: "awq"`를 보고 자동 로딩하므로 괜찮을 것으로 예상.

---

### [우선순위 3] INT8 W8A8 (SmoothQuant) — 안전한 대안

**FP8가 L4에서 문제가 있을 경우 대비:**
- INT8은 SM75(Turing)부터 지원 → L4에서 확실히 동작
- SmoothQuant으로 activation 분포 안정화
- 캘리브레이션 데이터 필요

**구현:**
```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

oneshot(
    model=model,
    dataset=calibration_ds,
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)
```

---

### [즉시 테스트] 기존 GPTQ 모델 fp16 전환

기존 GPTQ 4bit 모델의 config.json에서 `torch_dtype`만 `"float16"`으로 바꿔서 벤치마크 재실행. Marlin+bf16 문제가 원인이면 이것만으로 TPOT가 개선될 수 있음.

---

## 실행 계획

### Step 1: 기존 GPTQ fp16 전환 테스트 (즉시, 10분)
- `models/EXAONE-4.0-1.2B-gptq-4bit-g128-sym/config.json` 에서 `torch_dtype` → `"float16"` 변경
- 벤치마크 실행하여 ms/token 변화 확인
- H1(Marlin+bf16 비효율) 가설 검증

### Step 2: FP8 W8A8 Dynamic 양자화 스크립트 작성 및 실행
- `experiments/quantize_fp8_dynamic.py` 작성
- llmcompressor의 `QuantizationModifier(scheme="FP8_DYNAMIC")` 사용
- 캘리브레이션 불필요 → 빠른 실행
- 결과 모델의 config.json에서 `torch_dtype`을 `"float16"`으로 확인/수정

### Step 3: FP8 모델 벤치마크
- `python benchmark.py gsm8k -m models/EXAONE-4.0-1.2B-fp8-w8a8`
- ms/token, 정확도 확인
- PerfNorm, SpeedNorm 계산

### Step 4: (FP8 결과가 부족하면) AWQ 4bit 양자화
- `experiments/quantize_awq.py` 작성
- autoawq 라이브러리 사용
- 캘리브레이션 데이터: MANTA-1M
- 반드시 fp16 dtype 설정

### Step 5: 최종 모델 선정 및 submit.zip 패키징
- 가장 Score가 높은 모델 선택
- `model/` 디렉토리로 패키징
- 제출 용량 확인 (10GB 이내)

---

## 수정/생성할 파일

| 파일 | 작업 |
|------|------|
| `experiments/quantize_fp8_dynamic.py` | **신규 생성** - FP8 W8A8 양자화 스크립트 |
| `experiments/quantize_awq.py` | **신규 생성** - AWQ 4bit 양자화 스크립트 (Step 4) |
| 기존 GPTQ 모델 `config.json` | `torch_dtype` → `"float16"` 변경 (Step 1 테스트) |

---

## 검증 방법

1. **속도 검증:** `python benchmark.py gsm8k -m models/<모델>` 실행하여 ms/token 측정
2. **정확도 검증:** GSM8K strict-match로 PerfNorm 계산
3. **Score 계산:** `Score = 0.5 * (정확도/베이스정확도) + 0.5 * (1 - ms_per_token/베이스ms_per_token)`
4. **제출 호환성:** vLLM이 런타임 플래그 없이 모델을 자동 로딩하는지 확인
5. **용량 확인:** 모델 디렉토리 크기 < 10GB (압축 전)

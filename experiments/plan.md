아래 계획서는 **(1) GPTQ 4bit로 양자화한 체크포인트를 만들고**, **(2) vLLM이 서버에서 플래그 없이도 자동으로 “Marlin 커널 경로”를 타게** 하도록 **모델 파일(`config.json`)에 필요한 메타데이터를 내장**하는 것을 목표로 합니다. (vLLM은 `--quantization`이 None이면 `config.json`의 `quantization_config`를 먼저 확인합니다. ([vLLM][1]))

---

# EXAONE-4.0-1.2B GPTQ 4bit(마릴린) 양자화 계획서

## 0) 목표 / 성공 기준

### 목표

* L4에서 **decode latency/token 감소** (토큰당 추론 시간 단축)
* 평가 서버에서 **런타임 플래그 없이** 업로드한 모델만으로 **GPTQ 4bit + (가능 시) Marlin 커널**을 사용하도록 구성

### 성공 기준(측정 지표)

* 동일 조건(동일 max_tokens / temperature=0 / 동일 프롬프트 길이)에서

  * **latency/token**: FP16 대비 유의미한 감소

---

## 1) 핵심 제약과 설계 원칙

### 제약

* 평가 서버에서 `--quantization gptq_marlin` 같은 **플래그 지정 불가**
* **모델 파일만** 업로드 가능

### 설계 원칙

* 원본 모델의 위치는 models/EXAONE-4.0-1.2B에 위치할 예정 (평가서버에는 이미 존재함)
* vLLM이 **모델의 `config.json`에 포함된 `quantization_config`를 읽어 자동 인식**하도록 만든다.
* `quant_method`는 기본적으로 **"gptq"로 두고**, vLLM이 런타임에 **“convertible to gptq_marlin… Using gptq_marlin kernel.”** 경로로 자동 전환되게 유도한다. (실제 L4 환경 사례)

  * 이유: 강제로 `gptq_marlin`을 지정하면 **모델 config의 quant_method와 불일치**로 에러가 나는 케이스가 보고되어 있음.

---

## 2) Marlin 호환 GPTQ 4bit 설정(권장 기본값)

Marlin 경로를 타려면 **“Marlin이 지원하는 형태”**로 GPTQ를 만들어야 합니다.

### 필수 성격(권장)

* **bits = 4**
* **sym = true (대칭 양자화)**

  * vLLM의 GPTQ-Marlin 구성은 (4bit, sym=True) 같은 케이스를 타입 맵으로 갖고 있어, 대칭이 핵심 전제입니다. ([vLLM][3])
  * 또한 GPTQModel 문서/안내에서도 **Marlin이 비대칭(asymmetric)을 항상 지원하는 건 아니다**는 맥락이 있습니다. ([Hugging Face][4])
* **desc_act = false** (Marlin 경로/변환에서 자주 요구되는 설정으로 알려져 있고, 실제 예시 config에도 false로 들어가 있음) ([GitHub][2])
* **group_size = 128** (속도/품질/호환성의 표준적인 타협점)

### “실전적으로 무난한” 베이스 config (예시)

아래처럼 `quantization_config`가 `config.json`에 들어가게 만드는 것이 목표입니다(형태는 GPTQ로 저장한 체크포인트가 보통 자동 생성).

```json
"quantization_config": {
  "bits": 4,
  "group_size": 128,
  "desc_act": false,
  "sym": true,
  "true_sequential": true,
  "damp_percent": 0.1,
  "quant_method": "gptq"
}
```

> 주의: 위 키들 중 일부(예: exllama_config, use_exllama 등)는 툴체인에 따라 자동으로 추가되기도 합니다. 핵심은 **bits / group_size / desc_act / sym / quant_method**가 명확히 들어가고, `quant_method`를 **gptq**로 유지하는 것입니다.

---

## 3) 구현 절차(오프라인 양자화 → 업로드 산출물 생성)

### Step 1. 캘리브레이션 데이터 준비

* 목적: GPTQ는 캘리브레이션 샘플이 품질에 직접 영향
* 권장:

  * 512~2k 샘플
  * 한국어/영어 혼합 + 평가 도메인(수학/추론/지시문 등) 비중 반영
  * 입력 길이도 짧은/긴 케이스 섞기 (짧은 질문만 쓰면 긴 컨텍스트에서 오차가 커질 수 있음)

### Step 2. GPTQModel 기반 4bit 양자화 수행

* vLLM은 GPTQModel 기반 양자화 체크포인트를 공식적으로 다룹니다.
* 실무 권장: **GPTQModel로 “GPTQ 포맷” 저장** → vLLM이 읽는 표준 HF 디렉토리 형태로 저장

* 설정값:
  * bits=4, group_size=128, sym=True, desc_act=False
  * true_sequential=True(품질 쪽 방어)
  * damp_percent=0.01~0.1 (0.1이 예시로 자주 보임)

---

## 4) “플래그 없이 Marlin 경로”를 보장하기 위한 파일 내장 전략

### 4.1 vLLM 동작 기대치

* 서버에서 `--quantization`을 안 주면(=None) vLLM은 **모델 config의 `quantization_config`를 먼저 본다**
* 이때 `quant_method: "gptq"`로 로딩하되, 모델이 Marlin 변환 가능하면 런타임에
  **“convertible to gptq_marlin… Using gptq_marlin kernel.”**로 자동 전환되는 사례가 있습니다(L4에서 실제로 관측).

### 4.2 왜 `quant_method`를 gptq로 유지하나?

* `quant_method`가 gptq인 모델을 `gptq_marlin`로 강제 지정하면 **불일치 에러**가 날 수 있음.
* 우리는 플래그를 못 주는 환경이므로, **애초에 불일치 가능성을 최소화하는 구성이 안전**합니다.

---

## 5) 산출물(최종 제출물)

* `models/EXAONE-4.0-1.2B-gptq-4bit-g128-sym/` (폴더)
# vLLM NVIDIA GPU 커널별 양자화 기법 체계적 정리

vLLM은 양자화 방식에 따라 **Marlin, Machete, CUTLASS, DeepGEMM, Triton, cuBLAS(torch.scaled_mm)** 등 6개 이상의 커널 백엔드를 자동 선택하여 추론을 수행한다. GPU 세대(SM 버전)에 따라 사용 가능한 커널이 달라지며, **Ampere(SM 8.0)에서는 Marlin이 주력이고 Hopper(SM 9.0)에서는 Machete가 최적, Blackwell(SM 10.0)에서는 NVFP4 CUTLASS와 FP8이 핵심**이다. 특히 Blackwell에서 INT8 W8A8이 하드웨어 수준에서 미지원되는 점은 마이그레이션 시 반드시 고려해야 한다.

---

## 커널별 지원 양자화 포맷과 핵심 특성

vLLM의 양자화 커널은 크게 **weight-only 커널**(Marlin, Machete)과 **weight+activation 커널**(CUTLASS W8A8, DeepGEMM)로 나뉜다. 각 커널은 서로 다른 GPU 아키텍처에 최적화되어 있으며, vLLM 런타임이 자동으로 최적의 커널을 선택한다.

### 주요 커널 개요

| 커널 | 개발 주체 | 기반 기술 | 최소 SM | 최적 대상 GPU | 양자화 유형 | 핵심 특성 |
|------|-----------|-----------|---------|--------------|------------|-----------|
| **Marlin** | IST-DASLab + Neural Magic | 수작업 CUDA, `mma` 명령어 | SM 7.5 (제한적), SM 8.0 (전체) | Ampere (A100) | Weight-only | 비동기 메모리 로드, 4단계 파이프라인 |
| **Machete** | Neural Magic (Red Hat) | CUTLASS 3.5.1, `wgmma` 명령어 | SM 9.0 | Hopper (H100) | Weight-only | TMA, warp-specialization |
| **CUTLASS W8A8** | NVIDIA CUTLASS | CUTLASS scaled_mm | SM 7.5 (INT8), SM 8.9 (FP8) | Ada/Hopper | Weight+Activation | per-tensor/per-token/per-channel 스케일링 |
| **DeepGEMM** | DeepSeek | CUTLASS 기반 | SM 9.0 | Hopper/Blackwell | Weight+Activation (Block FP8) | 128×128 블록 단위 양자화 |
| **NVFP4 CUTLASS** | NVIDIA | CUTLASS | SM 10.0 | Blackwell (B200) | Weight+Activation (FP4) | 네이티브 FP4 텐서 코어 |
| **Triton** | OpenAI | Triton 컴파일러 | SM 9.0 | Hopper | 블록 FP8, MXFP4 | 범용 폴백 |
| **cuBLAS (torch.scaled_mm)** | NVIDIA/PyTorch | cuBLAS | SM 7.5+ (INT8), SM 8.9+ (FP8) | 범용 | Weight+Activation | CUTLASS 미사용 시 폴백 |
| **FlashInfer** | FlashInfer 프로젝트 | CUTLASS 기반 | SM 9.0 | Hopper/Blackwell | FP8 MoE, NVFP4 MoE | MoE 전용 최적화 |

### Marlin 커널 상세

Marlin은 "Mixed Auto-Regressive LINear kernel"의 약자로, **IST-DASLab**(GPTQ 개발팀)이 만든 고성능 weight-only 양자화 커널이다. Ampere GPU에서 이론적 최대 **3.87배 속도 향상**(FP16 대비 W4A16)을 달성하며, 배치 크기 16~32에서 가장 효율적이다.

Marlin이 지원하는 양자화 포맷은 다음과 같다:

| 포맷 | 가중치 타입 | 활성화 타입 | 그룹 크기 | 비고 |
|------|-----------|-----------|----------|------|
| GPTQ-Marlin INT4 | INT4 | FP16/BF16 | 32, 64, 128, channelwise | act_order 지원 |
| GPTQ-Marlin INT8 | INT8 | FP16/BF16 | 32, 64, 128, channelwise | act_order 지원 |
| AWQ-Marlin | INT4 (AWQ) | FP16/BF16 | 128 (일반적) | act_order 미지원 |
| FP8-Marlin | FP8 E4M3 | FP16/BF16 | per-channel | SM 8.0~8.8에서 FP8 W8A16 폴백 |
| HQQ-Marlin | INT4 (HQQ) | FP16/BF16 | 64, 128 | Half-Quadratic 양자화 |
| Sparse-Marlin (2:4) | INT4 + 2:4 sparsity | FP16/BF16 | 128 | NVIDIA 구조적 희소성 |
| W4A8-FP8 | INT4 | FP8 E4M3 | 128 | **SM 8.9와 SM 12.0만 지원** (SM 9.0/10.0 미지원) |

**핵심 제약 사항**: Marlin의 W4A8-FP8 경로(INT4 가중치 + FP8 활성화)는 SM 9.0(Hopper)과 SM 10.0(Blackwell DC)에서 `QMMA.16832.F32.E4M3.E4M3` 명령어가 없어 사용 불가하다. SM 8.9(Ada Lovelace)와 SM 12.0(RTX Blackwell)에서만 동작한다.

### Machete 커널 상세

Machete는 Marlin의 "정신적 후속작"으로, **Hopper GPU의 `wgmma` 명령어**를 활용한다. Marlin이 Hopper에서 구형 `mma` 명령어만 사용하여 **최대 연산 성능의 ~37%를 손실**하는 문제를 해결한다. CUTLASS 3.5.1 기반으로 구축되어 새로운 아키텍처 확장이 용이하다.

| 특성 | Marlin | Machete |
|------|--------|---------|
| 텐서 코어 명령어 | `mma` (레거시) | `wgmma` (Hopper 네이티브) |
| 메모리 전송 | 비동기 copy | **TMA (Tensor Memory Accelerator)** |
| 구현 방식 | 수작업 CUDA | CUTLASS 3.5.1 + CUTE 레이아웃 |
| 최적 배치 크기 | **1~32** (메모리 바운드) | **64~256+** (컴퓨트 바운드) |
| Hopper 성능 | 피크 대비 ~63% | **피크 대비 ~100%** |
| 2:4 구조적 희소성 | ✅ 지원 | ❌ 미지원 |
| 지원 포맷 | INT4, INT8, FP8, FP4 | INT4, INT8 (FP8 예정) |

**실제 서빙 성능 비교**: Llama 3.1 405B 4-bit 모델을 4×H100에서 3 req/s로 서빙할 때, Machete는 이전 커널 대비 입력 토큰 처리량 **29% 향상**, 출력 토큰 처리량 **32% 향상**을 달성했다.

---

## 양자화 방식별 커널 매핑과 자동 선택 로직

vLLM은 `choose_mp_linear_kernel()` 함수를 통해 런타임에 GPU 컴퓨트 캐파빌리티를 확인하고 최적 커널을 자동 선택한다. 아래 표는 각 양자화 방식이 GPU별로 어떤 커널에 매핑되는지를 정리한 것이다.

| 양자화 방식 | 유형 | SM 7.5 (Turing) | SM 8.0/8.6 (Ampere) | SM 8.9 (Ada) | SM 9.0 (Hopper) | SM 10.0 (Blackwell DC) |
|------------|------|----------------|---------------------|-------------|-----------------|----------------------|
| **FP8 W8A8** (per-tensor) | W+A | ❌ | ❌ (→W8A16 Marlin) | CUTLASS | CUTLASS | CUTLASS |
| **FP8 W8A8** (block 128×128) | W+A | ❌ | ❌ | ❌ | DeepGEMM/CUTLASS | DeepGEMM/CUTLASS |
| **FP8 W8A16** | Weight-only | ❌ | Marlin | Marlin | Marlin | Marlin |
| **INT8 W8A8** | W+A | CUTLASS | CUTLASS | CUTLASS | CUTLASS | **❌ 미지원** |
| **GPTQ INT4** (W4A16) | Weight-only | Marlin (제한) | **Marlin** | Marlin | **Machete** | Machete |
| **GPTQ INT8** (W8A16) | Weight-only | Marlin (제한) | Marlin | Marlin | Machete | Machete |
| **AWQ INT4** (W4A16) | Weight-only | Marlin (제한) | **Marlin** | Marlin | **Machete** | Machete |
| **GPTQ-Marlin 2:4** | Weight-only+Sparse | ❌ | Marlin Sparse | Marlin Sparse | Marlin Sparse | Marlin Sparse |
| **MXFP4/NVFP4** (W4A4) | W+A | ❌ | ❌ | ❌ | Triton | **NVFP4 CUTLASS** |
| **MXFP4** (W4A16 폴백) | Weight-only | Marlin | Marlin | Marlin | Triton | NVFP4 CUTLASS |
| **BitsAndBytes** (NF4/INT8) | Weight-only | BnB 라이브러리 | BnB 라이브러리 | BnB 라이브러리 | BnB 라이브러리 | BnB 라이브러리 |
| **GGUF** (Q4_K_M 등) | Weight-only | GGUF CUDA | GGUF CUDA | GGUF CUDA | GGUF CUDA | GGUF CUDA |
| **Compressed Tensors** | 유동적 | 하위 방식에 따라 자동 디스패치 ||||
| **ModelOpt FP8** | W+A | ❌ | ❌ | CUTLASS | CUTLASS | CUTLASS |
| **FBGEMM FP8** | W+A | ❌ | ❌ | CUTLASS | CUTLASS | CUTLASS |
| **TorchAO** | 다양 | torch.compile | torch.compile | torch.compile | torch.compile | torch.compile |

### 커널 자동 선택 결정 트리

vLLM의 커널 디스패치 로직은 다음과 같은 우선순위로 동작한다:

**Weight-only 양자화 (GPTQ, AWQ, W4A16, W8A16)**:
1. SM ≥ 9.0 → `MacheteLinearKernel.can_implement()` 확인 → **Machete** 선택
2. SM 8.0~8.9 → `MarlinLinearKernel.can_implement()` 확인 → **Marlin** 선택
3. SM < 8.0 → 레거시 커널 (ExLlamaV2 등) 폴백

**FP8 W8A8 양자화**:
1. SM < 8.9 → 네이티브 FP8 미지원 → **FP8 Marlin** (W8A16으로 디퀀트)
2. SM ≥ 8.9 + 블록 양자화 → **DeepGEMM** 또는 **CUTLASS 블록 FP8**
3. SM ≥ 8.9 + per-tensor → **CUTLASS scaled_mm**
4. CUTLASS 미사용 시 → **torch.scaled_mm** (cuBLAS) 폴백

**FP8 MoE 양자화**:
1. SM 10.0 (Blackwell) → **FlashInfer TRTLLM**
2. SM 9.0 (Hopper) → **FlashInfer CUTLASS** 또는 **DeepGEMM**
3. SM 7.5~8.9 → **Marlin** (weight-only FP8)
4. 범용 폴백 → **Triton**

**중요**: 사용자가 `--quantization gptq`처럼 명시적으로 양자화 방식을 지정하면, vLLM이 Marlin/Machete로 자동 업그레이드하지 않는다. 최적 커널 자동 선택을 위해서는 `quantization` 파라미터를 생략하고 모델 설정에서 자동 감지하도록 하는 것이 권장된다.

---

## NVIDIA GPU 세대별 양자화 지원 종합 매트릭스

아래 표는 각 GPU 세대에서 실제로 동작하는 양자화 방식과 사용되는 커널을 종합한 것이다. **Blackwell의 INT8 미지원**과 **Volta/Turing의 Marlin 제한**이 가장 주의해야 할 포인트다.

| 양자화 방식 | Volta SM 7.0 | Turing SM 7.5 | Ampere SM 8.0/8.6 | Ada SM 8.9 | Hopper SM 9.0 | Blackwell DC SM 10.0 | Blackwell RTX SM 12.0 |
|------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| FP8 W8A8 (per-tensor) | ❌ | ❌ | ❌¹ | ✅ | ✅ | ✅ | ✅ |
| FP8 W8A8 (block) | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| FP8 W8A16 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| INT8 W8A8 | ❌ | ✅ | ✅ | ✅ | ✅ | **❌** | **❌** |
| GPTQ/AWQ W4A16 | ⚠️² | ✅³ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GPTQ/AWQ W8A16 | ⚠️² | ✅³ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GPTQ 2:4 Sparse | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MXFP4/NVFP4 (네이티브) | ❌ | ❌ | ❌ | ❌ | ⚠️⁴ | ✅ | ⚠️⁵ |
| BitsAndBytes | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GGUF | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| AQLM | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

> ¹ Ampere에서 FP8 W8A8은 불가하며, FP8 Marlin을 통한 W8A16 폴백 사용  
> ² Volta에서는 Marlin 미지원, 레거시 GPTQ/AWQ 커널만 동작  
> ³ Turing의 Marlin은 FP16 활성화 전용, BF16 미지원, 2단계 파이프라인(Ampere의 4단계 대비 성능 저하)  
> ⁴ Hopper에서 MXFP4는 Triton 백엔드로 동작 (네이티브 FP4 텐서 코어 없음)  
> ⁵ SM 12.0에서 NVFP4 MoE 백엔드 선택 버그 존재 (SM120 패밀리 감지 문제, PR #31085, #33416 참고)

---

## 커널별 성능 특성과 최적 사용 시나리오

각 커널의 성능은 **배치 크기**, **모델 크기**, **GPU 세대**에 따라 크게 달라진다. 다음은 실측 벤치마크 기반의 성능 비교이다.

### 처리량 벤치마크 (Qwen2.5-32B, H200 기준)

| 양자화 방식 | 사용 커널 | 처리량 (tok/s) | FP16 대비 비율 | 비고 |
|------------|----------|:---:|:---:|------|
| FP16 (baseline) | cuBLAS | 461 | 1.0x | 기준선 |
| **AWQ-Marlin** W4A16 | Marlin/Machete | **741** | **1.61x** | 최고 처리량 |
| **GPTQ-Marlin** W4A16 | Marlin/Machete | **712** | **1.54x** | AWQ와 유사 |
| FP8 W8A8 (동적) | CUTLASS | ~550 | ~1.19x | 캘리브레이션 불필요 |
| FP8 W8A8 (정적) | CUTLASS | ~600 | ~1.30x | 최적 레이턴시 |
| BitsAndBytes NF4 | BnB 라이브러리 | 168 | 0.36x | 가장 느림 |
| GGUF Q4_K_M | GGUF CUDA | 93 | 0.20x | vLLM에서 비권장 |

### 배치 크기별 최적 커널

배치 크기는 커널 선택에 결정적 영향을 미친다. **소규모 배치에서는 메모리 대역폭이 병목**이고, **대규모 배치에서는 연산 처리량이 병목**이 된다.

| 배치 크기 | 최적 커널 (Hopper) | 최적 커널 (Ampere) | 설명 |
|:---------:|:------------------:|:------------------:|------|
| 1 | ExLlamaV2 / Marlin | Marlin | 디코딩 단계, 메모리 바운드 |
| 4~16 | Marlin | **Marlin (최적)** | Marlin의 최적 영역 |
| 32~64 | Machete ≈ Marlin | Marlin | 전환 구간 |
| 64~256 | **Machete (최적)** | Marlin | Machete의 `wgmma` 이점 극대화 |
| 256+ | **Machete** | Marlin | 컴퓨트 바운드, Machete 우세 |

### Weight-only vs Weight+Activation 양자화 비교

| 특성 | Weight-only (W4A16, W8A16) | Weight+Activation (W8A8, W4A8) |
|------|---------------------------|-------------------------------|
| 대표 커널 | Marlin, Machete | CUTLASS, DeepGEMM |
| 메모리 절감 | ✅ 가중치 크기 50~75% 감소 | ✅ 가중치 + 활성화 모두 감소 |
| 연산 절감 | ❌ 활성화는 FP16으로 연산 | ✅ 저정밀도 텐서 코어 활용 |
| 최적 시나리오 | **메모리 바운드** (소배치, 대형 모델) | **컴퓨트 바운드** (대배치, 높은 동시성) |
| 정확도 영향 | 낮음 (가중치만 양자화) | 중간 (활성화 양자화로 추가 오차) |
| 캘리브레이션 | GPTQ/AWQ 필요 | FP8 동적은 불필요, 정적은 필요 |

---

## Blackwell GPU의 양자화 이슈와 권장 사항

Blackwell 아키텍처는 vLLM 양자화에서 세 가지 주요 이슈가 존재한다.

**첫째, INT8 W8A8이 하드웨어 수준에서 미지원**된다. SM 10.0 이상의 CUTLASS 커널이 INT8 연산을 지원하지 않으며, vLLM은 `"Int8 not supported on SM120. Use FP8 quantization instead"` 에러를 반환한다. 기존 INT8 모델을 Blackwell로 마이그레이션할 때는 반드시 **FP8으로 재양자화**해야 한다.

**둘째, SM 10.0(데이터센터)과 SM 12.0(RTX) 간 비대칭**이 존재한다. SM 12.0은 Marlin의 W4A8-FP8 경로를 지원하지만 SM 10.0은 지원하지 않는다. 이는 `QMMA.16832.F32.E4M3.E4M3` 명령어의 가용성 차이 때문이다. NVFP4 MoE 백엔드 선택에서도 SM 12.0 패밀리 감지 버그(`is_device_capability_family(100)` 조건이 SM 12.0에 매칭되지 않음)가 보고되었다.

**셋째, RTX 5090에서 동적 FP8 양자화가 BF16보다 느린 현상**이 보고되었다. 온라인 동적 양자화(`--quantization="fp8"`)의 오버헤드가 FP8의 연산 이득을 상쇄하여, 벽시계 시간이 약 **17.9% 증가**하고 토큰 처리량이 약 **17.2% 감소**하는 것으로 나타났다. Blackwell에서의 FP8 사용은 정적 양자화(llm-compressor 캘리브레이션) 방식이 권장된다.

---

## Compressed Tensors가 통합하는 커널 디스패치 체계

`compressed-tensors`는 Neural Magic(현 Red Hat)이 개발한 **통합 양자화 포맷**으로, 모델의 양자화 설정을 자동 감지하여 최적 커널로 라우팅한다. 이 포맷은 llm-compressor의 기본 출력 형식이며, 하나의 포맷으로 다양한 양자화 방식을 커버한다.

| 감지된 양자화 스킴 | 라우팅 대상 클래스 | 사용 커널 |
|---|---|---|
| W4A16 / W8A16 (weight N-bit) | `CompressedTensorsWNA16` | Machete (SM 9.0+) → Marlin (SM 8.0+) |
| W8A8 INT8 | `CompressedTensorsW8A8Int8` | CUTLASS `cutlass_scaled_mm` |
| W8A8 FP8 | `CompressedTensorsW8A8Fp8` | CUTLASS → DeepGEMM → torch.scaled_mm |
| W4A4 NVFP4 | `CompressedTensorsW4A4Nvfp4` | NVFP4 CUTLASS (SM 10.0+) |
| MoE W_NA16 | `CompressedTensorsWNA16MarlinMoEMethod` | Marlin MoE |

이 아키텍처 덕분에 사용자는 양자화 방식이나 커널을 직접 지정할 필요 없이, 모델을 로드하기만 하면 vLLM이 **GPU 컴퓨트 캐파빌리티와 양자화 설정을 동시에 고려**하여 최적 경로를 선택한다.

---

## 결론: 커널 선택 전략과 실무 권장 사항

vLLM의 양자화 커널 생태계는 **GPU 세대별 최적화의 계층 구조**로 이해하는 것이 가장 정확하다. Marlin이 Ampere의 `mma` 명령어에 최적화된 범용 기반이라면, Machete는 Hopper의 `wgmma`+TMA를 활용하는 진화형이고, NVFP4 CUTLASS는 Blackwell의 네이티브 FP4 텐서 코어를 직접 활용하는 차세대 커널이다.

실무적으로 가장 중요한 판단 기준 세 가지는 다음과 같다. **GPU 세대에 따른 커널 가용성**(특히 Blackwell의 INT8 미지원), **배치 크기에 따른 커널 성능 특성**(소배치는 Marlin, 대배치는 Machete), 그리고 **weight-only vs weight+activation의 정확도-성능 트레이드오프**(메모리 바운드면 W4A16, 컴퓨트 바운드면 FP8 W8A8)이다. vLLM의 자동 커널 선택 기능을 최대한 활용하되, Blackwell 마이그레이션이나 특수한 배치 패턴에서는 커널 동작을 명시적으로 이해하고 검증해야 한다.
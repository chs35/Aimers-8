# vLLM 양자화 + 희소성 결합: 커널 제약과 GPU 요구사항 전체 정리

vLLM에서 양자화(Quantization)와 2:4 구조적 희소성(Structured Sparsity)을 결합하면 **최대 3.0배의 추론 속도 향상**을 달성할 수 있으나, 커널별로 지원하는 포맷과 GPU 아키텍처 요구사항이 크게 다르다. vLLM은 현재 두 개의 독립적인 sparse 커널 경로—**Marlin 2:4 (INT4 전용, SM 8.0+)**와 **CUTLASS 2:4 (INT8/FP8, SM 9.0+)**—를 운용하며, 각 경로의 지원 범위, 제약사항, 그리고 최근 진행 중인 Marlin24 deprecation까지 정확히 이해해야 올바른 배포 전략을 세울 수 있다.

## vLLM의 두 가지 sparse 커널 아키텍처

vLLM은 양자화+희소성 결합을 위해 완전히 분리된 두 개의 커널 경로를 제공한다. 이 둘은 서로 다른 양자화 포맷, 다른 GPU 요구사항, 다른 코드베이스를 기반으로 동작한다.

**첫 번째 경로는 Marlin 2:4 Sparse 커널**이다. IST-DASLab의 Sparse-Marlin 프로젝트에서 유래했으며, `csrc/quantization/marlin/sparse/marlin_24_cuda_kernel.cu`에 CUDA 커널이 구현되어 있다. NVIDIA의 `mma.sp` Sparse Tensor Core 명령어(m16×n8×k32, FP32 누적기)를 사용하여 **INT4 가중치 + 2:4 희소성** 조합만을 처리한다. FP16 활성화와 INT4 희소 가중치 간의 행렬곱을 수행하며, 최소 **SM 8.0 (Ampere)** 이상의 GPU가 필요하다.

**두 번째 경로는 CUTLASS 3.x 기반 2:4 Sparse 커널**이다. `csrc/sparse/cutlass/sparse_scaled_mm_c3x.cu`에 구현되어 있으며, 파일명의 "c3x"가 시사하듯 **Hopper 아키텍처 전용(SM 9.0+)**으로 설계되었다. `c2x` 버전(Ampere용)은 존재하지 않는다. 이 커널은 **W8A8 INT8** 및 **W8A8 FP8** 양자화와 2:4 희소성의 결합을 처리한다. CUDA 12.2 이상의 툴킷도 요구된다.

| 커널 경로 | 양자화 포맷 | 최소 SM | CUDA 요구 | Python 클래스 | 상태 |
|-----------|------------|---------|-----------|--------------|------|
| Marlin 2:4 Sparse | W4A16 (INT4) | SM 8.0 (Ampere) | — | `GPTQMarlin24LinearMethod` | ⚠️ Deprecation 진행 중 |
| CompressedTensorsW4A16Sparse24 | W4A16 (INT4) | SM 8.0 (Ampere) | — | `CompressedTensorsW4A16Sparse24` | ✅ 사용 가능 |
| CUTLASS 2:4 Sparse | W8A8 INT8 | **SM 9.0 (Hopper)** | CUDA 12.2+ | `CompressedTensors24` | ✅ 사용 가능 |
| CUTLASS 2:4 Sparse | W8A8 FP8 | **SM 9.0 (Hopper)** | CUDA 12.2+ | `CompressedTensors24` | ✅ 사용 가능 |
| W16A16 Sparse (양자화 없음) | None | — | — | Fallback to dense | ❌ 비활성화 |

## 양자화 + 희소성 조합별 지원 현황 상세

### INT4 (W4A16) + 2:4 구조적 희소성

가장 성숙한 조합이다. Marlin 2:4 Sparse 커널이 이 조합을 처리하며, Ampere(A100, A6000), Ada Lovelace(RTX 4090, L40), Hopper(H100, H200) GPU에서 모두 동작한다. `CompressedTensorsW4A16Sparse24` 스킴이 compressed-tensors 프레임워크 내에서 이 커널을 호출한다.

**제약사항이 명확하다.** `act_order`는 지원되지 않는다—2:4 희소성 메타데이터가 원본 열 순서에 의존하기 때문에 GPTQ의 열 재배치(act_order)와 구조적으로 충돌한다. 지원되는 `group_size`는 **-1(채널별)과 128**이다. 32, 64 등 다른 group_size는 dense Marlin에서는 지원되지만 sparse 변형에서는 사용할 수 없다.

성능 면에서는 Dense FP16 대비 커널 레벨에서 **~5.3배 속도 향상**을 달성한다(dense Marlin의 ~4배 대비 약 33% 추가 향상). 배치 크기 32까지 최적에 가까운 성능을 유지하며, 대형 배치에서도 Dense FP16 대비 일관된 2배 속도 향상을 제공한다.

### FP8 (W8A8) + 2:4 구조적 희소성

이 조합은 **SM 9.0 이상(Hopper)을 반드시 요구**한다. Ada Lovelace(SM 8.9)에서는 dense FP8가 지원되지만, sparse FP8는 동작하지 않는다. Neural Magic/Red Hat 팀이 CUTLASS v3.6 기반으로 개발한 sparse FP8 커널이 vLLM PR #10995를 통해 통합되었다.

`CompressedTensors24` 스킴이 이 조합을 처리하며, `supports_cutlass_24()` 함수가 다음 조건을 모두 검증한다: 희소성 구조가 2:4일 것, 가중치 양자화 전략이 TENSOR 또는 CHANNEL일 것, 입력 양자화 전략이 TENSOR 또는 TOKEN일 것, **8비트 양자화만 지원**(4비트 불가). Weight-only 양자화(W8A16 등)는 지원되지 않으며, 반드시 activation도 양자화된 W8A8 형태여야 한다.

Dense BF16 대비 **1.7배 낮은 지연시간**과 **1.5배 높은 처리량**을 달성한다. Dense FP8와 비교하면 matmul 커널 레벨에서 약 22%의 추가 속도 향상을 제공한다(HPC-AI.com 벤치마크 기준).

### INT8 (W8A8) + 2:4 구조적 희소성

FP8와 동일한 CUTLASS 2:4 커널 경로를 사용하며, 마찬가지로 **SM 9.0 이상(Hopper)**을 요구한다. Ampere의 하드웨어가 INT8 sparse를 지원함에도 불구하고, vLLM의 CUTLASS sparse 구현이 c3x(Hopper 전용)로만 존재하기 때문이다. 이 점은 Ampere 사용자들에게 중요한 제약이다—Ampere에서 INT8+2:4를 사용하려면 별도의 커널 구현이 필요하지만 현재 제공되지 않는다.

### NVFP4/MXFP4 + 희소성: 현재 미지원

**vLLM에서 NVFP4 또는 MXFP4와 2:4 희소성의 결합은 현재 지원되지 않는다.** `supports_cutlass_24()` 함수가 명시적으로 8비트 양자화만 허용하며, 4비트 부동소수점은 이 조건을 통과하지 못한다.

하드웨어 레벨에서 Blackwell(SM 10.0)은 NVFP4를 위한 새로운 **4:8 pairwise 구조적 희소성**을 도입했다. 이는 기존 2:4와 다른 패턴으로, 8개 요소를 4개의 연속 쌍(pair)으로 그룹화하고 그 중 정확히 2개 쌍이 0이어야 한다. Blackwell의 5세대 Tensor Core가 이를 네이티브로 지원하지만, vLLM의 소프트웨어 스택에는 아직 이 조합이 구현되지 않았다.

| 포맷 조합 | vLLM 지원 | 하드웨어 지원 | 비고 |
|-----------|-----------|-------------|------|
| NVFP4 + 2:4 | ❌ | Blackwell (4:8 pairwise) | 소프트웨어 미구현 |
| MXFP4 + 2:4 | ❌ | Blackwell | 실험적 단계 |
| NVFP4 Dense | ✅ | SM 10.0+ | vLLM 지원 |
| MXFP4 Dense | ⚠️ 실험적 | SM 10.0+ | 검증 대기 중 |

## GPU 아키텍처별 sparse Tensor Core 지원 데이터 타입

NVIDIA의 Sparse Tensor Core 하드웨어 지원과 vLLM의 소프트웨어 지원은 구분해야 한다. 하드웨어가 지원하더라도 vLLM에 커널이 구현되지 않으면 사용할 수 없다.

| 데이터 타입 | Ampere (SM 8.0) | Ada (SM 8.9) | Hopper (SM 9.0) | Blackwell (SM 10.0) |
|------------|:-:|:-:|:-:|:-:|
| FP16 2:4 | HW ✅ / vLLM ❌* | HW ✅ / vLLM ❌* | HW ✅ / vLLM ❌* | HW ✅ / vLLM ❌* |
| BF16 2:4 | HW ✅ / vLLM ❌* | HW ✅ / vLLM ❌* | HW ✅ / vLLM ❌* | HW ✅ / vLLM ❌* |
| INT8 2:4 | HW ✅ / vLLM ❌ | HW ✅ / vLLM ❌ | HW ✅ / **vLLM ✅** | HW ✅ / **vLLM ✅** |
| INT4 2:4 | HW ✅ / **vLLM ✅** | HW ✅ / **vLLM ✅** | HW ❌† | HW ❌† |
| FP8 2:4 | HW ❌ | HW ❌ | HW ✅ / **vLLM ✅** | HW ✅ / **vLLM ✅** |
| FP4 4:8 | HW ❌ | HW ❌ | HW ❌ | HW ✅ / vLLM ❌ |

*\* W16A16 sparse는 PR #12417에서 정확성 문제로 비활성화됨. Dense fallback으로 처리.*
*† INT4는 Hopper부터 deprecated.*

## Sparse-Marlin 커널의 정밀 제약사항

Marlin 2:4 Sparse 커널의 제약사항을 dense Marlin과 비교하면 그 차이가 뚜렷하다.

| 파라미터 | Dense Marlin | Sparse Marlin 2:4 |
|---------|:---:|:---:|
| 지원 비트 수 | 4, 8 | **4만 지원** |
| act_order | ✅ 지원 | **❌ 미지원** |
| group_size | -1, 32, 64, 128 | **-1, 128만 지원** |
| 최소 SM | 8.0 | 8.0 |
| FP8 입력 | SM 8.9+ | ❌ |
| 최대 효율 배치 크기 | ~16 | **~32** |
| Dense FP16 대비 속도 | ~3.87배 | **~5.3배** |

act_order 미지원은 구조적 한계에서 비롯된다. `mma.sp` 명령어는 메타데이터 인덱스가 원본 열 순서에 고정되어야 하며, GPTQ의 act_order가 요구하는 열 재배치와 양립할 수 없다. 따라서 **sparse 양자화 모델 생성 시 반드시 `act_order=False`로 설정**해야 한다.

## 비구조적 희소성(Unstructured Sparsity) 지원 현황

vLLM에서 비구조적 희소성은 **하드웨어 가속 없이** 처리된다. 과거 `--sparsity sparse_w16a16` 플래그를 통해 Neural Magic SparseML 모델(예: `neuralmagic/OpenHermes-2.5-Mistral-7B-pruned50`)을 로드할 수 있었지만, 0값이 여전히 연산에 참여하므로 실질적인 속도 향상은 없었다. 이 기능은 모델 포맷 호환성을 위한 것이었으며, 최신 vLLM 버전에서는 deprecated 상태이다.

compressed-tensors 포맷은 `sparsity_structure: "unstructured"`를 메타데이터에 기록할 수 있지만, vLLM의 커널 디스패치 로직에서 이를 활용한 최적화 커널로 라우팅하지 않는다. 모든 하드웨어 가속 sparse 커널은 **2:4 구조적 희소성 전용**이다.

## compressed-tensors에서의 sparsity 메타데이터와 디스패치 로직

### sparsity_config 구조

모델의 `config.json`에서 `quantization_config` 내부에 `sparsity_config`가 중첩된다:

```json
{
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "config_groups": { ... },
    "sparsity_config": {
      "format": "sparse-24-bitmask",
      "sparsity_structure": "2:4",
      "global_sparsity": 50.0
    }
  }
}
```

`format` 필드는 디스크 저장 방식을 결정한다. `"sparse-24-bitmask"`는 비트마스크로 비영 위치를 인코딩하고, `"dense"`는 0을 포함한 전체 가중치를 저장하며, `"marlin-24"`는 deprecated된 Marlin 전용 포맷이다.

### 커널 선택 디스패치 트리

`CompressedTensorsConfig.from_config()`이 호출되면 두 개의 맵을 파싱한다: `target_scheme_map`(양자화 설정)과 `sparsity_scheme_map`(희소성 설정). 이후 레이어별로 `get_scheme()` 메서드가 다음 결정 트리를 따른다:

1. **sparsity_structure == "2:4"인가?** → Yes이면 sparse 경로 진입
2. **supports_cutlass_24() 통과하는가?** (W8A8 INT8/FP8, TENSOR/CHANNEL/TOKEN 전략, 8비트만) → Yes이면 `CompressedTensors24` 선택
3. **W4A16 + 2:4인가?** → Yes이면 `CompressedTensorsW4A16Sparse24` 선택
4. **W16A16 + 2:4인가?** → `UnquantizedLinearMethod`로 fallback (비활성화됨)
5. **그 외** → 표준 dense 양자화 스킴으로 라우팅

부분적으로 sparse한 모델(예: MLP 레이어만 sparse)도 지원된다. PR #11889에서 수정되어, sparse 레이어는 `CompressedTensors24`로, non-sparse 레이어는 적절한 dense 스킴으로 독립적으로 라우팅된다. `sparsity_config.ignore`에 명시된 레이어는 sparse 경로를 건너뛴다.

### 가중치 로딩 프로세스

로딩 시 두 단계 압축 변환이 발생한다. 먼저 디스크의 `sparse-24-bitmask` 형식에서 **비트마스크 기반 복원**(decompression)이 수행되어 dense 텐서로 변환된다. 이후 `cutlass_sparse_compress`가 호출되어 **CUTLASS sparse 포맷**(압축 가중치 + 2비트 메타데이터 텐서)으로 재압축된다. 이 과정은 모델 로드 시 자동으로 수행된다.

## llm-compressor의 sparsity + quantization 워크플로

### 단일 패스 방식 (FP8 + 2:4)

`SparseGPTModifier`와 `QuantizationModifier`를 하나의 recipe 리스트에 넣고 `oneshot()`으로 한 번에 실행한다. SparseGPT가 먼저 2:4 패턴으로 가지치기를 수행한 후, FP8_DYNAMIC 양자화가 적용된다.

### 다단계 파이프라인 방식 (INT4 + 2:4)

3단계로 진행된다: (1) `oneshot()`으로 SparseGPT 2:4 pruning, (2) `train()`으로 sparse 복구 미세조정, (3) `oneshot()`으로 GPTQ W4A16 양자화. 각 단계의 출력이 다음 단계의 입력이 되며, 최종 모델은 compressed-tensors 포맷으로 저장된다.

| 알고리즘 | 용도 | 출력 포맷 | vLLM 커널 |
|---------|------|----------|----------|
| SparseGPT + FP8_DYNAMIC | Pruning + 양자화 | compressed-tensors (W8A8 FP8 + 2:4) | CUTLASS sparse (SM 9.0+) |
| SparseGPT + GPTQ W4A16 | Pruning + 양자화 | compressed-tensors (W4A16 + 2:4) | Marlin 2:4 (SM 8.0+) |
| SparseGPT only | Pruning만 | compressed-tensors (sparse-only) | Dense fallback |
| Wanda | 경량 pruning | compressed-tensors | 위와 동일 |

## 실제 성능 벤치마크 요약

Red Hat/Neural Magic의 Sparse-Llama-3.1-8B 모델 기준, vLLM에서 측정된 엔드투엔드 성능은 다음과 같다.

**INT4 + 2:4 sparse (Marlin 커널, vLLM 0.6.4):**

| GPU | Single-stream 지연시간 개선 | 처리량 개선 | 정확도 복구 |
|-----|:---:|:---:|:---:|
| A5000 (SM 8.6) | **3.0배** | 1.8배 | 98.4% (OpenLLM v1) |
| A6000 (SM 8.6) | **3.0배** | 1.4배 | 99.4% (미세조정 후) |
| A100 (SM 8.0) | **2.1배** | 1.2배 | — |

**FP8 + 2:4 sparse (CUTLASS 커널, H100):**

| 메트릭 | Dense BF16 대비 | Dense FP8 대비 |
|--------|:---:|:---:|
| 지연시간 | 1.7배 감소 | ~30% 감소 |
| 처리량 | 1.5배 증가 | ~22% 증가 |
| 총 속도 향상 (BF16 기준) | **최대 2.5배** | — |

2:4 희소성 단독으로는 처리량 **최대 30% 향상**, 지연시간 **~20% 감소** 수준이다. 양자화와 결합해야 극적인 효과를 얻는다.

## Marlin24 deprecation과 마이그레이션 경로

**vLLM PR #32688**에서 standalone `gptq_marlin_24` 양자화 메서드가 제거되고 있으며, llm-compressor issue #2267에서 `marlin24` 압축기 코드에 deprecation 경고가 추가되고 있다. 이는 2026년 1월 기준 진행 중인 작업이다.

그러나 `CompressedTensorsW4A16Sparse24` 스킴은 제거 대상이 아니다. 이 스킴은 내부적으로 Marlin-24 커널을 호출하지만, compressed-tensors 프레임워크의 일부로서 계속 유지된다. 따라서 **INT4 + 2:4 sparse 모델은 compressed-tensors 포맷으로 저장·배포하는 것이 권장**된다. `quantization="gptq_marlin_24"`를 명시적으로 지정하는 방식은 향후 동작하지 않을 수 있다.

## 결론: 실무 적용을 위한 핵심 가이드

첫째, **Ampere/Ada GPU 사용자**는 INT4 + 2:4 sparse 조합만 하드웨어 가속을 받을 수 있다. `CompressedTensorsW4A16Sparse24` 경로를 사용하되, `act_order=False`, `group_size=128` 또는 channelwise로 설정해야 한다.

둘째, **Hopper GPU 사용자**는 FP8 + 2:4 sparse가 최적의 선택이다. CUTLASS 2:4 sparse 커널이 FP8와 INT8 모두를 지원하며, 별도의 양자화 설정 없이 llm-compressor의 `FP8_DYNAMIC` 스킴으로 단일 패스 생성이 가능하다.

셋째, **Blackwell GPU의 NVFP4 + sparsity**는 하드웨어가 4:8 pairwise 희소성을 지원하지만, vLLM 소프트웨어 스택에 아직 구현되지 않았다. 이 조합의 실제 사용은 향후 CUTLASS/vLLM 업데이트를 기다려야 한다.

넷째, W16A16 sparse-only(양자화 없는 순수 희소성)는 정확성 버그(PR #12417)로 비활성화된 상태이며, dense fallback으로 처리되어 성능 이점이 없다. 실용적 관점에서 2:4 희소성은 반드시 양자화와 결합하여 사용해야 한다.

마지막으로, `cutlass_sparse_scaled_mm_supported()` 함수가 런타임에 GPU 호환성을 자동 검증하므로, 비호환 GPU에서는 graceful fallback이 발생한다. 그러나 이 경우 sparse 가속 없이 dense 연산으로 전환되므로, 배포 전 대상 GPU의 SM 버전을 반드시 확인해야 한다.
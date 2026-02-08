# Issue Report: vLLM 0.14.1 EngineCore Segfault on EXAONE-4.0-1.2B (L4)

## Summary

`EXAONE-4.0-1.2B` 실행 시 vLLM EngineCore 초기화 단계에서 Segmentation Fault가 발생합니다.  
크래시는 `torch.compile + TorchInductor` 경로에서 재현되며, `CUDAGraph` 자체 문제는 아닌 것으로 확인되었습니다.

## Reproduction

Command:

```bash
python script.py --model-path EXAONE-4.0-1.2B --quick --skip-eval
```

Observed in logs:

- `Start compiling function ... exaone4.py:338`
- `Not enough SMs to use max_autotune_gemm mode`
- `!!!!!!! Segfault encountered !!!!!!!`
- `RuntimeError: Engine core initialization failed`

## Test Matrix (A/B)

| Case | Setting | Result | Interpretation |
|------|---------|--------|----------------|
| 1 | `enforce_eager=True` | Success | compile/inductor/cudagraph 전체 비활성 시 정상 |
| 2 | `CompilationConfig(mode=CompilationMode.NONE)` | Success | torch.compile 비활성 시 정상 |
| 3 | `CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE)` | **Fail (Segfault)** | cudagraph만 꺼도 compile+inductor 경로에서 크래시 |
| 4 | `CompilationConfig(backend="eager")` | Success | Inductor 비활성 시 정상 |

### Decision

문제 범위는 **CUDAGraph가 아니라 TorchInductor 경로**입니다.

## Environment

- GPU: NVIDIA L4 (22.4GiB, SM 58)
- vLLM: 0.14.1
- Python: 3.11
- Triton: 3.5.1
- torch: 2.9.0+cu128 / 2.9.1 모두에서 재현됨
- Runtime logs show mixed CUDA signals (example: `VLLM_MAIN_CUDA_VERSION=12.9`)

## Why This Is Likely Not CUDAGraph

- `cudagraph_mode=NONE` 설정에서도 동일 크래시 발생.
- 크래시 직전 로그가 Inductor 컴파일 단계(`compilation/backends.py`, `vllm_inductor_pass.py`)에 집중됨.
- `mode=NONE` 또는 `backend=eager`에서는 안정 동작.

## Additional Notes

- `--quick`는 값 생략 시 `100`개 샘플을 처리합니다.
- 이전 tlparse 산출물(`tl_out/-_-_-_-/aotautograd_cache_miss_5.json`)은 과거 런(`combo_kernels=True`)과 연결되어 있어, 최신 런과 1:1 대응 여부 확인이 필요합니다.

## Temporary Workarounds

1. `CompilationConfig(backend="eager")` 사용
2. `CompilationConfig(mode=CompilationMode.NONE)` 사용
3. `enforce_eager=True` 사용

성능 손해가 허용된다면 1 또는 2가 안정적입니다.

## Open Questions

1. 평가 서버에서 동일 모델이 통과되는 이유가 드라이버/캐시 차이인지
2. `torch 2.9.0+cu128` 고정 환경에서도 왜 특정 런타임 조합에서만 Inductor segfault가 나는지

## Artifacts

- Torch trace / tlparse:
  - `tmp/torch_trace/`
  - `tl_out/`
- Debug:
  - `tmp/debug/core_bt.txt`
  - `tmp/debug/nvidia_smi_q.txt`

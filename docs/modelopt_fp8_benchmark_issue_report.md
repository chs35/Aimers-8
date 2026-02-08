# 이슈 보고서: ModelOpt FP8 + KV Cache FP8 벤치마크 실패

## 1) 요약

- 발생 시각(로그 기준): 2026-02-08 23:12:30
- 증상: `script.py --model-path EXAONE-4.0-1.2B-ModelOpt-FP8 --quick --skip-eval` 실행 시 vLLM 엔진 초기화 실패
- 직접 원인: `KeyError: 'layers.0.self_attn.qkv_proj.k_scale'`
- 결론: OOM/컴파일(segfault) 문제가 아니라, **ModelOpt FP8 체크포인트와 vLLM EXAONE4 로더 간 스케일 파라미터 매핑 불일치**로 판단됨

## 2) 재현 절차

### 2.1 양자화

```bash
python experiments/quantize_modelopt_fp8.py \
  --model models/EXAONE-4.0-1.2B \
  --output models/EXAONE-4.0-1.2B-ModelOpt-FP8
```

양자화는 성공했으며 아래가 확인됨.
- `quant_algo: FP8`
- `kv_cache_quant_algo: FP8`
- 압축비 약 `1.72x`

근거: `experiments/modelopt.log:1`, `experiments/modelopt.log:47`, `experiments/modelopt.log:55`

### 2.2 벤치마크

```bash
python script.py --model-path EXAONE-4.0-1.2B-ModelOpt-FP8 --quick --skip-eval
```

벤치마크 시작 후 모델 로딩 단계에서 실패.

근거: `experiments/modelopt.log:56`, `experiments/modelopt.log:358`

## 3) 실패 시그니처

핵심 에러:

```text
KeyError: 'layers.0.self_attn.qkv_proj.k_scale'
```

스택 위치:
- `vllm/model_executor/models/exaone4.py`의 `load_weights` 경로에서 `params_dict[name]` 조회 시 실패
- 최종적으로 `RuntimeError: Engine core initialization failed`로 전파

근거:
- `experiments/modelopt.log:397`
- `experiments/modelopt.log:400`
- `experiments/modelopt.log:483`

## 4) 분석

1. vLLM은 체크포인트를 ModelOpt FP8로 정상 인식함.
- `quantization=modelopt`, `kv_cache_dtype=fp8_e4m3` 로 초기화됨
- 근거: `experiments/modelopt.log:330`

2. 실패 지점은 추론 실행 전, **가중치 로딩 시작 직후(샤드 0%)**임.
- 즉, 추론 로직/데이터셋/평가 단계 문제가 아님
- 근거: `experiments/modelopt.log:357`, `experiments/modelopt.log:399`

3. 기존 segfault 이슈와 성격이 다름.
- 이번 런은 `compilation_config.backend="eager"` 상태에서도 실패
- 따라서 TorchInductor segfault가 아니라 체크포인트 키 매핑 문제로 분리됨
- 근거: `src/runtime/model_loader.py:50`, `experiments/modelopt.log:330`

4. `model.safetensors` 키 구조 확인 결과, KV-cache scale이 `qkv_proj.*`가 아니라 `k_proj/v_proj`에 붙어 저장됨.
- 존재: `model.layers.0.self_attn.k_proj.k_scale`, `model.layers.0.self_attn.v_proj.v_scale`
- 부재: `model.layers.0.self_attn.qkv_proj.k_scale` (동일 패턴 0개)
- 이 때문에 vLLM EXAONE4 로더의 QKV 스태킹( `.k_proj` -> `.qkv_proj`) 로직이 KV-scale 텐서까지 오인 처리할 수 있는 상태임

## 5) 추정 원인

가장 가능성 높은 원인(우선순위 순):

1. **vLLM EXAONE4 ModelOpt 로더가 요구하는 scale 키(`...qkv_proj.k_scale`)와 실제 export 산출 키가 다름**
2. modelopt 버전(로그상 `0.41.0`)과 vLLM `0.14.1` 조합에서 EXAONE4용 ModelOpt 포맷 호환성이 완전하지 않음
3. 동일 아키텍처에서도 특정 레이어(QKV fused 구조)의 scale 키 생성 규칙이 다를 가능성

## 6) 영향 범위

- 영향 있음:
  - `quantize_modelopt_fp8.py`로 만든 EXAONE4 ModelOpt FP8 체크포인트의 vLLM 로딩
  - KV cache FP8 자동 감지 경로 포함
- 영향 없음(이번 로그 기준):
  - 데이터셋 로딩
  - 벤치마크 파이프라인 로직 자체
  - 평가(judge) 단계 (`--skip-eval`에서도 동일 실패)

## 7) 단기 우회책

1. ModelOpt FP8 대신 현재 통과 경로(예: compressed-tensors FP8) 사용
2. `--no-kv-cache`로 KV cache FP8 양자화를 끄고(ModelOpt weight FP8만) 제출
3. **모델 파일만으로 패치(권장)**: `model.safetensors`에서 KV-scale 키를 vLLM 로더가 오인하지 않도록 rename/remove 처리
   - rename 예: `*.self_attn.k_proj.k_scale -> *.self_attn.k_scale`, `*.self_attn.v_proj.v_scale -> *.self_attn.v_scale`
   - 리포 내 스크립트: `experiments/patch_modelopt_kv_scales_for_vllm.py`
4. vLLM/modelopt 버전 조합을 바꿔 호환 매트릭스 확인

## 8) 권장 후속 조치

1. **체크포인트 키 검증**
   - 산출 `model.safetensors`에서 `layers.0.self_attn.qkv_proj.k_scale` 실제 존재 여부 확인
2. **로더 기대 키 검증**
   - vLLM EXAONE4 ModelOpt 로더가 어떤 키 집합을 필수로 요구하는지 확인
3. **호환성 매트릭스 문서화**
   - `(vLLM 0.14.1, modelopt 0.41.0, EXAONE4)` 조합을 “불안정/실패”로 기록
4. **필요 시 임시 패치**
   - 누락 키 fallback 로직(예: 키 alias 매핑)을 로더에 추가해 재검증

## 9) 근거 파일

- 실행/오류 통합 로그: `experiments/modelopt.log`
- 양자화 스크립트: `experiments/quantize_modelopt_fp8.py`
- 모델 패치 스크립트: `experiments/patch_modelopt_kv_scales_for_vllm.py`
- 벤치마크 로더 설정: `src/runtime/model_loader.py`

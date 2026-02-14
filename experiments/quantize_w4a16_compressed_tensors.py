"""
EXAONE-4.0-1.2B Weight-only INT4 PTQ (compressed-tensors 포맷)

vLLM 0.14.1이 바로 읽을 수 있는 compressed-tensors 포맷으로 저장합니다.
대회 평가 서버에 compressed-tensors==0.13.0이 설치되어 있어 추가 패키지 없이 동작합니다.

두 가지 양자화 알고리즘 지원:
  1. GPTQ (기본): Hessian 기반 최적 라운딩, 최고 품질, 느림
  2. RTN:  Round-to-Nearest, 빠름, 약간 낮은 품질

출력 포맷: compressed-tensors (pack-quantized)
  - config.json에 quant_method="compressed-tensors" 메타데이터 포함
  - safetensors에 weight_packed(int32) + weight_scale + weight_shape 저장
  - vLLM이 자동으로 CompressedTensorsWNA16 → Marlin INT4 커널 사용
  - L4(SM89)에서 Marlin 커널 네이티브 지원

핵심 설정 (Marlin 호환 필수 조건):
  - num_bits: 4 (INT4)
  - symmetric: True (Marlin 필수)
  - group_size: 128 (Marlin 호환, 128의 배수)
  - strategy: group

선택 옵션:
  - actorder: "weight" → 정확도 개선, 추론 속도 영향 없음
  - KV cache FP8: kv_cache_scheme 추가 가능

캘리브레이션 데이터: LGAI-EXAONE/MANTA-1M (EXAONE 공식)

사전 설치: pip install llmcompressor

사용 예:
  # GPTQ (기본, 권장)
  python experiments/quantize_awq_int4.py \\
    --model-path LGAI-EXAONE/EXAONE-4.0-1.2B \\
    --output-path models/EXAONE-4.0-1.2B-W4A16-G128

  # RTN (빠른 실험용)
  python experiments/quantize_awq_int4.py \\
    --model-path LGAI-EXAONE/EXAONE-4.0-1.2B \\
    --output-path models/EXAONE-4.0-1.2B-W4A16-RTN \\
    --method rtn

  # GPTQ + actorder + KV cache FP8
  python experiments/quantize_awq_int4.py \\
    --model-path LGAI-EXAONE/EXAONE-4.0-1.2B \\
    --output-path models/EXAONE-4.0-1.2B-W4A16-G128-KV \\
    --actorder weight \\
    --kv-cache-fp8
"""

import os
import sys
import json
import logging
import argparse
import random
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
except ImportError:
    raise ImportError(
        "llmcompressor가 설치되지 않았습니다. "
        "다음 명령어로 설치하세요: pip install llmcompressor"
    )

SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR.parent / "logs"


# ──────────────────────────────────────────────────────────
# 로깅
# ──────────────────────────────────────────────────────────

def setup_logging(tag: str = "W4A16"):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{timestamp}_{tag}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)

    return log_path


# ──────────────────────────────────────────────────────────
# 캘리브레이션 데이터
# ──────────────────────────────────────────────────────────

def prepare_calibration_dataset(
    tokenizer,
    num_samples: int = 512,
    max_length: int = 2048,
    seed: int = 42,
):
    """
    MANTA-1M 데이터셋으로 캘리브레이션 데이터셋 준비 (llmcompressor 형식).

    LGAI-EXAONE/MANTA-1M: EXAONE 모델 공식 캘리브레이션 데이터셋
    - 한국어/영어 혼합, 다양한 도메인
    """
    random.seed(seed)

    print(f"Loading calibration data from LGAI-EXAONE/MANTA-1M...")
    dataset = load_dataset(
        "LGAI-EXAONE/MANTA-1M",
        split="train",
        trust_remote_code=True,
    )

    if len(dataset) > num_samples:
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = dataset.select(indices)

    print(f"Selected {len(dataset)} samples for calibration")

    texts = []
    for sample in dataset:
        conversations = sample.get("conversations", [])
        if conversations:
            text = "\n".join([turn.get("content", "") for turn in conversations])
        else:
            text = sample.get("text", sample.get("content", ""))
        if text and text.strip():
            texts.append(text)

    ds = Dataset.from_dict({"text": texts})

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    print(f"Total calibration samples: {len(ds)}")
    return ds


# ──────────────────────────────────────────────────────────
# 양자화 레시피
# ──────────────────────────────────────────────────────────

def build_gptq_recipe(
    group_size: int = 128,
    actorder: str = "weight",
    dampening_frac: float = 0.01,
    kv_cache_fp8: bool = False,
):
    """
    GPTQ 기반 W4A16 레시피 (compressed-tensors 포맷).

    Marlin 커널 호환을 위한 필수 조건:
    - symmetric=True
    - group_size=128 (128의 배수)
    - num_bits=4
    """
    weights_args = {
        "num_bits": 4,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": group_size,
    }

    if actorder:
        weights_args["actorder"] = actorder

    config_groups = {
        "group_0": {
            "targets": ["Linear"],
            "input_activations": None,
            "output_activations": None,
            "weights": weights_args,
        }
    }

    kwargs = dict(
        config_groups=config_groups,
        ignore=["lm_head"],
        dampening_frac=dampening_frac,
    )

    if kv_cache_fp8:
        kwargs["kv_cache_scheme"] = {
            "type": "float",
            "num_bits": 8,
            "strategy": "tensor",
            "dynamic": True,
        }

    return GPTQModifier(**kwargs)


def build_rtn_recipe(
    group_size: int = 128,
    kv_cache_fp8: bool = False,
):
    """
    RTN (Round-to-Nearest) 기반 W4A16 레시피.
    GPTQ보다 빠르지만 정확도가 약간 낮음.
    """
    config_groups = {
        "group_0": {
            "targets": ["Linear"],
            "input_activations": None,
            "output_activations": None,
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": group_size,
            },
        }
    }

    kwargs = dict(
        config_groups=config_groups,
        ignore=["lm_head"],
    )

    if kv_cache_fp8:
        kwargs["kv_cache_scheme"] = {
            "type": "float",
            "num_bits": 8,
            "strategy": "tensor",
            "dynamic": True,
        }

    return QuantizationModifier(**kwargs)


# ──────────────────────────────────────────────────────────
# 출력 검증
# ──────────────────────────────────────────────────────────

def verify_output(output_path: str):
    """
    양자화된 모델이 vLLM compressed-tensors 포맷에 맞는지 검증.

    체크 항목:
    1. config.json에 quant_method="compressed-tensors" 존재
    2. format="pack-quantized" (INT4 패킹)
    3. safetensors에 weight_packed, weight_scale, weight_shape 텐서 존재
    """
    output_path = Path(output_path)
    errors = []
    warnings = []

    # 1) config.json 검증
    config_path = output_path / "config.json"
    if not config_path.exists():
        errors.append("config.json이 없습니다")
        return errors, warnings

    with open(config_path) as f:
        config = json.load(f)

    qc = config.get("quantization_config", {})

    quant_method = qc.get("quant_method", "")
    if quant_method != "compressed-tensors":
        errors.append(
            f"quant_method이 'compressed-tensors'가 아닙니다: '{quant_method}'"
        )

    fmt = qc.get("format", "")
    if fmt != "pack-quantized":
        warnings.append(
            f"format이 'pack-quantized'가 아닙니다: '{fmt}' "
            "(INT4 패킹이 안 되어 있을 수 있음)"
        )

    # config_groups 검증
    config_groups = qc.get("config_groups", {})
    if not config_groups:
        errors.append("config_groups가 비어 있습니다")
    else:
        for name, group in config_groups.items():
            weights = group.get("weights", {})
            if weights.get("num_bits") != 4:
                errors.append(f"{name}: num_bits가 4가 아닙니다: {weights.get('num_bits')}")
            if weights.get("symmetric") is not True:
                errors.append(f"{name}: symmetric이 True가 아닙니다 (Marlin 필수)")
            if weights.get("group_size", 0) % 128 != 0:
                warnings.append(f"{name}: group_size가 128의 배수가 아닙니다: {weights.get('group_size')}")
            if group.get("input_activations") is not None:
                warnings.append(f"{name}: input_activations이 None이 아닙니다 (W4A16이 아닐 수 있음)")

    # 2) safetensors 검증
    try:
        from safetensors import safe_open

        st_files = sorted(output_path.glob("*.safetensors"))
        if not st_files:
            errors.append("safetensors 파일이 없습니다")
        else:
            has_packed = False
            has_scale = False
            has_shape = False

            for st_file in st_files:
                with safe_open(str(st_file), framework="pt", device="cpu") as f:
                    keys = f.keys()
                    for k in keys:
                        if k.endswith(".weight_packed"):
                            has_packed = True
                        elif k.endswith(".weight_scale"):
                            has_scale = True
                        elif k.endswith(".weight_shape"):
                            has_shape = True

            if not has_packed:
                errors.append("weight_packed 텐서가 없습니다 (save_compressed=True 확인)")
            if not has_scale:
                errors.append("weight_scale 텐서가 없습니다")
            if not has_shape:
                errors.append("weight_shape 텐서가 없습니다")

    except ImportError:
        warnings.append("safetensors 라이브러리가 없어 텐서 검증을 건너뜁니다")

    return errors, warnings


# ──────────────────────────────────────────────────────────
# 메인 양자화
# ──────────────────────────────────────────────────────────

def quantize_model(
    model_path: str,
    output_path: str,
    method: str = "gptq",
    num_samples: int = 512,
    max_length: int = 2048,
    seed: int = 42,
    group_size: int = 128,
    actorder: str = "weight",
    dampening_frac: float = 0.01,
    kv_cache_fp8: bool = False,
):
    log_path = setup_logging(f"W4A16-{method.upper()}")

    print("=" * 60)
    print(f"EXAONE-4.0-1.2B Weight-only INT4 PTQ ({method.upper()})")
    print("  Output format: compressed-tensors (pack-quantized)")
    print("  vLLM kernel:   Marlin INT4 (auto-detected)")
    print("=" * 60)
    print(f"  Model:      {model_path}")
    print(f"  Output:     {output_path}")
    print(f"  Method:     {method}")
    print(f"  Group size: {group_size}")
    print(f"  Symmetric:  True (Marlin required)")
    if method == "gptq":
        print(f"  Actorder:   {actorder}")
        print(f"  Dampening:  {dampening_frac}")
    print(f"  KV FP8:     {kv_cache_fp8}")
    print(f"  Calib:      {num_samples} samples, max_len={max_length}")
    print(f"  Log:        {log_path}")

    # ── 토크나이저 ──
    print(f"\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # ── 모델 ──
    print(f"[2/4] Loading model (float16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── 캘리브레이션 데이터 ──
    print(f"[3/4] Preparing calibration dataset...")
    calibration_ds = prepare_calibration_dataset(
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_length=max_length,
        seed=seed,
    )

    # ── 레시피 ──
    if method == "gptq":
        recipe = build_gptq_recipe(
            group_size=group_size,
            actorder=actorder,
            dampening_frac=dampening_frac,
            kv_cache_fp8=kv_cache_fp8,
        )
    else:
        recipe = build_rtn_recipe(
            group_size=group_size,
            kv_cache_fp8=kv_cache_fp8,
        )

    # ── 양자화 ──
    print(f"\n[4/4] Applying {method.upper()} quantization...")
    if method == "gptq":
        print("  (GPTQ는 Hessian 계산이 필요해 시간이 걸립니다)")
    oneshot(
        model=model,
        dataset=calibration_ds,
        recipe=recipe,
        max_seq_length=max_length,
        num_calibration_samples=num_samples,
    )

    # ── 저장 (compressed-tensors pack-quantized 포맷) ──
    print(f"\nSaving model (compressed-tensors pack-quantized)...")
    os.makedirs(output_path, exist_ok=True)

    # save_compressed=True가 핵심: INT4를 INT32에 패킹하여 저장
    model.save_pretrained(output_path, save_compressed=True)
    tokenizer.save_pretrained(output_path)

    # ── 출력 검증 ──
    print("\n" + "-" * 60)
    print("Verifying output format...")
    errors, warnings = verify_output(output_path)

    for w in warnings:
        print(f"  [WARN] {w}")
    for e in errors:
        print(f"  [ERROR] {e}")

    if not errors:
        print("  [OK] compressed-tensors 포맷 검증 통과")
    else:
        print("\n  !! 검증 실패 — vLLM에서 로드 시 문제가 발생할 수 있습니다")

    # ── config.json 내용 출력 ──
    config_path = Path(output_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        qc = config.get("quantization_config", {})
        print(f"\nconfig.json quantization_config:")
        print(json.dumps(qc, indent=2, ensure_ascii=False))

    # ── 파일 목록 ──
    print(f"\nGenerated files:")
    total_size = 0
    for f in sorted(Path(output_path).glob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  {f.name}: {size_mb:.2f} MB")
    print(f"  Total: {total_size:.2f} MB")

    # 원본 대비 압축률
    # EXAONE-4.0-1.2B: 1.28B params, BF16=2bytes → ~2560MB
    original_mb = 2560
    print(f"\n  Compression: ~{original_mb} MB → {total_size:.0f} MB "
          f"({total_size / original_mb * 100:.1f}%)")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    print(f"""
vLLM 로딩 확인:
  from vllm import LLM
  llm = LLM("{output_path}")
  # 로그에서 다음을 확인:
  #   "Using MarlinLinearKernel for CompressedTensorsWNA16"

대회 제출:
  1. {output_path}/ 내용을 model/ 폴더로 복사
  2. cd <제출디렉토리> && zip -r submit.zip model/
  3. submit.zip 제출 (max 10GB)

벤치마크:
  python benchmark.py gsm8k -m {output_path}
""")


def main():
    parser = argparse.ArgumentParser(
        description="EXAONE-4.0-1.2B Weight-only INT4 PTQ (compressed-tensors)"
    )
    parser.add_argument(
        "--model-path", type=str,
        default="LGAI-EXAONE/EXAONE-4.0-1.2B",
        help="원본 모델 경로 (HuggingFace 모델 ID 또는 로컬 경로)",
    )
    parser.add_argument(
        "--output-path", type=str,
        default="models/EXAONE-4.0-1.2B-W4A16-G128",
        help="양자화된 모델 저장 경로",
    )
    parser.add_argument(
        "--method", type=str,
        default="gptq",
        choices=["gptq", "rtn"],
        help="양자화 알고리즘: gptq(권장) 또는 rtn(빠름)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=512,
        help="캘리브레이션 샘플 수 (GPTQ 권장: 512~1024)",
    )
    parser.add_argument(
        "--max-length", type=int, default=2048,
        help="최대 시퀀스 길이",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드",
    )
    parser.add_argument(
        "--group-size", type=int, default=128,
        help="양자화 그룹 사이즈 (Marlin 호환: 128)",
    )
    parser.add_argument(
        "--actorder", type=str, default="weight",
        choices=["weight", "group", ""],
        help="GPTQ activation order (weight: 정확도↑ 속도 동일, group: 정확도↑↑ 속도↓, 빈값: off)",
    )
    parser.add_argument(
        "--dampening-frac", type=float, default=0.01,
        help="GPTQ 댐핑 비율 (기본 0.01)",
    )
    parser.add_argument(
        "--kv-cache-fp8", action="store_true",
        help="KV cache FP8 양자화 추가 (decode 속도 개선)",
    )

    args = parser.parse_args()

    quantize_model(
        model_path=args.model_path,
        output_path=args.output_path,
        method=args.method,
        num_samples=args.num_samples,
        max_length=args.max_length,
        seed=args.seed,
        group_size=args.group_size,
        actorder=args.actorder if args.actorder else None,
        dampening_frac=args.dampening_frac,
        kv_cache_fp8=args.kv_cache_fp8,
    )


if __name__ == "__main__":
    main()

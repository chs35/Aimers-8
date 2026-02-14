"""
EXAONE-4.0-1.2B KV Cache FP8 모델 준비 (modelopt 경로)

vLLM 0.14.1에서 kv_cache_dtype="auto"가 "fp8_e4m3"로 resolve되도록
config.json에 modelopt quantization_config를 설정.

핵심 원리:
  1. quant_method="modelopt" → vLLM이 modelopt 경로로 KV cache dtype 해석
  2. kv_cache_quant_algo="FP8" → "fp8_e4m3"로 매핑
  3. exclude_modules=["*"] → 모든 weight 양자화 스킵 (BF16 그대로)
  4. Attention 레이어는 exclude 체크 전에 처리 → KV cache FP8 적용

결과: Weight BF16 + KV cache FP8 → TPOT ~26% 개선 기대

사용법:
  python experiments/prepare_kv_fp8_modelopt.py
  python experiments/prepare_kv_fp8_modelopt.py --model-path models/EXAONE-4.0-1.2B
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def prepare_model(model_path: str, output_path: str):
    model_path = Path(model_path)
    output_path = Path(output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"모델 경로가 존재하지 않습니다: {model_path}")

    print("=" * 60)
    print("EXAONE-4.0-1.2B KV Cache FP8 (modelopt path)")
    print("=" * 60)
    print(f"Source: {model_path}")
    print(f"Output: {output_path}")

    # 모델 파일 복사
    if output_path.exists():
        print(f"기존 출력 디렉토리 삭제: {output_path}")
        shutil.rmtree(output_path)

    print("모델 파일 복사 중...")
    shutil.copytree(model_path, output_path)

    # config.json 수정
    config_path = output_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # modelopt quantization_config 추가
    # Format B (traditional nested) - vLLM 0.14.1 호환
    config["quantization_config"] = {
        "quant_method": "modelopt",
        "quantization": {
            "quant_algo": "FP8",
            "kv_cache_quant_algo": "FP8",
            "exclude_modules": ["*"],
        },
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\nconfig.json 수정 완료:")
    print('  quant_method: "modelopt"')
    print('  quant_algo: "FP8"')
    print('  kv_cache_quant_algo: "FP8"')
    print('  exclude_modules: ["*"]')

    # hf_quant_config.json도 생성 (ModelOpt <= 0.29.0 호환)
    hf_quant_config_path = output_path / "hf_quant_config.json"
    hf_quant_config = {
        "quant_method": "modelopt",
        "quantization": {
            "quant_algo": "FP8",
            "kv_cache_quant_algo": "FP8",
            "exclude_modules": ["*"],
        },
    }
    with open(hf_quant_config_path, "w") as f:
        json.dump(hf_quant_config, f, indent=2, ensure_ascii=False)

    print("hf_quant_config.json 생성 완료")

    # 파일 목록 출력
    print(f"\n생성된 파일:")
    total_size = 0
    for f in sorted(output_path.glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {f.name}: {size_mb:.2f} MB")
    print(f"  Total: {total_size:.2f} MB")

    print(f"\n벤치마크: python benchmark.py gsm8k -m {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="EXAONE-4.0-1.2B KV Cache FP8 준비 (modelopt path)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/EXAONE-4.0-1.2B",
        help="원본 모델 경로",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/EXAONE-4.0-1.2B-kv-fp8-modelopt",
        help="출력 모델 경로",
    )
    args = parser.parse_args()
    prepare_model(args.model_path, args.output_path)


if __name__ == "__main__":
    main()

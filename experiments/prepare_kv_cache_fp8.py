"""
EXAONE-4.0-1.2B KV Cache FP8 모델 준비 스크립트

Weight 양자화 없이 config.json에 kv_cache_scheme만 추가하여
vLLM이 FP8 KV cache를 자동으로 사용하도록 설정.

- Weight: 원본 BF16 그대로 (정확도 손실 없음)
- KV cache: FP8 E5M2 dynamic (decode 대역폭 절반 → TPOT 개선)
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

    config["quantization_config"] = {
        "kv_cache_scheme": {
            "type": "float",
            "num_bits": 8,
            "strategy": "tensor",
            "dynamic": True,
        },
        "quant_method": "compressed-tensors",
        "quantization_status": "frozen",
        "config_groups": {},
        "format": "dense",
        "ignore": [],
        "sparsity_config": {},
        "transform_config": {},
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("config.json에 kv_cache_scheme 추가 완료")

    # 파일 목록 출력
    print(f"\n생성된 파일:")
    total_size = 0
    for f in sorted(output_path.glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {f.name}: {size_mb:.2f} MB")
    print(f"  Total: {total_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="EXAONE-4.0-1.2B KV Cache FP8 모델 준비"
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
        default="models/EXAONE-4.0-1.2B-kv-fp8",
        help="출력 모델 경로",
    )
    args = parser.parse_args()
    prepare_model(args.model_path, args.output_path)


if __name__ == "__main__":
    main()

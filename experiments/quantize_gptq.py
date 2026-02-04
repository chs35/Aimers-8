"""
EXAONE-4.0-1.2B GPTQ 4bit 양자화 스크립트

계획서(plan.md)에 따른 설정:
- bits: 4
- group_size: 128
- sym: True (대칭 양자화)
- desc_act: False
- quant_method: "gptq"

캘리브레이션 데이터: LGAI-EXAONE/MANTA-1M (EXAONE 공식 데이터셋)

이 설정으로 vLLM이 자동으로 gptq_marlin 커널을 사용하도록 유도합니다.
"""

import os
import argparse
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# GPTQModel 임포트 (gptqmodel 패키지)
try:
    from gptqmodel import GPTQModel, QuantizeConfig
except ImportError:
    raise ImportError(
        "GPTQModel이 설치되지 않았습니다. "
        "다음 명령어로 설치하세요: pip install gptqmodel --no-build-isolation"
    )


def prepare_calibration_data(
    tokenizer,
    num_samples: int = 1024,
    max_length: int = 2048,
    seed: int = 42,
):
    """
    MANTA-1M 데이터셋으로 캘리브레이션 데이터 준비

    LGAI-EXAONE/MANTA-1M: EXAONE 모델 공식 캘리브레이션 데이터셋
    - 한국어/영어 혼합
    - 다양한 도메인 포함
    """
    random.seed(seed)

    print(f"Loading calibration data from LGAI-EXAONE/MANTA-1M...")
    dataset = load_dataset(
        "LGAI-EXAONE/MANTA-1M",
        split="train",
        trust_remote_code=True,
    )

    # 샘플 선택
    if len(dataset) > num_samples:
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = dataset.select(indices)

    print(f"Selected {len(dataset)} samples for calibration")

    # 토크나이즈
    print("Tokenizing calibration data...")
    tokenized_data = []
    for sample in dataset:
        text = sample.get("text", sample.get("content", ""))
        if not text:
            continue

        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized_data.append(tokens)

    print(f"Total calibration samples: {len(tokenized_data)}")
    return tokenized_data


def quantize_model(
    model_path: str,
    output_path: str,
    num_samples: int = 1024,
    max_length: int = 2048,
    seed: int = 42,
):
    """
    GPTQ 4bit 양자화 수행

    Args:
        model_path: 원본 모델 경로 (HuggingFace 모델 ID 또는 로컬 경로)
        output_path: 양자화된 모델 저장 경로
        num_samples: 캘리브레이션 샘플 수
        max_length: 최대 시퀀스 길이
        seed: 랜덤 시드
    """
    print("=" * 60)
    print("EXAONE-4.0-1.2B GPTQ 4bit Quantization")
    print("=" * 60)

    # 양자화 설정 (계획서 기반)
    quantize_config = QuantizeConfig(
        bits=4,                    # 4bit 양자화
        group_size=128,            # Marlin 호환 그룹 사이즈
        desc_act=False,            # Marlin 호환성을 위해 False
        sym=True,                  # 대칭 양자화 (Marlin 필수)
        true_sequential=True,      # 품질 향상을 위한 순차적 양자화
        damp_percent=0.1,          # 양자화 안정성을 위한 댐핑
    )

    print("\nQuantization Config:")
    print(f"  bits: {quantize_config.bits}")
    print(f"  group_size: {quantize_config.group_size}")
    print(f"  desc_act: {quantize_config.desc_act}")
    print(f"  sym: {quantize_config.sym}")
    print(f"  true_sequential: {quantize_config.true_sequential}")
    print(f"  damp_percent: {quantize_config.damp_percent}")

    # 토크나이저 로드
    print(f"\nLoading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # 캘리브레이션 데이터 준비
    print("\nPreparing calibration data...")
    calibration_data = prepare_calibration_data(
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_length=max_length,
        seed=seed,
    )

    # 모델 로드 및 양자화
    print(f"\nLoading model from {model_path}...")
    model = GPTQModel.load(
        model_path,
        quantize_config=quantize_config,
        trust_remote_code=True,
    )

    print("\nStarting quantization (this may take a while)...")
    model.quantize(calibration_data)

    # 양자화된 모델 저장
    print(f"\nSaving quantized model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)

    model.save(output_path)
    tokenizer.save_pretrained(output_path)

    print("\n" + "=" * 60)
    print("Quantization completed!")
    print("=" * 60)
    print(f"\nOutput directory: {output_path}")
    print("\nExpected config.json quantization_config:")
    print("""
{
  "quantization_config": {
    "bits": 4,
    "group_size": 128,
    "desc_act": false,
    "sym": true,
    "quant_method": "gptq"
  }
}
""")

    # 생성된 파일 목록 출력
    print("Generated files:")
    for f in sorted(Path(output_path).glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="EXAONE-4.0-1.2B GPTQ 4bit Quantization"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="LGAI-EXAONE/EXAONE-4.0-1.2B",
        help="원본 모델 경로 (HuggingFace 모델 ID 또는 로컬 경로)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/EXAONE-4.0-1.2B-gptq-4bit-g128-sym",
        help="양자화된 모델 저장 경로",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1024,
        help="캘리브레이션 샘플 수 (권장: 512~2048)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="최대 시퀀스 길이",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드",
    )

    args = parser.parse_args()

    quantize_model(
        model_path=args.model_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

"""
EXAONE-4.0-1.2B GPTQ 4bit 양자화 스크립트

계획서(plan.md)에 따른 설정:
- bits: 4
- group_size: 128
- sym: True (대칭 양자화)
- desc_act: False
- quant_method: "gptq"

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
        "다음 명령어로 설치하세요: pip install gptqmodel"
    )


def prepare_calibration_data(
    tokenizer,
    num_samples: int = 1024,
    max_length: int = 2048,
    seed: int = 42,
):
    """
    캘리브레이션 데이터 준비

    계획서 권장사항:
    - 512~2k 샘플
    - 한국어/영어 혼합 + 평가 도메인(수학/추론/지시문 등) 비중 반영
    - 입력 길이도 짧은/긴 케이스 섞기
    """
    random.seed(seed)
    calibration_data = []

    # 1. 영어 데이터: c4 데이터셋 (일반 텍스트)
    print("Loading English calibration data (c4)...")
    try:
        c4_dataset = load_dataset(
            "allenai/c4",
            "en",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        c4_samples = []
        for i, sample in enumerate(c4_dataset):
            if i >= num_samples // 2:
                break
            text = sample["text"]
            if len(text) > 100:  # 너무 짧은 텍스트 제외
                c4_samples.append(text)
        calibration_data.extend(c4_samples)
        print(f"  Loaded {len(c4_samples)} English samples")
    except Exception as e:
        print(f"  Warning: Failed to load c4 dataset: {e}")

    # 2. 한국어 데이터: 한국어 위키 또는 대화 데이터
    print("Loading Korean calibration data...")
    try:
        # 한국어 위키 데이터
        ko_dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.ko",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        ko_samples = []
        for i, sample in enumerate(ko_dataset):
            if i >= num_samples // 4:
                break
            text = sample["text"]
            if len(text) > 100:
                ko_samples.append(text)
        calibration_data.extend(ko_samples)
        print(f"  Loaded {len(ko_samples)} Korean samples")
    except Exception as e:
        print(f"  Warning: Failed to load Korean dataset: {e}")

    # 3. 수학/추론 데이터: GSM8K
    print("Loading math calibration data (GSM8K)...")
    try:
        gsm8k = load_dataset("openai/gsm8k", "main", split="train")
        gsm8k_samples = []
        indices = random.sample(range(len(gsm8k)), min(num_samples // 4, len(gsm8k)))
        for idx in indices:
            sample = gsm8k[idx]
            # 질문과 답변을 함께 사용
            text = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
            gsm8k_samples.append(text)
        calibration_data.extend(gsm8k_samples)
        print(f"  Loaded {len(gsm8k_samples)} math samples")
    except Exception as e:
        print(f"  Warning: Failed to load GSM8K dataset: {e}")

    # 데이터 셔플
    random.shuffle(calibration_data)

    # 최종 샘플 수 조정
    calibration_data = calibration_data[:num_samples]
    print(f"Total calibration samples: {len(calibration_data)}")

    # 토크나이즈
    print("Tokenizing calibration data...")
    tokenized_data = []
    for text in calibration_data:
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized_data.append(tokens)

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

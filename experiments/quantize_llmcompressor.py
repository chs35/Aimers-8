"""
EXAONE-4.0-1.2B GPTQ 4bit 양자화 스크립트 (LLM Compressor 버전)

LLM Compressor는 vLLM 팀에서 관리하는 공식 양자화 도구입니다.
vLLM과의 호환성이 보장되어 더 안정적입니다.

계획서(plan.md)에 따른 설정:
- bits: 4
- group_size: 128
- sym: True (대칭 양자화)
- desc_act: False (LLM Compressor에서는 기본값)
- quant_method: "gptq"

캘리브레이션 데이터: LGAI-EXAONE/MANTA-1M (EXAONE 공식 데이터셋)
"""

import os
import argparse
import random
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# LLM Compressor 임포트
try:
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
except ImportError:
    raise ImportError(
        "llmcompressor가 설치되지 않았습니다. "
        "다음 명령어로 설치하세요: pip install llmcompressor"
    )


def prepare_calibration_dataset(
    tokenizer,
    num_samples: int = 1024,
    max_length: int = 2048,
    seed: int = 42,
):
    """
    MANTA-1M 데이터셋으로 캘리브레이션 데이터셋 준비 (LLM Compressor 형식)

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

    # 텍스트 추출 (MANTA-1M은 conversations 형식)
    texts = []
    for sample in dataset:
        # MANTA-1M 데이터셋: conversations 컬럼에서 텍스트 추출
        conversations = sample.get("conversations", [])
        if conversations:
            # 모든 대화 내용을 하나의 텍스트로 합침
            text = "\n".join([turn.get("content", "") for turn in conversations])
            if text:
                texts.append(text)
        else:
            # 다른 형식의 데이터셋 대비
            text = sample.get("text", sample.get("content", ""))
            if text:
                texts.append(text)

    # HuggingFace Dataset 형식으로 변환
    ds = Dataset.from_dict({"text": texts})

    # 토크나이즈
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


def quantize_model(
    model_path: str,
    output_path: str,
    num_samples: int = 1024,
    max_length: int = 2048,
    seed: int = 42,
    use_custom_config: bool = False,
):
    """
    LLM Compressor를 사용한 GPTQ 4bit 양자화

    Args:
        model_path: 원본 모델 경로
        output_path: 양자화된 모델 저장 경로
        num_samples: 캘리브레이션 샘플 수
        max_length: 최대 시퀀스 길이
        seed: 랜덤 시드
        use_custom_config: 커스텀 config_groups 사용 여부
    """
    print("=" * 60)
    print("EXAONE-4.0-1.2B GPTQ 4bit Quantization (LLM Compressor)")
    print("=" * 60)

    # 토크나이저 로드
    print(f"\nLoading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # 모델 로드
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 캘리브레이션 데이터셋 준비
    print("\nPreparing calibration dataset...")
    calibration_ds = prepare_calibration_dataset(
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_length=max_length,
        seed=seed,
    )

    # 양자화 레시피 설정
    if use_custom_config:
        # 명시적인 config_groups 설정 (계획서 기반)
        print("\nUsing custom quantization config...")
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
                    "group_size": 128,
                },
            }
        }
        recipe = GPTQModifier(
            config_groups=config_groups,
            ignore=["lm_head"],
        )
        print("  num_bits: 4")
        print("  symmetric: True")
        print("  strategy: group")
        print("  group_size: 128")
    else:
        # W4A16 scheme 사용 (기본 4bit + group_size=128 + symmetric)
        print("\nUsing W4A16 scheme (4bit, group_size=128, symmetric)...")
        recipe = GPTQModifier(
            targets="Linear",
            scheme="W4A16",
            ignore=["lm_head"],
        )

    print(f"  ignore: ['lm_head']")

    # 양자화 적용
    print("\nApplying quantization (this may take a while)...")
    oneshot(
        model=model,
        dataset=calibration_ds,
        recipe=recipe,
        max_seq_length=max_length,
        num_calibration_samples=num_samples,
    )

    # 저장
    print(f"\nSaving quantized model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)

    model.save_pretrained(output_path, save_compressed=True)
    tokenizer.save_pretrained(output_path)

    print("\n" + "=" * 60)
    print("Quantization completed!")
    print("=" * 60)
    print(f"\nOutput directory: {output_path}")

    # 생성된 파일 목록 출력
    print("\nGenerated files:")
    for f in sorted(Path(output_path).glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="EXAONE-4.0-1.2B GPTQ 4bit Quantization (LLM Compressor)"
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
    parser.add_argument(
        "--custom-config",
        action="store_true",
        help="커스텀 config_groups 사용 (명시적 설정)",
    )

    args = parser.parse_args()

    quantize_model(
        model_path=args.model_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        max_length=args.max_length,
        seed=args.seed,
        use_custom_config=args.custom_config,
    )


if __name__ == "__main__":
    main()

"""
EXAONE-4.0-1.2B Weight-only INT4 PTQ (AWQ 방식)

AWQ (Activation-aware Weight Quantization):
- Weight-only INT4 양자화 (W4A16): 가중치만 INT4로 양자화, 활성값은 FP16 유지
- 활성값 분포 기반으로 중요한 weight 채널을 보호하여 양자화 오차 최소화
- GPTQ 대비 빠른 양자화 속도 (~3-5x)
- vLLM 0.14.1에서 AWQ-Marlin 커널로 빠른 추론 지원

설정:
- w_bit: 4 (INT4)
- q_group_size: 128 (Marlin 호환)
- zero_point: True (비대칭 양자화 - AWQ 기본값)
- version: "GEMM" (vLLM AWQ-Marlin 호환)

캘리브레이션 데이터: LGAI-EXAONE/MANTA-1M (EXAONE 공식 데이터셋)

사전 설치 필요:
  pip install autoawq

대회 평가 서버 호환:
- vLLM 0.14.1: AWQ 모델 네이티브 지원
- config.json의 quantization_config.quant_method = "awq"로 자동 인식
- Marlin 커널 사용 시 group_size=128, zero_point=True 필수

사용 예:
  python experiments/quantize_awq_int4.py \\
    --model-path LGAI-EXAONE/EXAONE-4.0-1.2B \\
    --output-path models/EXAONE-4.0-1.2B-awq-int4 \\
    --num-samples 512 \\
    --max-length 2048
"""

import os
import argparse
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

try:
    from awq import AutoAWQForCausalLM
except ImportError:
    raise ImportError(
        "AutoAWQ가 설치되지 않았습니다. "
        "다음 명령어로 설치하세요: pip install autoawq"
    )


def prepare_calibration_data(
    tokenizer,
    num_samples: int = 512,
    max_length: int = 2048,
    seed: int = 42,
):
    """
    MANTA-1M 데이터셋으로 캘리브레이션 텍스트 준비

    LGAI-EXAONE/MANTA-1M: EXAONE 모델 공식 캘리브레이션 데이터셋
    - 한국어/영어 혼합
    - 다양한 도메인 포함

    AutoAWQ는 List[str] 형태의 캘리브레이션 데이터를 받음
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

    # 텍스트 추출
    texts = []
    for sample in dataset:
        conversations = sample.get("conversations", [])
        if conversations:
            text = "\n".join([turn.get("content", "") for turn in conversations])
        else:
            text = sample.get("text", sample.get("content", ""))

        if not text or not text.strip():
            continue

        # AutoAWQ는 충분한 길이의 텍스트가 필요
        # 너무 짧은 텍스트는 제외
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= 64:
            texts.append(text)

    print(f"Total calibration texts (len >= 64 tokens): {len(texts)}")
    return texts


def quantize_model(
    model_path: str,
    output_path: str,
    num_samples: int = 512,
    max_length: int = 2048,
    seed: int = 42,
    w_bit: int = 4,
    q_group_size: int = 128,
    zero_point: bool = True,
    version: str = "GEMM",
):
    """
    AWQ Weight-only INT4 양자화 수행

    Args:
        model_path: 원본 모델 경로 (HuggingFace 모델 ID 또는 로컬 경로)
        output_path: 양자화된 모델 저장 경로
        num_samples: 캘리브레이션 샘플 수
        max_length: 최대 시퀀스 길이
        seed: 랜덤 시드
        w_bit: 양자화 비트 수 (기본 4)
        q_group_size: 양자화 그룹 사이즈 (기본 128, Marlin 호환)
        zero_point: 비대칭 양자화 여부 (기본 True)
        version: AWQ 버전 ("GEMM" 또는 "GEMV", vLLM은 GEMM 사용)
    """
    print("=" * 60)
    print("EXAONE-4.0-1.2B Weight-only INT4 PTQ (AWQ)")
    print("=" * 60)

    # 양자화 설정
    quant_config = {
        "zero_point": zero_point,
        "q_group_size": q_group_size,
        "w_bit": w_bit,
        "version": version,
    }

    print("\nQuantization Config (AWQ):")
    print(f"  w_bit: {w_bit}")
    print(f"  q_group_size: {q_group_size}")
    print(f"  zero_point: {zero_point}")
    print(f"  version: {version}")

    # 토크나이저 로드
    print(f"\nLoading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # 캘리브레이션 데이터 준비
    print("\nPreparing calibration data...")
    calib_texts = prepare_calibration_data(
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_length=max_length,
        seed=seed,
    )

    if len(calib_texts) == 0:
        raise ValueError(
            "캘리브레이션 텍스트가 없습니다. "
            "MANTA-1M 데이터셋 접근 또는 텍스트 추출을 확인하세요."
        )

    # 모델 로드 (AutoAWQ용)
    print(f"\nLoading model from {model_path} (AutoAWQ)...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        safetensors=True,
    )

    # AWQ 양자화 수행
    print(f"\nStarting AWQ quantization with {len(calib_texts)} calibration samples...")
    print("(This may take 10-30 minutes depending on GPU...)")
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_texts,
        max_calib_samples=num_samples,
        max_calib_seq_len=max_length,
    )

    # 양자화된 모델 저장
    print(f"\nSaving quantized model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)

    model.save_quantized(output_path, safetensors=True)
    tokenizer.save_pretrained(output_path)

    print("\n" + "=" * 60)
    print("Quantization completed!")
    print("=" * 60)

    # 예상 config 출력
    print("\nExpected config.json quantization_config:")
    print(f"""
{{
  "quantization_config": {{
    "quant_method": "awq",
    "bits": {w_bit},
    "group_size": {q_group_size},
    "zero_point": {str(zero_point).lower()},
    "version": "{version}"
  }}
}}
""")

    # 생성된 파일 목록 출력
    print("Generated files:")
    total_size = 0
    for f in sorted(Path(output_path).glob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  {f.name}: {size_mb:.2f} MB")
    print(f"\nTotal size: {total_size:.2f} MB")

    # 원본 대비 압축률 추정
    original_size_mb = 1280 * 2 / 1024  # ~2.5GB (1.28B params * 2 bytes BF16)
    print(f"Estimated compression: {original_size_mb:.0f} MB -> {total_size:.0f} MB "
          f"({total_size / original_size_mb * 100:.1f}%)")

    print("\n제출 방법:")
    print(f"  1. {output_path} 폴더의 내용을 submit.zip/model/ 에 복사")
    print("  2. zip -r submit.zip model/")
    print("  3. 대회 페이지에서 submit.zip 제출")


def main():
    parser = argparse.ArgumentParser(
        description="EXAONE-4.0-1.2B Weight-only INT4 PTQ (AWQ)"
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
        default="models/EXAONE-4.0-1.2B-awq-int4",
        help="양자화된 모델 저장 경로",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="캘리브레이션 샘플 수 (권장: 256~1024)",
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
        "--w-bit",
        type=int,
        default=4,
        choices=[4],
        help="양자화 비트 수 (INT4 고정)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="양자화 그룹 사이즈 (Marlin 호환: 128)",
    )
    parser.add_argument(
        "--no-zero-point",
        action="store_true",
        help="대칭 양자화 사용 (기본: 비대칭/zero_point=True)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="GEMM",
        choices=["GEMM", "GEMV"],
        help="AWQ 커널 버전 (GEMM: batch 추론용, GEMV: single-batch용)",
    )

    args = parser.parse_args()

    quantize_model(
        model_path=args.model_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        max_length=args.max_length,
        seed=args.seed,
        w_bit=args.w_bit,
        q_group_size=args.group_size,
        zero_point=not args.no_zero_point,
        version=args.version,
    )


if __name__ == "__main__":
    main()

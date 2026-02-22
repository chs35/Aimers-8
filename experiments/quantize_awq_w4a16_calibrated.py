"""
AWQ W4A16 양자화 스크립트 - 커스텀 캘리브레이션 데이터 사용

필요 패키지:
  pip install llmcompressor transformers datasets torch

설명:
  AWQ (Activation-Weighted Quantization) 알고리즘으로 W4A16 양자화
  - 활성화 기반으로 중요 가중치 채널 1%를 보호하여 양자화 오차 최소화
  - 가중치: 4-bit integer, 비대칭(asymmetric) 양자화
  - 활성화: 16-bit (FP16) 유지
  - 로컬 캘리브레이션 데이터셋 사용 (chat template 적용)
"""
import argparse
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier


def prepare_calibration_dataset(tokenizer, data_dir="data/calibration", max_length=2048):
    """저장된 캘리브레이션 데이터 로드 → chat template 적용 → 토크나이즈"""
    ds = load_from_disk(str(Path(data_dir) / "dataset"))

    def apply_chat_template_and_tokenize(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return tokenizer(
            text,
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(apply_chat_template_and_tokenize, remove_columns=ds.column_names)
    print(f"Loaded {len(ds)} calibration samples from {data_dir}")
    return ds


def quantize_awq_w4a16(
    model_path: str,
    output_path: str,
    calibration_data: str = "data/calibration",
    max_length: int = 2048,
):
    """AWQ W4A16 양자화 실행"""

    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Calibration data: {calibration_data}")
    print(f"Max length: {max_length}")
    print(f"Algorithm: AWQ W4A16 (Asymmetric)")
    print("=" * 60)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading calibration data...")
    ds = prepare_calibration_dataset(
        tokenizer,
        data_dir=calibration_data,
        max_length=max_length,
    )

    # AWQ W4A16 레시피 (비대칭 양자화)
    recipe = AWQModifier(
        targets=["Linear"],
        scheme="W4A16_ASYM",
        ignore=["lm_head"],
    )

    print("Running AWQ W4A16 quantization...")

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_length,
        num_calibration_samples=len(ds),
    )

    model.save_pretrained(output_path, save_compressed=True)
    tokenizer.save_pretrained(output_path)

    print("=" * 60)
    print(f"Quantization complete: {output_path}")

    original_size = sum(f.stat().st_size for f in Path(model_path).glob("*.safetensors"))
    quantized_size = sum(f.stat().st_size for f in Path(output_path).glob("*.safetensors"))

    if original_size > 0 and quantized_size > 0:
        print(f"Original size: {original_size / 1e9:.2f} GB")
        print(f"Compressed size: {quantized_size / 1e9:.2f} GB")
        print(f"Compression ratio: {original_size / quantized_size:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="AWQ W4A16 quantization with custom calibration data"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="models/EXAONE-4.0-1.2B",
        help="Input model path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path (default: {model}-AWQ-W4A16-calibrated)"
    )
    parser.add_argument(
        "--calibration-data", "-c",
        type=str,
        default="data/calibration",
        help="Calibration dataset path (default: data/calibration)"
    )
    parser.add_argument(
        "--max-length", "-l",
        type=int,
        default=2048,
        help="Max sequence length for calibration (default: 2048)"
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.model}-AWQ-W4A16-calibrated"

    quantize_awq_w4a16(
        model_path=args.model,
        output_path=args.output,
        calibration_data=args.calibration_data,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()

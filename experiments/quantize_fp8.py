"""
FP8 (E4M3) 양자화 스크립트 - L4 GPU용
llmcompressor + compressed-tensors 사용 (대회 환경 호환)
Static Scaling으로 Activation 스케일링 팩터 고정
"""
import argparse
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot


# FP8 양자화 레시피 (Static Scaling)
FP8_RECIPE = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
"""


def prepare_calibration_dataset(tokenizer, data_dir="data/calibration", max_length=16384):
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


def quantize_fp8(
    model_path: str,
    output_path: str,
    calibration_data: str = "data/calibration",
    max_length: int = 2048,
):
    """FP8 양자화 실행 (llmcompressor 사용)"""

    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Calibration data: {calibration_data}")
    print(f"Max length: {max_length}")
    print("=" * 60)

    # 토크나이저 로드
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 모델 로드
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    # 캘리브레이션 데이터 로드
    print("Loading calibration data...")
    ds = prepare_calibration_dataset(
        tokenizer,
        data_dir=calibration_data,
        max_length=max_length,
    )

    # 양자화 실행 (oneshot calibration)
    print("Running FP8 quantization with static calibration...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=FP8_RECIPE,
        save_compressed=True,
        output_dir=output_path,
        max_seq_length=max_length,
        num_calibration_samples=len(ds),
    )

    # 토크나이저 저장
    tokenizer.save_pretrained(output_path)

    print("=" * 60)
    print(f"FP8 quantization complete: {output_path}")

    # 모델 크기 비교
    original_size = sum(f.stat().st_size for f in Path(model_path).glob("*.safetensors"))
    quantized_size = sum(f.stat().st_size for f in Path(output_path).glob("*.safetensors"))

    if original_size > 0 and quantized_size > 0:
        print(f"Original size: {original_size / 1e9:.2f} GB")
        print(f"Quantized size: {quantized_size / 1e9:.2f} GB")
        print(f"Compression ratio: {original_size / quantized_size:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="FP8 (E4M3) Quantization with llmcompressor (compressed-tensors)"
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
        help="Output path (default: {model}-FP8)"
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
        help="Max sequence length for calibration (default: 16384)"
    )
    args = parser.parse_args()

    # 출력 경로 기본값
    if args.output is None:
        args.output = f"{args.model}-FP8-calibrated"

    quantize_fp8(
        model_path=args.model,
        output_path=args.output,
        calibration_data=args.calibration_data,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()

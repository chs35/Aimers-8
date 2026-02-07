"""
W8A8 INT8 양자화 스크립트 - SmoothQuant + GPTQ
llmcompressor + compressed-tensors 사용 (대회 환경 호환)
Manta 1M 데이터셋으로 캘리브레이션
"""
import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier


def prepare_calibration_dataset(tokenizer, num_samples=2048, max_length=4096):
    """Manta 1M 데이터셋에서 캘리브레이션 데이터 준비"""
    # Manta 1M 데이터셋 로드 (EXAONE-3.5-32B-Instruct로 생성된 합성 데이터)
    # https://huggingface.co/datasets/LGAI-EXAONE/MANTA-1M
    ds = load_dataset(
        "LGAI-EXAONE/MANTA-1M",
        split=f"train[:{num_samples}]"
    )
    ds = ds.shuffle(seed=42)

    def process_and_tokenize(example):
        # Manta 데이터셋 구조에 맞게 텍스트 추출
        text = example.get("text") or example.get("content") or str(example)
        return tokenizer(
            text,
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False
        )

    ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)
    print(f"Prepared {len(ds)} calibration samples from MANTA-1M dataset")
    return ds


def quantize_int8(
    model_path: str,
    output_path: str,
    num_samples: int = 2048,
    max_length: int = 4096,
    smoothing_strength: float = 0.8,
):
    """W8A8 INT8 양자화 실행 (SmoothQuant + GPTQ)"""

    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Calibration samples: {num_samples}")
    print(f"Max length: {max_length}")
    print(f"Smoothing strength: {smoothing_strength}")
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

    # 캘리브레이션 데이터 준비
    print("Preparing calibration data from MANTA-1M...")
    ds = prepare_calibration_dataset(
        tokenizer,
        num_samples=num_samples,
        max_length=max_length
    )

    # W8A8 INT8 레시피 (SmoothQuant + GPTQ)
    recipe = [
        SmoothQuantModifier(smoothing_strength=smoothing_strength),
        GPTQModifier(
            scheme="W8A8",
            targets="Linear",
            ignore=["lm_head"],
        ),
    ]

    # 양자화 실행
    print("Running W8A8 INT8 quantization (SmoothQuant + GPTQ)...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        save_compressed=True,
        output_dir=output_path,
        max_seq_length=max_length,
        num_calibration_samples=num_samples,
    )

    # 토크나이저 저장
    tokenizer.save_pretrained(output_path)

    print("=" * 60)
    print(f"W8A8 INT8 quantization complete: {output_path}")

    # 모델 크기 비교
    original_size = sum(f.stat().st_size for f in Path(model_path).glob("*.safetensors"))
    quantized_size = sum(f.stat().st_size for f in Path(output_path).glob("*.safetensors"))

    if original_size > 0 and quantized_size > 0:
        print(f"Original size: {original_size / 1e9:.2f} GB")
        print(f"Quantized size: {quantized_size / 1e9:.2f} GB")
        print(f"Compression ratio: {original_size / quantized_size:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="W8A8 INT8 Quantization with SmoothQuant + GPTQ"
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
        help="Output path (default: {model}-INT8)"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=2048,
        help="Number of calibration samples (default: 2048)"
    )
    parser.add_argument(
        "--max-length", "-l",
        type=int,
        default=4096,
        help="Max sequence length for calibration (default: 4096)"
    )
    parser.add_argument(
        "--smoothing-strength", "-s",
        type=float,
        default=0.8,
        help="SmoothQuant smoothing strength (default: 0.8)"
    )

    args = parser.parse_args()

    # 출력 경로 기본값
    if args.output is None:
        args.output = f"{args.model}-INT8"

    quantize_int8(
        model_path=args.model,
        output_path=args.output,
        num_samples=args.num_samples,
        max_length=args.max_length,
        smoothing_strength=args.smoothing_strength,
    )


if __name__ == "__main__":
    main()

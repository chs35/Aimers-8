"""
W4A16 INT4 양자화 스크립트 - GPTQ 기반
llmcompressor 사용 (대회 환경 호환)
vLLM에서 Marlin 커널로 자동 로드
Manta 1M 데이터셋으로 캘리브레이션
"""
import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier


def prepare_calibration_dataset(tokenizer, num_samples=512, max_length=2048):
    """Manta 1M 데이터셋에서 캘리브레이션 데이터 준비"""
    ds = load_dataset(
        "LGAI-EXAONE/MANTA-1M",
        split=f"train[:{num_samples}]"
    )
    ds = ds.shuffle(seed=42)

    def process_and_tokenize(example):
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


def quantize_w4a16(
    model_path: str,
    output_path: str,
    num_samples: int = 512,
    max_length: int = 2048,
    group_size: int = 128,
    symmetric: bool = True,
):
    """W4A16 INT4 양자화 실행 (GPTQ 기반)"""

    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Calibration samples: {num_samples}")
    print(f"Max length: {max_length}")
    print(f"Group size: {group_size}")
    print(f"Symmetric: {symmetric}")
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

    # W4A16 레시피 (GPTQ 기반 INT4 양자화)
    # Marlin 커널 호환을 위해 symmetric=True 권장
    scheme = "W4A16" if symmetric else "W4A16_ASYM"

    recipe = GPTQModifier(
        targets="Linear",
        scheme=scheme,
        ignore=["lm_head"],
    )

    print(f"Quantization scheme: {scheme}")
    print("Running W4A16 INT4 quantization (GPTQ)...")

    # 양자화 실행
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_length,
        num_calibration_samples=num_samples,
        save_compressed=True,
        output_dir=output_path,
    )

    # 토크나이저 저장
    tokenizer.save_pretrained(output_path)

    print("=" * 60)
    print(f"W4A16 INT4 quantization complete: {output_path}")

    # 모델 크기 비교
    original_size = sum(f.stat().st_size for f in Path(model_path).glob("*.safetensors"))
    quantized_size = sum(f.stat().st_size for f in Path(output_path).glob("*.safetensors"))

    if original_size > 0 and quantized_size > 0:
        print(f"Original size: {original_size / 1e9:.2f} GB")
        print(f"Quantized size: {quantized_size / 1e9:.2f} GB")
        print(f"Compression ratio: {original_size / quantized_size:.2f}x")

    print("\n[vLLM 사용법]")
    print("  vLLM이 자동으로 Marlin 커널 사용")
    print(f"  python benchmark.py gsm8k --model {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="W4A16 INT4 Quantization with GPTQ (Marlin kernel compatible)"
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
        help="Output path (default: {model}-W4A16)"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=512,
        help="Number of calibration samples (default: 512)"
    )
    parser.add_argument(
        "--max-length", "-l",
        type=int,
        default=2048,
        help="Max sequence length for calibration (default: 2048)"
    )
    parser.add_argument(
        "--group-size", "-g",
        type=int,
        default=128,
        help="Quantization group size (default: 128)"
    )
    parser.add_argument(
        "--asymmetric",
        action="store_true",
        help="Use asymmetric quantization (default: symmetric for Marlin)"
    )

    args = parser.parse_args()

    # 출력 경로 기본값
    if args.output is None:
        args.output = f"{args.model}-W4A16"

    quantize_w4a16(
        model_path=args.model,
        output_path=args.output,
        num_samples=args.num_samples,
        max_length=args.max_length,
        group_size=args.group_size,
        symmetric=not args.asymmetric,
    )


if __name__ == "__main__":
    main()

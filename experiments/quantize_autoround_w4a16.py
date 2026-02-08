"""
AutoRound W4A16 양자화 스크립트

필요 패키지:
  pip install llmcompressor transformers datasets torch auto-round

설명:
  AutoRound (Intel) 알고리즘으로 W4A16 양자화
  - 디코딩 레이어별 반복 최적화로 양자화 파라미터 튜닝
  - 가중치: 4-bit integer, group_size=128
  - 활성화: 16-bit (FP16) 유지
  - iters: 최적화 반복 횟수 (기본 200, 높을수록 정확도 향상)
  - 약 3.7x 가중치 크기 감소
  - 캘리브레이션 데이터 필요 (MANTA-1M 사용)
  - 모든 NVIDIA GPU 지원
"""
import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier


def prepare_calibration_dataset(tokenizer, num_samples=512, max_length=2048):
    """MANTA-1M 데이터셋에서 캘리브레이션 데이터 준비"""
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


def quantize_autoround_w4a16(
    model_path: str,
    output_path: str,
    num_samples: int = 512,
    max_length: int = 2048,
    iters: int = 200,
):
    """AutoRound W4A16 양자화 실행"""

    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Calibration samples: {num_samples}")
    print(f"Max length: {max_length}")
    print(f"AutoRound iters: {iters}")
    print(f"Algorithm: AutoRound W4A16")
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

    print("Preparing calibration data from MANTA-1M...")
    ds = prepare_calibration_dataset(
        tokenizer,
        num_samples=num_samples,
        max_length=max_length
    )

    # AutoRound W4A16 레시피
    recipe = AutoRoundModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
        iters=iters,
    )

    print("Running AutoRound W4A16 quantization...")

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_length,
        num_calibration_samples=num_samples,
        shuffle_calibration_samples=False,
    )

    SAVE_DIR = output_path
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)

    print("=" * 60)
    print(f"Quantization complete: {SAVE_DIR}")

    original_size = sum(f.stat().st_size for f in Path(model_path).glob("*.safetensors"))
    quantized_size = sum(f.stat().st_size for f in Path(SAVE_DIR).glob("*.safetensors"))

    if original_size > 0 and quantized_size > 0:
        print(f"Original size: {original_size / 1e9:.2f} GB")
        print(f"Compressed size: {quantized_size / 1e9:.2f} GB")
        print(f"Compression ratio: {original_size / quantized_size:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="AutoRound W4A16 quantization with llmcompressor"
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
        help="Output path (default: {model}-AutoRound-W4A16)"
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
        "--iters",
        type=int,
        default=200,
        help="AutoRound optimization iterations (default: 200)"
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.model}-AutoRound-W4A16"

    quantize_autoround_w4a16(
        model_path=args.model,
        output_path=args.output,
        num_samples=args.num_samples,
        max_length=args.max_length,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()

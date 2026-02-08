"""
AWQ W4A16 + 2:4 Sparsity 양자화 스크립트

필요 패키지:
  pip install llmcompressor transformers datasets torch

설명:
  AWQ W4A16 비대칭 양자화 + SparseGPT 2:4 구조적 프루닝 결합
  - 2:4 구조적 프루닝: 4개 중 2개 가중치를 0으로 (50% sparsity)
  - AWQ: 활성화 기반 중요 가중치 보호로 양자화 오차 최소화
  - 가중치: 4-bit integer, 비대칭(asymmetric) 양자화
  - 활성화: 16-bit (FP16) 유지
  - 캘리브레이션 데이터 필요 (MANTA-1M 사용)
  - SM80+ (Ampere 이상) GPU 지원, L4 (SM89) 호환
"""
import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier


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


def quantize_awq_w4a16_sparse24(
    model_path: str,
    output_path: str,
    num_samples: int = 512,
    max_length: int = 2048,
):
    """AWQ W4A16 + 2:4 Sparsity 양자화 실행"""

    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Calibration samples: {num_samples}")
    print(f"Max length: {max_length}")
    print(f"Algorithm: AWQ W4A16 + 2:4 Sparsity")
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

    # 2:4 Sparsity + AWQ W4A16 레시피
    recipe = [
        SparseGPTModifier(
            sparsity=0.5,
            mask_structure="2:4",
            targets=[r"re:model.layers.\d*$"],
        ),
        AWQModifier(
            targets=["Linear"],
            scheme="W4A16_ASYM",
            ignore=["lm_head"],
        ),
    ]

    print("Running AWQ W4A16 + 2:4 Sparsity compression...")

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_length,
        num_calibration_samples=num_samples,
    )

    SAVE_DIR = output_path
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)

    print("=" * 60)
    print(f"Compression complete: {SAVE_DIR}")

    original_size = sum(f.stat().st_size for f in Path(model_path).glob("*.safetensors"))
    quantized_size = sum(f.stat().st_size for f in Path(SAVE_DIR).glob("*.safetensors"))

    if original_size > 0 and quantized_size > 0:
        print(f"Original size: {original_size / 1e9:.2f} GB")
        print(f"Compressed size: {quantized_size / 1e9:.2f} GB")
        print(f"Compression ratio: {original_size / quantized_size:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="AWQ W4A16 + 2:4 Sparsity quantization with llmcompressor"
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
        help="Output path (default: {model}-AWQ-W4A16-Sparse24)"
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

    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.model}-AWQ-W4A16-Sparse24"

    quantize_awq_w4a16_sparse24(
        model_path=args.model,
        output_path=args.output,
        num_samples=args.num_samples,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()

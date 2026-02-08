"""
INT8 W8A8 + 2:4 Sparsity (Sparse24) 양자화 스크립트
llmcompressor의 SparseGPTModifier + QuantizationModifier 사용
- 2:4 구조적 프루닝: 4개 중 2개 가중치를 0으로 (50% sparsity)
- INT8 W8A8 양자화: 채널별 가중치 + 동적 per-token 활성화 양자화
- SM89 (L4) 이상 모든 NVIDIA GPU 지원
- 결합 시 ~4x 압축률
"""
import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
from llmcompressor.modifiers.quantization import QuantizationModifier


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


def quantize_int8_sparse24(
    model_path: str,
    output_path: str,
    num_samples: int = 512,
    max_length: int = 2048,
    int8_enabled: bool = True,
):
    """INT8 W8A8 + 2:4 Sparsity 양자화 실행"""

    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Calibration samples: {num_samples}")
    print(f"Max length: {max_length}")
    print(f"INT8 quantization: {int8_enabled}")
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

    # 레시피 구성: 2:4 Sparsity + INT8 W8A8 Quantization
    recipe = [
        SparseGPTModifier(
            sparsity=0.5,
            mask_structure="2:4",
            targets=[r"re:model.layers.\d*$"],
        )
    ]

    if int8_enabled:
        recipe.append(
            QuantizationModifier(
                targets=["Linear"],
                ignore=["lm_head"],
                scheme="W8A8",
            ),
        )

    mode = "INT8 W8A8 + 2:4 Sparsity" if int8_enabled else "2:4 Sparsity only"
    print(f"Running {mode} compression...")

    # 양자화 + 프루닝 실행
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
    print(f"Compression complete: {output_path}")

    # 모델 크기 비교
    original_size = sum(f.stat().st_size for f in Path(model_path).glob("*.safetensors"))
    quantized_size = sum(f.stat().st_size for f in Path(output_path).glob("*.safetensors"))

    if original_size > 0 and quantized_size > 0:
        print(f"Original size: {original_size / 1e9:.2f} GB")
        print(f"Compressed size: {quantized_size / 1e9:.2f} GB")
        print(f"Compression ratio: {original_size / quantized_size:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="INT8 W8A8 + 2:4 Sparsity (Sparse24) with llmcompressor"
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
        help="Output path (default: {model}-INT8-Sparse24)"
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
        "--no-int8",
        action="store_true",
        help="Disable INT8 quantization (2:4 sparsity only)"
    )

    args = parser.parse_args()

    # 출력 경로 기본값
    if args.output is None:
        suffix = "Sparse24" if args.no_int8 else "INT8-Sparse24"
        args.output = f"{args.model}-{suffix}"

    quantize_int8_sparse24(
        model_path=args.model,
        output_path=args.output,
        num_samples=args.num_samples,
        max_length=args.max_length,
        int8_enabled=not args.no_int8,
    )


if __name__ == "__main__":
    main()

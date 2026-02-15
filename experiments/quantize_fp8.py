"""
FP8 (E4M3) 양자화 스크립트 - L4 GPU용
llmcompressor + compressed-tensors 사용 (대회 환경 호환)
Static Scaling으로 Activation 스케일링 팩터 고정
Manta 1M 데이터셋으로 캘리브레이션
"""
import argparse
from pathlib import Path

from datasets import load_dataset, load_from_disk
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

# FP8 + KV Cache FP8 레시피
FP8_KV_CACHE_RECIPE = """
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
            kv_cache_scheme:
                num_bits: 8
                type: float
                strategy: tensor
                dynamic: false
                symmetric: true
"""


def prepare_calibration_dataset(tokenizer, num_samples=512, max_length=2048,
                                local_dataset_path=None):
    """캘리브레이션 데이터 준비

    Args:
        local_dataset_path: 로컬 Arrow 데이터셋 경로 (prepare_dataset.py의 fp8_calib).
                           지정 시 이 경로에서 로드, 미지정 시 MANTA-1M 사용.
    """
    if local_dataset_path:
        print(f"Loading calibration data from local dataset: {local_dataset_path}")
        ds = load_from_disk(local_dataset_path)
        # 로컬 데이터셋은 prepare_dataset.py에서 이미 샘플 수가 결정됨 → 전체 사용

        def process_local(example):
            text = tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=True,
                tokenize=False,
            )
            return tokenizer(
                text,
                padding=False,
                max_length=max_length,
                truncation=True,
                add_special_tokens=False,
            )

        ds = ds.map(process_local, remove_columns=ds.column_names)
        print(f"Prepared {len(ds)} calibration samples from local dataset")
        return ds

    # 기본: Manta 1M 데이터셋 로드
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


def quantize_fp8(
    model_path: str,
    output_path: str,
    num_samples: int = 512,
    max_length: int = 2048,
    kv_cache_fp8: bool = False,
    dataset_path: str = None,
):
    """FP8 양자화 실행 (llmcompressor 사용)"""

    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Calibration samples: {num_samples}")
    print(f"Max length: {max_length}")
    print(f"KV Cache FP8: {kv_cache_fp8}")
    if dataset_path:
        print(f"Dataset: {dataset_path}")
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
    print("Preparing calibration data...")
    ds = prepare_calibration_dataset(
        tokenizer,
        num_samples=num_samples,
        max_length=max_length,
        local_dataset_path=dataset_path,
    )

    # 레시피 선택
    recipe = FP8_KV_CACHE_RECIPE if kv_cache_fp8 else FP8_RECIPE

    # 양자화 실행 (oneshot calibration)
    actual_samples = len(ds)
    print(f"Running FP8 quantization with static calibration ({actual_samples} samples)...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        save_compressed=True,
        output_dir=output_path,
        max_seq_length=max_length,
        num_calibration_samples=actual_samples,
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
        "--kv-cache-fp8",
        action="store_true",
        help="Enable FP8 KV cache quantization"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Local calibration dataset path (from prepare_dataset.py fp8_calib)"
    )

    args = parser.parse_args()

    # 출력 경로 기본값
    if args.output is None:
        args.output = f"{args.model}-FP8"

    quantize_fp8(
        model_path=args.model,
        output_path=args.output,
        num_samples=args.num_samples,
        max_length=args.max_length,
        kv_cache_fp8=args.kv_cache_fp8,
        dataset_path=args.dataset,
    )


if __name__ == "__main__":
    main()

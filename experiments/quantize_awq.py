"""
AWQ INT4 양자화 스크립트 - Marlin 커널 지원
AutoAWQ 사용, vLLM에서 awq_marlin으로 로드
Manta 1M 데이터셋으로 캘리브레이션
"""
import argparse
from pathlib import Path

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset


def prepare_calibration_data(tokenizer, num_samples=512, max_length=2048):
    """Manta 1M 데이터셋에서 캘리브레이션 데이터 준비"""
    # Manta 1M 데이터셋 로드
    ds = load_dataset(
        "LGAI-EXAONE/MANTA-1M",
        split=f"train[:{num_samples}]"
    )
    ds = ds.shuffle(seed=42)

    calibration_data = []
    for example in ds:
        text = example.get("text") or example.get("content") or str(example)
        # 토큰화하여 max_length에 맞게 자르기
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        # 다시 텍스트로 디코딩
        text = tokenizer.decode(tokens)
        if len(text.strip()) > 0:
            calibration_data.append(text)

    print(f"Prepared {len(calibration_data)} calibration samples from MANTA-1M dataset")
    return calibration_data


def quantize_awq(
    model_path: str,
    output_path: str,
    num_samples: int = 512,
    max_length: int = 2048,
    group_size: int = 128,
    use_marlin: bool = True,
):
    """AWQ INT4 양자화 실행"""

    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Calibration samples: {num_samples}")
    print(f"Max length: {max_length}")
    print(f"Group size: {group_size}")
    print(f"Marlin format: {use_marlin}")
    print("=" * 60)

    # 토크나이저 로드
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 캘리브레이션 데이터 준비
    print("Preparing calibration data from MANTA-1M...")
    calibration_data = prepare_calibration_data(
        tokenizer,
        num_samples=num_samples,
        max_length=max_length
    )

    # 모델 로드
    print("Loading model for AWQ quantization...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        safetensors=True,
    )

    # AWQ 양자화 설정
    # Marlin 커널 사용 시 zero_point=False 필수
    quant_config = {
        "zero_point": not use_marlin,  # Marlin은 zero_point=False
        "q_group_size": group_size,
        "w_bit": 4,
        "version": "Marlin" if use_marlin else "GEMM",
    }

    print(f"Quantization config: {quant_config}")
    print("Running AWQ INT4 quantization...")

    # 양자화 실행
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calibration_data,
    )

    # 모델 저장
    print(f"Saving quantized model to {output_path}...")
    model.save_quantized(output_path, safetensors=True)
    tokenizer.save_pretrained(output_path)

    print("=" * 60)
    print(f"AWQ INT4 quantization complete: {output_path}")

    # 모델 크기 비교
    original_size = sum(f.stat().st_size for f in Path(model_path).glob("*.safetensors"))
    quantized_size = sum(f.stat().st_size for f in Path(output_path).glob("*.safetensors"))

    if original_size > 0 and quantized_size > 0:
        print(f"Original size: {original_size / 1e9:.2f} GB")
        print(f"Quantized size: {quantized_size / 1e9:.2f} GB")
        print(f"Compression ratio: {original_size / quantized_size:.2f}x")

    print("\n[vLLM 사용법]")
    print(f"  python -m vllm.entrypoints.openai.api_server \\")
    print(f"    --model {output_path} \\")
    print(f"    --quantization awq_marlin \\")
    print(f"    --trust-remote-code")


def main():
    parser = argparse.ArgumentParser(
        description="AWQ INT4 Quantization with Marlin kernel support"
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
        help="Output path (default: {model}-AWQ)"
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
        "--no-marlin",
        action="store_true",
        help="Disable Marlin format (use GEMM instead)"
    )

    args = parser.parse_args()

    # 출력 경로 기본값
    if args.output is None:
        args.output = f"{args.model}-AWQ"

    quantize_awq(
        model_path=args.model,
        output_path=args.output,
        num_samples=args.num_samples,
        max_length=args.max_length,
        group_size=args.group_size,
        use_marlin=not args.no_marlin,
    )


if __name__ == "__main__":
    main()

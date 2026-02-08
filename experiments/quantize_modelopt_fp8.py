"""
nvidia-modelopt FP8 + KV Cache FP8 양자화 스크립트

llm-compressor(compressed-tensors)와 달리, modelopt로 양자화하면
hf_quant_config.json이 생성되어 vLLM이 별도 플래그 없이
자동으로 KV cache FP8을 감지합니다.

설치:
    pip install "nvidia-modelopt[hf]"

사용 예:
    python experiments/quantize_modelopt_fp8.py \
        --model models/EXAONE-4.0-1.2B \
        --output models/EXAONE-4.0-1.2B-ModelOpt-FP8 \
        --num-samples 512
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint


def prepare_calibration_dataloader(tokenizer, num_samples=512, max_length=2048):
    """Manta 1M 데이터셋에서 캘리브레이션 DataLoader 준비"""
    ds = load_dataset(
        "LGAI-EXAONE/MANTA-1M",
        split=f"train[:{num_samples}]",
    )
    ds = ds.shuffle(seed=42)

    def process_and_tokenize(example):
        text = example.get("text") or example.get("content") or str(example)
        return tokenizer(
            text,
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)
    ds.set_format("torch")
    print(f"Prepared {len(ds)} calibration samples from MANTA-1M dataset")

    dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    return dataloader


def quantize_modelopt_fp8(
    model_path: str,
    output_path: str,
    num_samples: int = 512,
    max_length: int = 2048,
    kv_cache_fp8: bool = True,
):
    """nvidia-modelopt FP8 양자화 실행"""

    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Calibration samples: {num_samples}")
    print(f"Max length: {max_length}")
    print(f"KV Cache FP8: {kv_cache_fp8}")
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
    calib_dataloader = prepare_calibration_dataloader(
        tokenizer,
        num_samples=num_samples,
        max_length=max_length,
    )

    # 양자화 config 준비
    quant_cfg = mtq.FP8_DEFAULT_CFG.copy()

    if kv_cache_fp8:
        print("Merging KV cache FP8 config...")
        quant_cfg = mtq.utils.update_quant_cfg_with_kv_cache_quant(
            quant_cfg,
            mtq.FP8_KV_CFG["quant_cfg"],
        )

    # 캘리브레이션 forward loop 정의
    def forward_loop(model):
        for batch in calib_dataloader:
            model(input_ids=batch["input_ids"].to("cuda"))

    # 양자화 실행
    print("Running FP8 quantization with modelopt...")
    mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

    # HF 체크포인트로 내보내기
    print(f"Exporting HF checkpoint to {output_path}...")
    export_hf_checkpoint(model, export_dir=output_path)

    # 토크나이저 저장
    tokenizer.save_pretrained(output_path)

    print("=" * 60)
    print(f"ModelOpt FP8 quantization complete: {output_path}")

    # 출력 검증
    verify_output(model_path, output_path, kv_cache_fp8)


def verify_output(model_path: str, output_path: str, kv_cache_fp8: bool):
    """양자화 출력 검증"""
    output_dir = Path(output_path)

    # hf_quant_config.json 존재 여부 확인
    quant_config_path = output_dir / "hf_quant_config.json"
    if quant_config_path.exists():
        with open(quant_config_path) as f:
            quant_config = json.load(f)
        print(f"\nhf_quant_config.json: {json.dumps(quant_config, indent=2)}")

        # KV cache FP8 설정 확인
        if kv_cache_fp8:
            quant_info = quant_config.get("quantization", {})
            kv_algo = quant_info.get("kv_cache_quant_algo", "")
            if kv_algo == "FP8":
                print("[OK] kv_cache_quant_algo: FP8")
            else:
                print(f"[WARN] kv_cache_quant_algo: {kv_algo!r} (expected 'FP8')")
    else:
        print("[WARN] hf_quant_config.json not found in output directory")

    # 모델 크기 비교
    original_size = sum(
        f.stat().st_size for f in Path(model_path).glob("*.safetensors")
    )
    quantized_size = sum(
        f.stat().st_size for f in output_dir.glob("*.safetensors")
    )

    if original_size > 0 and quantized_size > 0:
        print(f"\nOriginal size: {original_size / 1e9:.2f} GB")
        print(f"Quantized size: {quantized_size / 1e9:.2f} GB")
        print(f"Compression ratio: {original_size / quantized_size:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="nvidia-modelopt FP8 + KV Cache FP8 Quantization"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="models/EXAONE-4.0-1.2B",
        help="Input model path",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path (default: {model}-ModelOpt-FP8)",
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=512,
        help="Number of calibration samples (default: 512)",
    )
    parser.add_argument(
        "--max-length", "-l",
        type=int,
        default=2048,
        help="Max sequence length for calibration (default: 2048)",
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable KV cache FP8 quantization",
    )

    args = parser.parse_args()

    # 출력 경로 기본값
    if args.output is None:
        args.output = f"{args.model}-ModelOpt-FP8"

    quantize_modelopt_fp8(
        model_path=args.model,
        output_path=args.output,
        num_samples=args.num_samples,
        max_length=args.max_length,
        kv_cache_fp8=not args.no_kv_cache,
    )


if __name__ == "__main__":
    main()

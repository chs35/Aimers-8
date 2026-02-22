"""
FP8 (E4M3) 양자화 스크립트 - modelopt 버전
NVIDIA TensorRT Model Optimizer (modelopt) 사용
vLLM 배포 시 quantization="modelopt" 으로 로드

Usage:
    python experiments/quantize_fp8_modelopt.py --model models/EXAONE-4.0-1.2B
"""
import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint


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
    ds.set_format("torch")
    print(f"Loaded {len(ds)} calibration samples from {data_dir}")
    return ds


def make_forward_loop(ds):
    """캘리브레이션용 forward loop 생성 (mtq.quantize에 전달)"""
    def forward_loop(model):
        device = next(model.parameters()).device
        for i in range(len(ds)):
            input_ids = ds[i]["input_ids"].unsqueeze(0).to(device)
            attention_mask = ds[i]["attention_mask"].unsqueeze(0).to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)
    return forward_loop


def quantize_fp8(
    model_path: str,
    output_path: str,
    calibration_data: str = "data/calibration",
    max_length: int = 16384,
):
    """FP8 양자화 실행 (modelopt 사용)"""

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

    # FP8 양자화 config (lm_head 제외)
    quant_cfg = mtq.FP8_DEFAULT_CFG.copy()
    quant_cfg["quant_cfg"]["*lm_head*"] = {"enable": False}

    # 양자화 실행
    print("Running FP8 quantization with modelopt...")
    forward_loop = make_forward_loop(ds)
    model = mtq.quantize(model, quant_cfg, forward_loop)
    mtq.print_quant_summary(model)

    # HF checkpoint으로 export (vLLM quantization="modelopt" 호환)
    print(f"Exporting quantized model to {output_path}...")
    export_hf_checkpoint(model, export_dir=output_path)
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
        description="FP8 (E4M3) Quantization with NVIDIA modelopt"
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
        help="Output path (default: {model}-FP8-modelopt)"
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
        default=16384,
        help="Max sequence length for calibration (default: 16384)"
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.model}-FP8-modelopt"

    quantize_fp8(
        model_path=args.model,
        output_path=args.output,
        calibration_data=args.calibration_data,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()

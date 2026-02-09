"""
EXAONE-4.0-1.2B FP8 W8A8 Static 양자화 스크립트

FP8 Static = Weight(FP8) + Activation(FP8), 캘리브레이션 기반 고정 스케일
- L4(SM89, Ada Lovelace)에서 FP8 텐서코어 네이티브 지원
- Static: 스케일이 캘리브레이션으로 고정 → decode 시 런타임 오버헤드 없음
- TPOT(Time Per Output Token) 개선에 최적

캘리브레이션 데이터: LGAI-EXAONE/MANTA-1M
"""

import os
import sys
import logging
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
except ImportError:
    raise ImportError(
        "llmcompressor가 설치되지 않았습니다. "
        "다음 명령어로 설치하세요: pip install llmcompressor"
    )

SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR / "logs"


def prepare_calibration_dataset(
    tokenizer,
    num_samples: int = 512,
    max_length: int = 2048,
    seed: int = 42,
):
    """
    MANTA-1M 데이터셋으로 캘리브레이션 데이터셋 준비

    LGAI-EXAONE/MANTA-1M: EXAONE 모델 공식 캘리브레이션 데이터셋
    - 한국어/영어 혼합
    - 다양한 도메인 포함
    """
    random.seed(seed)

    print(f"Loading calibration data from LGAI-EXAONE/MANTA-1M...")
    dataset = load_dataset(
        "LGAI-EXAONE/MANTA-1M",
        split="train",
        trust_remote_code=True,
    )

    if len(dataset) > num_samples:
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = dataset.select(indices)

    print(f"Selected {len(dataset)} samples for calibration")

    texts = []
    for sample in dataset:
        conversations = sample.get("conversations", [])
        if conversations:
            text = "\n".join([turn.get("content", "") for turn in conversations])
            if text:
                texts.append(text)
        else:
            text = sample.get("text", sample.get("content", ""))
            if text:
                texts.append(text)

    ds = Dataset.from_dict({"text": texts})

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    print(f"Total calibration samples: {len(ds)}")
    return ds


def setup_logging(log_path: str):
    """모든 디버그 로그를 파일 + 콘솔에 동시 출력"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # 파일 핸들러 — DEBUG 레벨 전부 기록
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root.addHandler(fh)

    # 콘솔 핸들러 — INFO 이상만 표시
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)

    # stdout/stderr도 파일로 tee
    class Tee:
        def __init__(self, stream, log_file):
            self.stream = stream
            self.log_file = log_file
        def write(self, msg):
            self.stream.write(msg)
            self.log_file.write(msg)
            self.log_file.flush()
        def flush(self):
            self.stream.flush()
            self.log_file.flush()
        def fileno(self):
            return self.stream.fileno()

    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    return log_file


def quantize_model(
    model_path: str,
    output_path: str,
    num_samples: int = 512,
    max_length: int = 2048,
    seed: int = 42,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{timestamp}_quantize_fp8_static.log"
    log_file = setup_logging(str(log_path))

    print("=" * 60)
    print("EXAONE-4.0-1.2B FP8 W8A8 Static Quantization")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Calibration samples: {num_samples}")
    print(f"Max sequence length: {max_length}")
    print(f"Log: {log_path}")

    # 토크나이저 로드
    print(f"\nLoading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # 모델 로드 (fp16 — L4에서 양자화 커널 호환성)
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 캘리브레이션 데이터셋 준비
    print("\nPreparing calibration dataset (MANTA-1M)...")
    calibration_ds = prepare_calibration_dataset(
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_length=max_length,
        seed=seed,
    )

    # FP8 Static W8A8 양자화 레시피 + KV cache FP8
    print("\nQuantization config:")
    print("  scheme: FP8 (Static W8A8)")
    print("  targets: Linear")
    print("  ignore: ['lm_head']")
    print("  kv_cache_scheme: FP8 E4M3 dynamic (per-tensor)")

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8",
        ignore=["lm_head"],
        kv_cache_scheme={
            "type": "float",
            "num_bits": 8,
            "strategy": "tensor",
            "dynamic": True,
        },
    )

    # 양자화 적용
    print("\nApplying FP8 Static quantization...")
    oneshot(
        model=model,
        dataset=calibration_ds,
        recipe=recipe,
        max_seq_length=max_length,
        num_calibration_samples=num_samples,
    )

    # 저장
    print(f"\nSaving quantized model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, save_compressed=True)
    tokenizer.save_pretrained(output_path)

    # config.json의 torch_dtype을 float16으로 강제 설정
    config_path = os.path.join(output_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        config["torch_dtype"] = "float16"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print("  config.json: torch_dtype set to 'float16'")

    print("\n" + "=" * 60)
    print("Quantization completed!")
    print("=" * 60)
    print(f"\nOutput directory: {output_path}")

    print("\nGenerated files:")
    for f in sorted(Path(output_path).glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")

    log_file.close()


def main():
    parser = argparse.ArgumentParser(
        description="EXAONE-4.0-1.2B FP8 W8A8 Static Quantization"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="LGAI-EXAONE/EXAONE-4.0-1.2B",
        help="원본 모델 경로 (HuggingFace 모델 ID 또는 로컬 경로)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/EXAONE-4.0-1.2B-fp8-w8a8-kv",
        help="양자화된 모델 저장 경로",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="캘리브레이션 샘플 수",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="최대 시퀀스 길이",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드",
    )

    args = parser.parse_args()

    quantize_model(
        model_path=args.model_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

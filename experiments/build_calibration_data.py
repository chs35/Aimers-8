"""
캘리브레이션 데이터셋 빌드 스크립트
gsm8k (512) + smol-koreantalk 한국어 (1024) + smol-koreantalk 영어 (512) = 2048

Usage:
    python experiments/build_calibration_data.py
    python experiments/build_calibration_data.py --output data/calibration --seed 42
"""
import json
import argparse
from datetime import datetime
from pathlib import Path

from datasets import load_dataset, Dataset


def build_gsm8k(num_samples=512, seed=42):
    """GSM8K train에서 샘플링 → chat messages 형태로 변환"""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=seed).select(range(num_samples))

    samples = []
    for ex in ds:
        samples.append({
            "messages": [
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["answer"]},
            ],
            "source": "gsm8k",
            "source_id": None,
        })
    return samples


def build_koreantalk(num_ko=1024, num_en=512, seed=42):
    """smol-koreantalk에서 한국어/영어 샘플링 (custom_id 중복 없음)"""
    ds = load_dataset("lemon-mint/smol-koreantalk", split="train")
    ds = ds.shuffle(seed=seed)

    # 앞쪽 num_ko개 → 한국어, 그 다음 num_en개 → 영어 (custom_id 중복 방지)
    ko_subset = ds.select(range(num_ko))
    en_subset = ds.select(range(num_ko, num_ko + num_en))

    ko_ids = [ex["custom_id"] for ex in ko_subset]
    en_ids = [ex["custom_id"] for ex in en_subset]

    # 중복 검증
    overlap = set(ko_ids) & set(en_ids)
    assert len(overlap) == 0, f"custom_id 중복 발견: {overlap}"

    ko_samples = []
    for ex in ko_subset:
        ko_samples.append({
            "messages": [
                {"role": m["role"], "content": m["content"]}
                for m in ex["messages"]
            ],
            "source": "koreantalk-ko",
            "source_id": ex["custom_id"],
        })

    en_samples = []
    for ex in en_subset:
        en_samples.append({
            "messages": [
                {"role": m["role"], "content": m["content_en"]}
                for m in ex["messages"]
            ],
            "source": "koreantalk-en",
            "source_id": ex["custom_id"],
        })

    return ko_samples, en_samples, ko_ids, en_ids


def build(output_dir="data/calibration", seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading gsm8k...")
    gsm8k_samples = build_gsm8k(num_samples=512, seed=seed)

    print("Loading smol-koreantalk...")
    ko_samples, en_samples, ko_ids, en_ids = build_koreantalk(
        num_ko=1024, num_en=512, seed=seed
    )

    # 합치기 & 셔플
    all_samples = gsm8k_samples + ko_samples + en_samples
    ds = Dataset.from_list(all_samples)
    ds = ds.shuffle(seed=seed)

    # 데이터셋 저장
    ds.save_to_disk(str(output_dir / "dataset"))

    # 메타데이터 저장
    metadata = {
        "total_samples": len(ds),
        "seed": seed,
        "created_at": datetime.now().isoformat(),
        "sources": {
            "gsm8k": {
                "dataset": "openai/gsm8k",
                "config": "main",
                "split": "train",
                "count": len(gsm8k_samples),
            },
            "koreantalk-ko": {
                "dataset": "lemon-mint/smol-koreantalk",
                "split": "train",
                "field": "content",
                "count": len(ko_samples),
                "custom_ids": ko_ids,
            },
            "koreantalk-en": {
                "dataset": "lemon-mint/smol-koreantalk",
                "split": "train",
                "field": "content_en",
                "count": len(en_samples),
                "custom_ids": en_ids,
            },
        },
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Built {len(ds)} samples → {output_dir}")
    print(f"  gsm8k: {len(gsm8k_samples)}")
    print(f"  koreantalk-ko: {len(ko_samples)}")
    print(f"  koreantalk-en: {len(en_samples)}")
    print(f"Metadata saved → {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build calibration dataset from multiple HF sources"
    )
    parser.add_argument("--output", "-o", default="data/calibration")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.output, args.seed)

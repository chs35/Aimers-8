"""
prepare_dataset: 7개 HuggingFace 데이터셋에서 Disjoint 3-Split 생성

파이프라인의 각 단계(Width Pruning, KD Recovery, FP8 Quantization)가
서로 겹치지 않는(disjoint) 데이터를 사용하도록 보장한다.

통합 포맷: {"conversations": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}], "source": "dataset_name"}

데이터셋:
  1. Open-Orca/SlimOrca (영어 일반)
  2. microsoft/orca-math-word-problems-200k (영어 수학)
  3. kuotient/orca-math-korean-preference (한글 수학)
  4. beomi/KoAlpaca-RealQA (한글 일반)
  5. TIGER-Lab/MMLU-STEM (영어 STEM)
  6. daekeun-ml/naver-news-summarization-ko (한글 요약)
  7. davidkim205/ko-ged (한글 STEM)

사용법:
  python generate/prepare_dataset.py --output dataset --wp-samples 500 --kd-samples 2000 --fp8-samples 512 --seed 42
  # → dataset/dataset_1/{wp_calib, kd_train, fp8_calib, metadata.json}
"""

import os
import json
import argparse
import random
from datetime import datetime

from datasets import load_dataset, Dataset


# ╔══════════════════════════════════════════════════════════════╗
# ║  데이터셋별 정규화 함수                                       ║
# ╚══════════════════════════════════════════════════════════════╝

DATASET_CONFIGS = [
    {
        "id": "Open-Orca/SlimOrca",
        "name": "SlimOrca",
        "split": "train",
        "default_samples": 800,
    },
    {
        "id": "microsoft/orca-math-word-problems-200k",
        "name": "orca-math",
        "split": "train",
        "default_samples": 600,
    },
    {
        "id": "kuotient/orca-math-korean-preference",
        "name": "orca-math-korean",
        "split": "train",
        "default_samples": 500,
    },
    {
        "id": "beomi/KoAlpaca-RealQA",
        "name": "KoAlpaca-RealQA",
        "split": "train",
        "default_samples": 400,
    },
    {
        "id": "TIGER-Lab/MMLU-STEM",
        "name": "MMLU-STEM",
        "split": "test",
        "default_samples": 300,
    },
    {
        "id": "daekeun-ml/naver-news-summarization-ko",
        "name": "naver-news",
        "split": "train",
        "default_samples": 300,
    },
    {
        "id": "davidkim205/ko-ged",
        "name": "ko-ged",
        "split": "train",
        "default_samples": 200,
    },
]


def normalize_slimorca(example):
    """SlimOrca: from/value → role/content 변환 (human→user, gpt→assistant)"""
    ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}
    conversations = []
    for turn in example["conversations"]:
        role = ROLE_MAP.get(turn["from"], turn["from"])
        conversations.append({"role": role, "content": turn["value"]})
    return conversations


def normalize_orca_math(example):
    """orca-math-word-problems-200k: question/answer → user/assistant"""
    return [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]


def normalize_orca_math_korean(example):
    """orca-math-korean-preference: question/answer → user/assistant"""
    return [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]


def normalize_koalpaca(example):
    """KoAlpaca-RealQA: 필드 동적 탐지 (Q/A, instruction/output 등)"""
    # Q/A 패턴
    if "Q" in example and "A" in example:
        return [
            {"role": "user", "content": example["Q"]},
            {"role": "assistant", "content": example["A"]},
        ]
    # instruction/output 패턴
    if "instruction" in example and "output" in example:
        user_content = example["instruction"]
        if example.get("input"):
            user_content += "\n\n" + example["input"]
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]
    # question/answer 패턴
    if "question" in example and "answer" in example:
        return [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    # 대화 패턴
    if "conversations" in example:
        return normalize_slimorca(example)
    raise ValueError(f"KoAlpaca-RealQA: 알 수 없는 필드 구조: {list(example.keys())}")


def normalize_mmlu_stem(example):
    """MMLU-STEM: MCQ 포맷팅 (question + choices → user, answer → assistant)"""
    choices = example["choices"]
    labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))
    user_content = f"{example['question']}\n\n{choices_text}"

    answer_idx = example["answer"]
    if isinstance(answer_idx, int) and 0 <= answer_idx < len(labels):
        answer_text = f"{labels[answer_idx]}. {choices[answer_idx]}"
    else:
        answer_text = str(answer_idx)

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer_text},
    ]


def normalize_naver_news(example):
    """naver-news: document → user (요약 요청), summary → assistant"""
    return [
        {"role": "user", "content": f"다음 기사를 요약해주세요.\n\n{example['document']}"},
        {"role": "assistant", "content": example["summary"]},
    ]


def normalize_ko_ged(example):
    """ko-ged: 필드 동적 탐지"""
    # question/answer 패턴
    if "question" in example and "answer" in example:
        return [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": str(example["answer"])},
        ]
    # instruction/output 패턴
    if "instruction" in example and "output" in example:
        user_content = example["instruction"]
        if example.get("input"):
            user_content += "\n\n" + example["input"]
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]
    # problem/solution 패턴
    if "problem" in example and "solution" in example:
        return [
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": str(example["solution"])},
        ]
    # conversations 패턴
    if "conversations" in example:
        return normalize_slimorca(example)
    raise ValueError(f"ko-ged: 알 수 없는 필드 구조: {list(example.keys())}")


NORMALIZERS = {
    "SlimOrca": normalize_slimorca,
    "orca-math": normalize_orca_math,
    "orca-math-korean": normalize_orca_math_korean,
    "KoAlpaca-RealQA": normalize_koalpaca,
    "MMLU-STEM": normalize_mmlu_stem,
    "naver-news": normalize_naver_news,
    "ko-ged": normalize_ko_ged,
}


# ╔══════════════════════════════════════════════════════════════╗
# ║  핵심 로직                                                    ║
# ╚══════════════════════════════════════════════════════════════╝

def load_and_normalize(config, num_samples, seed=42):
    """단일 데이터셋 로드 → 정규화 → conversations 형식으로 변환"""
    ds_id = config["id"]
    ds_name = config["name"]
    ds_split = config["split"]
    normalizer = NORMALIZERS[ds_name]

    print(f"  [{ds_name}] {ds_id} (split={ds_split}) 에서 {num_samples}개 로드 중...")

    try:
        # 스트리밍으로 필요한 만큼만 로드
        raw_ds = load_dataset(ds_id, split=ds_split, streaming=True)
        raw_ds = raw_ds.shuffle(seed=seed, buffer_size=5000)
        samples = list(raw_ds.take(num_samples))
    except Exception as e:
        print(f"  [{ds_name}] 스트리밍 로드 실패, 일반 로드 시도: {e}")
        try:
            raw_ds = load_dataset(ds_id, split=ds_split)
            raw_ds = raw_ds.shuffle(seed=seed)
            samples = list(raw_ds.select(range(min(num_samples, len(raw_ds)))))
        except Exception as e2:
            print(f"  [{ds_name}] 로드 실패, 건너뜀: {e2}")
            return []

    normalized = []
    skipped = 0
    for sample in samples:
        try:
            conversations = normalizer(sample)
            # 빈 대화 필터링
            if conversations and all(turn.get("content") for turn in conversations):
                normalized.append({
                    "conversations": conversations,
                    "source": ds_name,
                })
            else:
                skipped += 1
        except Exception:
            skipped += 1

    print(f"  [{ds_name}] {len(normalized)}개 정규화 완료 (건너뜀: {skipped})")
    return normalized


def find_next_dataset_dir(base_dir):
    """dataset/dataset_N에서 N을 자동 증가"""
    n = 1
    while os.path.exists(os.path.join(base_dir, f"dataset_{n}")):
        n += 1
    return n, os.path.join(base_dir, f"dataset_{n}")


def main():
    parser = argparse.ArgumentParser(description="Disjoint 3-Split 데이터셋 생성")
    parser.add_argument("--output", type=str, default="dataset",
                        help="출력 base 디렉토리 (기본값: dataset)")
    parser.add_argument("--wp-samples", type=int, default=500,
                        help="Width Pruning 캘리브레이션 샘플 수 (기본값: 500)")
    parser.add_argument("--kd-samples", type=int, default=2000,
                        help="KD 학습 샘플 수 (기본값: 2000)")
    parser.add_argument("--fp8-samples", type=int, default=512,
                        help="FP8 캘리브레이션 샘플 수 (기본값: 512)")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드 (기본값: 42)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="각 데이터셋의 기본 샘플 수에 곱할 배율 (기본값: 1.0)")
    args = parser.parse_args()

    total_needed = args.wp_samples + args.kd_samples + args.fp8_samples

    print("=" * 60)
    print("Disjoint 3-Split 데이터셋 생성")
    print("=" * 60)
    print(f"  WP 캘리브레이션: {args.wp_samples}")
    print(f"  KD 학습:        {args.kd_samples}")
    print(f"  FP8 캘리브레이션: {args.fp8_samples}")
    print(f"  총 필요:         {total_needed}")
    print(f"  시드:            {args.seed}")
    print()

    # ── Step 1: 각 데이터셋 로드 + 정규화 ──
    print("=" * 60)
    print("Step 1: 데이터셋 로드 + 정규화")
    print("=" * 60)

    global_pool = []
    source_counts = {}

    for config in DATASET_CONFIGS:
        num = int(config["default_samples"] * args.scale)
        # 최소한 total_needed를 채울 수 있도록 스케일 업
        normalized = load_and_normalize(config, num, seed=args.seed)
        global_pool.extend(normalized)
        source_counts[config["name"]] = len(normalized)

    print()
    print(f"  글로벌 풀 크기: {len(global_pool)}")
    for name, count in source_counts.items():
        print(f"    {name}: {count}")

    if len(global_pool) < total_needed:
        print(f"\n  [경고] 풀 크기({len(global_pool)}) < 필요 샘플 수({total_needed})")
        print(f"  --scale 값을 높이거나 샘플 수를 줄여주세요.")
        print(f"  현재 풀 크기만큼 분할합니다.")
    print()

    # ── Step 2: 셔플 + 3-Split 분할 ──
    print("=" * 60)
    print("Step 2: 셔플 + Disjoint 3-Split 분할")
    print("=" * 60)

    random.seed(args.seed)
    random.shuffle(global_pool)

    pool_size = len(global_pool)
    wp_end = min(args.wp_samples, pool_size)
    kd_end = min(wp_end + args.kd_samples, pool_size)
    fp8_end = min(kd_end + args.fp8_samples, pool_size)

    wp_split = global_pool[:wp_end]
    kd_split = global_pool[wp_end:kd_end]
    fp8_split = global_pool[kd_end:fp8_end]

    print(f"  wp_calib:  {len(wp_split)} 샘플")
    print(f"  kd_train:  {len(kd_split)} 샘플")
    print(f"  fp8_calib: {len(fp8_split)} 샘플")
    print(f"  미사용:    {pool_size - fp8_end} 샘플")
    print()

    # ── Step 3: 저장 ──
    print("=" * 60)
    print("Step 3: Arrow 포맷으로 저장")
    print("=" * 60)

    dataset_n, dataset_dir = find_next_dataset_dir(args.output)
    os.makedirs(dataset_dir, exist_ok=True)

    splits = {
        "wp_calib": wp_split,
        "kd_train": kd_split,
        "fp8_calib": fp8_split,
    }

    for split_name, split_data in splits.items():
        split_dir = os.path.join(dataset_dir, split_name)
        ds = Dataset.from_list(split_data)
        ds.save_to_disk(split_dir)
        print(f"  {split_name}: {len(ds)} 샘플 → {split_dir}/")

    # 메타데이터 저장
    metadata = {
        "created_at": datetime.now().isoformat(),
        "seed": args.seed,
        "scale": args.scale,
        "total_pool_size": pool_size,
        "splits": {
            "wp_calib": len(wp_split),
            "kd_train": len(kd_split),
            "fp8_calib": len(fp8_split),
        },
        "source_counts": source_counts,
        "dataset_configs": [
            {"id": c["id"], "name": c["name"], "split": c["split"],
             "requested": int(c["default_samples"] * args.scale)}
            for c in DATASET_CONFIGS
        ],
    }
    meta_path = os.path.join(dataset_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  metadata: {meta_path}")

    print()
    print("=" * 60)
    print(f"완료! 데이터셋 저장 위치: {dataset_dir}/")
    print("=" * 60)
    print(f"  wp_calib:  {len(wp_split)} 샘플 (Width Pruning 캘리브레이션)")
    print(f"  kd_train:  {len(kd_split)} 샘플 (KD 학습)")
    print(f"  fp8_calib: {len(fp8_split)} 샘플 (FP8 캘리브레이션)")


if __name__ == "__main__":
    main()

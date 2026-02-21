"""
prepare_dataset: QAT 전용 단일 Split 데이터셋 생성

generate/prepare_dataset.py와 동일한 데이터셋 및 normalizer를 재사용하되,
QAT에 필요한 qat_train split 하나만 생성한다.

사용법:
  python QAT/prepare_dataset.py --output dataset --samples 2000 --seed 42
  # → dataset/dataset_N/{qat_train, metadata.json}
"""

import os
import json
import argparse
import random
from datetime import datetime

from datasets import load_dataset, Dataset


# ╔══════════════════════════════════════════════════════════════╗
# ║  데이터셋별 정규화 함수 (generate/prepare_dataset.py 그대로)  ║
# ╚══════════════════════════════════════════════════════════════╝

DATASET_CONFIGS = [
    # ── SFT 메인 ──
    {
        "id": "microsoft/orca-math-word-problems-200k",
        "name": "orca-math",
        "split": "train",
        "default_samples": 600,
    },
    {
        "id": "Open-Orca/SlimOrca",
        "name": "SlimOrca",
        "split": "train",
        "default_samples": 600,
    },
    {
        "id": "kuotient/orca-math-word-problems-193k-korean",
        "name": "orca-math-193k-korean",
        "split": "train",
        "default_samples": 500,
    },
    {
        "id": "nlpai-lab/kullm-v2",
        "name": "kullm-v2",
        "split": "train",
        "default_samples": 500,
    },
    # ── Calibration / 보강 ──
    {
        "id": "openai/gsm8k",
        "name": "gsm8k",
        "config": "main",
        "split": "train",
        "default_samples": 400,
    },
    {
        "id": "LGAI-EXAONE/MANTA-1M",
        "name": "MANTA-1M",
        "split": "train",
        "default_samples": 500,
    },
    {
        "id": "kuotient/orca-math-korean-preference",
        "name": "orca-math-korean-pref",
        "split": "train",
        "default_samples": 300,
    },
    {
        "id": "TIGER-Lab/MMLU-Pro",
        "name": "MMLU-Pro",
        "split": "test",
        "default_samples": 300,
    },
]


def normalize_orca_math(example):
    """orca-math-word-problems-200k: question/answer → user/assistant"""
    return [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]


def normalize_slimorca(example):
    """SlimOrca: from/value → role/content 변환 (human→user, gpt→assistant)"""
    ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}
    conversations = []
    for turn in example["conversations"]:
        role = ROLE_MAP.get(turn["from"], turn["from"])
        conversations.append({"role": role, "content": turn["value"]})
    return conversations


def normalize_orca_math_193k_korean(example):
    """orca-math-word-problems-193k-korean: question/answer → user/assistant"""
    return [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]


def normalize_kullm_v2(example):
    """kullm-v2: instruction/input/output → user/assistant"""
    user_content = example["instruction"]
    if example.get("input"):
        user_content += "\n\n" + example["input"]
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]


def normalize_gsm8k(example):
    """gsm8k: question/answer → user/assistant (CoT reasoning 포함)"""
    return [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]


def normalize_manta_1m(example):
    """MANTA-1M: conversations 필드가 이미 role/content 포맷"""
    conversations = example["conversations"]
    if conversations and isinstance(conversations[0], dict) and "role" in conversations[0]:
        return conversations
    ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}
    return [
        {"role": ROLE_MAP.get(t.get("from", ""), t.get("from", "")),
         "content": t.get("value", t.get("content", ""))}
        for t in conversations
    ]


def normalize_orca_math_korean_pref(example):
    """orca-math-korean-preference: question/answer → user/assistant"""
    return [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]


def normalize_mmlu_pro(example):
    """MMLU-Pro: MCQ 포맷팅 (question + options → user, answer → assistant)"""
    options = example["options"]
    labels = [chr(ord("A") + i) for i in range(len(options))]
    choices_text = "\n".join(f"{labels[i]}. {opt}" for i, opt in enumerate(options))
    user_content = f"{example['question']}\n\n{choices_text}"

    answer = example.get("answer", "")
    if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
        idx = ord(answer.upper()) - ord("A")
        if 0 <= idx < len(options):
            answer_text = f"{answer.upper()}. {options[idx]}"
        else:
            answer_text = answer
    elif isinstance(answer, int) and 0 <= answer < len(options):
        answer_text = f"{labels[answer]}. {options[answer]}"
    else:
        answer_text = str(answer)

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer_text},
    ]


NORMALIZERS = {
    "orca-math": normalize_orca_math,
    "SlimOrca": normalize_slimorca,
    "orca-math-193k-korean": normalize_orca_math_193k_korean,
    "kullm-v2": normalize_kullm_v2,
    "gsm8k": normalize_gsm8k,
    "MANTA-1M": normalize_manta_1m,
    "orca-math-korean-pref": normalize_orca_math_korean_pref,
    "MMLU-Pro": normalize_mmlu_pro,
}


# ╔══════════════════════════════════════════════════════════════╗
# ║  핵심 로직                                                    ║
# ╚══════════════════════════════════════════════════════════════╝

def load_and_normalize(config, num_samples, seed=42):
    """단일 데이터셋 로드 → 정규화 → conversations 형식으로 변환"""
    ds_id = config["id"]
    ds_name = config["name"]
    ds_split = config["split"]
    ds_config = config.get("config")
    normalizer = NORMALIZERS[ds_name]

    print(f"  [{ds_name}] {ds_id} (split={ds_split}) 에서 {num_samples}개 로드 중...")

    try:
        raw_ds = load_dataset(ds_id, ds_config, split=ds_split, streaming=True)
        raw_ds = raw_ds.shuffle(seed=seed, buffer_size=5000)
        samples = list(raw_ds.take(num_samples))
    except Exception as e:
        print(f"  [{ds_name}] 스트리밍 로드 실패, 일반 로드 시도: {e}")
        try:
            raw_ds = load_dataset(ds_id, ds_config, split=ds_split)
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
    parser = argparse.ArgumentParser(description="QAT 전용 단일 Split 데이터셋 생성")
    parser.add_argument("--output", type=str, default="dataset",
                        help="출력 base 디렉토리 (기본값: dataset)")
    parser.add_argument("--samples", type=int, default=2000,
                        help="QAT 학습 샘플 수 (기본값: 2000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드 (기본값: 42)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="각 데이터셋의 기본 샘플 수에 곱할 배율 (기본값: 1.0)")
    args = parser.parse_args()

    print("=" * 60)
    print("QAT 전용 데이터셋 생성")
    print("=" * 60)
    print(f"  QAT 학습 샘플: {args.samples}")
    print(f"  시드:          {args.seed}")
    print()

    # ── Step 1: 각 데이터셋 로드 + 정규화 ──
    print("=" * 60)
    print("Step 1: 데이터셋 로드 + 정규화")
    print("=" * 60)

    global_pool = []
    source_counts = {}

    default_total = sum(c["default_samples"] for c in DATASET_CONFIGS)
    needed_with_buffer = args.samples * 1.2
    if default_total * args.scale < needed_with_buffer:
        auto_scale = needed_with_buffer / default_total
        print(f"  [자동 스케일] 기본 풀({default_total})이 필요량({args.samples})에 부족할 수 있어 "
              f"scale {args.scale:.1f} → {auto_scale:.2f}로 조정")
        effective_scale = max(args.scale, auto_scale)
    else:
        effective_scale = args.scale

    for config in DATASET_CONFIGS:
        num = int(config["default_samples"] * effective_scale)
        normalized = load_and_normalize(config, num, seed=args.seed)
        global_pool.extend(normalized)
        source_counts[config["name"]] = len(normalized)

    print()
    print(f"  글로벌 풀 크기: {len(global_pool)}")
    for name, count in source_counts.items():
        print(f"    {name}: {count}")

    if len(global_pool) < args.samples:
        print(f"\n  [경고] 풀 크기({len(global_pool)}) < 필요 샘플 수({args.samples})")
        print(f"  --scale 값을 높이거나 --samples를 줄여주세요.")
        print(f"  가용한 샘플 전체를 qat_train으로 사용합니다.")
    print()

    # ── Step 2: 셔플 + Split ──
    print("=" * 60)
    print("Step 2: 셔플 + 단일 Split")
    print("=" * 60)

    random.seed(args.seed)
    random.shuffle(global_pool)

    qat_split = global_pool[:args.samples]
    print(f"  qat_train: {len(qat_split)} 샘플")
    print()

    # ── Step 3: 저장 ──
    print("=" * 60)
    print("Step 3: Arrow 포맷으로 저장")
    print("=" * 60)

    dataset_n, dataset_dir = find_next_dataset_dir(args.output)
    os.makedirs(dataset_dir, exist_ok=True)

    split_dir = os.path.join(dataset_dir, "qat_train")
    ds = Dataset.from_list(qat_split)
    ds.save_to_disk(split_dir)
    print(f"  qat_train: {len(ds)} 샘플 → {split_dir}/")

    metadata = {
        "created_at": datetime.now().isoformat(),
        "seed": args.seed,
        "scale": args.scale,
        "total_pool_size": len(global_pool),
        "splits": {
            "qat_train": len(qat_split),
        },
        "source_counts": source_counts,
        "dataset_configs": [
            {"id": c["id"], "name": c["name"], "split": c["split"],
             "requested": int(c["default_samples"] * effective_scale)}
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
    print(f"  qat_train: {len(qat_split)} 샘플 (QAT 학습)")


if __name__ == "__main__":
    main()

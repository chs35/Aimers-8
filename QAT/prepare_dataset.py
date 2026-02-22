"""
prepare_dataset: QAT 전용 단일 Split 데이터셋 생성

--samples N 을 지정하면 전체 N개를 13개 데이터셋에서 공평하게 분할하여 추출한다.
(N // 13 개씩, 나머지는 앞 데이터셋에 +1)

사용법:
  python QAT/prepare_dataset.py --output dataset --samples 13000 --seed 42
  # → dataset/dataset_N/{qat_train, metadata.json}
"""

import os
import json
import argparse
import random
from datetime import datetime

from datasets import load_dataset, Dataset


# ╔══════════════════════════════════════════════════════════════╗
# ║  데이터셋 설정 (13개)                                         ║
# ╚══════════════════════════════════════════════════════════════╝

DATASET_CONFIGS = [
    # ── 수학 / 추론 ──
    {"id": "microsoft/orca-math-word-problems-200k", "name": "orca-math",              "split": "train"},
    {"id": "openai/gsm8k",                           "name": "gsm8k",                 "split": "train", "config": "main"},
    {"id": "MegaScience/TextbookReasoning",           "name": "TextbookReasoning",     "split": "train"},
    # ── 영어 SFT ──
    {"id": "TIGER-Lab/MMLU-Pro",                     "name": "MMLU-Pro",              "split": "test"},
    # ── 한국어 수학 / 추론 ──
    {"id": "kuotient/orca-math-word-problems-193k-korean", "name": "orca-math-193k-korean", "split": "train"},
    {"id": "kuotient/orca-math-korean-preference",   "name": "orca-math-korean-pref", "split": "train"},
    # ── 한국어 SFT ──
    {"id": "nlpai-lab/kullm-v2",                     "name": "kullm-v2",              "split": "train"},
    {"id": "LGAI-EXAONE/MANTA-1M",                   "name": "MANTA-1M",              "split": "train"},
    {"id": "lemon-mint/Korean-FineTome-100k",         "name": "Korean-FineTome",       "split": "train"},
    # ── 한국어 특화 ──
    {"id": "daekeun-ml/naver-news-summarization-ko", "name": "naver-news-ko",         "split": "train"},
    {"id": "KETI-AIR/korquad",                       "name": "korquad",               "split": "train"},
    {"id": "HAERAE-HUB/KMMLU",                       "name": "KMMLU",                 "split": "test"},
]


# ╔══════════════════════════════════════════════════════════════╗
# ║  데이터셋별 정규화 함수                                        ║
# ╚══════════════════════════════════════════════════════════════╝

def normalize_orca_math(example):
    """orca-math-word-problems-200k: question/answer → user/assistant"""
    return [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]


def normalize_gsm8k(example):
    """gsm8k: question/answer → user/assistant (CoT reasoning 포함)"""
    return [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]


def normalize_textbook_reasoning(example):
    """MegaScience/TextbookReasoning: problem/solution 또는 question/answer"""
    # 가능한 필드명 순서대로 시도
    question = (
        example.get("problem") or example.get("question") or
        example.get("query") or example.get("input") or ""
    )
    answer = (
        example.get("solution") or example.get("answer") or
        example.get("response") or example.get("output") or ""
    )
    # conversations 형식이면 그대로 반환
    if not question and "conversations" in example:
        return normalize_sharegpt(example)
    if not question and "text" in example:
        # text 단일 필드면 건너뜀 (QA 쌍 추출 불가)
        return []
    return [
        {"role": "user", "content": str(question)},
        {"role": "assistant", "content": str(answer)},
    ]


def normalize_slimorca(example):
    """SlimOrca: from/value → role/content 변환 (human→user, gpt→assistant)"""
    ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}
    conversations = []
    for turn in example["conversations"]:
        role = ROLE_MAP.get(turn["from"], turn["from"])
        conversations.append({"role": role, "content": turn["value"]})
    return conversations


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


def normalize_orca_math_193k_korean(example):
    """orca-math-word-problems-193k-korean: question/answer → user/assistant"""
    return [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]


def normalize_orca_math_korean_pref(example):
    """orca-math-korean-preference: question/answer → user/assistant"""
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


def normalize_sharegpt(example):
    """ShareGPT 공통: conversations 필드의 from/value → role/content"""
    ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}
    conversations = example.get("conversations", [])
    result = []
    for turn in conversations:
        role = ROLE_MAP.get(turn.get("from", ""), turn.get("from", ""))
        content = turn.get("value", turn.get("content", ""))
        result.append({"role": role, "content": content})
    return result


def normalize_korean_finetome(example):
    """lemon-mint/Korean-FineTome-100k: ShareGPT conversations 포맷"""
    # conversations 또는 messages 필드
    if "conversations" in example:
        return normalize_sharegpt(example)
    if "messages" in example:
        # role/content 포맷이면 그대로
        messages = example["messages"]
        if messages and "role" in messages[0]:
            return messages
    # instruction/output 단일 쌍
    if "instruction" in example:
        return normalize_kullm_v2(example)
    return []


def normalize_naver_news_ko(example):
    """daekeun-ml/naver-news-summarization-ko: 뉴스 기사 → 요약"""
    # 필드: id, title, content, summary (또는 document/abstractive)
    content = (
        example.get("content") or example.get("document") or
        example.get("article") or example.get("text") or ""
    )
    summary = (
        example.get("summary") or example.get("abstractive") or
        example.get("highlights") or ""
    )
    title = example.get("title", "")
    user_content = f"다음 기사를 요약해 주세요.\n\n제목: {title}\n\n{content}" if title else f"다음 기사를 요약해 주세요.\n\n{content}"
    return [
        {"role": "user", "content": user_content.strip()},
        {"role": "assistant", "content": str(summary).strip()},
    ]


def normalize_korquad(example):
    """KETI-AIR/korquad: SQuAD 스타일 한국어 QA (context + question → answer)"""
    question = example.get("question", "")
    context = example.get("context", "")
    answers = example.get("answers", {})
    # answers: {"text": [...], "answer_start": [...]}
    if isinstance(answers, dict):
        texts = answers.get("text", [])
        answer = texts[0] if texts else ""
    elif isinstance(answers, list):
        answer = answers[0] if answers else ""
    else:
        answer = str(answers)
    return [
        {"role": "user", "content": f"{question}\n\n지문:\n{context}"},
        {"role": "assistant", "content": str(answer)},
    ]


def normalize_kmmlu(example):
    """HAERAE-HUB/KMMLU: 한국어 4지선다 (A/B/C/D → answer)"""
    question = example.get("question", "")
    options = [
        example.get("A", ""), example.get("B", ""),
        example.get("C", ""), example.get("D", ""),
    ]
    labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(
        f"{labels[i]}. {opt}" for i, opt in enumerate(options) if opt
    )
    user_content = f"{question}\n\n{choices_text}"

    answer = example.get("answer", "")
    # answer가 1~4 정수인 경우
    if isinstance(answer, int) and 1 <= answer <= 4:
        idx = answer - 1
        answer_text = f"{labels[idx]}. {options[idx]}"
    # answer가 알파벳 문자인 경우
    elif isinstance(answer, str) and len(answer) == 1 and answer.upper() in "ABCD":
        idx = ord(answer.upper()) - ord("A")
        if 0 <= idx < len(options):
            answer_text = f"{answer.upper()}. {options[idx]}"
        else:
            answer_text = answer
    else:
        answer_text = str(answer)

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer_text},
    ]


NORMALIZERS = {
    "orca-math":            normalize_orca_math,
    "gsm8k":                normalize_gsm8k,
    "TextbookReasoning":    normalize_textbook_reasoning,
    "SlimOrca":             normalize_slimorca,
    "MMLU-Pro":             normalize_mmlu_pro,
    "orca-math-193k-korean": normalize_orca_math_193k_korean,
    "orca-math-korean-pref": normalize_orca_math_korean_pref,
    "kullm-v2":             normalize_kullm_v2,
    "MANTA-1M":             normalize_manta_1m,
    "Korean-FineTome":      normalize_korean_finetome,
    "naver-news-ko":        normalize_naver_news_ko,
    "korquad":              normalize_korquad,
    "KMMLU":                normalize_kmmlu,
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
    parser.add_argument("--samples", type=int, default=13000,
                        help=f"QAT 학습 총 샘플 수 — {len(DATASET_CONFIGS)}개 데이터셋에서 공평 분할 (기본값: 13000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드 (기본값: 42)")
    args = parser.parse_args()

    n_datasets = len(DATASET_CONFIGS)
    per_dataset = args.samples // n_datasets
    remainder   = args.samples % n_datasets

    print("=" * 60)
    print("QAT 전용 데이터셋 생성")
    print("=" * 60)
    print(f"  총 목표 샘플:    {args.samples}")
    print(f"  데이터셋 수:     {n_datasets}개")
    print(f"  데이터셋당 샘플: {per_dataset} (나머지 {remainder}개는 앞 데이터셋에 +1)")
    print(f"  시드:            {args.seed}")
    print()

    # ── Step 1: 각 데이터셋 로드 + 정규화 (공평 분할) ──
    print("=" * 60)
    print("Step 1: 데이터셋 로드 + 정규화")
    print("=" * 60)

    global_pool  = []
    source_counts = {}
    per_dataset_requested = {}

    for i, config in enumerate(DATASET_CONFIGS):
        num = per_dataset + (1 if i < remainder else 0)
        per_dataset_requested[config["name"]] = num
        normalized = load_and_normalize(config, num, seed=args.seed)
        global_pool.extend(normalized)
        source_counts[config["name"]] = len(normalized)

    print()
    print(f"  글로벌 풀 크기: {len(global_pool)}")
    for name, count in source_counts.items():
        req = per_dataset_requested[name]
        print(f"    {name}: {count} / {req} 요청")

    if len(global_pool) < args.samples:
        print(f"\n  [경고] 풀 크기({len(global_pool)}) < 목표 샘플 수({args.samples})")
        print(f"  일부 데이터셋이 부족합니다. 가용한 샘플 전체를 qat_train으로 사용합니다.")
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
        "target_samples": args.samples,
        "per_dataset": per_dataset,
        "total_pool_size": len(global_pool),
        "splits": {
            "qat_train": len(qat_split),
        },
        "source_counts": source_counts,
        "dataset_configs": [
            {
                "id": c["id"],
                "name": c["name"],
                "split": c["split"],
                "requested": per_dataset_requested[c["name"]],
                "actual": source_counts[c["name"]],
            }
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

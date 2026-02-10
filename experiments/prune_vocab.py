"""
Vocab Pruning for EXAONE-4.0-1.2B

명확하게 다른 언어의 고유 문자체계(non-Latin script) 토큰을 제거하여
임베딩 레이어를 축소합니다.

동작:
  1. vocab.json에서 각 토큰을 Unicode 문자체계 기반으로 분류
  2. PRUNE_SCRIPTS에 지정된 문자체계의 토큰 ID를 수집
  3. 임베딩 행렬에서 해당 행 제거, 토큰 ID 재매핑
  4. 토크나이저(vocab.json, merges.txt, tokenizer.json) 업데이트
  5. config.json의 vocab_size 업데이트 후 저장

주의:
  - tie_word_embeddings=true이므로 embed_tokens만 수정하면 lm_head도 반영됨
  - 라틴 계열(é, ñ, ° 등)은 한국어/영어 문서에서도 사용되므로 제거하지 않음
  - special token (PAD, BOS, EOS, UNK 등)은 절대 제거하지 않음

사용법:
  python experiments/prune_vocab.py --model models/EXAONE-4.0-1.2B
  python experiments/prune_vocab.py --model models/EXAONE-4.0-1.2B --output models/EXAONE-4.0-1.2B-VocabPruned
"""

import argparse
import json
import copy
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# 제거 대상 문자체계 설정
# 여기에 나열된 스크립트에 해당하는 토큰만 제거합니다.
# 추가/제거하려면 이 집합을 수정하세요.
# ============================================================================
PRUNE_SCRIPTS = {
    "cjk",                # 중국어 (CJK Unified Ideographs)
    "japanese_hiragana",  # 일본어 히라가나
    "japanese_katakana",  # 일본어 카타카나
    "cyrillic",           # 러시아어 등 키릴 문자
    "arabic",             # 아랍어
    "hebrew",             # 히브리어
    "thai",               # 태국어
    "devanagari",         # 힌디어 (데바나가리)
    "bengali",            # 벵골어
    "tamil",              # 타밀어
    "telugu",             # 텔루구어
    "kannada",            # 칸나다어
    "malayalam",          # 말라얄람어
    "gujarati",           # 구자라트어
    "gurmukhi",           # 구르무키 (펀자브어)
    "oriya",              # 오리야어
    "georgian",           # 조지아어
    "armenian",           # 아르메니아어
    "myanmar",            # 미얀마어
    "sinhala",            # 싱할라어
    "ethiopic",           # 에티오피아어
    "khmer",              # 크메르어
    "lao",                # 라오어
    "tibetan",            # 티베트어
}

# ============================================================================
# 보존 대상 (절대 제거하지 않음)
# ============================================================================
ALWAYS_PRESERVE = {
    "english",
    "korean",
    "korean_english_mixed",
    "greek",              # 수학: α, β, π, σ...
    "math_symbol",        # 수학 연산자, 첨자
    "digit",
    "ascii_punctuation",
    "punctuation",
    "symbol",
    "whitespace",
    "punctuation_digit_symbol",
    # 라틴 계열 - 한국어/영어 문서에서도 사용 (°, ±, ×, · 등 포함)
    "latin1_supplement",
    "latin_extended",
    "latin_ext_additional",
    "combining_diacritical",
    "ipa_modifier",
    "other",              # 불확실한 것은 보존
}


# ============================================================================
# 토큰 분류 함수
# ============================================================================
def build_byte_decoder():
    """GPT-2 스타일 byte-level BPE 디코더 생성."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def decode_bpe_token(token: str, byte_decoder: dict) -> str:
    """BPE 토큰을 실제 UTF-8 문자열로 디코딩."""
    try:
        byte_seq = bytes([byte_decoder[c] for c in token])
        return byte_seq.decode("utf-8", errors="replace")
    except (KeyError, UnicodeDecodeError):
        return token


def classify_char(ch: str) -> str:
    """단일 문자를 Unicode 문자체계로 분류."""
    cp = ord(ch)

    if cp < 0x80:
        if ch.isascii() and ch.isalpha():
            return "english"
        elif ch.isdigit():
            return "digit"
        elif ch.isspace():
            return "whitespace"
        else:
            return "ascii_punctuation"

    # Korean
    if (0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F or
        0xAC00 <= cp <= 0xD7AF or 0xA960 <= cp <= 0xA97F or
        0xD7B0 <= cp <= 0xD7FF):
        return "korean"

    # Greek
    if 0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF:
        return "greek"

    # Math symbols
    if (0x2200 <= cp <= 0x22FF or 0x27C0 <= cp <= 0x27EF or
        0x2980 <= cp <= 0x29FF or 0x2A00 <= cp <= 0x2AFF or
        0x2100 <= cp <= 0x214F or 0x2070 <= cp <= 0x209F):
        return "math_symbol"

    # Latin-1 Supplement
    if 0x0080 <= cp <= 0x00FF:
        return "latin1_supplement"

    # Latin Extended A/B
    if 0x0100 <= cp <= 0x024F:
        return "latin_extended"

    # Latin Extended Additional (Vietnamese)
    if 0x1E00 <= cp <= 0x1EFF:
        return "latin_ext_additional"

    # IPA / Spacing Modifiers
    if 0x0250 <= cp <= 0x02FF:
        return "ipa_modifier"

    # Combining Diacritical Marks
    if 0x0300 <= cp <= 0x036F:
        return "combining_diacritical"

    # CJK
    if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
        0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F or
        0x2B740 <= cp <= 0x2B81F or 0xF900 <= cp <= 0xFAFF or
        cp == 0x3005):
        return "cjk"

    if 0x3040 <= cp <= 0x309F:
        return "japanese_hiragana"
    if 0x30A0 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
        return "japanese_katakana"
    if 0x0400 <= cp <= 0x04FF or 0x0500 <= cp <= 0x052F:
        return "cyrillic"
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF:
        return "arabic"
    if 0x0590 <= cp <= 0x05FF:
        return "hebrew"
    if 0x0E00 <= cp <= 0x0E7F:
        return "thai"
    if 0x0E80 <= cp <= 0x0EFF:
        return "lao"
    if 0x0900 <= cp <= 0x097F:
        return "devanagari"
    if 0x0980 <= cp <= 0x09FF:
        return "bengali"
    if 0x0A00 <= cp <= 0x0A7F:
        return "gurmukhi"
    if 0x0A80 <= cp <= 0x0AFF:
        return "gujarati"
    if 0x0B00 <= cp <= 0x0B7F:
        return "oriya"
    if 0x0B80 <= cp <= 0x0BFF:
        return "tamil"
    if 0x0C00 <= cp <= 0x0C7F:
        return "telugu"
    if 0x0C80 <= cp <= 0x0CFF:
        return "kannada"
    if 0x0D00 <= cp <= 0x0D7F:
        return "malayalam"
    if 0x0D80 <= cp <= 0x0DFF:
        return "sinhala"
    if 0x1000 <= cp <= 0x109F:
        return "myanmar"
    if 0x10A0 <= cp <= 0x10FF:
        return "georgian"
    if 0x1200 <= cp <= 0x137F:
        return "ethiopic"
    if 0x1780 <= cp <= 0x17FF:
        return "khmer"
    if 0x0530 <= cp <= 0x058F:
        return "armenian"
    if 0x0F00 <= cp <= 0x0FFF:
        return "tibetan"

    import unicodedata
    cat = unicodedata.category(ch)
    if cat.startswith("P"):
        return "punctuation"
    if cat.startswith("S"):
        return "symbol"
    if cat.startswith("N"):
        return "digit"
    if cat.startswith("Z"):
        return "whitespace"

    return "other"


def classify_token(decoded: str) -> str:
    """디코딩된 토큰을 문자체계 기반으로 분류."""
    from collections import Counter

    text = decoded.lstrip(" ")
    if not text:
        return "whitespace"

    char_classes = Counter()
    for ch in text:
        cls = classify_char(ch)
        if cls not in ("whitespace", "digit", "ascii_punctuation", "punctuation", "symbol"):
            char_classes[cls] += 1

    if not char_classes:
        has_alpha = any(c.isalpha() for c in text)
        if not has_alpha:
            return "punctuation_digit_symbol"
        return "english"

    top = char_classes.most_common()
    dominant, count = top[0]
    total_alpha = sum(char_classes.values())

    if count / total_alpha > 0.5:
        return dominant

    scripts = set(char_classes.keys())
    if scripts <= {"korean", "english"}:
        return "korean_english_mixed"

    return f"mixed_{'+'.join(sorted(scripts))}"


# ============================================================================
# Pruning 로직
# ============================================================================
def get_prune_token_ids(vocab: dict, byte_decoder: dict, special_ids: set) -> tuple:
    """제거할 토큰 ID 목록을 반환.

    Returns:
        (prune_ids, stats): 제거 대상 ID set, 카테고리별 통계 dict
    """
    from collections import defaultdict

    prune_ids = set()
    stats = defaultdict(lambda: {"count": 0, "action": "", "samples": []})

    for bpe_token, token_id in vocab.items():
        decoded = decode_bpe_token(bpe_token, byte_decoder)
        category = classify_token(decoded)

        # mixed 카테고리 처리: 구성 스크립트 중 하나라도 PRUNE 대상이면 제거
        if category.startswith("mixed_"):
            scripts_in_token = set(category.replace("mixed_", "").split("+"))
            should_prune = any(s in PRUNE_SCRIPTS for s in scripts_in_token)
        else:
            should_prune = category in PRUNE_SCRIPTS

        # special token은 절대 보존
        if token_id in special_ids:
            should_prune = False

        action = "PRUNE" if should_prune else "PRESERVE"
        stats[category]["count"] += 1
        stats[category]["action"] = action
        if len(stats[category]["samples"]) < 5:
            stats[category]["samples"].append(
                {"id": token_id, "decoded": decoded}
            )

        if should_prune:
            prune_ids.add(token_id)

    return prune_ids, dict(stats)


def prune_model(model_path: str, output_path: str):
    """모델 vocab pruning 실행."""

    model_dir = Path(model_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 토크나이저 & vocab 로드 ----
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    vocab_path = model_dir / "vocab.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Special token IDs (절대 제거하지 않음)
    config_path = model_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    special_ids = set()
    for key in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        tid = config.get(key)
        if tid is not None:
            special_ids.add(tid)
    # added_tokens (special tokens from tokenizer_config)
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
        tok_config = json.load(f)
    for tid_str, info in tok_config.get("added_tokens_decoder", {}).items():
        if info.get("special", False):
            special_ids.add(int(tid_str))

    print(f"Original vocab size: {len(vocab):,}")
    print(f"Special tokens (protected): {len(special_ids)} IDs")
    print(f"Prune target scripts: {sorted(PRUNE_SCRIPTS)}")

    # ---- 2. 제거 대상 식별 ----
    byte_decoder = build_byte_decoder()
    prune_ids, stats = get_prune_token_ids(vocab, byte_decoder, special_ids)

    print(f"\nTokens to prune: {len(prune_ids):,}")
    print(f"Tokens to keep:  {len(vocab) - len(prune_ids):,}")
    print(f"Reduction:       {len(prune_ids) / len(vocab) * 100:.2f}%")

    # 카테고리별 통계 출력
    print(f"\n{'Category':<35} {'Count':>7} {'Action':<10}")
    print("-" * 55)
    for cat in sorted(stats.keys(), key=lambda x: -stats[x]["count"]):
        info = stats[cat]
        print(f"  {cat:<33} {info['count']:>7,} {info['action']}")

    if not prune_ids:
        print("\nNo tokens to prune. Exiting.")
        return

    # ---- 3. ID 재매핑 테이블 생성 ----
    old_vocab_size = len(vocab)
    preserve_ids = sorted(set(range(old_vocab_size)) - prune_ids)
    new_vocab_size = len(preserve_ids)

    # old_id -> new_id
    id_remap = {old_id: new_id for new_id, old_id in enumerate(preserve_ids)}

    print(f"\nNew vocab size: {new_vocab_size:,} (was {old_vocab_size:,})")

    # ---- 4. 모델 로드 & 임베딩 pruning ----
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    embed_weight = model.get_input_embeddings().weight.data  # [vocab_size, hidden_size]
    print(f"Original embedding shape: {embed_weight.shape}")
    print(f"tie_word_embeddings: {config.get('tie_word_embeddings', False)}")

    # 보존할 행만 추출 (순서 유지)
    keep_indices = torch.tensor(preserve_ids, dtype=torch.long)
    new_embed_weight = embed_weight[keep_indices].clone()
    print(f"Pruned embedding shape:   {new_embed_weight.shape}")

    # 임베딩 교체: resize → 올바른 데이터 덮어쓰기 → weight tying 보장
    model.resize_token_embeddings(new_vocab_size)
    model.get_input_embeddings().weight.data.copy_(new_embed_weight)

    # weight tying 명시적으로 재적용
    model.tie_weights()

    # tying 검증
    input_emb = model.get_input_embeddings().weight
    output_emb = model.get_output_embeddings()
    if output_emb is not None:
        if output_emb.weight.data_ptr() == input_emb.data_ptr():
            print("Weight tying verified: embed_tokens and lm_head share the same tensor")
        else:
            print("WARNING: Weight tying FAILED - embed_tokens and lm_head are NOT tied!")
            print("  Manually copying embed_tokens -> lm_head")
            output_emb.weight = input_emb

    # config에 tie 설정 보존 확인
    model.config.tie_word_embeddings = True
    model.config.vocab_size = new_vocab_size

    # ---- 5. 토크나이저 파일 업데이트 ----
    print("\nUpdating tokenizer files...")

    # 5a. vocab.json 업데이트
    new_vocab = {}
    for bpe_token, old_id in vocab.items():
        if old_id not in prune_ids:
            new_vocab[bpe_token] = id_remap[old_id]

    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(new_vocab, f, ensure_ascii=False)

    # 5b. merges.txt 업데이트 (pruned 토큰을 생성하는 merge 제거)
    merges_path = model_dir / "merges.txt"
    if merges_path.exists():
        with open(merges_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_merges_lines = []
        removed_merges = 0
        for line in lines:
            stripped = line.strip()
            # 첫 줄 (#version 등) 또는 빈 줄은 유지
            if not stripped or stripped.startswith("#"):
                new_merges_lines.append(line)
                continue
            parts = stripped.split(" ")
            if len(parts) == 2:
                merged = parts[0] + parts[1]
                # merge 결과가 pruned vocab에 없으면 해당 merge도 제거
                if merged in vocab and vocab[merged] in prune_ids:
                    removed_merges += 1
                    continue
            new_merges_lines.append(line)

        with open(output_dir / "merges.txt", "w", encoding="utf-8") as f:
            f.writelines(new_merges_lines)
        print(f"  Removed {removed_merges:,} merges")

    # 5c. tokenizer.json 업데이트 (HuggingFace fast tokenizer)
    tokenizer_json_path = model_dir / "tokenizer.json"
    if tokenizer_json_path.exists():
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            tok_json = json.load(f)

        # vocab 업데이트
        if "model" in tok_json and "vocab" in tok_json["model"]:
            old_tok_vocab = tok_json["model"]["vocab"]
            new_tok_vocab = {}
            for token, old_id in old_tok_vocab.items():
                if old_id not in prune_ids:
                    new_tok_vocab[token] = id_remap[old_id]
            tok_json["model"]["vocab"] = new_tok_vocab

        # merges 업데이트 (list of str "a b" 또는 list of list ["a","b"] 모두 처리)
        if "model" in tok_json and "merges" in tok_json["model"]:
            old_merges = tok_json["model"]["merges"]
            new_merges = []
            for merge_entry in old_merges:
                if isinstance(merge_entry, list):
                    parts = merge_entry
                else:
                    parts = merge_entry.split(" ")
                if len(parts) == 2:
                    merged = parts[0] + parts[1]
                    if merged in vocab and vocab[merged] in prune_ids:
                        continue
                new_merges.append(merge_entry)
            tok_json["model"]["merges"] = new_merges

        # added_tokens ID 재매핑
        if "added_tokens" in tok_json:
            new_added = []
            for at in tok_json["added_tokens"]:
                old_id = at["id"]
                if old_id in id_remap:
                    at_copy = copy.deepcopy(at)
                    at_copy["id"] = id_remap[old_id]
                    new_added.append(at_copy)
            tok_json["added_tokens"] = new_added

        with open(output_dir / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(tok_json, f, ensure_ascii=False)

    # 5d. config.json 업데이트
    new_config = copy.deepcopy(config)
    new_config["vocab_size"] = new_vocab_size
    for key in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        old_id = config.get(key)
        if old_id is not None and old_id in id_remap:
            new_config[key] = id_remap[old_id]

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(new_config, f, ensure_ascii=False, indent=2)

    # 5e. tokenizer_config.json 업데이트
    new_tok_config = copy.deepcopy(tok_config)
    if "added_tokens_decoder" in new_tok_config:
        new_atd = {}
        for tid_str, info in new_tok_config["added_tokens_decoder"].items():
            old_id = int(tid_str)
            if old_id in id_remap:
                new_atd[str(id_remap[old_id])] = info
        new_tok_config["added_tokens_decoder"] = new_atd

    with open(output_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(new_tok_config, f, ensure_ascii=False, indent=2)

    # 5f. special_tokens_map.json 복사
    stm_path = model_dir / "special_tokens_map.json"
    if stm_path.exists():
        with open(stm_path, "r", encoding="utf-8") as f:
            stm = json.load(f)
        with open(output_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
            json.dump(stm, f, ensure_ascii=False, indent=2)

    # 5g. chat_template.jinja 복사
    chat_tpl_path = model_dir / "chat_template.jinja"
    if chat_tpl_path.exists():
        with open(chat_tpl_path, "r", encoding="utf-8") as f:
            content = f.read()
        with open(output_dir / "chat_template.jinja", "w", encoding="utf-8") as f:
            f.write(content)

    # ---- 6. 모델 저장 ----
    print(f"\nSaving pruned model to {output_dir}...")
    model.save_pretrained(output_dir)

    # generation_config.json 업데이트
    gen_config_path = output_dir / "generation_config.json"
    if gen_config_path.exists():
        with open(gen_config_path, "r", encoding="utf-8") as f:
            gen_config = json.load(f)
        for key in ["bos_token_id", "eos_token_id", "pad_token_id"]:
            old_id = gen_config.get(key)
            if old_id is not None and old_id in id_remap:
                gen_config[key] = id_remap[old_id]
        with open(gen_config_path, "w", encoding="utf-8") as f:
            json.dump(gen_config, f, ensure_ascii=False, indent=2)

    # ---- 7. 결과 요약 ----
    original_safetensors = list(model_dir.glob("*.safetensors"))
    pruned_safetensors = list(output_dir.glob("*.safetensors"))

    original_size = sum(f.stat().st_size for f in original_safetensors)
    pruned_size = sum(f.stat().st_size for f in pruned_safetensors)

    print("\n" + "=" * 60)
    print("Vocab Pruning Complete")
    print("=" * 60)
    print(f"  Vocab: {old_vocab_size:,} -> {new_vocab_size:,} ({len(prune_ids):,} removed)")
    print(f"  Embedding: [{old_vocab_size}, {embed_weight.shape[1]}] -> [{new_vocab_size}, {embed_weight.shape[1]}]")
    if original_size > 0 and pruned_size > 0:
        saved = original_size - pruned_size
        print(f"  Model size: {original_size / 1e9:.3f} GB -> {pruned_size / 1e9:.3f} GB ({saved / 1e6:.1f} MB saved)")
    print(f"  Output: {output_dir}")

    # 재매핑 정보 저장 (디버깅/검증용)
    remap_path = output_dir / "id_remap.json"
    with open(remap_path, "w") as f:
        json.dump(
            {
                "description": "old_token_id -> new_token_id mapping",
                "pruned_count": len(prune_ids),
                "pruned_scripts": sorted(PRUNE_SCRIPTS),
                "remap": {str(k): v for k, v in id_remap.items()},
            },
            f,
        )
    print(f"  ID remap saved to: {remap_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Vocab pruning: remove non-Korean/English script tokens"
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
        help="Output path (default: {model}-VocabPruned)",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.model}-VocabPruned"

    prune_model(
        model_path=args.model,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import re

UNANSWERABLE_REFERENCES = (
    "unanswerable",
    "주어진 정보로 답할 수 없다",
    "주어진 문서내에서 답할 수 있는 정보가 충분하지 않습니다",
    "문서 내에서 답할 수 없다",
)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'`“”‘’]", "", text)
    text = re.sub(r"[.,!?;:(){}\[\]<>]", "", text)
    return text.strip()


NORMALIZED_UNANSWERABLE_REFERENCES = frozenset(
    normalize_text(ref) for ref in UNANSWERABLE_REFERENCES
)


def is_unanswerable_reference(text: str) -> bool:
    return normalize_text(text) in NORMALIZED_UNANSWERABLE_REFERENCES


def extract_short_answer(text: str) -> str:
    s = text.strip()
    s = re.sub(r"^(답변은\s*다음과\s*같습니다\s*[:：]\s*)", "", s)
    s = re.sub(r"^(정답\s*[:：]\s*)", "", s)
    s = re.sub(r"^(a\s*[:：]\s*)", "", s, flags=re.IGNORECASE)
    s = s.split("\n")[0].strip()
    s = re.split(r"[.?!]\s+", s)[0].strip()
    return s if s else text.strip()

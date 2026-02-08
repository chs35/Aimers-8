from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalRecord:
    idx: int
    sample_id: str
    task: str
    question: str
    reference: str
    prediction: str
    short_ans: str
    judge_long: str | None
    judge_short: str | None
    is_correct: bool | None
    output_tokens: int


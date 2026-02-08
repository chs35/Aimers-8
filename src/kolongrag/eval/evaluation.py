from __future__ import annotations

from typing import Any

from .answer_text import extract_short_answer, is_unanswerable_reference
from .judge import OpenAIJudge
from .types import EvalRecord


def build_eval_records(
    samples: list[dict[str, Any]],
    outputs: list[Any],
    skip_eval: bool,
    judge: OpenAIJudge | None,
    safe_output_text_fn,
) -> tuple[list[EvalRecord], int]:
    records: list[EvalRecord] = []
    judge_errors = 0
    for idx, (sample, out) in enumerate(zip(samples, outputs)):
        prediction, output_tokens = safe_output_text_fn(out)
        short_ans = extract_short_answer(prediction)
        judge_long: str | None = None
        judge_short: str | None = None
        is_correct: bool | None = None
        answer = str(sample.get("answer", ""))

        if not skip_eval:
            try:
                if judge is None:
                    raise RuntimeError("judge instance is not initialized")
                judge_eval = judge.evaluate(
                    question=str(sample.get("question", "")),
                    answer=answer,
                    long_ans=prediction,
                    short_ans=short_ans,
                )
                judge_long = judge_eval.judge_long
                judge_short = judge_eval.judge_short
                is_correct = judge_eval.is_correct
            except Exception:
                judge_errors += 1
                judge_long = "error"
                judge_short = "error"
                is_correct = None

        records.append(
            EvalRecord(
                idx=idx,
                sample_id=str(sample.get("id", f"row_{idx:04d}")),
                task=str(sample.get("task", "")),
                question=str(sample.get("question", "")),
                reference=answer,
                prediction=prediction,
                short_ans=short_ans,
                judge_long=judge_long,
                judge_short=judge_short,
                is_correct=is_correct,
                output_tokens=output_tokens,
            )
        )
    return records, judge_errors


def accuracy(records: list[EvalRecord], task: str | None = None) -> float | None:
    scoped = records
    if task is not None:
        scoped = [r for r in scoped if r.task == task]
    judged = [r for r in scoped if r.is_correct is not None]
    if not judged:
        return None
    return sum(1 for r in judged if r.is_correct) / len(judged)


def unanswerable_accuracy(records: list[EvalRecord]) -> float | None:
    scoped = [r for r in records if is_unanswerable_reference(r.reference)]
    judged = [r for r in scoped if r.is_correct is not None]
    if not judged:
        return None
    return sum(1 for r in judged if r.is_correct) / len(judged)

from __future__ import annotations

import ast
import hashlib
import logging
import re
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class JudgeEval:
    judge_long: str
    judge_short: str
    is_correct: bool


class JudgeParseError(ValueError):
    pass


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:python)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_judge_response(raw: str) -> JudgeEval:
    cleaned = _strip_code_fence(raw)
    # ast.literal_eval이 직접 되면 사용, 아니면 {} 블록 추출 시도
    try:
        obj = ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        m = re.search(r"\{[^}]+\}", cleaned)
        if m is None:
            raise JudgeParseError(f"no dict found in judge output: {cleaned!r}")
        obj = ast.literal_eval(m.group())
    if not isinstance(obj, dict):
        raise JudgeParseError("judge output is not a dictionary")

    judge_long = str(obj.get("long_ans", "")).strip().lower()
    judge_short = str(obj.get("short_ans", "")).strip().lower()
    if judge_long not in {"correct", "incorrect"}:
        raise JudgeParseError("long_ans must be correct/incorrect")
    if judge_short not in {"correct", "incorrect"}:
        raise JudgeParseError("short_ans must be correct/incorrect")
    return JudgeEval(
        judge_long=judge_long,
        judge_short=judge_short,
        is_correct=(judge_long == "correct" or judge_short == "correct"),
    )


class OpenAIJudge:
    def __init__(
        self,
        model: str = "gpt-4o",
        max_retries: int = 3,
        system_prompt: str = "",
        user_template: str = "",
    ):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.max_retries = max_retries
        self.system_prompt = system_prompt
        self.user_template = user_template
        if not self.system_prompt.strip():
            raise ValueError("judge system prompt is empty")
        if not self.user_template.strip():
            raise ValueError("judge user template is empty")
        self.cache: dict[str, JudgeEval] = {}

    @staticmethod
    def _cache_key(question: str, answer: str, long_ans: str, short_ans: str) -> str:
        payload = f"{question}|{answer}|{long_ans}|{short_ans}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def evaluate(self, question: str, answer: str, long_ans: str, short_ans: str) -> JudgeEval:
        key = self._cache_key(question, answer, long_ans, short_ans)
        if key in self.cache:
            logger.debug("Judge cache hit")
            return self.cache[key]

        user_prompt = (
            self.user_template
            .replace("{question}", question)
            .replace("{answer}", answer)
            .replace("{long_ans}", long_ans)
            .replace("{short_ans}", short_ans)
        )
        last_error: Exception | None = None
        for i in range(self.max_retries):
            try:
                logger.debug("Calling judge model=%s attempt=%d/%d", self.model, i + 1, self.max_retries)
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = (resp.choices[0].message.content or "").strip()
                parsed = parse_judge_response(content)
                self.cache[key] = parsed
                return parsed
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.debug("Judge attempt failed: %s", exc)
                if i < self.max_retries - 1:
                    time.sleep(0.5 * (2**i))

        if last_error is not None:
            raise last_error
        raise RuntimeError("judge failed without explicit error")

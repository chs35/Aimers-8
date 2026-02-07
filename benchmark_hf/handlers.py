"""Ko-LongRAG 데이터셋 핸들러 (GPT-4o judge 평가)"""

import ast
import os
from abc import ABC, abstractmethod
from pathlib import Path

import yaml
from datasets import load_dataset
from openai import OpenAI

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 프롬프트 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"
with open(_PROMPTS_PATH, encoding="utf-8") as _f:
    PROMPTS = yaml.safe_load(_f)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 핸들러
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DatasetHandler(ABC):
    """데이터셋별 로드·프롬프트·평가 로직 추상 클래스"""

    name: str = ""
    dataset_id: str = ""

    @abstractmethod
    def load(self, subset=None, split=None, max_samples=None):
        ...

    @abstractmethod
    def format_messages(self, sample: dict) -> list[dict]:
        ...

    @abstractmethod
    def evaluate(self, sample: dict, prediction: str) -> dict:
        ...

    @abstractmethod
    def get_reference(self, sample: dict) -> str:
        ...


class KoLongRAGHandler(DatasetHandler):
    """
    LGAI-EXAONE/Ko-LongRAG (Korean Long-Context RAG Benchmark)
    - 600 samples (singledocQA 300 + multidocQA 300)
    - ~16.6% unanswerable
    - GPT-4o LLM-as-a-Judge 평가
    """

    name = "Ko-LongRAG"
    dataset_id = "LGAI-EXAONE/Ko-LongRAG"

    def __init__(self):
        self._judge = None

    def _get_judge(self) -> OpenAI:
        if self._judge is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY 환경변수가 설정되지 않았습니다. "
                    ".env 파일을 확인하세요."
                )
            self._judge = OpenAI()
        return self._judge

    def load(self, subset=None, split=None, max_samples=None, quick=False):
        ds = load_dataset(self.dataset_id, split=split or "test")
        if quick:
            ds = ds.filter(
                lambda x: x["task"] == "singledocQA"
                and x["id"].startswith("singledocqa_")
                and 100 <= int(x["id"].split("_")[1]) <= 199
            )
        elif subset in ("singledocQA", "multidocQA"):
            ds = ds.filter(lambda x: x["task"] == subset)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        return ds

    def format_messages(self, sample):
        task = sample.get("task", "singledocQA")
        titles = sample.get("titles", [])
        titles_str = ", ".join(titles) if isinstance(titles, list) else str(titles)

        template = PROMPTS["inference"].get(task, PROMPTS["inference"]["singledocQA"])
        user = template.format(
            titles=titles_str,
            context=sample["context"],
            question=sample["question"],
        )

        return [
            {"role": "system", "content": PROMPTS["inference"]["system"]},
            {"role": "user", "content": user.strip()},
        ]

    def evaluate(self, sample, prediction):
        question = sample["question"]
        answer = sample["answer"].strip()

        user_msg = PROMPTS["judge"]["user"].format(
            question=question,
            answer=repr([answer]),
            long_ans=prediction,
            short_ans=prediction,
        )

        try:
            response = self._get_judge().chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": PROMPTS["judge"]["system"].strip()},
                    {"role": "user", "content": user_msg.strip()},
                ],
                temperature=0.0,
                max_tokens=64,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            return {"correct": False, "judge_result": {"error": str(e)}}

        try:
            result = ast.literal_eval(raw)
            correct = (
                result.get("long_ans") == "correct"
                or result.get("short_ans") == "correct"
            )
        except (ValueError, SyntaxError):
            correct = False
            result = {"raw": raw, "parse_error": True}

        return {"correct": correct, "judge_result": result}

    def get_reference(self, sample):
        return sample["answer"]


DATASET_REGISTRY: dict[str, type[DatasetHandler]] = {
    "LGAI-EXAONE/Ko-LongRAG": KoLongRAGHandler,
}


def get_handler(dataset_id: str) -> DatasetHandler:
    """데이터셋 ID로 핸들러 인스턴스 반환"""
    if dataset_id in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_id]()
    for key, cls in DATASET_REGISTRY.items():
        short = key.split("/")[-1].lower()
        if dataset_id.lower() == short or dataset_id.lower() in key.lower():
            return cls()
    raise ValueError(
        f"지원하지 않는 데이터셋: {dataset_id}\n"
        f"지원 목록: {', '.join(DATASET_REGISTRY.keys())}"
    )

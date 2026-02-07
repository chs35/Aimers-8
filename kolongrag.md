---
pretty_name: Ko-LongRAG
license: cc-by-nc-4.0
task_categories:
- question-answering
language:
- ko
size_categories:
- 1K<n<10K
dataset_info:
  features:
    - name: id
      dtype: string
    - name: titles
      sequence: string
    - name: context
      dtype: string
    - name: question
      dtype: string
    - name: answer
      dtype: string
    - name: prompt
      dtype: string
    - name: task
      dtype: string
configs:
  - config_name: ko_longrag
    data_files:
      - split: test
        path: data/test-*.parquet
---

<p align="center">
  <img src="image.png" alt="Ko-LongRAG" width="50%">
</p>

## **Abstract**

The rapid advancement of large language models (LLMs) significantly enhances long-context Retrieval-Augmented Generation (RAG), yet existing benchmarks focus primarily on English. This leaves low-resource languages without comprehensive evaluation frameworks, limiting their progress in retrieval-based tasks. To bridge this gap, we introduce **Ko-LongRAG**, the first **Ko**rean **long**-context **RAG** benchmark. Unlike conventional benchmarks that depend on external retrievers, Ko-LongRAG adopts a retrieval-free approach designed around Specialized Content Knowledge (SCK), enabling controlled and high-quality QA pair generation without the need for an extensive retrieval infrastructure. Our evaluation shows that o1 model achieves the highest performance among proprietary models, while EXAONE 3.5 leads among open-sourced models. Additionally, various findings confirm Ko-LongRAG as a reliable benchmark for assessing Korean long-context RAG capabilities and highlight its potential for advancing multilingual RAG research.

## **Dataset Details**

- **Composition**: 600 total examples  
  - **singledocQA** (300): extraction-style QA grounded in a single document  
  - **multidocQA** (300): comparison/bridge reasoning across documents within a domain cluster  
- **Fields (schema)**: `id`, `titles` (list[str]), `context` (str), `question` (str), `answer` (str), `prompt` (str), `task` (str; `"singledocQA"` or `"multidocQA"`)  
- **Context lengths (approx.)**: single ≈ 2,915 tokens; multi ≈ 14,092 tokens  
- **Unanswerable share**: ≈ 16.6%  

> For construction protocol, prompts, human verification checks, and extended statistics, please refer to the accompanying paper and repository notes. Guidance for dataset cards and their structure follows the Hugging Face documentation.

## **Usage**

```python
from datasets import load_dataset

ds = load_dataset("LGAI-EXAONE/Ko-LongRAG", split="test")
print(ds)
print(ds[0]["task"], ds[0]["question"])

```
## **Data Fields**

- `id` — unique identifier (string)  
- `titles` — list of section titles included in the context (list[string])  
- `context` — concatenated long passages (string)  
- `question` — Korean question (string)  
- `answer` — short answer string (or "unanswerable" when appropriate)  
- `prompt` — prompt used during data creation/evaluation (string, optional)  
- `task` — `"singledocQA"` or `"multidocQA"`

## **Benchmark Design (Brief)**

- **Domain-aware clustering** groups documents by topic/keywords to form long contexts suitable for QA.  
- **Question generation** distinguishes extraction-style (single) from cross-document comparison/bridge (multi).  
- **Quality control** uses a human checklist to validate question–answer–context consistency.  
- **Unanswerable cases** are systematically included to assess reliability and calibration under retrieval failure.

## **License**

This dataset is released under **CC BY-NC 4.0** (Attribution–NonCommercial 4.0). Please review the Creative Commons BY-NC 4.0 terms before reuse.  

**Additional Terms (model usage):**  
This dataset was created using **OpenAI GPT-4o**. In addition to the license above, **the dataset is subject to OpenAI’s Terms of Use and related policies** governing use of model outputs. This means the dataset **must not be used to develop competing models** where such use conflicts with those terms.  
> This dataset is licensed under CC BY-NC 4.0, and is subject to the Terms of Use of the model (OpenAI GPT-4o) used in its creation.

## **Citation**

```bibtex
@misc{KoLongRAG-2025,
  title  = {Ko-LongRAG: A Korean Long-Context RAG Benchmark Built with a Retrieval-Free Approach},
  author = {Ko-LongRAG Authors},
  year   = {2025},
  note   = {Preprint},
}
```

## **Contact**
For questions or issues, please open an issue on the dataset repository or contact the maintainers.
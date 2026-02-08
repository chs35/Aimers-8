# script.py 벤치마크 사용법 (Ko-LongRAG MVP)

## 1. 목적
`script.py`는 Ko-LongRAG 벤치마크를 로컬에서 실행해 아래 산출물을 생성합니다.

- 추론/평가 상세 CSV
- 제출용 CSV (`submission.csv`)
- 실행 메타데이터 JSON

## 2. 실행 진입점
프로젝트 루트에서 실행합니다.

```bash
python script.py --model-path <MODEL_DIR_NAME>
```

예:

```bash
python script.py --model-path EXAONE-4.0-1.2B
```

## 3. CLI 옵션
현재 지원되는 옵션은 아래만 제공합니다.

- `--model-path`: `models/` 하위 모델 디렉토리명 (기본값: `EXAONE-4.0-1.2B`)
- `--prompt-config`: 프롬프트/런타임 YAML 경로 (기본값: `src/kolongrag/config/prompt.yaml`)
- `--quick [N]`: `singledocQA` 앞쪽 N개만 실행 (숫자 생략 시 100)
- `--single-only`: `singledocQA`만 실행
- `--multi-only`: `multidocQA`만 실행
- `--skip-eval`: GPT-4o Judge 채점을 건너뜀

제약:

- `--single-only`와 `--multi-only`는 동시에 사용 불가
- `--quick`과 `--multi-only`는 동시에 사용 불가
- `--model-path`에는 경로가 아닌 디렉토리명만 허용 (`models/` prefix 입력 금지)

## 4. 고정 정책 (README 준수)
아래 항목은 코드에서 고정되어 있으며 CLI로 변경할 수 없습니다.

- dataset: `LGAI-EXAONE/Ko-LongRAG`
- split: `test`
- tokenizer: `local_files_only=True`
- vLLM
  - `tensor_parallel_size=1`
  - `gpu_memory_utilization=0.85`
  - `batch_size=auto`
  - `max_gen_toks=16384`
  - `apply_chat_template=true`

## 5. YAML 설정 항목
`src/kolongrag/config/prompt.yaml`의 `runtime` 섹션에서 아래만 설정합니다.

- `judge_model`
- `judge_max_retries`
- `temperature`
- `top_p`
- `results_dir`
- `output_dir`

예:

```yaml
runtime:
  judge_model: "gpt-4o"
  judge_max_retries: 3
  temperature: 0.0
  top_p: 1.0
  results_dir: "results"
  output_dir: "output"
```

## 6. 실행 예시
전체 실행:

```bash
python script.py --model-path EXAONE-4.0-1.2B
```

`singledocQA` 100개 quick:

```bash
python script.py --model-path EXAONE-4.0-1.2B --quick --skip-eval
```

`multidocQA`만 추론(채점 스킵):

```bash
python script.py --model-path EXAONE-4.0-1.2B --multi-only --skip-eval
```

## 7. 출력 파일
기본 `runtime` 기준으로 아래 파일이 생성됩니다.

- `results/<run_name>.json`
- `results/<run_name>.csv`
- `output/<run_name>_submission.csv`
- `output/submission.csv`
- `logs/<timestamp>_<model_dir_name>.log`

`<run_name>`은 `YYYYMMDD_HHMMSS_<model_dir_name>` 형식으로 자동 생성됩니다.

## 8. 로그 정책
- 벤치마크 실행 시 DEBUG 레벨 로그를 기본 출력합니다.
- 코어 엔진(vLLM/transformers/datasets 등) 로그도 DEBUG 레벨로 출력되도록 설정됩니다.
- CLI에 출력된 `stdout/stderr`는 모두 `logs/` 파일로 동시 저장됩니다.

## 9. Judge 사용 시 환경 변수
`--skip-eval` 없이 실행하면 OpenAI Judge를 사용하므로 환경 변수가 필요합니다.

```bash
export OPENAI_API_KEY=...
```

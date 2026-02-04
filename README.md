# EXAONE-4.0-1.2B 모델 경량화 해커톤
> [대회 홈페이지↗](https://dacon.io/competitions/official/236673/overview/description) <br>
> 26년 2월 26일 오전 10시 마감

## 주제
LLM 경량화 (Large Language Model Compression)

## 설명
본 해커톤의 기본 모델인 `EXAONE-4.0-1.2B` 모델을 대상으로 한 경량화를 수행해야합니다. <br>
참가자는 모델 구조와 특성을 분석한 뒤, 다양한 경량화 기법과 필요 시 추가 학습·튜닝을 적용하여 모델 크기, 추론 속도, 성능 간의 균형을 최적화하는 것을 목표로 합니다. <br>
경량화된 모델은 vLLM 기반 추론 엔진 환경에서 평가되며, 동일한 추론 파이프라인 조건 하에서 모델 성능과 효율을 비교하게 됩니다. <br>
이를 통해 단순한 파라미터 감소에 국한되지 않고, 실제 서비스 환경에서의 추론 효율과 성능을 함께 고려한 의미 있는 경량화 전략을 설계·검증하는 경험을 제공하고자 합니다.

## 대회 규칙
### 대회 지정 기본 모델
`LGAI-EXAONE/EXAONE-4.0-1.2B` [허깅페이스 링크↗](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B) <br>
참가자는 반드시 기본 모델인 `EXAONE-4.0-1.2B` 모델만을 사용해야 하며, 이 외의 다른 모델을 어떠한 방식으로든 사용하는 것은 허용되지 않으며 규칙 위반에 해당합니다.

### vLLM 커스컴 수정 불가
vLLM 추론 엔진을 고정된 환경으로 사용합니다. <br>
참가자는 vLLM 소스 코드, CUDA 커널, 내부 로직을 수정할 수 없습니다.

### 외부 데이터 사용 가능
참가자가 모델 학습 또는 튜닝이 필요한 경우 외부 데이터 사용이 가능합니다. <br>
단, 최소한 비상업적 이용이 허용된 라이선스(CC BY-NC, CC0 등)로 배포된 외부 데이터만 사용할 수 있습니다. 해당 조건을 충족하지 않는 외부 데이터는 사용이 불가능합니다.

### 모델 경량화 및 성능 유지 및 향상을 위한 추가 학습/튜닝
참가자는 제공된 기본 모델인 `EXAONE-4.0-1.2B` 모델을 대상으로, 모델 경량화뿐만 아니라 기본 모델 성능의 유지 또는 향상을 목적으로 추가 학습 또는 파라미터 튜닝을 수행할 수 있습니다. <br>
모델 경량화를 위한 기법에는 제한이 없으며, 적용 방식과 무관하게 최종 결과물이 허깅페이스(Hugging Face) 표준 형식의 모델 가중치 및 설정 파일로 제출 가능한 경우에 한해 평가 대상이 됩니다. <br>
제출된 모델은 운영진이 제공하는 동일한 vLLM 추론 환경에서 평가됩니다.

## 리더보드
### 평가 산식
1. 전체 점수 산식 (Score): <br>
$$Score = \max(0.5 \times PerfNorm_{model} + 0.5 \times SpeedNorm_{model}, 0)$$

2. 성능 정규화 지표 ($PerfNorm_{model}$): <br>
기본 모델(EXAONE-4.0-1.2B) 대비 성능 비율을 나타냅니다. <br>
$$PerfNorm_{model} = \frac{Perf_{model}}{Perf_{base\_model}}$$

3. 속도 정규화 지표 ($SpeedNorm_{model}$): <br>
기본 모델 대비 토큰당 추론 시간 감소 비율을 나타냅니다. <br>
$$SpeedNorm_{model} = 1 - \frac{\left(\frac{Time_{model}}{Tokens_{model}}\right)}{\left(\frac{Time_{base\_model}}{Tokens_{base\_model}}\right)}$$

### 테스트 데이터
Public score는 전체 테스트 데이터 100%로 평가되며 테스트 데이터는 비공개 벤치셋으로 구성되어 있습니다.

### 기타 안내사항
제출 탭에 표시되는 **시간**은 평가 서버에서 제출물이 실행되는 **전체 추론 실행 시간**입니다. <br>
해당 시간에는 모델 로드, 평가 데이터에 대한 추론, 결과값 파싱 등 모든 처리 과정이 포함되며, 이 **전체 추론 실행 시간**은 **규칙으로 지정한 시간 제한(20분)** 이내에 완료되어야 합니다. <br>
반면, 평가 산식에 반영되는 추론 시간은 모델이 평가용 벤치마크 데이터셋에 대해 실제로 추론을 수행한 시간만을 기준으로 측정한 **토큰 당 추론 시간**입니다.

## 제출 가이드
본 대회는 `submit.zip` 파일을 제출하는 방식의 **코드 제출 대회**로 진행됩니다. <br>
아래와 같은 구조로 `submit.zip`을 구성하여 제출해야 합니다. <br>
아래의 구조와 동일하고 디렉토리 명과 파일 명을 모두 일치 시켜야합니다.

### 참가자 제출 파일 구조 (`submit.zip`)
```
submit.zip
 └── model/        # 허깅 페이스(HF) 방식의 모델 가중치 파일 디렉토리
       └── (예: config.json, model.safetensors 등)
```
`submit.zip` 내 구조는 반드시 일치해야하며, 추가 최상위 폴더가 `zip` 구조 내 존재하는 경우 등 구조가 불일치하는 경우 설치 오류가 발생합니다.

### 평가 서버에서 추가되는 항목
제출 시, 평가 서버에서 참가자가 제출한 `submit.zip` 파일에는 아래 항목이 자동으로 추가됩니다.
```
submit.zip
├── model/        # 참가자 구성
├── script.py       # 평가에 사용될 추론 코드 (운영진 고정, 평가 시 자동 생성)
├── requirements.txt   # 평가에 사용될 환경 구성 (운영진 고정, 평가 시 자동 생성)
├── data/         # 평가에 사용될 테스트 데이터 (운영진 고정, 평가 시 자동 생성)
└── output/submission.csv        # 참가자 모델의 추론 결과가 저장되는 경로 (운영진 고정, 평가 시 자동 생성)
```
`model/` 디렉토리만 참가자 제출 파일(`zip`)에서 구성할 수 있습니다.

### 주의사항
1. 제출 파일(`zip`) 용량 제한: 최대 10GB 이내 (*압축해제 후 32GB 이내) <br>

2. 전체 추론 코드 실행 시간: 최대 20분 이내 (시간 초과 시 제출 오류)

### 평가 서버 정보
1. 평가 서버 사양: <br>
    ```
    OS : Ubuntu 22.04.5 LTS
    CPU: 6 vCPU
    RAM: 28GB
    GPU : L4 (VRAM 22.4GiB)
    Python : 3.11.14
    CUDA : 12.8
    ```

2. 평가 서버 기본 설치 패키지(라이브러리) 목록: <br>
아래에 명시된 패키지(라이브러리)는 평가 서버에 사전 설치된 기본 환경이며, 참가자는 `requirements.txt`를 제출하거나 패키지 버전을 변경할 수 없습니다. <br>
**모든 제출물은 아래에 안내된 고정된 라이브러리 환경에서 그대로 실행 및 채점**되며, 해당 환경과 호환되지 않는 코드 또는 다른 버전의 라이브러리를 전제로 한 구현은 실행 오류가 발생할 수 있습니다. <br>
따라서, **참가자는 평가 서버에 기본 설치된 패키지(라이브러리 및 버전)를 기준으로 코드를 구현해야 하며, 아래 환경 외의 추가 라이브러리나 특정 버전에 의존하는 구현은 지원되지 않습니다.**
    ```
    # 설치 패키지(라이브러리)
    torch==2.9.0+cu128 
    numpy==2.2.6 
    pandas==2.3.3 
    transformers==4.57.3 
    tokenizers==0.22.1 
    accelerate==1.10.1 
    datasets==4.4.1 
    huggingface-hub==0.36.0 
    safetensors==0.7.0 
    sentencepiece==0.2.1 
    protobuf==6.33.2 
    evaluate==0.4.6 
    rouge_score==0.1.2 
    sacrebleu==2.5.1 
    tqdm==4.67.1 
    regex==2025.11.3 
    nltk==3.9.2 
    compressed-tensors==0.13.0 
    math-verify==0.8.0 
    antlr4-python3-runtime==4.11.0 
    sympy==1.14.0 
    langdetect==1.0.9 
    immutabledict==4.2.2 
    torch-c-dlpack-ext==0.1.4
    vllm==0.14.1
    ```
    ```
    # 설치 시스템 패키지
    python3.11 
    python3.11-distutils 
    python3-pip 
    python3.11-dev 
    build-essential 
    git 
    git-lfs
    ninja-build 
    libomp-dev 
    libblas3 
    liblapack3 
    gfortran 
    libatlas-base-dev 
    curl 
    wget 
    ca-certificates 
    unzip 
    procps
    cmake
    tzdata
    libxcb1
    ```

3. 모델 서빙 스펙: <br>
본 대회는 참가자가 제출한 `EXAONE-4.0-1.2B` 경량화 모델을 대상으로, 고정된 vLLM 서빙 환경에서 추론을 수행하여 평가합니다. <br>
추론 엔진 (Inference Engine):`vLLM (version: 0.14.1)` <br>
Model Interface: `HuggingFace AutoModelForCausalLM 호환`<br>
제출된 모델은 다음 호출이 가능해야 하며, 토크나이저는 모델과 함께 제공되어야 합니다. 아래 vLLM Serving 옵션은 채점 서버에서 고정 적용됩니다.
    ```python
    # tokenizer
    AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    # model
    AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True)
    ```
    ```
    # vLLM Serving 옵션
    tensor_parallel_size = 1
    gpu_memory_utilization = 0.85
    batch_size = auto
    max_gen_toks = 16384
    apply_chat_template = true
    ```

4. 제출 주의사항: <br>
제출 시 발생하는 오류의 종류는 두 가지로 정의되며, 일일 제출 횟수 반영에 대한 기준이 다르므로 반드시 숙지하여 진행해야 합니다. <br>
**설치 오류 : 제출하는 submit.zip 내부 구조가 불일치한 경우** <br>
예) submit.zip 파일 안에 ./model 폴더로 되어 있지 않은 경우, 폴더명이 model이 아닌 경우 <br>
**제출 오류 : script.py 코드 실행 후 발생하는 모든 오류** <br>
예) tokenizer 로드가 안되는 경우, vLLM 모델 로딩에 실패한 경우, 전체 추론 시간이 20분을 초과하는 경우 등 <br>
**평가 서버 환경은 인터넷 접속이 불가능하므로, 패키지 설치 이후 외부 다운로드가 필요한 코드나 모델은 작동하지 않습니다.**
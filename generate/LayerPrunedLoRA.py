"""
LayerPrunedLoRA: Layer Pruning (Shortened LLaMA) + LoRA Recovery

파이프라인:
  1. 캘리브레이션 데이터로 각 레이어의 중요도 점수 계산
  2. 중요도가 낮은 레이어 제거 (depth pruning)
  3. LoRA fine-tuning으로 프루닝 손실 복구
  4. LoRA 가중치를 base 모델에 병합
  5. 최종 모델 저장

참고 논문:
  - Shortened LLaMA (arXiv 2402.02834): Depth Pruning for LLMs
    · PPL 기반: 레이어 제거 시 perplexity 변화로 중요도 측정
    · BI 기반: Block Influence (입출력 cosine similarity)로 중요도 측정
    · <5B 모델에서는 전체 블록 단위 제거가 최적
    · LoRA retraining: rank=8, lr=1e-4, epochs=2, all projection modules

GPU 호환: L4(sm_89) 이상 모든 GPU에서 동작
필요 패키지: pip install peft  (Dockerfile에 미포함)
"""

import os
import json
import math
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    import subprocess
    print("peft 패키지가 없습니다. 설치 중...")
    subprocess.check_call(["pip", "install", "peft", "-q"])
    from peft import LoraConfig, get_peft_model, TaskType

# ╔══════════════════════════════════════════════════════════════╗
# ║  HYPERPARAMETERS — 이 섹션만 수정하면 됩니다                  ║
# ╚══════════════════════════════════════════════════════════════╝

# ── 모델 & 데이터셋 ──
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
SAVE_DIR = "EXAONE-4.0-1.2B-LayerPruned-LoRA"

# ── Layer Pruning (Shortened LLaMA) ──
NUM_LAYERS_TO_REMOVE = 3           # 제거할 레이어 수 (EXAONE 1.2B = 30 레이어)
IMPORTANCE_METRIC = "ppl"          # "ppl" (정확, 느림) 또는 "bi" (빠름, Block Influence)
PROTECT_FIRST = 2                  # 앞쪽 보호 레이어 수 (PPL 사용 시 0으로 설정 가능)
PROTECT_LAST = 2                   # 뒤쪽 보호 레이어 수 (PPL 사용 시 0으로 설정 가능)
NUM_CALIB_SAMPLES_PRUNING = 10     # 프루닝 중요도 계산용 샘플 수 (논문: 10이면 충분)
MAX_SEQ_LEN_PRUNING = 128          # 프루닝용 최대 시퀀스 길이

# ── LoRA Fine-tuning (Recovery, Shortened LLaMA 논문 설정) ──
LORA_ENABLED = True                # LoRA 파인튜닝 활성화/비활성화
LORA_RANK = 8                      # LoRA rank (논문: 8)
LORA_ALPHA = 16                    # LoRA alpha (보통 rank의 2배)
LORA_DROPOUT = 0.05                # LoRA dropout
LORA_LR = 1e-4                     # 학습률 (논문: 0.0001)
LORA_EPOCHS = 2                    # 에폭 수 (논문: 2)
LORA_BATCH_SIZE = 4                # per-device 배치 크기 (GPU 메모리에 따라 조절)
LORA_GRAD_ACCUM = 16               # gradient accumulation (effective batch = 4*16=64, 논문: 64)
LORA_TARGET_MODULES = [            # LoRA 적용 대상 모듈 (논문: 모든 projection)
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
NUM_CALIB_SAMPLES_LORA = 1000      # LoRA 학습용 샘플 수
MAX_SEQ_LEN_LORA = 512             # LoRA 학습용 최대 시퀀스 길이

# ── 환경 ──
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ╔══════════════════════════════════════════════════════════════╗
# ║  LAYER IMPORTANCE FUNCTIONS                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def prepare_calib_texts(tokenizer, dataset_id, dataset_split, num_samples):
    """캘리브레이션 텍스트 준비 (채팅 템플릿 적용)"""
    raw_ds = load_dataset(dataset_id, split=dataset_split, streaming=True)
    samples = list(raw_ds.take(num_samples))
    texts = []
    for sample in samples:
        text = tokenizer.apply_chat_template(
            sample["conversations"],
            add_generation_prompt=True,
            tokenize=False,
        )
        texts.append(text)
    return texts


def compute_perplexity(model, tokenizer, texts, max_seq_len):
    """주어진 텍스트들에 대한 perplexity 계산"""
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=max_seq_len,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        num_tokens = inputs["input_ids"].numel()
        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens

    return math.exp(total_loss / total_tokens)


def compute_ppl_importance(model, tokenizer, calib_texts, max_seq_len):
    """
    PPL 기반 레이어 중요도 (Shortened LLaMA 논문 방식)

    각 레이어를 하나씩 제거하고 perplexity를 측정.
    PPL이 낮을수록 = 해당 레이어 없어도 됨 = 덜 중요.
    """
    num_layers = len(model.model.layers)
    original_layers = list(model.model.layers)
    scores = {}

    print(f"    PPL 기반 중요도 계산 ({num_layers}개 레이어, 각 레이어마다 forward pass)...")
    for skip_idx in range(num_layers):
        # skip_idx 레이어를 제외한 새 ModuleList 생성
        new_layers = torch.nn.ModuleList(
            [original_layers[i] for i in range(num_layers) if i != skip_idx]
        )
        model.model.layers = new_layers

        ppl = compute_perplexity(model, tokenizer, calib_texts, max_seq_len)
        scores[skip_idx] = ppl
        print(f"      Layer {skip_idx:2d} 제거 시 PPL: {ppl:.2f}")

    # 원래 레이어 복원
    model.model.layers = torch.nn.ModuleList(original_layers)
    return scores


def compute_bi_importance(model, tokenizer, calib_texts, max_seq_len):
    """
    Block Influence (BI) 기반 레이어 중요도

    BI(i) = 1 - cosine_similarity(layer_input, layer_output)
    BI가 낮을수록 = 입출력이 비슷함 = 레이어가 거의 아무것도 안 함 = 덜 중요.
    단일 forward pass로 계산 가능 (PPL 대비 빠름).
    """
    num_layers = len(model.model.layers)
    layer_inputs = {}
    hooks = []

    # 각 레이어에 hook 등록
    for idx, layer in enumerate(model.model.layers):
        def make_hook(layer_idx):
            def hook_fn(module, inp, out):
                # inp: tuple, 첫 번째 원소가 hidden_states
                h_in = inp[0].detach().float()
                h_out = (out[0] if isinstance(out, tuple) else out).detach().float()

                # 토큰별 cosine similarity 계산
                cos_sim = F.cosine_similarity(h_in, h_out, dim=-1)  # (batch, seq_len)
                bi = 1.0 - cos_sim.mean().item()

                if layer_idx not in layer_inputs:
                    layer_inputs[layer_idx] = []
                layer_inputs[layer_idx].append(bi)
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    # Forward pass (여러 샘플)
    print(f"    BI 기반 중요도 계산 (단일 forward pass)...")
    for text in calib_texts:
        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=max_seq_len,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)

    # Hook 제거
    for h in hooks:
        h.remove()

    # 평균 BI 점수
    scores = {}
    for idx in range(num_layers):
        avg_bi = sum(layer_inputs[idx]) / len(layer_inputs[idx])
        scores[idx] = avg_bi
        print(f"      Layer {idx:2d} BI score: {avg_bi:.6f}")

    return scores


def select_layers_to_remove(scores, num_to_remove, num_layers, protect_first, protect_last):
    """중요도 점수 기반으로 제거할 레이어 선택"""
    # 보호 레이어 제외
    prunable = {
        idx: score for idx, score in scores.items()
        if idx >= protect_first and idx < num_layers - protect_last
    }

    if len(prunable) < num_to_remove:
        raise ValueError(
            f"보호 레이어를 제외하면 {len(prunable)}개만 프루닝 가능하지만, "
            f"{num_to_remove}개를 제거하려 합니다. "
            f"PROTECT_FIRST/PROTECT_LAST를 줄이거나 NUM_LAYERS_TO_REMOVE를 줄이세요."
        )

    # 점수가 낮은 순서대로 정렬 (낮은 점수 = 덜 중요 = 먼저 제거)
    sorted_layers = sorted(prunable.items(), key=lambda x: x[1])
    layers_to_remove = [idx for idx, _ in sorted_layers[:num_to_remove]]
    layers_to_remove.sort()

    return layers_to_remove


def prune_layers(model, layers_to_remove):
    """모델에서 지정된 레이어를 제거하고 config 업데이트 (per-layer 설정 포함)"""
    original_num = len(model.model.layers)
    original_layers = list(model.model.layers)

    layers_to_keep = [i for i in range(original_num) if i not in layers_to_remove]
    new_layers = torch.nn.ModuleList([original_layers[i] for i in layers_to_keep])
    model.model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)

    # per-layer config 필드 업데이트 (예: layer_types, head_dim 등)
    # EXAONE 등 일부 모델은 레이어별 설정(layer_types)을 가지고 있음
    # num_hidden_layers와 길이가 일치해야 vLLM이 정상 로드함
    config_dict = model.config.to_dict()
    for key, value in config_dict.items():
        if isinstance(value, list) and len(value) == original_num:
            pruned_value = [value[i] for i in layers_to_keep]
            setattr(model.config, key, pruned_value)
            print(f"    config.{key} 업데이트: {original_num}개 → {len(pruned_value)}개")

    return layers_to_keep


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE                                              ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":

    # ===========================================================
    # Step 1: 모델 및 토크나이저 로드
    # ===========================================================
    print("=" * 60)
    print("Step 1: 모델 및 토크나이저 로드")
    print("=" * 60)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    num_layers = len(model.model.layers)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  모델: {MODEL_ID}")
    print(f"  레이어 수: {num_layers}")
    print(f"  총 파라미터: {total_params:,}")
    print()

    # ===========================================================
    # Step 2: 레이어 중요도 계산
    # ===========================================================
    print("=" * 60)
    print(f"Step 2: 레이어 중요도 계산 (metric: {IMPORTANCE_METRIC})")
    print("=" * 60)

    calib_texts = prepare_calib_texts(
        tokenizer, DATASET_ID, DATASET_SPLIT, NUM_CALIB_SAMPLES_PRUNING
    )
    print(f"  캘리브레이션 샘플 수: {len(calib_texts)}")

    if IMPORTANCE_METRIC == "ppl":
        importance_scores = compute_ppl_importance(
            model, tokenizer, calib_texts, MAX_SEQ_LEN_PRUNING
        )
    elif IMPORTANCE_METRIC == "bi":
        importance_scores = compute_bi_importance(
            model, tokenizer, calib_texts, MAX_SEQ_LEN_PRUNING
        )
    else:
        raise ValueError(f"알 수 없는 metric: {IMPORTANCE_METRIC}. 'ppl' 또는 'bi'를 사용하세요.")
    print()

    # ===========================================================
    # Step 3: 레이어 프루닝
    # ===========================================================
    print("=" * 60)
    print(f"Step 3: 레이어 프루닝 ({NUM_LAYERS_TO_REMOVE}개 제거)")
    print("=" * 60)

    layers_to_remove = select_layers_to_remove(
        importance_scores, NUM_LAYERS_TO_REMOVE,
        num_layers, PROTECT_FIRST, PROTECT_LAST,
    )
    print(f"  보호 레이어: 앞 {PROTECT_FIRST}개, 뒤 {PROTECT_LAST}개")
    print(f"  제거 대상 레이어: {layers_to_remove}")

    # 제거 대상 레이어의 점수 출력
    for idx in layers_to_remove:
        label = "PPL" if IMPORTANCE_METRIC == "ppl" else "BI"
        print(f"    Layer {idx:2d}: {label} = {importance_scores[idx]:.4f}")

    layers_kept = prune_layers(model, layers_to_remove)
    new_num_layers = len(model.model.layers)
    pruned_params = sum(p.numel() for p in model.parameters())

    print(f"\n  프루닝 결과:")
    print(f"    {num_layers} → {new_num_layers} 레이어")
    print(f"    {total_params:,} → {pruned_params:,} 파라미터")
    print(f"    파라미터 감소율: {(1 - pruned_params / total_params) * 100:.1f}%")
    print(f"    유지된 레이어 원본 인덱스: {layers_kept}")
    print()

    # ===========================================================
    # Step 4: LoRA Fine-tuning (프루닝 손실 복구)
    # ===========================================================
    if LORA_ENABLED:
        print("=" * 60)
        print("Step 4: LoRA Fine-tuning (프루닝 손실 복구)")
        print("=" * 60)

        # 4-1. LoRA 학습 데이터 준비
        print("  학습 데이터 준비...")
        raw_ds_lora = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)
        lora_data = list(raw_ds_lora.take(NUM_CALIB_SAMPLES_LORA))
        lora_ds = Dataset.from_list(lora_data)

        def preprocess_lora(example):
            return {
                "text": tokenizer.apply_chat_template(
                    example["conversations"],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            }

        lora_ds = lora_ds.map(preprocess_lora)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize_fn(example):
            return tokenizer(
                example["text"],
                truncation=True,
                max_length=MAX_SEQ_LEN_LORA,
                padding=False,
            )

        lora_ds = lora_ds.map(tokenize_fn, remove_columns=lora_ds.column_names)
        print(f"  학습 샘플 수: {len(lora_ds)}")

        # 4-2. LoRA 적용
        print(f"  LoRA 적용 (rank={LORA_RANK}, alpha={LORA_ALPHA}, modules={LORA_TARGET_MODULES})...")
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # 4-3. 학습
        model.config.use_cache = False  # 학습 시 KV cache 비활성화
        model.enable_input_require_grads()  # gradient_checkpointing + LoRA 호환성

        training_args = TrainingArguments(
            output_dir="./lora_recovery_output",
            per_device_train_batch_size=LORA_BATCH_SIZE,
            gradient_accumulation_steps=LORA_GRAD_ACCUM,
            num_train_epochs=LORA_EPOCHS,
            learning_rate=LORA_LR,
            bf16=True,
            gradient_checkpointing=True,   # activation 메모리 절약 (L4 24GB 안전)
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lora_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        print("  LoRA 학습 시작...")
        trainer.train()
        print("  LoRA 학습 완료!")

        # 4-4. LoRA 가중치를 base 모델에 병합
        print("  LoRA 가중치 병합 (merge_and_unload)...")
        model = model.merge_and_unload()
        model.config.use_cache = True  # 추론용으로 다시 활성화

        # 학습 임시 파일 정리
        import shutil
        if os.path.exists("./lora_recovery_output"):
            shutil.rmtree("./lora_recovery_output")

        print("  LoRA 병합 완료! 모델이 복구되었습니다.")
        print()
    else:
        print("=" * 60)
        print("Step 4: LoRA Fine-tuning 건너뜀 (LORA_ENABLED=False)")
        print("=" * 60)
        print()

    # ===========================================================
    # Step 5: 결과 저장
    # ===========================================================
    print("=" * 60)
    print("Step 5: 결과 저장")
    print("=" * 60)

    # .gitignore 처리
    GITIGNORE_PATH = ".gitignore"
    if os.path.exists(GITIGNORE_PATH):
        with open(GITIGNORE_PATH, "r") as f:
            gitignore_content = f.read()
        if SAVE_DIR not in gitignore_content:
            with open(GITIGNORE_PATH, "a") as f:
                f.write(f"\n{SAVE_DIR}\n")
    else:
        with open(GITIGNORE_PATH, "w") as f:
            f.write(f"{SAVE_DIR}\n")

    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    # 메타데이터 저장
    pruning_meta = {
        "method": "Shortened LLaMA (Layer Pruning) + LoRA Recovery",
        "original_model": MODEL_ID,
        "importance_metric": IMPORTANCE_METRIC,
        "original_num_layers": num_layers,
        "pruned_num_layers": new_num_layers,
        "layers_removed": layers_to_remove,
        "layers_kept": layers_kept,
        "importance_scores": {str(k): v for k, v in importance_scores.items()},
        "protect_first": PROTECT_FIRST,
        "protect_last": PROTECT_LAST,
        "lora_enabled": LORA_ENABLED,
        "lora_rank": LORA_RANK if LORA_ENABLED else None,
        "lora_alpha": LORA_ALPHA if LORA_ENABLED else None,
        "lora_epochs": LORA_EPOCHS if LORA_ENABLED else None,
        "lora_lr": LORA_LR if LORA_ENABLED else None,
        "lora_samples": NUM_CALIB_SAMPLES_LORA if LORA_ENABLED else None,
        "original_params": total_params,
        "pruned_params": pruned_params,
        "param_reduction_pct": round((1 - pruned_params / total_params) * 100, 1),
    }
    meta_path = os.path.join(SAVE_DIR, "pruning_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(pruning_meta, f, indent=2, ensure_ascii=False)

    print(f"  모델 저장 완료: {SAVE_DIR}/")
    print(f"  프루닝 메타데이터: {meta_path}")

    # ===========================================================
    # 최종 요약
    # ===========================================================
    print()
    print("=" * 60)
    print("최종 요약")
    print("=" * 60)
    print(f"  원본 모델:     {MODEL_ID}")
    print(f"  레이어:        {num_layers} → {new_num_layers} ({NUM_LAYERS_TO_REMOVE}개 제거)")
    print(f"  파라미터:      {total_params:,} → {pruned_params:,} ({pruning_meta['param_reduction_pct']}% 감소)")
    print(f"  LoRA 복구:     {'rank=' + str(LORA_RANK) + ', epochs=' + str(LORA_EPOCHS) if LORA_ENABLED else '비활성화'}")
    print(f"  제거된 레이어: {layers_to_remove}")
    print(f"  저장 위치:     {SAVE_DIR}/")

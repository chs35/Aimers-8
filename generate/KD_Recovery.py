"""
KD_Recovery: Knowledge Distillation을 사용한 모델 성능 복구

프루닝된 모델(student)을 원본 모델(teacher)로부터 Knowledge Distillation하여
성능을 복구하는 독립 스크립트.

사용법:
  python generate/KD_Recovery.py --model <student 모델 경로> --teacher <teacher 모델 경로>

지원 기능:
  - Logit KD: Forward/Reverse/Adaptive KL divergence (AKL 논문)
  - Hidden State KD: Cosine Similarity 또는 MSE (CoDIR 논문)
  - Attention Map Transfer: teacher attention pattern KL 매칭
  - Embedding Output 매칭: 입력 표현 정렬
  - 동적 α 가중치 (Minitron: loss 비율 기반 자동 조절)
  - LoRA / Direct Training 두 가지 학습 모드

논문 기반:
  - Temperature 1.0 (Minitron: LLM에서 ≤1.0 최적)
  - CE 제거, Logit KLD 주력 (Minitron: logit-only가 최고 성능)
  - Adaptive KL: FKL→RKL 동적 전환 (AKL 논문)
  - Cosine Similarity: MSE 대비 scale-invariant, gradient 안정 (CoDIR)

GPU 호환: L4(sm_89) 이상 모든 GPU에서 동작
필요 패키지: pip install peft  (LoRA 모드에서만 필요)

# Student + Teacher 지정 (필수: --model, 선택: --teacher)
python generate/KD_Recovery.py --model models/EXAONE-4.0-1.2B-WP5152 --teacher models/EXAONE-4.0-1.2B

# Teacher 기본값은 LGAI-EXAONE/EXAONE-4.0-1.2B (HF에서 자동 다운로드)
python generate/KD_Recovery.py --model models/EXAONE-4.0-1.2B

# 출력 경로 직접 지정
python generate/KD_Recovery.py --model models/EXAONE-4.0-1.2B-WP3584 -o models/my-custom-name

기본 출력 이름
models/<student모델명>-KD


"""

import os
import json
import argparse
import shutil
from datetime import datetime
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def _import_peft():
    """LoRA 모드에서만 peft를 import"""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        return LoraConfig, get_peft_model, TaskType
    except ImportError:
        import subprocess
        print("peft 패키지가 없습니다. 설치 중...")
        subprocess.check_call(["pip", "install", "peft", "-q"])
        from peft import LoraConfig, get_peft_model, TaskType
        return LoraConfig, get_peft_model, TaskType


# ╔══════════════════════════════════════════════════════════════╗
# ║  HYPERPARAMETERS — 이 섹션만 수정하면 됩니다                  ║
# ╚══════════════════════════════════════════════════════════════╝

# ── 데이터셋 ──
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# ── Knowledge Distillation ──
KD_TEMPERATURE = 1.0               # soft target temperature (Minitron: LLM에서 ≤1.0이 최적)
KD_ALPHA_TASK = 0.0                # task loss (CE) 가중치 (Minitron: CE 제거가 최고 성능)
KD_ALPHA_LOGIT = 0.7               # logit distillation (KL div) 가중치 (주력 loss)
KD_ALPHA_HIDDEN = 0.3              # hidden state distillation 가중치
KD_ALPHA_ATTN = 0.0                # attention map transfer 가중치 (0이면 비활성화)
KD_ALPHA_EMBED = 0.0               # embedding output 매칭 가중치 (0이면 비활성화)
KD_USE_COSINE_HIDDEN = True        # True: cosine similarity, False: MSE (기존)
KD_USE_AKL = True                  # Adaptive KL: FKL+RKL 동적 결합 (AKL 논문)
KD_USE_DYNAMIC_ALPHA = True        # 동적 loss 가중치 (Minitron 방식: logit/hidden 비율)

# ── 학습 모드 선택 ──
USE_LORA = False

# ── LoRA 설정 (USE_LORA=True 일 때만 적용) ──
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Direct Training 설정 (USE_LORA=False 일 때만 적용) ──
DIRECT_TRAIN_MODULES = [
    "gate_proj", "down_proj",
]

# ── 공통 학습 설정 ──
KD_LR = 2e-4                       # 학습률 (LoRA 모드)
DIRECT_LR = 5e-5                   # 학습률 (Direct 모드, LoRA보다 낮게 설정)
KD_EPOCHS = 12
KD_BATCH_SIZE = 4
KD_GRAD_ACCUM = 8
NUM_TRAIN_SAMPLES = 1000
MAX_SEQ_LEN_TRAIN = 512

# ── 환경 ──
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ╔══════════════════════════════════════════════════════════════╗
# ║  KD TRAINER                                                  ║
# ╚══════════════════════════════════════════════════════════════╝

class KDTrainer(Trainer):
    """
    Knowledge Distillation Trainer (논문 기반 개선 버전).

    Teacher 모델의 logit, hidden state, attention map, embedding을 사용하여 student를 학습.
    """

    def __init__(self, teacher_model, temperature, alpha_task, alpha_logit, alpha_hidden,
                 alpha_attn=0.0, alpha_embed=0.0,
                 use_cosine_hidden=True, use_akl=True, use_dynamic_alpha=True,
                 total_train_steps=None, loss_history=None, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha_task = alpha_task
        self.alpha_logit = alpha_logit
        self.alpha_hidden = alpha_hidden
        self.alpha_attn = alpha_attn
        self.alpha_embed = alpha_embed
        self.use_cosine_hidden = use_cosine_hidden
        self.use_akl = use_akl
        self.use_dynamic_alpha = use_dynamic_alpha
        self.total_train_steps = total_train_steps
        self._current_step = 0
        self.loss_history = loss_history if loss_history is not None else {
            'total': [], 'kd_logit': [], 'hidden': [], 'task': [],
        }

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._current_step += 1

        # Student forward
        need_attn = self.alpha_attn > 0
        student_outputs = model(
            **inputs, output_hidden_states=True,
            output_attentions=need_attn,
        )
        task_loss = student_outputs.loss

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                **inputs, output_hidden_states=True,
                output_attentions=need_attn,
            )

        # ── 1) Logit KD ──
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        # vocab 크기가 다를 경우 최소 크기로 맞춤
        min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
        if student_logits.size(-1) != teacher_logits.size(-1):
            student_logits = student_logits[..., :min_vocab]
            teacher_logits = teacher_logits[..., :min_vocab]
        T = self.temperature

        if self.use_akl and self.total_train_steps and self.total_train_steps > 0:
            step_ratio = min(self._current_step / self.total_train_steps, 1.0)
            beta = 1.0 - step_ratio

            s_log_prob = F.log_softmax(student_logits / T, dim=-1)
            t_prob = F.softmax(teacher_logits / T, dim=-1)
            t_log_prob = F.log_softmax(teacher_logits / T, dim=-1)
            s_prob = F.softmax(student_logits / T, dim=-1)

            fkl = F.kl_div(s_log_prob, t_prob, reduction="batchmean")
            rkl = F.kl_div(t_log_prob, s_prob, reduction="batchmean")
            kd_logit_loss = (beta * fkl + (1.0 - beta) * rkl) * (T ** 2)
        else:
            kd_logit_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T ** 2)

        # ── 2) Hidden State KD ──
        student_hidden = student_outputs.hidden_states[1:]
        teacher_hidden = teacher_outputs.hidden_states[1:]

        # student/teacher 레이어 수가 다를 경우 균등 매핑
        if len(student_hidden) != len(teacher_hidden):
            n_s = len(student_hidden)
            n_t = len(teacher_hidden)
            indices = [round(i * (n_t - 1) / (n_s - 1)) for i in range(n_s)]
            teacher_hidden = [teacher_hidden[i] for i in indices]

        if self.use_cosine_hidden:
            hidden_loss = sum(
                1.0 - F.cosine_similarity(s.float(), t.float(), dim=-1).mean()
                for s, t in zip(student_hidden, teacher_hidden)
            ) / len(student_hidden)
        else:
            hidden_loss = sum(
                F.mse_loss(s.float(), t.float())
                for s, t in zip(student_hidden, teacher_hidden)
            ) / len(student_hidden)

        # ── 3) Attention Map Transfer ──
        attn_loss = torch.tensor(0.0, device=student_logits.device)
        if need_attn:
            student_attns = student_outputs.attentions
            teacher_attns = teacher_outputs.attentions
            if len(student_attns) != len(teacher_attns):
                n_s = len(student_attns)
                n_t = len(teacher_attns)
                indices = [round(i * (n_t - 1) / (n_s - 1)) for i in range(n_s)]
                teacher_attns = [teacher_attns[i] for i in indices]
            num_layers = len(student_attns)
            attn_loss = sum(
                F.kl_div(
                    torch.log(s_attn.float() + 1e-8),
                    t_attn.float(),
                    reduction="batchmean",
                )
                for s_attn, t_attn in zip(student_attns, teacher_attns)
            ) / num_layers

        # ── 4) Embedding Output 매칭 ──
        embed_loss = torch.tensor(0.0, device=student_logits.device)
        if self.alpha_embed > 0:
            embed_loss = F.mse_loss(
                student_outputs.hidden_states[0].float(),
                teacher_outputs.hidden_states[0].float(),
            )

        # ── 5) Combined loss ──
        if self.use_dynamic_alpha and self.alpha_hidden > 0:
            alpha_dynamic = kd_logit_loss.detach() / (hidden_loss.detach() + 1e-8)
            total_loss = (
                self.alpha_task * task_loss
                + kd_logit_loss
                + alpha_dynamic * hidden_loss
                + self.alpha_attn * attn_loss
                + self.alpha_embed * embed_loss
            )
        else:
            total_loss = (
                self.alpha_task * task_loss
                + self.alpha_logit * kd_logit_loss
                + self.alpha_hidden * hidden_loss
                + self.alpha_attn * attn_loss
                + self.alpha_embed * embed_loss
            )

        # 개별 loss 로깅
        self._kd_metrics = {
            "task_loss": task_loss.detach().item(),
            "kd_logit_loss": kd_logit_loss.detach().item(),
            "hidden_loss": hidden_loss.detach().item(),
        }
        if need_attn:
            self._kd_metrics["attn_loss"] = attn_loss.detach().item()
        if self.alpha_embed > 0:
            self._kd_metrics["embed_loss"] = embed_loss.detach().item()
        if self.use_akl and self.total_train_steps and self.total_train_steps > 0:
            self._kd_metrics["akl_beta"] = beta
        if self.use_dynamic_alpha and self.alpha_hidden > 0:
            self._kd_metrics["dynamic_alpha"] = alpha_dynamic.item()

        # Loss 히스토리 기록
        self.loss_history['total'].append(total_loss.detach().item())
        if self.alpha_logit > 0 or (self.use_dynamic_alpha and self.alpha_hidden > 0):
            self.loss_history['kd_logit'].append(kd_logit_loss.detach().item())
        if self.alpha_hidden > 0:
            self.loss_history['hidden'].append(hidden_loss.detach().item())
        if self.alpha_task > 0:
            self.loss_history['task'].append(task_loss.detach().item())

        return (total_loss, student_outputs) if return_outputs else total_loss

    def log(self, logs, *args, **kwargs):
        if hasattr(self, "_kd_metrics"):
            logs.update(self._kd_metrics)
        super().log(logs, *args, **kwargs)


def plot_losses(loss_history, config, save_dir='KD_Loss'):
    """학습 Loss 그래프를 개별 파일로 생성하여 타임스탬프 폴더에 저장"""
    plots = []

    # 종합 Loss (항상 표시)
    if loss_history['total']:
        plots.append(('Total_Loss', 'Total Loss', loss_history['total']))

    # KL Divergence (alpha_logit > 0 또는 dynamic alpha 사용 시)
    show_kl = config['alpha_logit'] > 0 or (config['use_dynamic_alpha'] and config['alpha_hidden'] > 0)
    if show_kl and loss_history['kd_logit']:
        if config['use_akl']:
            kl_fname = 'KL_Divergence_AKL'
            kl_label = 'KL Divergence (AKL)'
        else:
            kl_fname = 'KL_Divergence_FKL'
            kl_label = 'KL Divergence (FKL)'
        plots.append((kl_fname, kl_label, loss_history['kd_logit']))

    # Hidden State Loss (alpha_hidden > 0)
    if config['alpha_hidden'] > 0 and loss_history['hidden']:
        if config['use_cosine_hidden']:
            h_fname = 'Hidden_State_CosineSimilarity'
            h_label = 'Hidden State Loss (Cosine Similarity)'
        else:
            h_fname = 'Hidden_State_MSE'
            h_label = 'Hidden State Loss (MSE)'
        plots.append((h_fname, h_label, loss_history['hidden']))

    # Task Loss (alpha_task > 0)
    if config['alpha_task'] > 0 and loss_history['task']:
        plots.append(('Task_Loss_CE', 'Task Loss (CE)', loss_history['task']))

    if not plots:
        print("  표시할 Loss 데이터가 없습니다.")
        return

    # 타임스탬프 하위 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = os.path.join(save_dir, timestamp)
    os.makedirs(sub_dir, exist_ok=True)

    # 각 Loss를 개별 파일로 저장
    for fname, title, data in plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(len(data)), data, linewidth=0.8, color='#1f77b4')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fpath = os.path.join(sub_dir, f"{fname}.png")
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Loss 그래프 저장 ({len(plots)}개): {sub_dir}/")


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE                                              ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":

    # ===========================================================
    # CLI Arguments
    # ===========================================================
    parser = argparse.ArgumentParser(description="Knowledge Distillation Recovery")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Student 모델 경로 (프루닝된 모델)")
    parser.add_argument("--teacher", "-t", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B",
                        help="Teacher 모델 경로 (기본값: LGAI-EXAONE/EXAONE-4.0-1.2B)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="출력 경로 (기본값: models/<student모델명>-KD)")
    parser.add_argument("--dataset", "-d", type=str, default=None,
                        help="로컬 학습 데이터셋 경로 (prepare_dataset.py의 kd_train)")
    args = parser.parse_args()

    # Student 경로 해석
    MODEL_ID = args.model
    model_name = os.path.basename(MODEL_ID.rstrip("/\\"))
    if os.path.isdir(MODEL_ID):
        LOCAL_MODEL_DIR = MODEL_ID
    else:
        LOCAL_MODEL_DIR = os.path.join("models", model_name)

    # Teacher 경로 해석
    TEACHER_MODEL_ID = args.teacher
    teacher_name = os.path.basename(TEACHER_MODEL_ID.rstrip("/\\"))
    if os.path.isdir(TEACHER_MODEL_ID):
        TEACHER_LOCAL_MODEL_DIR = TEACHER_MODEL_ID
    else:
        TEACHER_LOCAL_MODEL_DIR = os.path.join("models", teacher_name)

    SAVE_DIR = args.output or os.path.join("models", f"{model_name}-KD")

    # ===========================================================
    # Step 1: Student 모델 로드
    # ===========================================================
    print("=" * 60)
    print("Step 1: Student 모델 로드")
    print("=" * 60)

    if os.path.isdir(LOCAL_MODEL_DIR) and os.path.isfile(os.path.join(LOCAL_MODEL_DIR, "config.json")):
        model_path = LOCAL_MODEL_DIR
        print(f"  로컬 모델 사용: {LOCAL_MODEL_DIR}")
    else:
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        print(f"  모델 다운로드: {MODEL_ID} → {LOCAL_MODEL_DIR}")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_MODEL_DIR)
        model_path = LOCAL_MODEL_DIR

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    student_params = sum(p.numel() for p in model.parameters())
    print(f"  Student 모델: {MODEL_ID}")
    print(f"  파라미터: {student_params:,}")
    print()

    # ===========================================================
    # Step 2: Teacher 모델 로드
    # ===========================================================
    print("=" * 60)
    print("Step 2: Teacher 모델 로드")
    print("=" * 60)

    if os.path.isdir(TEACHER_LOCAL_MODEL_DIR) and os.path.isfile(os.path.join(TEACHER_LOCAL_MODEL_DIR, "config.json")):
        teacher_path = TEACHER_LOCAL_MODEL_DIR
        print(f"  Teacher 로컬 모델 사용: {TEACHER_LOCAL_MODEL_DIR}")
    else:
        os.makedirs(TEACHER_LOCAL_MODEL_DIR, exist_ok=True)
        print(f"  Teacher 모델 다운로드: {TEACHER_MODEL_ID} → {TEACHER_LOCAL_MODEL_DIR}")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=TEACHER_MODEL_ID, local_dir=TEACHER_LOCAL_MODEL_DIR)
        teacher_path = TEACHER_LOCAL_MODEL_DIR

    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    teacher_model.eval()
    teacher_model.requires_grad_(False)

    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"  Teacher 모델: {TEACHER_MODEL_ID}")
    print(f"  파라미터: {teacher_params:,}")
    print()

    # ===========================================================
    # Step 3: 학습 데이터 준비
    # ===========================================================
    print("=" * 60)
    print("Step 3: 학습 데이터 준비")
    print("=" * 60)

    if args.dataset:
        print(f"  로컬 데이터셋 사용: {args.dataset}")
        train_ds = load_from_disk(args.dataset)
        # 로컬 데이터셋은 prepare_dataset.py에서 이미 샘플 수가 결정됨 → 전체 사용
    else:
        raw_ds = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)
        kd_data = list(raw_ds.take(NUM_TRAIN_SAMPLES))
        train_ds = Dataset.from_list(kd_data)

    def preprocess_kd(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=True,
                tokenize=False,
            )
        }

    train_ds = train_ds.map(preprocess_kd)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN_TRAIN,
            padding=False,
        )

    train_ds = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
    print(f"  학습 샘플 수: {len(train_ds)}")
    print()

    # ===========================================================
    # Step 4: 학습 설정 (LoRA vs Direct Training)
    # ===========================================================
    print("=" * 60)
    print("Step 4: 학습 설정")
    print("=" * 60)

    if USE_LORA:
        LoraConfig, get_peft_model, TaskType = _import_peft()
        print(f"  [LoRA 모드] rank={LORA_RANK}, alpha={LORA_ALPHA}, modules={LORA_TARGET_MODULES}")
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
        learning_rate = KD_LR
    else:
        print(f"  [Direct Training 모드] 학습 대상 모듈: {DIRECT_TRAIN_MODULES}")
        model.requires_grad_(False)
        unfrozen_params = 0
        total_params_model = sum(p.numel() for p in model.parameters())
        for name, param in model.named_parameters():
            if any(mod in name for mod in DIRECT_TRAIN_MODULES):
                param.requires_grad = True
                unfrozen_params += param.numel()
        print(f"  학습 파라미터: {unfrozen_params:,} / {total_params_model:,} "
              f"({unfrozen_params / total_params_model * 100:.2f}%)")
        learning_rate = DIRECT_LR
    print()

    # ===========================================================
    # Step 5: KD 학습
    # ===========================================================
    print("=" * 60)
    print("Step 5: Knowledge Distillation 학습")
    print("=" * 60)

    model.config.use_cache = False
    model.enable_input_require_grads()

    CHECKPOINT_DIR = "./kd_recovery_output"

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=KD_BATCH_SIZE,
        gradient_accumulation_steps=KD_GRAD_ACCUM,
        num_train_epochs=KD_EPOCHS,
        learning_rate=learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=False,
    )

    steps_per_epoch = max(len(train_ds) // (KD_BATCH_SIZE * KD_GRAD_ACCUM), 1)
    total_train_steps = steps_per_epoch * KD_EPOCHS

    loss_history = {'total': [], 'kd_logit': [], 'hidden': [], 'task': []}

    trainer = KDTrainer(
        teacher_model=teacher_model,
        temperature=KD_TEMPERATURE,
        alpha_task=KD_ALPHA_TASK,
        alpha_logit=KD_ALPHA_LOGIT,
        alpha_hidden=KD_ALPHA_HIDDEN,
        alpha_attn=KD_ALPHA_ATTN,
        alpha_embed=KD_ALPHA_EMBED,
        use_cosine_hidden=KD_USE_COSINE_HIDDEN,
        use_akl=KD_USE_AKL,
        use_dynamic_alpha=KD_USE_DYNAMIC_ALPHA,
        total_train_steps=total_train_steps,
        loss_history=loss_history,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    mode_str = f"LoRA(rank={LORA_RANK})" if USE_LORA else f"Direct({DIRECT_TRAIN_MODULES})"
    kd_features = []
    if KD_USE_AKL:
        kd_features.append("AKL")
    if KD_USE_COSINE_HIDDEN:
        kd_features.append("CosineHidden")
    if KD_USE_DYNAMIC_ALPHA:
        kd_features.append("DynamicAlpha")
    if KD_ALPHA_ATTN > 0:
        kd_features.append(f"AttnTransfer(α={KD_ALPHA_ATTN})")
    if KD_ALPHA_EMBED > 0:
        kd_features.append(f"EmbedMatch(α={KD_ALPHA_EMBED})")
    features_str = ", ".join(kd_features) if kd_features else "none"

    print(f"  [{mode_str}] T={KD_TEMPERATURE}, α_task={KD_ALPHA_TASK}, "
          f"α_logit={KD_ALPHA_LOGIT}, α_hidden={KD_ALPHA_HIDDEN}, lr={learning_rate}")
    print(f"  개선 기법: {features_str}")
    print(f"  총 학습 스텝: {total_train_steps}")

    trainer.train()
    print("  KD 학습 완료!")

    # Optimizer/Scheduler state 보존을 위한 checkpoint 저장
    trainer.save_state()
    ckpt_path = CHECKPOINT_DIR

    # ===========================================================
    # Step 5.5: 인터랙티브 추가 학습 루프
    # ===========================================================
    total_epochs_trained = KD_EPOCHS
    kd_config = {
        'alpha_logit': KD_ALPHA_LOGIT,
        'alpha_hidden': KD_ALPHA_HIDDEN,
        'alpha_task': KD_ALPHA_TASK,
        'use_akl': KD_USE_AKL,
        'use_cosine_hidden': KD_USE_COSINE_HIDDEN,
        'use_dynamic_alpha': KD_USE_DYNAMIC_ALPHA,
    }

    while True:
        # Loss 그래프 표시
        plot_losses(loss_history, kd_config)

        # 사용자에게 추가 학습 여부 질문
        print()
        print("-" * 60)
        print(f"  현재까지 총 {total_epochs_trained} epoch 학습 완료.")
        try:
            extra_input = input("  추가 학습할 epoch 수를 입력하세요 (0: 저장 후 종료): ").strip()
            extra_epochs = int(extra_input)
        except (ValueError, EOFError, KeyboardInterrupt):
            extra_epochs = 0

        if extra_epochs <= 0:
            print("  학습 종료. 모델 저장을 시작합니다...")
            break

        print(f"\n  {extra_epochs} epoch 추가 학습 시작...")
        total_epochs_trained += extra_epochs
        current_step = trainer._current_step

        # AcceleratorState 리셋 후 재호출 불가 → 새 Trainer + resume_from_checkpoint
        # resume_from_checkpoint로 optimizer/scheduler state를 복원하여 loss 스파이크 방지
        training_args = TrainingArguments(
            output_dir=CHECKPOINT_DIR,
            per_device_train_batch_size=KD_BATCH_SIZE,
            gradient_accumulation_steps=KD_GRAD_ACCUM,
            num_train_epochs=extra_epochs,
            learning_rate=learning_rate,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=5,
            logging_first_step=True,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            disable_tqdm=False,
            ignore_data_skip=True,
        )
        trainer = KDTrainer(
            teacher_model=teacher_model,
            temperature=KD_TEMPERATURE,
            alpha_task=KD_ALPHA_TASK,
            alpha_logit=KD_ALPHA_LOGIT,
            alpha_hidden=KD_ALPHA_HIDDEN,
            alpha_attn=KD_ALPHA_ATTN,
            alpha_embed=KD_ALPHA_EMBED,
            use_cosine_hidden=KD_USE_COSINE_HIDDEN,
            use_akl=KD_USE_AKL,
            use_dynamic_alpha=KD_USE_DYNAMIC_ALPHA,
            total_train_steps=steps_per_epoch * total_epochs_trained,
            loss_history=loss_history,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer._current_step = current_step
        trainer.train(resume_from_checkpoint=ckpt_path)
        print(f"  추가 {extra_epochs} epoch 학습 완료! (총 {total_epochs_trained} epoch)")

        # 다음 라운드를 위해 checkpoint 갱신
        trainer.save_state()
        ckpt_path = CHECKPOINT_DIR

    # ===========================================================
    # Step 6: 후처리 및 저장
    # ===========================================================
    print()
    print("=" * 60)
    print("Step 6: 후처리 및 저장")
    print("=" * 60)

    # Teacher 메모리 해제
    del teacher_model
    torch.cuda.empty_cache()
    print("  Teacher 모델 메모리 해제 완료")

    # LoRA merge / cache 복원
    if USE_LORA:
        print("  LoRA 가중치 병합 (merge_and_unload)...")
        model = model.merge_and_unload()
    model.config.use_cache = True

    # 학습 임시 파일 정리
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)

    # 모델 저장
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    # 메타데이터 저장
    metadata = {
        "method": "Knowledge Distillation Recovery",
        "student_model": MODEL_ID,
        "teacher_model": TEACHER_MODEL_ID,
        "temperature": KD_TEMPERATURE,
        "alpha_task": KD_ALPHA_TASK,
        "alpha_logit": KD_ALPHA_LOGIT,
        "alpha_hidden": KD_ALPHA_HIDDEN,
        "alpha_attn": KD_ALPHA_ATTN,
        "alpha_embed": KD_ALPHA_EMBED,
        "use_cosine_hidden": KD_USE_COSINE_HIDDEN,
        "use_akl": KD_USE_AKL,
        "use_dynamic_alpha": KD_USE_DYNAMIC_ALPHA,
        "use_lora": USE_LORA,
        "lora_rank": LORA_RANK if USE_LORA else None,
        "direct_train_modules": DIRECT_TRAIN_MODULES if not USE_LORA else None,
        "learning_rate": learning_rate,
        "epochs": total_epochs_trained,
        "train_samples": len(train_ds),
        "student_params": student_params,
    }
    meta_path = os.path.join(SAVE_DIR, "kd_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  모델 저장 완료: {SAVE_DIR}/")
    print(f"  메타데이터: {meta_path}")

    # ===========================================================
    # 최종 요약
    # ===========================================================
    print()
    print("=" * 60)
    print("최종 요약")
    print("=" * 60)
    print(f"  Student:   {MODEL_ID}")
    print(f"  Teacher:   {TEACHER_MODEL_ID}")
    print(f"  학습 모드: {mode_str}")
    print(f"  총 epochs: {total_epochs_trained}")
    print(f"  개선 기법: {features_str}")
    print(f"  저장 위치: {SAVE_DIR}/")

"""
WidthPrunedKD: Width Pruning (구조적 채널 프루닝) + Knowledge Distillation Recovery

파이프라인:
  1. 모델 로드 + 아키텍처 정보 출력
  2. 캘리브레이션 데이터로 활성화 통계 수집 (채널별 중요도)
  3. 채널 중요도 기반 프루닝 대상 선정
  4. FFN Weight Surgery (gate_proj, up_proj, down_proj 차원 축소)
  5. Attention Head Pruning (선택적, GQA 고려)
  6. Knowledge Distillation Recovery (원본 모델을 teacher로 사용)
  7. 최종 모델 저장

Width Pruning은 FFN의 intermediate 차원을 줄여 작은 dense 행렬을 만들어
직접적인 추론 속도 향상을 제공한다. Depth pruning(레이어 제거)과 달리
각 레이어 내부의 채널을 제거하는 방식.

중요도 측정 방식:
  - wanda: ||W_row[ch]||₂ × ||activation[ch]||₂ (가중치 + 활성화 결합)
  - activation_norm: 캘리브레이션 데이터에서 채널별 활성화 L2 norm 평균
  - weight_norm: gate_proj, up_proj 가중치의 행 L2 norm 합

Knowledge Distillation Recovery:
  - 원본(프루닝 전) 모델을 teacher, 프루닝된 모델을 student로 사용
  - 3가지 loss를 결합하여 학습:
    · Task loss (CE): 일반적인 language modeling loss
    · Logit KD (KL divergence): teacher/student의 softmax 출력 분포 매칭
    · Hidden State KD (MSE): 모든 디코더 레이어의 hidden state 1:1 매칭
  - Width pruning은 hidden_size(2048)를 유지하므로 projection layer 불필요
  - 두 가지 학습 모드 지원 (USE_LORA로 선택):
    · LoRA 모드: 저랭크 어댑터로 메모리 효율적 학습 (rank ≤ 128 권장)
    · Direct Training 모드: 특정 projection layer만 unfreeze하여 직접 학습 (rank > 128이면 권장)
  - LoRA 모드에서는 학습 후 merge_and_unload()로 가중치 병합

GPU 호환: L4(sm_89) 이상 모든 GPU에서 동작
필요 패키지: pip install peft  (LoRA 모드에서만 필요, Dockerfile에 미포함)
"""

import os
import json
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

def _import_peft():
    """LoRA 모드에서만 peft를 import (Direct Training 모드에서는 불필요)"""
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

# ── 모델 & 데이터셋 ──
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
SAVE_DIR = "EXAONE-4.0-1.2B-WidthPruned-KD-Trial2"

# ── Width Pruning (FFN) ──
PRUNE_FFN = True
FFN_PRUNING_RATIO = 0.08           # FFN intermediate 차원 제거 비율 (0.25 = 25% 제거) / 7.4% 감소 시 14.8% 감소 시보다 나은 성능
IMPORTANCE_METRIC = "wanda"        # "wanda" | "activation_norm" | "weight_norm"
NUM_CALIB_SAMPLES_PRUNING = 128    # 채널 중요도 계산용 캘리브레이션 샘플 수
MAX_SEQ_LEN_PRUNING = 512          # 캘리브레이션 최대 시퀀스 길이
WARP_ALIGN = True                  # 프루닝 후 차원을 32의 배수로 정렬

# ── Width Pruning (Attention Head) ──
PRUNE_ATTN_HEADS = False
NUM_HEADS_TO_REMOVE = 2            # 제거할 attention head 수 (KV 그룹 단위), 1.2B 모델은 디코더 당 8그룹(GQA)

# ── Knowledge Distillation ──
KD_ENABLED = True                  # KD 학습 활성화/비활성화
KD_TEMPERATURE = 2.0               # soft target temperature
KD_ALPHA_TASK = 0.3                # task loss (CE) 가중치
KD_ALPHA_LOGIT = 0.2               # logit distillation (KL div) 가중치
KD_ALPHA_HIDDEN = 0.5              # hidden state distillation (MSE) 가중치

# ── 학습 모드 선택 ──
# True: LoRA 사용 (rank ≤ 128 권장, 메모리 효율적)
# False: 특정 projection layer만 직접 학습 (rank > 128이면 이 모드가 더 효과적)
#
# LoRA rank별 파라미터 수 (1.2B 모델, FFN 3모듈 × 16레이어 기준):
#   rank  8  → 2.2M (0.6%)   |  rank  64 → 17.3M (4.9%)
#   rank 128 → 34.6M (9.8%)  |  rank 256 → 69.2M (19.6%)
# Direct training (gate+up+down × 16레이어): 352M (100%) → 2.8GB optimizer mem
# Direct training (gate+down × 16레이어): 235M (67%) → 1.9GB optimizer mem
#
# 권장: rank ≤ 128이면 LoRA, rank > 128이면 Direct Training으로 전환
USE_LORA = True

# ── LoRA 설정 (USE_LORA=True 일 때만 적용) ──
LORA_RANK = 64                     # LoRA rank (KD에서는 더 큰 rank 사용)
LORA_ALPHA = 128                   # LoRA alpha (보통 rank의 2배, scaling = alpha/rank)
LORA_DROPOUT = 0.05                # LoRA dropout (과적합 방지)
LORA_TARGET_MODULES_FFN = [        # FFN만 프루닝 시 LoRA 대상 (프루닝된 모듈만)
    "gate_proj", "up_proj", "down_proj",
]
LORA_TARGET_MODULES_ALL = [        # Attention + FFN 모두 프루닝 시 LoRA 대상
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Direct Training 설정 (USE_LORA=False 일 때만 적용) ──
# 학습할 projection layer 이름 목록 (나머지는 freeze)
# FFN: "gate_proj", "up_proj", "down_proj"
# Attention: "q_proj", "k_proj", "v_proj", "o_proj"
DIRECT_TRAIN_MODULES_FFN = [       # FFN만 프루닝 시 직접 학습 대상
    "gate_proj", "down_proj",      # up_proj 제외로 메모리 절약 (gate와 up은 유사 역할)
]
DIRECT_TRAIN_MODULES_ALL = [       # Attention + FFN 모두 프루닝 시 직접 학습 대상
    "q_proj", "v_proj",            # k_proj, o_proj 제외로 메모리 절약
    "gate_proj", "down_proj",
]

# ── 공통 학습 설정 ──
KD_LR = 2e-4                       # 학습률 (LoRA 모드)
DIRECT_LR = 5e-5                   # 학습률 (Direct 모드, LoRA보다 낮게 설정)
KD_EPOCHS = 5                      # 에폭 수, 기본은 3, 늘리면 성능 상승 / 과적합 가능성 상승 및 시간 소요 상승
KD_BATCH_SIZE = 4                  # per-device 배치 크기 (GPU 메모리에 따라 조절)
KD_GRAD_ACCUM = 8                  # gradient accumulation (effective batch = 4*8=32)
NUM_TRAIN_SAMPLES = 1000           # KD 학습용 샘플 수
MAX_SEQ_LEN_TRAIN = 512            # KD 학습용 최대 시퀀스 길이

# ── 환경 ──
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ╔══════════════════════════════════════════════════════════════╗
# ║  HELPER FUNCTIONS                                            ║
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


def round_to_multiple(value, multiple):
    """값을 multiple의 배수로 내림"""
    return (value // multiple) * multiple


def collect_ffn_activations(model, tokenizer, calib_texts, max_seq_len):
    """
    각 레이어 MLP의 채널별 활성화 통계 수집.

    gate_proj와 up_proj의 출력에서 silu(gate) * up 의 채널별 L2 norm을 누적한다.
    Returns: dict[layer_idx] -> tensor of shape (intermediate_size,) with accumulated norms
    """
    num_layers = len(model.model.layers)
    intermediate_size = model.config.intermediate_size

    # 채널별 활성화 누적 저장소
    activation_accum = {i: torch.zeros(intermediate_size, device="cpu") for i in range(num_layers)}
    sample_count = {i: 0 for i in range(num_layers)}
    hooks = []

    for idx, layer in enumerate(model.model.layers):
        def make_gate_hook(layer_idx, store):
            def hook_fn(module, inp, out):
                # gate_proj 출력을 임시 저장 (silu 적용 전 raw output)
                store["gate_out"] = out.detach()
            return hook_fn

        def make_up_hook(layer_idx, store):
            def hook_fn(module, inp, out):
                # silu(gate) * up 계산
                gate_out = store.get("gate_out")
                if gate_out is not None:
                    activated = F.silu(gate_out) * out.detach()  # (batch, seq, intermediate_size)
                    # 채널별 L2 norm: (batch, seq) 차원에서 합산
                    ch_norm = activated.float().pow(2).sum(dim=(0, 1)).sqrt().cpu()
                    activation_accum[layer_idx] += ch_norm
                    sample_count[layer_idx] += 1
                    store.clear()
            return hook_fn

        store = {}
        hooks.append(layer.mlp.gate_proj.register_forward_hook(make_gate_hook(idx, store)))
        hooks.append(layer.mlp.up_proj.register_forward_hook(make_up_hook(idx, store)))

    # Forward pass
    print(f"    활성화 수집 중 ({len(calib_texts)}개 샘플)...")
    for i, text in enumerate(calib_texts):
        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=max_seq_len,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)

        if (i + 1) % 32 == 0:
            print(f"      {i + 1}/{len(calib_texts)} 완료")

    # Hook 제거
    for h in hooks:
        h.remove()

    # 평균 계산
    for idx in range(num_layers):
        if sample_count[idx] > 0:
            activation_accum[idx] /= sample_count[idx]

    return activation_accum


def compute_channel_importance(model, activation_norms, metric):
    """
    채널별 중요도 점수 계산.

    Args:
        model: 모델
        activation_norms: collect_ffn_activations()의 결과
        metric: "wanda" | "activation_norm" | "weight_norm"

    Returns:
        dict[layer_idx] -> tensor of shape (intermediate_size,) with importance scores
    """
    num_layers = len(model.model.layers)
    importance = {}

    for idx in range(num_layers):
        layer = model.model.layers[idx]

        if metric == "activation_norm":
            importance[idx] = activation_norms[idx]

        elif metric == "weight_norm":
            # gate_proj.weight[ch,:] 와 up_proj.weight[ch,:] 의 L2 norm 합
            gate_norm = layer.mlp.gate_proj.weight.data.float().pow(2).sum(dim=1).sqrt().cpu()
            up_norm = layer.mlp.up_proj.weight.data.float().pow(2).sum(dim=1).sqrt().cpu()
            importance[idx] = gate_norm + up_norm

        elif metric == "wanda":
            # Wanda: ||W_row[ch]||₂ × ||activation[ch]||₂
            gate_norm = layer.mlp.gate_proj.weight.data.float().pow(2).sum(dim=1).sqrt().cpu()
            up_norm = layer.mlp.up_proj.weight.data.float().pow(2).sum(dim=1).sqrt().cpu()
            w_norm = gate_norm + up_norm
            importance[idx] = w_norm * activation_norms[idx]

        else:
            raise ValueError(f"알 수 없는 metric: {metric}. 'wanda', 'activation_norm', 'weight_norm' 중 하나를 사용하세요.")

    return importance


def select_channels_to_keep(importance, intermediate_size, pruning_ratio, warp_align):
    """
    각 레이어에서 유지할 채널 인덱스를 선정 (균일 프루닝).

    모든 레이어에서 동일한 new_intermediate_size를 사용하여 vLLM 호환성을 보장한다.

    Returns:
        new_intermediate_size: int
        keep_indices_per_layer: dict[layer_idx] -> tensor of keep indices
    """
    num_to_keep = int(intermediate_size * (1 - pruning_ratio))

    if warp_align:
        new_intermediate_size = round_to_multiple(num_to_keep, 32)
    else:
        new_intermediate_size = num_to_keep

    # 최소 32 채널은 유지
    new_intermediate_size = max(new_intermediate_size, 32)

    keep_indices_per_layer = {}
    for layer_idx, scores in importance.items():
        # 중요도가 높은 상위 N개 채널 선택
        _, top_indices = torch.topk(scores, new_intermediate_size)
        # 정렬 (메모리 접근 패턴 최적화)
        keep_indices_per_layer[layer_idx] = top_indices.sort().values

    return new_intermediate_size, keep_indices_per_layer


def prune_ffn_weights(model, keep_indices_per_layer, new_intermediate_size):
    """
    각 레이어의 FFN 가중치를 실제로 슬라이싱하여 차원을 축소한다.

    gate_proj: (intermediate_size, hidden_size) → (new_intermediate_size, hidden_size)
    up_proj:   (intermediate_size, hidden_size) → (new_intermediate_size, hidden_size)
    down_proj: (hidden_size, intermediate_size) → (hidden_size, new_intermediate_size)
    """
    for layer_idx, layer in enumerate(model.model.layers):
        keep_idx = keep_indices_per_layer[layer_idx].to(layer.mlp.gate_proj.weight.device)

        # gate_proj: output dim pruning
        layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[keep_idx]
        layer.mlp.gate_proj.out_features = new_intermediate_size
        if layer.mlp.gate_proj.bias is not None:
            layer.mlp.gate_proj.bias.data = layer.mlp.gate_proj.bias.data[keep_idx]

        # up_proj: output dim pruning
        layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[keep_idx]
        layer.mlp.up_proj.out_features = new_intermediate_size
        if layer.mlp.up_proj.bias is not None:
            layer.mlp.up_proj.bias.data = layer.mlp.up_proj.bias.data[keep_idx]

        # down_proj: input dim pruning
        layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[:, keep_idx]
        layer.mlp.down_proj.in_features = new_intermediate_size
        if layer.mlp.down_proj.bias is not None:
            pass  # down_proj bias는 hidden_size 차원이므로 변경 불필요

    # config 업데이트 (vLLM이 config.json에서 읽음)
    model.config.intermediate_size = new_intermediate_size


def prune_attention_heads(model, num_groups_to_remove):
    """
    GQA 그룹 단위로 attention head를 프루닝한다.

    모든 레이어에서 동일한 그룹을 제거하여 vLLM 호환성을 보장한다.
    그룹 중요도는 전체 레이어에서 평균하여 결정한다.
    """
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads
    heads_per_group = num_heads // num_kv_heads
    num_layers = len(model.model.layers)

    # 먼저 head importance를 가중치 norm으로 계산
    # 모든 레이어에서 동일한 그룹을 제거해야 하므로 전체 평균 사용
    avg_group_scores = [0.0] * num_kv_heads
    for layer_idx, layer in enumerate(model.model.layers):
        q_weight = layer.self_attn.q_proj.weight.data.float()
        k_weight = layer.self_attn.k_proj.weight.data.float()
        v_weight = layer.self_attn.v_proj.weight.data.float()

        for kv_idx in range(num_kv_heads):
            total_norm = 0.0
            q_start = kv_idx * heads_per_group
            q_end = (kv_idx + 1) * heads_per_group
            for q_idx in range(q_start, q_end):
                row_start = q_idx * head_dim
                row_end = (q_idx + 1) * head_dim
                total_norm += q_weight[row_start:row_end, :].norm().item()

            kv_row_start = kv_idx * head_dim
            kv_row_end = (kv_idx + 1) * head_dim
            total_norm += k_weight[kv_row_start:kv_row_end, :].norm().item()
            total_norm += v_weight[kv_row_start:kv_row_end, :].norm().item()
            avg_group_scores[kv_idx] += total_norm

    for kv_idx in range(num_kv_heads):
        avg_group_scores[kv_idx] /= num_layers

    # 중요도가 낮은 그룹 선택
    scored_groups = [(kv_idx, score) for kv_idx, score in enumerate(avg_group_scores)]
    scored_groups.sort(key=lambda x: x[1])
    groups_to_remove = set(kv_idx for kv_idx, _ in scored_groups[:num_groups_to_remove])
    groups_to_keep = sorted(set(range(num_kv_heads)) - groups_to_remove)

    print(f"    제거할 KV 그룹: {sorted(groups_to_remove)}")
    print(f"    유지할 KV 그룹: {groups_to_keep}")
    for kv_idx, score in scored_groups:
        marker = " ← 제거" if kv_idx in groups_to_remove else ""
        print(f"      KV Group {kv_idx}: importance = {score:.4f}{marker}")

    new_num_kv_heads = num_kv_heads - num_groups_to_remove
    new_num_heads = new_num_kv_heads * heads_per_group

    # 유지할 Q head 인덱스
    q_keep_indices = []
    for kv_idx in groups_to_keep:
        q_start = kv_idx * heads_per_group
        for q_offset in range(heads_per_group):
            q_keep_indices.append(q_start + q_offset)

    # 유지할 KV head 인덱스
    kv_keep_indices = groups_to_keep

    for layer_idx, layer in enumerate(model.model.layers):
        # q_proj: (num_heads * head_dim, hidden_size) → keep Q heads
        q_rows = []
        for q_idx in q_keep_indices:
            row_start = q_idx * head_dim
            row_end = (q_idx + 1) * head_dim
            q_rows.append(layer.self_attn.q_proj.weight.data[row_start:row_end])
        layer.self_attn.q_proj.weight.data = torch.cat(q_rows, dim=0)
        layer.self_attn.q_proj.out_features = new_num_heads * head_dim
        if layer.self_attn.q_proj.bias is not None:
            q_bias_parts = []
            for q_idx in q_keep_indices:
                row_start = q_idx * head_dim
                row_end = (q_idx + 1) * head_dim
                q_bias_parts.append(layer.self_attn.q_proj.bias.data[row_start:row_end])
            layer.self_attn.q_proj.bias.data = torch.cat(q_bias_parts, dim=0)

        # k_proj: (num_kv_heads * head_dim, hidden_size) → keep KV heads
        k_rows = []
        for kv_idx in kv_keep_indices:
            row_start = kv_idx * head_dim
            row_end = (kv_idx + 1) * head_dim
            k_rows.append(layer.self_attn.k_proj.weight.data[row_start:row_end])
        layer.self_attn.k_proj.weight.data = torch.cat(k_rows, dim=0)
        layer.self_attn.k_proj.out_features = new_num_kv_heads * head_dim
        if layer.self_attn.k_proj.bias is not None:
            k_bias_parts = []
            for kv_idx in kv_keep_indices:
                row_start = kv_idx * head_dim
                row_end = (kv_idx + 1) * head_dim
                k_bias_parts.append(layer.self_attn.k_proj.bias.data[row_start:row_end])
            layer.self_attn.k_proj.bias.data = torch.cat(k_bias_parts, dim=0)

        # v_proj: same as k_proj
        v_rows = []
        for kv_idx in kv_keep_indices:
            row_start = kv_idx * head_dim
            row_end = (kv_idx + 1) * head_dim
            v_rows.append(layer.self_attn.v_proj.weight.data[row_start:row_end])
        layer.self_attn.v_proj.weight.data = torch.cat(v_rows, dim=0)
        layer.self_attn.v_proj.out_features = new_num_kv_heads * head_dim
        if layer.self_attn.v_proj.bias is not None:
            v_bias_parts = []
            for kv_idx in kv_keep_indices:
                row_start = kv_idx * head_dim
                row_end = (kv_idx + 1) * head_dim
                v_bias_parts.append(layer.self_attn.v_proj.bias.data[row_start:row_end])
            layer.self_attn.v_proj.bias.data = torch.cat(v_bias_parts, dim=0)

        # o_proj: (hidden_size, num_heads * head_dim) → keep columns for kept Q heads
        o_cols = []
        for q_idx in q_keep_indices:
            col_start = q_idx * head_dim
            col_end = (q_idx + 1) * head_dim
            o_cols.append(layer.self_attn.o_proj.weight.data[:, col_start:col_end])
        layer.self_attn.o_proj.weight.data = torch.cat(o_cols, dim=1)
        layer.self_attn.o_proj.in_features = new_num_heads * head_dim

    # Config 업데이트
    model.config.num_attention_heads = new_num_heads
    model.config.num_key_value_heads = new_num_kv_heads

    return new_num_heads, new_num_kv_heads


class KDTrainer(Trainer):
    """
    Knowledge Distillation Trainer.

    Teacher 모델의 logit과 hidden state를 사용하여 student 모델을 학습한다.
    3가지 loss를 결합: task loss (CE) + logit KD (KL div) + hidden state KD (MSE)
    """

    def __init__(self, teacher_model, temperature, alpha_task, alpha_logit, alpha_hidden, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha_task = alpha_task
        self.alpha_logit = alpha_logit
        self.alpha_hidden = alpha_hidden

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Student forward (with grad, output_hidden_states=True)
        student_outputs = model(**inputs, output_hidden_states=True)
        task_loss = student_outputs.loss  # standard CE loss

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs, output_hidden_states=True)

        # 1) Logit KD: KL divergence
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        T = self.temperature
        kd_logit_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T ** 2)

        # 2) Hidden State KD: MSE on all decoder layers
        student_hidden = student_outputs.hidden_states[1:]  # skip embedding
        teacher_hidden = teacher_outputs.hidden_states[1:]
        hidden_loss = sum(
            F.mse_loss(s.float(), t.float())
            for s, t in zip(student_hidden, teacher_hidden)
        ) / len(student_hidden)

        # 3) Combined loss
        total_loss = (
            self.alpha_task * task_loss
            + self.alpha_logit * kd_logit_loss
            + self.alpha_hidden * hidden_loss
        )

        # 개별 loss를 로깅용으로 저장
        self._kd_metrics = {
            "task_loss": task_loss.detach().item(),
            "kd_logit_loss": kd_logit_loss.detach().item(),
            "hidden_loss": hidden_loss.detach().item(),
        }

        return (total_loss, student_outputs) if return_outputs else total_loss

    def log(self, logs, *args, **kwargs):
        """학습 로그에 개별 KD loss 컴포넌트를 추가"""
        if hasattr(self, "_kd_metrics"):
            logs.update(self._kd_metrics)
        super().log(logs, *args, **kwargs)


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE                                              ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":

    # ===========================================================
    # Step 1: 모델 로드 + 아키텍처 정보 출력
    # ===========================================================
    print("=" * 60)
    print("Step 1: 모델 로드 + 아키텍처 정보 출력")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 아키텍처 정보 읽기
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    num_attention_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    num_layers = len(model.model.layers)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  모델: {MODEL_ID}")
    print(f"  레이어 수: {num_layers}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  num_attention_heads: {num_attention_heads}")
    print(f"  num_key_value_heads: {num_kv_heads}")
    print(f"  head_dim: {hidden_size // num_attention_heads}")
    print(f"  총 파라미터: {total_params:,}")
    print()

    # ===========================================================
    # Step 2: 캘리브레이션 데이터로 활성화 통계 수집
    # ===========================================================
    if PRUNE_FFN:
        print("=" * 60)
        print(f"Step 2: 캘리브레이션 데이터로 활성화 통계 수집 (metric: {IMPORTANCE_METRIC})")
        print("=" * 60)

        calib_texts = prepare_calib_texts(
            tokenizer, DATASET_ID, DATASET_SPLIT, NUM_CALIB_SAMPLES_PRUNING
        )
        print(f"  캘리브레이션 샘플 수: {len(calib_texts)}")

        if IMPORTANCE_METRIC in ("wanda", "activation_norm"):
            activation_norms = collect_ffn_activations(
                model, tokenizer, calib_texts, MAX_SEQ_LEN_PRUNING
            )
        else:
            activation_norms = None

        print("  활성화 수집 완료!")
        print()

        # ===========================================================
        # Step 3: 채널 중요도 기반 프루닝 대상 선정
        # ===========================================================
        print("=" * 60)
        print(f"Step 3: 채널 중요도 기반 프루닝 대상 선정 (ratio: {FFN_PRUNING_RATIO})")
        print("=" * 60)

        importance = compute_channel_importance(model, activation_norms, IMPORTANCE_METRIC)

        new_intermediate_size, keep_indices_per_layer = select_channels_to_keep(
            importance, intermediate_size, FFN_PRUNING_RATIO, WARP_ALIGN,
        )

        print(f"  원본 intermediate_size: {intermediate_size}")
        print(f"  프루닝 후 intermediate_size: {new_intermediate_size}")
        print(f"  실제 제거 비율: {(1 - new_intermediate_size / intermediate_size) * 100:.1f}%")
        if WARP_ALIGN:
            print(f"  32의 배수 정렬 적용됨")

        # 레이어별 중요도 통계 출력
        for layer_idx in [0, num_layers // 2, num_layers - 1]:
            scores = importance[layer_idx]
            print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")
        print()

        # ===========================================================
        # Step 4: FFN Weight Surgery
        # ===========================================================
        print("=" * 60)
        print("Step 4: FFN Weight Surgery")
        print("=" * 60)

        prune_ffn_weights(model, keep_indices_per_layer, new_intermediate_size)

        ffn_pruned_params = sum(p.numel() for p in model.parameters())
        print(f"  FFN 프루닝 완료!")
        print(f"    intermediate_size: {intermediate_size} → {new_intermediate_size}")
        print(f"    파라미터: {total_params:,} → {ffn_pruned_params:,}")
        print(f"    파라미터 감소율: {(1 - ffn_pruned_params / total_params) * 100:.1f}%")

        # 검증: 모든 레이어의 FFN 차원이 일관적인지 확인
        for layer_idx, layer in enumerate(model.model.layers):
            assert layer.mlp.gate_proj.out_features == new_intermediate_size, \
                f"Layer {layer_idx} gate_proj 불일치: {layer.mlp.gate_proj.out_features}"
            assert layer.mlp.up_proj.out_features == new_intermediate_size, \
                f"Layer {layer_idx} up_proj 불일치: {layer.mlp.up_proj.out_features}"
            assert layer.mlp.down_proj.in_features == new_intermediate_size, \
                f"Layer {layer_idx} down_proj 불일치: {layer.mlp.down_proj.in_features}"
        print("  차원 일관성 검증 통과!")
        print()
    else:
        print("=" * 60)
        print("Step 2-4: FFN Width Pruning 건너뜀 (PRUNE_FFN=False)")
        print("=" * 60)
        new_intermediate_size = intermediate_size
        ffn_pruned_params = total_params
        print()

    # ===========================================================
    # Step 5: Attention Head Pruning (선택적)
    # ===========================================================
    if PRUNE_ATTN_HEADS:
        print("=" * 60)
        print(f"Step 5: Attention Head Pruning ({NUM_HEADS_TO_REMOVE}개 KV 그룹 제거)")
        print("=" * 60)

        heads_per_group = num_attention_heads // num_kv_heads
        print(f"  GQA 구조: {num_kv_heads} KV heads, 각 {heads_per_group}개 Q heads")
        print(f"  KV 그룹 단위 프루닝 (그룹 제거 시 Q {heads_per_group}개 + KV 1개 함께 제거)")

        new_num_heads, new_num_kv_heads = prune_attention_heads(model, NUM_HEADS_TO_REMOVE)

        head_pruned_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Head 프루닝 완료!")
        print(f"    num_attention_heads: {num_attention_heads} → {new_num_heads}")
        print(f"    num_key_value_heads: {num_kv_heads} → {new_num_kv_heads}")
        print(f"    파라미터: {ffn_pruned_params:,} → {head_pruned_params:,}")
        print(f"    총 파라미터 감소율: {(1 - head_pruned_params / total_params) * 100:.1f}%")
        print()
    else:
        print("=" * 60)
        print("Step 5: Attention Head Pruning 건너뜀 (PRUNE_ATTN_HEADS=False)")
        print("=" * 60)
        head_pruned_params = ffn_pruned_params
        print()

    # ===========================================================
    # Step 6: Knowledge Distillation Recovery
    # ===========================================================
    # 원본(프루닝 전) 모델을 teacher로, 프루닝된 모델을 student로 사용하여
    # logit + hidden state distillation으로 성능을 복구한다.
    # LoRA를 사용하여 메모리 효율적으로 학습.
    if KD_ENABLED:
        print("=" * 60)
        print("Step 6: Knowledge Distillation Recovery")
        print("=" * 60)

        # 6-1. Teacher 모델 로드 (프루닝 전 원본)
        print("  Teacher 모델 로드 (원본 모델)...")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
        )
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        print(f"  Teacher 모델 로드 완료: {MODEL_ID}")

        # 6-2. 학습 데이터 준비
        print("  학습 데이터 준비...")
        raw_ds_kd = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)
        kd_data = list(raw_ds_kd.take(NUM_TRAIN_SAMPLES))
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

        # 6-3. Student 모델 학습 설정 (LoRA vs Direct Training)
        if USE_LORA:
            # ── LoRA 모드: 저랭크 어댑터로 메모리 효율적 학습 ──
            LoraConfig, get_peft_model, TaskType = _import_peft()
            lora_target = LORA_TARGET_MODULES_ALL if PRUNE_ATTN_HEADS else LORA_TARGET_MODULES_FFN
            print(f"  [LoRA 모드] rank={LORA_RANK}, alpha={LORA_ALPHA}, modules={lora_target}")
            lora_config = LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=lora_target,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            learning_rate = KD_LR
        else:
            # ── Direct Training 모드: 특정 projection layer만 unfreeze ──
            direct_target = DIRECT_TRAIN_MODULES_ALL if PRUNE_ATTN_HEADS else DIRECT_TRAIN_MODULES_FFN
            print(f"  [Direct Training 모드] 학습 대상 모듈: {direct_target}")

            # 전체 파라미터 freeze
            model.requires_grad_(False)

            # 지정된 모듈만 unfreeze
            unfrozen_params = 0
            total_params_model = sum(p.numel() for p in model.parameters())
            for name, param in model.named_parameters():
                if any(mod in name for mod in direct_target):
                    param.requires_grad = True
                    unfrozen_params += param.numel()

            print(f"  학습 파라미터: {unfrozen_params:,} / {total_params_model:,} "
                  f"({unfrozen_params / total_params_model * 100:.2f}%)")
            learning_rate = DIRECT_LR

        # 6-4. KD 학습
        model.config.use_cache = False
        if USE_LORA:
            model.enable_input_require_grads()
        else:
            # Direct Training에서도 gradient checkpointing 호환을 위해 필요
            model.enable_input_require_grads()

        training_args = TrainingArguments(
            output_dir="./kd_recovery_output",
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

        trainer = KDTrainer(
            teacher_model=teacher_model,
            temperature=KD_TEMPERATURE,
            alpha_task=KD_ALPHA_TASK,
            alpha_logit=KD_ALPHA_LOGIT,
            alpha_hidden=KD_ALPHA_HIDDEN,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        mode_str = f"LoRA(rank={LORA_RANK})" if USE_LORA else f"Direct({direct_target})"
        print(f"  KD 학습 시작 [{mode_str}] (T={KD_TEMPERATURE}, alpha_task={KD_ALPHA_TASK}, "
              f"alpha_logit={KD_ALPHA_LOGIT}, alpha_hidden={KD_ALPHA_HIDDEN}, lr={learning_rate})...")
        trainer.train()
        print("  KD 학습 완료!")

        # 6-5. Teacher 메모리 해제
        del teacher_model
        torch.cuda.empty_cache()
        print("  Teacher 모델 메모리 해제 완료")

        # 6-6. 후처리 (LoRA: merge, Direct: re-enable cache)
        if USE_LORA:
            print("  LoRA 가중치 병합 (merge_and_unload)...")
            model = model.merge_and_unload()
        model.config.use_cache = True

        # 학습 임시 파일 정리
        import shutil
        if os.path.exists("./kd_recovery_output"):
            shutil.rmtree("./kd_recovery_output")

        print(f"  KD {'+ LoRA 병합' if USE_LORA else '(Direct Training)'} 완료! 모델이 복구되었습니다.")
        print()
    else:
        print("=" * 60)
        print("Step 6: Knowledge Distillation 건너뜀 (KD_ENABLED=False)")
        print("=" * 60)
        print()

    # ===========================================================
    # Step 7: 저장
    # ===========================================================
    print("=" * 60)
    print("Step 7: 최종 모델 저장")
    print("=" * 60)

    # .gitignore 처리
    GITIGNORE_PATH = ".gitignore"
    if os.path.exists(GITIGNORE_PATH):
        with open(GITIGNORE_PATH, "r") as f:
            gitignore_content = f.read()
        if SAVE_DIR not in gitignore_content:
            with open(GITIGNORE_PATH, "a") as f:
                f.write(f"\n{SAVE_DIR}\n")
            print(f"  '{SAVE_DIR}'을(를) .gitignore에 추가했습니다.")
        else:
            print(f"  '{SAVE_DIR}'은(는) 이미 .gitignore에 있습니다.")
    else:
        with open(GITIGNORE_PATH, "w") as f:
            f.write(f"{SAVE_DIR}\n")
        print(f"  .gitignore 파일을 생성하고 '{SAVE_DIR}'을(를) 추가했습니다.")

    # 모델 저장
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    # 프루닝 메타데이터 저장
    final_params = sum(p.numel() for p in model.parameters())
    pruning_meta = {
        "method": "Width Pruning + Knowledge Distillation",
        "original_model": MODEL_ID,
        # FFN pruning info
        "prune_ffn": PRUNE_FFN,
        "importance_metric": IMPORTANCE_METRIC,
        "ffn_pruning_ratio": FFN_PRUNING_RATIO,
        "original_intermediate_size": intermediate_size,
        "pruned_intermediate_size": new_intermediate_size,
        "warp_aligned": WARP_ALIGN,
        # Attention head pruning info
        "prune_heads": PRUNE_ATTN_HEADS,
        "original_num_attention_heads": num_attention_heads,
        "original_num_kv_heads": num_kv_heads,
        "pruned_num_attention_heads": model.config.num_attention_heads,
        "pruned_num_kv_heads": model.config.num_key_value_heads,
        "num_heads_removed": NUM_HEADS_TO_REMOVE if PRUNE_ATTN_HEADS else 0,
        # KD info
        "kd_enabled": KD_ENABLED,
        "kd_temperature": KD_TEMPERATURE if KD_ENABLED else None,
        "kd_alpha_task": KD_ALPHA_TASK if KD_ENABLED else None,
        "kd_alpha_logit": KD_ALPHA_LOGIT if KD_ENABLED else None,
        "kd_alpha_hidden": KD_ALPHA_HIDDEN if KD_ENABLED else None,
        # Training mode info
        "use_lora": USE_LORA if KD_ENABLED else None,
        "lora_rank": LORA_RANK if (KD_ENABLED and USE_LORA) else None,
        "lora_alpha": LORA_ALPHA if (KD_ENABLED and USE_LORA) else None,
        "direct_train_modules": (DIRECT_TRAIN_MODULES_ALL if PRUNE_ATTN_HEADS else DIRECT_TRAIN_MODULES_FFN) if (KD_ENABLED and not USE_LORA) else None,
        "learning_rate": (KD_LR if USE_LORA else DIRECT_LR) if KD_ENABLED else None,
        "kd_epochs": KD_EPOCHS if KD_ENABLED else None,
        "train_samples": NUM_TRAIN_SAMPLES if KD_ENABLED else None,
        # Param counts
        "original_params": total_params,
        "after_ffn_pruning_params": ffn_pruned_params,
        "after_head_pruning_params": head_pruned_params,
        "final_params": final_params,
        "param_reduction_pct": round((1 - head_pruned_params / total_params) * 100, 1),
    }
    meta_path = os.path.join(SAVE_DIR, "pruning_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(pruning_meta, f, indent=2, ensure_ascii=False)

    print(f"  모델 저장 완료: {SAVE_DIR}/")
    print(f"  프루닝 메타데이터: {meta_path}")
    print()

    # ===========================================================
    # 최종 요약
    # ===========================================================
    print("=" * 60)
    print("최종 요약")
    print("=" * 60)
    print(f"  원본 모델:          {MODEL_ID}")
    print(f"  FFN Pruning:        {'intermediate_size ' + str(intermediate_size) + ' → ' + str(new_intermediate_size) + ' (' + str(round((1 - new_intermediate_size / intermediate_size) * 100, 1)) + '% 감소)' if PRUNE_FFN else '비활성화'}")
    print(f"  Head Pruning:       {'heads ' + str(num_attention_heads) + ' → ' + str(model.config.num_attention_heads) + ', kv_heads ' + str(num_kv_heads) + ' → ' + str(model.config.num_key_value_heads) if PRUNE_ATTN_HEADS else '비활성화'}")
    if KD_ENABLED:
        if USE_LORA:
            kd_info = f"LoRA(rank={LORA_RANK}, lr={KD_LR}), T={KD_TEMPERATURE}, epochs={KD_EPOCHS}"
        else:
            dt_mods = DIRECT_TRAIN_MODULES_ALL if PRUNE_ATTN_HEADS else DIRECT_TRAIN_MODULES_FFN
            kd_info = f"Direct({dt_mods}, lr={DIRECT_LR}), T={KD_TEMPERATURE}, epochs={KD_EPOCHS}"
    else:
        kd_info = "비활성화"
    print(f"  KD 복구:            {kd_info}")
    print(f"  파라미터:           {total_params:,} → {head_pruned_params:,} ({pruning_meta['param_reduction_pct']}% 감소)")
    print(f"  저장 위치:          {SAVE_DIR}/")

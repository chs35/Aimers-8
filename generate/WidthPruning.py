"""
WidthPruning: Width Pruning (구조적 채널 프루닝)

파이프라인:
  1. 모델 로드 + 아키텍처 정보 출력
  2. 캘리브레이션 데이터로 활성화 통계 수집 (채널별 중요도)
  3. 채널 중요도 기반 프루닝 대상 선정
  4. FFN Weight Surgery (gate_proj, up_proj, down_proj 차원 축소)
  5. Attention Head Pruning (선택적, GQA 고려)
  6. 최종 모델 저장

Width Pruning은 FFN의 intermediate 차원을 줄여 작은 dense 행렬을 만들어
직접적인 추론 속도 향상을 제공한다. Depth pruning(레이어 제거)과 달리
각 레이어 내부의 채널을 제거하는 방식.

중요도 측정 방식:
  - wanda: ||W_row[ch]||₂ × ||activation[ch]||₂ (가중치 + 활성화 결합)
  - activation_norm: 캘리브레이션 데이터에서 채널별 활성화 L2 norm 평균
  - weight_norm: gate_proj, up_proj 가중치의 행 L2 norm 합

KD Recovery는 별도 스크립트(KD_Recovery.py)에서 수행.

GPU 호환: L4(sm_89) 이상 모든 GPU에서 동작


#사용 예시
python generate/WidthPruning.py --model models/EXAONE-4.0-1.2B       
  --target-size 3072

"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# ╔══════════════════════════════════════════════════════════════╗
# ║  HYPERPARAMETERS — 이 섹션만 수정하면 됩니다                  ║
# ╚══════════════════════════════════════════════════════════════╝

# ── 모델 & 데이터셋 ──
# MODEL_ID, LOCAL_MODEL_DIR, SAVE_DIR 등은 argparse로 설정 (하단 __main__ 참조)
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# ── Width Pruning (FFN) ──
PRUNE_FFN = True
TARGET_INTERMEDIATE_SIZE = None    # 목표 intermediate 차원 (예: 3072, 3584). CLI --target-size로 지정.
IMPORTANCE_METRIC = "wanda"        # "wanda" | "activation_norm" | "weight_norm"
NUM_CALIB_SAMPLES_PRUNING = 128    # 채널 중요도 계산용 캘리브레이션 샘플 수
MAX_SEQ_LEN_PRUNING = 512          # 캘리브레이션 최대 시퀀스 길이
ALIGN_MULTIPLE = 128               # 프루닝 후 차원을 128의 배수로 정렬

# ── Width Pruning (Attention Head) ──
PRUNE_ATTN_HEADS = False
NUM_HEADS_TO_REMOVE = 2            # 제거할 attention head 수 (KV 그룹 단위), 1.2B 모델은 디코더 당 8그룹(GQA)

# ── 환경 ──
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ╔══════════════════════════════════════════════════════════════╗
# ║  HELPER FUNCTIONS                                            ║
# ╚══════════════════════════════════════════════════════════════╝

def prepare_calib_texts(tokenizer, dataset_id, dataset_split, num_samples,
                        local_dataset_path=None):
    """캘리브레이션 텍스트 준비 (채팅 템플릿 적용)

    Args:
        local_dataset_path: 로컬 Arrow 데이터셋 경로 (prepare_dataset.py로 생성).
                           지정 시 이 경로에서 로드, 미지정 시 HF 스트리밍.
    """
    if local_dataset_path:
        print(f"    로컬 데이터셋 사용: {local_dataset_path}")
        ds = load_from_disk(local_dataset_path)
        # 로컬 데이터셋은 prepare_dataset.py에서 이미 샘플 수가 결정됨 → 전체 사용                                 
        samples = list(ds)
    else:
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


def select_channels_to_keep(importance, intermediate_size, target_size, align_multiple):
    """
    각 레이어에서 유지할 채널 인덱스를 선정 (균일 프루닝).

    모든 레이어에서 동일한 new_intermediate_size를 사용하여 vLLM 호환성을 보장한다.

    Args:
        importance: 채널별 중요도 점수
        intermediate_size: 원본 intermediate 차원
        target_size: 목표 intermediate 차원 (128의 배수 권장)
        align_multiple: 정렬 배수 (128)

    Returns:
        new_intermediate_size: int
        keep_indices_per_layer: dict[layer_idx] -> tensor of keep indices
    """
    new_intermediate_size = round_to_multiple(target_size, align_multiple)

    # 최소 128 채널은 유지
    new_intermediate_size = max(new_intermediate_size, align_multiple)

    # 원본보다 클 수 없음
    new_intermediate_size = min(new_intermediate_size, intermediate_size)

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


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE                                              ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":

    # ===========================================================
    # CLI Arguments
    # ===========================================================
    parser = argparse.ArgumentParser(description="Width Pruning")
    parser.add_argument("--model", "-m", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B",
                        help="모델 경로 (HuggingFace ID 또는 로컬 경로)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="출력 경로 (기본값: models/<모델명>-WP<intermediate_size>)")
    parser.add_argument("--target-size", "-t", type=int, default=None,
                        help="목표 FFN intermediate 차원 (예: 3072, 3584). 128의 배수로 자동 정렬됨.")
    parser.add_argument("--dataset", "-d", type=str, default=None,
                        help="로컬 캘리브레이션 데이터셋 경로 (prepare_dataset.py의 wp_calib)")
    parser.add_argument("--prune-attn", action="store_true",
                        help="어텐션 헤드 프루닝 활성화 (기본: 비활성화)")
    args = parser.parse_args()

    # CLI --prune-attn으로 하이퍼파라미터 오버라이드
    if args.prune_attn:
        PRUNE_ATTN_HEADS = True

    # 모델 경로 해석
    MODEL_ID = args.model
    model_name = os.path.basename(MODEL_ID.rstrip("/\\"))
    if os.path.isdir(MODEL_ID):
        LOCAL_MODEL_DIR = MODEL_ID
    else:
        LOCAL_MODEL_DIR = os.path.join("models", model_name)

    # ===========================================================
    # Step 1: 모델 로드 + 아키텍처 정보 출력
    # ===========================================================
    print("=" * 60)
    print("Step 1: 모델 로드 + 아키텍처 정보 출력")
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
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

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
            tokenizer, DATASET_ID, DATASET_SPLIT, NUM_CALIB_SAMPLES_PRUNING,
            local_dataset_path=args.dataset,
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
        # 목표 intermediate 차원 결정 (CLI > 하이퍼파라미터 > 에러)
        target_size = args.target_size or TARGET_INTERMEDIATE_SIZE
        if target_size is None:
            print("[ERROR] --target-size를 지정해야 합니다. (예: --target-size 3072)")
            print(f"  현재 intermediate_size: {intermediate_size}")
            print(f"  128의 배수 예시: {', '.join(str(round_to_multiple(intermediate_size * r, ALIGN_MULTIPLE)) for r in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7])}")
            import sys; sys.exit(1)

        if target_size >= intermediate_size:
            print(f"[ERROR] target_size({target_size})가 원본 intermediate_size({intermediate_size})보다 크거나 같습니다.")
            import sys; sys.exit(1)

        print("=" * 60)
        print(f"Step 3: 채널 중요도 기반 프루닝 대상 선정 (target: {target_size})")
        print("=" * 60)

        importance = compute_channel_importance(model, activation_norms, IMPORTANCE_METRIC)

        new_intermediate_size, keep_indices_per_layer = select_channels_to_keep(
            importance, intermediate_size, target_size, ALIGN_MULTIPLE,
        )

        print(f"  원본 intermediate_size: {intermediate_size}")
        print(f"  목표 intermediate_size: {target_size}")
        print(f"  정렬 후 intermediate_size: {new_intermediate_size}")
        print(f"  실제 제거 비율: {(1 - new_intermediate_size / intermediate_size) * 100:.1f}%")
        print(f"  {ALIGN_MULTIPLE}의 배수 정렬 적용됨")

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
    # Step 6: 저장
    # ===========================================================
    SAVE_DIR = args.output or os.path.join("models", f"{model_name}-WP{model.config.intermediate_size}")

    print("=" * 60)
    print("Step 6: 최종 모델 저장")
    print("=" * 60)

    # 모델 저장
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    # 프루닝 메타데이터 저장
    final_params = sum(p.numel() for p in model.parameters())
    pruning_meta = {
        "method": "Width Pruning",
        "original_model": MODEL_ID,
        # FFN pruning info
        "prune_ffn": PRUNE_FFN,
        "importance_metric": IMPORTANCE_METRIC,
        "target_intermediate_size": target_size if PRUNE_FFN else None,
        "original_intermediate_size": intermediate_size,
        "pruned_intermediate_size": new_intermediate_size,
        "align_multiple": ALIGN_MULTIPLE,
        # Attention head pruning info
        "prune_heads": PRUNE_ATTN_HEADS,
        "original_num_attention_heads": num_attention_heads,
        "original_num_kv_heads": num_kv_heads,
        "pruned_num_attention_heads": model.config.num_attention_heads,
        "pruned_num_kv_heads": model.config.num_key_value_heads,
        "num_heads_removed": NUM_HEADS_TO_REMOVE if PRUNE_ATTN_HEADS else 0,
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
    print(f"  파라미터:           {total_params:,} → {head_pruned_params:,} ({pruning_meta['param_reduction_pct']}% 감소)")
    print(f"  저장 위치:          {SAVE_DIR}/")

"""
EXAONE-4.0-1.2B 레이어 중요도 분석 (Block Influence)

각 레이어의 입력/출력 hidden state 간 코사인 유사도를 측정하여
제거 가능한 레이어를 식별.

- 코사인 유사도가 높은 레이어 = 변환이 거의 없음 = 제거 가능
- 코사인 유사도가 낮은 레이어 = 중요한 변환 수행 = 보존 필요

사용법:
  python experiments/analyze_layers.py
  python experiments/analyze_layers.py --model-path LGAI-EXAONE/EXAONE-4.0-1.2B
  python experiments/analyze_layers.py --num-samples 32 --max-length 512
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random


def analyze_layer_importance(
    model_path: str,
    num_samples: int = 16,
    max_length: int = 512,
    seed: int = 42,
):
    random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 60)
    print("EXAONE-4.0-1.2B Layer Importance Analysis (Block Influence)")
    print("=" * 60)

    # 토크나이저 + 모델 로드
    print(f"\nLoading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    num_layers = len(model.model.layers)
    print(f"Total layers: {num_layers}")

    # 캘리브레이션 데이터 준비 (MANTA-1M에서 샘플링)
    print(f"\nLoading calibration data ({num_samples} samples)...")
    dataset = load_dataset(
        "LGAI-EXAONE/MANTA-1M",
        split="train",
        trust_remote_code=True,
    )
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    dataset = dataset.select(indices)

    texts = []
    for sample in dataset:
        conversations = sample.get("conversations", [])
        if conversations:
            text = "\n".join([turn.get("content", "") for turn in conversations])
        else:
            text = sample.get("text", sample.get("content", ""))
        if text:
            texts.append(text)

    # 레이어별 hidden state 수집을 위한 hook
    layer_inputs = {}
    layer_outputs = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # input: tuple, first element is hidden_states
            inp = input[0].detach().float()
            # output: tuple or tensor — first element is hidden_states
            if isinstance(output, tuple):
                out = output[0].detach().float()
            else:
                out = output.detach().float()

            if layer_idx not in layer_inputs:
                layer_inputs[layer_idx] = []
                layer_outputs[layer_idx] = []
            layer_inputs[layer_idx].append(inp)
            layer_outputs[layer_idx].append(out)

        return hook_fn

    # Hook 등록
    hooks = []
    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Forward pass
    print(f"Running forward pass on {len(texts)} samples...")
    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            ).to(model.device)
            model(**inputs)
            if (i + 1) % 4 == 0:
                print(f"  Processed {i + 1}/{len(texts)} samples")

    # Hook 제거
    for h in hooks:
        h.remove()

    # Block Influence (BI) 계산
    print("\n" + "=" * 60)
    print("Block Influence Analysis")
    print("=" * 60)
    print(f"{'Layer':>6} {'Cos Sim':>10} {'BI (1-cos)':>12} {'Importance':>12}")
    print("-" * 44)

    bi_scores = []
    for layer_idx in range(num_layers):
        cos_sims = []
        for inp, out in zip(layer_inputs[layer_idx], layer_outputs[layer_idx]):
            # Flatten to (batch * seq_len, hidden_dim)
            inp_flat = inp.reshape(-1, inp.shape[-1])
            out_flat = out.reshape(-1, out.shape[-1])
            # 토큰별 코사인 유사도 평균
            cos = F.cosine_similarity(inp_flat, out_flat, dim=-1).mean().item()
            cos_sims.append(cos)

        avg_cos = sum(cos_sims) / len(cos_sims)
        bi = 1.0 - avg_cos  # Block Influence = 1 - cosine_similarity

        bi_scores.append((layer_idx, avg_cos, bi))

    # BI 기준으로 정렬 (낮은 BI = 덜 중요 = 제거 후보)
    for layer_idx, cos, bi in bi_scores:
        bar = "#" * int(bi * 100)
        print(f"  {layer_idx:>3}    {cos:>8.5f}    {bi:>10.5f}    {bar}")

    # 제거 후보 추천
    print("\n" + "=" * 60)
    print("Layer Removal Candidates (sorted by BI, lowest = most removable)")
    print("=" * 60)

    sorted_by_bi = sorted(bi_scores, key=lambda x: x[2])

    print(f"\n{'Rank':>5} {'Layer':>6} {'Cos Sim':>10} {'BI':>10}")
    print("-" * 35)
    for rank, (layer_idx, cos, bi) in enumerate(sorted_by_bi):
        marker = " <-- REMOVABLE" if bi < sorted_by_bi[num_layers // 2][2] else ""
        print(f"  {rank+1:>3}    {layer_idx:>3}    {cos:>8.5f}    {bi:>8.5f}{marker}")

    # 프루닝 시나리오 제안
    print("\n" + "=" * 60)
    print("Suggested Pruning Scenarios")
    print("=" * 60)

    for n_remove in [1, 2, 3, 4, 5, 6]:
        removable = [s[0] for s in sorted_by_bi[:n_remove]]
        keep = sorted([i for i in range(num_layers) if i not in removable])
        keep_count = len(keep)
        avg_bi_removed = sum(s[2] for s in sorted_by_bi[:n_remove]) / n_remove
        print(f"\n  Remove {n_remove} layers ({num_layers}→{keep_count}L):")
        print(f"    Remove: {removable}")
        print(f"    Keep:   {keep}")
        print(f"    Avg BI of removed: {avg_bi_removed:.5f}")
        layer_str = ",".join(map(str, keep))
        print(f"    Command: python experiments/prune_layers_test.py --layer-indices {layer_str}")

    # 메모리 정리
    del layer_inputs, layer_outputs


def main():
    parser = argparse.ArgumentParser(
        description="EXAONE-4.0-1.2B Layer Importance Analysis"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="LGAI-EXAONE/EXAONE-4.0-1.2B",
        help="모델 경로",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="분석에 사용할 샘플 수",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="최대 시퀀스 길이",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드",
    )
    args = parser.parse_args()

    analyze_layer_importance(
        model_path=args.model_path,
        num_samples=args.num_samples,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

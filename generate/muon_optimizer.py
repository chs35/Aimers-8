"""
Muon Optimizer variants for KD Recovery

Core: Newton-Schulz orthogonalization (Keller Jordan's Muon, MIT License)
  https://github.com/KellerJordan/Muon

Custom wrappers:
  - HybridMuonAdamW: 2D weight → Muon/MuonClip, 1D/embed/norm → AdamW
  - QK-Clip: per-head weight norm clipping on Q/K projections (Kimi K2 inspired)

Usage:
  from muon_optimizer import create_hybrid_optimizer

  # Muon (2D → Muon, rest → AdamW)
  optimizer = create_hybrid_optimizer(model, 'muon', muon_lr=0.02, adamw_lr=3e-4)

  # MuonClip (Q/K에 QK-Clip 적용)
  optimizer = create_hybrid_optimizer(model, 'muon-clip', muon_lr=0.02, adamw_lr=3e-4,
                                      head_dim=64, clip_threshold=1.0)
"""

import torch


# ╔══════════════════════════════════════════════════════════════╗
# ║  Newton-Schulz Orthogonalization (from KellerJordan/Muon)   ║
# ╚══════════════════════════════════════════════════════════════╝

def zeropower_via_newtonschulz5(G, steps=5):
    """
    Quintic Newton-Schulz iteration for matrix orthogonalization.
    Coefficients (a, b, c) maximize slope at zero for fastest convergence.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# ╔══════════════════════════════════════════════════════════════╗
# ║  HybridMuonAdamW Optimizer                                  ║
# ╚══════════════════════════════════════════════════════════════╝

class HybridMuonAdamW(torch.optim.Optimizer):
    """
    Hybrid optimizer: Muon for 2D weight matrices, AdamW for the rest.

    Param groups should contain 'use_muon' flag:
      - use_muon=True:  Newton-Schulz orthogonalized Muon update
      - use_muon=False: Standard AdamW update
      - use_qk_clip=True (with use_muon=True): QK-Clip post-step on all params in group
    """

    def __init__(self, param_groups):
        for group in param_groups:
            if group.get('use_muon', False):
                group.setdefault('lr', 0.02)
                group.setdefault('momentum', 0.95)
                group.setdefault('weight_decay', 0.0)
                group.setdefault('ns_steps', 5)
                group.setdefault('nesterov', True)
                group.setdefault('use_qk_clip', False)
                group.setdefault('head_dim', 64)
                group.setdefault('clip_threshold', 1.0)
            else:
                group.setdefault('lr', 3e-4)
                group.setdefault('betas', (0.9, 0.95))
                group.setdefault('eps', 1e-8)
                group.setdefault('weight_decay', 0.0)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get('use_muon', False):
                self._muon_step(group)
            else:
                self._adam_step(group)

        return loss

    def _muon_step(self, group):
        lr = group['lr']
        momentum = group['momentum']
        wd = group['weight_decay']
        ns_steps = group['ns_steps']
        nesterov = group['nesterov']
        use_qk_clip = group.get('use_qk_clip', False)
        head_dim = group.get('head_dim', 64)
        clip_threshold = group.get('clip_threshold', 1.0)

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state['momentum_buffer'] = torch.zeros_like(p)

            buf = state['momentum_buffer']
            buf.lerp_(grad, 1 - momentum)

            if nesterov:
                update = grad.lerp(buf, momentum)
            else:
                update = buf.clone()

            # Newton-Schulz orthogonalization (2D params only)
            orig_shape = update.shape
            if update.ndim >= 2:
                if update.ndim > 2:
                    update = update.view(update.size(0), -1)
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
                update = update.view(orig_shape)

            # Weight decay
            if wd > 0:
                p.mul_(1 - lr * wd)

            p.add_(update, alpha=-lr)

            # QK-Clip: per-head weight norm clipping
            if use_qk_clip and p.ndim == 2:
                num_heads = p.size(0) // head_dim
                if num_heads > 0 and p.size(0) % head_dim == 0:
                    w = p.view(num_heads, head_dim, -1)
                    rms = w.pow(2).mean(dim=(1, 2)).sqrt()  # (num_heads,)
                    scale = torch.clamp(clip_threshold / (rms + 1e-8), max=1.0)
                    p.view(num_heads, head_dim, -1).mul_(scale.view(-1, 1, 1))

    def _adam_step(self, group):
        lr = group['lr']
        betas = group['betas']
        eps = group['eps']
        wd = group['weight_decay']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] = 0

            state['step'] += 1
            step = state['step']

            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']

            exp_avg.lerp_(grad, 1 - betas[0])
            exp_avg_sq.lerp_(grad.square(), 1 - betas[1])

            bc1 = exp_avg / (1 - betas[0] ** step)
            bc2 = exp_avg_sq / (1 - betas[1] ** step)

            update = bc1 / (bc2.sqrt() + eps)

            if wd > 0:
                p.mul_(1 - lr * wd)

            p.add_(update, alpha=-lr)


# ╔══════════════════════════════════════════════════════════════╗
# ║  Factory Function                                            ║
# ╚══════════════════════════════════════════════════════════════╝

def create_hybrid_optimizer(model, optimizer_type='muon', muon_lr=0.02, adamw_lr=3e-4,
                            weight_decay=0.0, head_dim=64, clip_threshold=1.0):
    """
    Create a HybridMuonAdamW optimizer for a model.

    Routing rules:
      - 2D weight matrices (gate_proj, down_proj, up_proj, q/k/v/o_proj 등) → Muon
      - Q/K projections (muon-clip mode only) → Muon + QK-Clip
      - Embeddings, LayerNorm, biases, lm_head → AdamW

    Args:
        model: The model to optimize (only requires_grad params are included)
        optimizer_type: 'muon' or 'muon-clip'
        muon_lr: Learning rate for Muon params
        adamw_lr: Learning rate for AdamW params
        weight_decay: Weight decay for all params
        head_dim: Attention head dim (for MuonClip QK-Clip)
        clip_threshold: Per-head RMS clip threshold (for MuonClip)

    Returns:
        HybridMuonAdamW optimizer instance
    """
    muon_params = []
    qk_clip_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_1d = param.ndim < 2
        is_embed = 'embed' in name.lower() or 'wte' in name.lower()
        is_norm = 'norm' in name.lower() or 'ln_' in name.lower()
        is_head = 'lm_head' in name.lower()

        if is_1d or is_embed or is_norm or is_head:
            adamw_params.append(param)
        elif optimizer_type == 'muon-clip' and ('q_proj' in name or 'k_proj' in name):
            qk_clip_params.append(param)
        else:
            muon_params.append(param)

    groups = []
    if muon_params:
        groups.append({
            'params': muon_params,
            'use_muon': True,
            'lr': muon_lr,
            'weight_decay': weight_decay,
            'use_qk_clip': False,
        })
    if qk_clip_params:
        groups.append({
            'params': qk_clip_params,
            'use_muon': True,
            'lr': muon_lr,
            'weight_decay': weight_decay,
            'use_qk_clip': True,
            'head_dim': head_dim,
            'clip_threshold': clip_threshold,
        })
    if adamw_params:
        groups.append({
            'params': adamw_params,
            'use_muon': False,
            'lr': adamw_lr,
            'weight_decay': weight_decay,
        })

    n_muon = sum(p.numel() for p in muon_params)
    n_qk = sum(p.numel() for p in qk_clip_params)
    n_adamw = sum(p.numel() for p in adamw_params)
    total = n_muon + n_qk + n_adamw

    print(f"  HybridMuonAdamW ({optimizer_type}):")
    print(f"    Muon params:     {n_muon:>12,} ({n_muon/total*100:.1f}%)")
    if qk_clip_params:
        print(f"    MuonClip params: {n_qk:>12,} ({n_qk/total*100:.1f}%)")
    print(f"    AdamW params:    {n_adamw:>12,} ({n_adamw/total*100:.1f}%)")
    print(f"    Muon LR: {muon_lr}, AdamW LR: {adamw_lr}")

    return HybridMuonAdamW(groups)

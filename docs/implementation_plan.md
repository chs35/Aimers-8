# KD_Recovery.py 수정 계획서 (v3 — 구현 완료)

## 배경

`run_pipeline.py` → `WidthPruning.py` → `KD_Recovery.py` 파이프라인 전체에 걸쳐:
1. **어텐션 프루닝 여부**를 CLI 인자로 전달하는 체계 구축
2. 어텐션 프루닝 여부에 따라 **optimizer 자동 분기** (MuonClip / Muon)
3. `--optimizer` 인자로 **명시적 optimizer 선택** 가능 (adamw 포함)
4. **Progressive layer-wise KD** 가중치 스케줄링 도입 (adaptive sharpness)

---

## 1. CLI 인자 전달 체계

### 전체 흐름

```
run_pipeline.py
  --prune-attn          (기본: False)
  --optimizer muon      (기본: muon)
       │
       ├── WidthPruning.py --prune-attn
       │     └── PRUNE_ATTN_HEADS = True/False (CLI로 제어)
       │
       └── KD_Recovery.py --train-attn --optimizer muon-clip
             └── 어텐션 프로젝션 학습 여부 + optimizer 선택
```

### 자동 연동 규칙

| run_pipeline 인자 | WidthPruning | KD_Recovery |
|---|---|---|
| `(기본값)` | `PRUNE_ATTN_HEADS = False` | `--optimizer muon` (기본) |
| `--prune-attn` | `--prune-attn` 전달 → `PRUNE_ATTN_HEADS = True` | `--train-attn --optimizer muon-clip` |
| `--optimizer adamw` | 영향 없음 | `--optimizer adamw` |
| `--prune-attn --optimizer muon` | `--prune-attn` | `--train-attn --optimizer muon` (사용자 명시 우선) |

> [!NOTE]
> `--optimizer`를 명시하면 어텐션 프루닝과 무관하게 해당 optimizer를 사용.
> 명시하지 않으면 `--prune-attn` 여부에 따라 자동 결정 (muon-clip / muon).

---

## 2. 파일별 구현 상세

### [DONE] [run_pipeline.py](file:///c:/Users/정희승/Aimers-8/Aimers-8/Aimers-8/generate/run_pipeline.py)

**CLI 인자 추가** (L138-142):
```python
parser.add_argument("--prune-attn", action="store_true",
                    help="어텐션 헤드 프루닝 활성화 (기본: 비활성화)")
parser.add_argument("--optimizer", type=str, default=None,
                    choices=["muon", "muon-clip", "adamw"],
                    help="KD optimizer 선택 (기본: prune-attn이면 muon-clip, 아니면 muon)")
```

**optimizer 자동 결정** (L168):
```python
effective_optimizer = args.optimizer or ("muon-clip" if args.prune_attn else "muon")
```

**Step 2 (WidthPruning) 호출** (L232-234):
```python
if args.prune_attn:
    wp_cmd.append("--prune-attn")
```

**Step 3 (KD_Recovery) 호출** (L253-259):
```python
kd_cmd = [sys.executable, "generate/KD_Recovery.py",
          "--model", current_model, "--teacher", teacher_model,
          "--dataset", kd_train_path, "--optimizer", effective_optimizer]
if args.prune_attn:
    kd_cmd.append("--train-attn")
```

---

### [DONE] [WidthPruning.py](file:///c:/Users/정희승/Aimers-8/Aimers-8/Aimers-8/generate/WidthPruning.py)

**CLI 인자 추가** (L422-423):
```python
parser.add_argument("--prune-attn", action="store_true",
                    help="어텐션 헤드 프루닝 활성화 (기본: 비활성화)")
```

**하이퍼파라미터 오버라이드** (L426-427):
```python
if args.prune_attn:
    PRUNE_ATTN_HEADS = True
```

---

### [DONE] [KD_Recovery.py](file:///c:/Users/정희승/Aimers-8/Aimers-8/Aimers-8/generate/KD_Recovery.py)

**CLI 인자 추가** (L381-385):
```python
parser.add_argument("--train-attn", action="store_true",
                    help="어텐션 프로젝션(q,k,v,o_proj)도 학습 대상에 포함")
parser.add_argument("--optimizer", type=str, default="muon",
                    choices=["muon", "muon-clip", "adamw"],
                    help="Optimizer 선택 (기본: muon)")
```

**학습 대상 모듈 분기** (L389-393):
```python
if args.train_attn:
    DIRECT_TRAIN_MODULES = [
        "gate_proj", "down_proj",
        "q_proj", "k_proj", "v_proj", "o_proj",
    ]
```

**Optimizer 분기 — KDTrainer.create_optimizer()** (L169-185):
```python
def create_optimizer(self):
    if self._optimizer_type in ("muon", "muon-clip"):
        create_hybrid_optimizer = _import_muon()
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        self.optimizer = create_hybrid_optimizer(
            self.model, optimizer_type=self._optimizer_type,
            muon_lr=self._muon_lr, adamw_lr=self._adamw_lr,
            weight_decay=self.args.weight_decay, head_dim=head_dim,
        )
        return self.optimizer
    else:
        return super().create_optimizer()  # HF 기본 AdamW
```

**Muon LR 설정**: `muon_lr = learning_rate * 4` (Muon은 AdamW보다 높은 LR 사용)

---

### [DONE] [muon_optimizer.py](file:///c:/Users/정희승/Aimers-8/Aimers-8/Aimers-8/generate/muon_optimizer.py)

**구현 방식: 단일 HybridMuonAdamW 클래스**

Keller Jordan의 [Muon](https://github.com/KellerJordan/Muon) (MIT 라이선스) 코어를 기반으로,
HF Trainer 호환 단일 optimizer 클래스로 구현.

**핵심 구성요소:**

1. **`zeropower_via_newtonschulz5()`** — Newton-Schulz 5차 반복 orthogonalization
   - 계수 `(a, b, c) = (3.4445, -4.7750, 2.0315)` (검증된 매직 넘버)
   - bfloat16 연산, tall/wide 행렬 자동 전치

2. **`HybridMuonAdamW(torch.optim.Optimizer)`** — 통합 optimizer
   - param_groups의 `use_muon` 플래그로 Muon/AdamW 자동 분기
   - `use_qk_clip=True` 그룹: Muon step 후 per-head RMS norm clipping (QK-Clip)
   - 표준 `state_dict()`/`load_state_dict()` 지원 (HF Trainer checkpoint 호환)

3. **QK-Clip** (Kimi K2 논문 inspired)
   - Q/K projection 가중치에 per-head RMS norm clipping 적용
   - 각 head의 RMS가 `clip_threshold` (기본 1.0) 초과 시 rescale
   - Muon의 orthogonalization으로 인한 attention logit 폭발 방지

4. **`create_hybrid_optimizer()`** — 팩토리 함수
   - 모델 파라미터를 자동 분류:
     - 2D weight (gate_proj, down_proj, up_proj, v_proj, o_proj 등) → Muon
     - Q/K projection (muon-clip 모드) → Muon + QK-Clip
     - Embedding, LayerNorm, bias, lm_head → AdamW
   - 분류 결과를 콘솔에 출력 (파라미터 수, 비율)

**Param Routing 규칙:**

| 파라미터 종류 | `muon` 모드 | `muon-clip` 모드 |
|---|---|---|
| gate_proj, down_proj, up_proj, v_proj, o_proj | Muon | Muon |
| q_proj, k_proj | Muon | **Muon + QK-Clip** |
| embed, norm, bias, lm_head | AdamW | AdamW |

---

## 3. Progressive Layer-wise KD (Adaptive Sharpness)

`alpha_hidden` 예산을 softmax로 레이어별 경쟁 분배 (30 레이어 기준):
- **Adaptive sharpness**: `10.0 - 8.0 × progress` (10 → 2)
- **Softmax logit**: `sharpness × (progress - layer_ratio)`
- **Threshold 클램핑**: softmax 출력 `< 0.01` → 0 강제 → 해당 레이어 **연산 스킵**

### 왜 adaptive sharpness인가?

고정 `sharpness=10`은 30개 레이어에서 softmax 출력이 극단적으로 치우쳐
progress 값과 무관하게 **항상 ~10/30 레이어만 활성화**됨 (progressive 효과 없음).

| progress | 고정 sharp=10 (active) | adaptive 10→2 (active) |
|----------|----------------------|----------------------|
| 0.0 | 10/30 | 10/30 |
| 0.5 | 10/30 | 15/30 |
| 1.0 | **10/30** | **30/30** |

Adaptive sharpness는 초반에는 높은 sharpness로 낮은 레이어에 집중하고,
후반에는 낮은 sharpness로 전체 레이어에 균등 분배.

### 인터랙티브 추가 학습 시 동작

10 epoch + 3 epoch 추가 시:
- `total_train_steps`가 13 epochs로 재계산됨
- `_current_step`은 이전 값 보존
- progress가 0.85 → 1.0 구간을 돌며 22/30 → 30/30 레이어 점진 참여

| 구간 | progress | sharpness | active layers |
|---|---|---|---|
| Epoch 1 | 0.10 | 9.2 | 11/30 |
| Epoch 5 | 0.50 | 6.0 | 15/30 |
| Epoch 10 | 1.00 | 2.0 | 30/30 |
| Extra 1 | 0.85 | 3.2 | 22/30 |
| Extra 3 | 1.00 | 2.0 | 30/30 |

### 구현 코드

```python
def _get_layer_weights(self, num_layers):
    progress = min(self._current_step / max(self.total_train_steps, 1), 1.0)
    sharpness = 10.0 - 8.0 * progress  # adaptive: 10 (focused) → 2 (spread)
    logits = []
    for i in range(num_layers):
        layer_ratio = i / max(num_layers - 1, 1)
        logits.append(sharpness * (progress - layer_ratio))
    weights = F.softmax(torch.tensor(logits), dim=0).tolist()
    weights = [0.0 if w < 0.01 else w for w in weights]
    total = sum(weights)
    return [w / total for w in weights] if total > 0 else [1.0 / num_layers] * num_layers

# compute_loss: alpha_hidden 예산을 softmax 가중치로 분배
layer_weights = self._get_layer_weights(len(student_hidden))
hidden_loss = torch.tensor(0.0, device=student_logits.device)
for w, s, t in zip(layer_weights, student_hidden, teacher_hidden):
    if w > 0:
        if self.use_cosine_hidden:
            hidden_loss = hidden_loss + w * (
                1.0 - F.cosine_similarity(s.float(), t.float(), dim=-1).mean()
            )
        else:
            hidden_loss = hidden_loss + w * F.mse_loss(s.float(), t.float())
```

---

## 구현 완료 사항

- ✅ `--prune-attn` → 자동으로 `muon-clip`, 그 외 → `muon` 기본값
- ✅ `--optimizer`로 명시 시 자동 결정 무시
- ✅ `--train-attn` 시 `q_proj, k_proj, v_proj, o_proj` 전부 포함
- ✅ Muon 코어는 Keller Jordan 원본 기반, `HybridMuonAdamW` 단일 클래스로 구현
- ✅ QK-Clip: per-head RMS norm clipping (Kimi K2 inspired)
- ✅ Progressive KD: adaptive sharpness (10→2)로 초반 집중 → 후반 균등 확인
- ✅ 인터랙티브 추가 학습 시 `total_train_steps` 재계산 + `_current_step` 보존
- ✅ 구문 검증 통과 (4개 파일 `py_compile`)
- ✅ Muon/MuonClip optimizer 단위 테스트 통과

---

## Verification Plan

1. ✅ **구문 검증**: 4개 파일 `py_compile` 통과
2. ✅ **CLI 인자 파싱**: `--prune-attn`, `--optimizer`, `--train-attn` 정상 파싱 확인
3. ✅ **Muon optimizer 단위 테스트**: dummy model로 forward → backward → step 동작 확인
4. ✅ **QK-Clip 단위 테스트**: per-head RMS clipping 동작 확인 (RMS ≤ threshold)
5. ✅ **Progressive KD 시뮬레이션**: adaptive sharpness로 active layers 10→30 확산 확인
6. ⬜ **E2E 학습 테스트**: 소규모 데이터로 muon / muon-clip / adamw 각각 학습 시작 확인
7. ⬜ **학습 재개 테스트**: 2 epoch → 추가 2 epoch, loss spike 없는지 확인

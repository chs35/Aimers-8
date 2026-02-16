# KD_Recovery.py 수정 계획서 (v2)

## 배경

`run_pipeline.py` → `WidthPruning.py` → `KD_Recovery.py` 파이프라인 전체에 걸쳐:
1. **어텐션 프루닝 여부**를 CLI 인자로 전달하는 체계 구축
2. 어텐션 프루닝 여부에 따라 **optimizer 자동 분기** (Muon Clip / Muon)
3. `--optimizer` 인자로 **명시적 optimizer 선택** 가능 (adamw 포함)
4. **Progressive layer-wise KD** 가중치 스케줄링 도입

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

## 2. 파일별 수정 상세

### [MODIFY] [run_pipeline.py](file:///c:/Users/정희승/Aimers-8/Aimers-8/Aimers-8/generate/run_pipeline.py)

**CLI 인자 추가:**
```python
parser.add_argument("--prune-attn", action="store_true",
                    help="어텐션 헤드 프루닝 활성화 (기본: 비활성화)")
parser.add_argument("--optimizer", type=str, default=None,
                    choices=["muon", "muon-clip", "adamw"],
                    help="KD optimizer 선택 (기본: prune-attn이면 muon-clip, 아니면 muon)")
```

**Step 2 (WidthPruning) 호출 수정:**
```python
wp_cmd = [sys.executable, "generate/WidthPruning.py",
          "--model", current_model, "--dataset", wp_calib_path,
          "--target-size", str(args.target_size)]
if args.prune_attn:
    wp_cmd.append("--prune-attn")
```

**Step 3 (KD_Recovery) 호출 수정:**
```python
# optimizer 결정: 명시 > 자동(prune-attn 기반)
optimizer = args.optimizer or ("muon-clip" if args.prune_attn else "muon")

kd_cmd = [sys.executable, "generate/KD_Recovery.py",
          "--model", current_model, "--teacher", teacher_model,
          "--dataset", kd_train_path, "--optimizer", optimizer]
if args.prune_attn:
    kd_cmd.append("--train-attn")
```

---

### [MODIFY] [WidthPruning.py](file:///c:/Users/정희승/Aimers-8/Aimers-8/Aimers-8/generate/WidthPruning.py)

**CLI 인자 추가:**
```python
parser.add_argument("--prune-attn", action="store_true",
                    help="어텐션 헤드 프루닝 활성화 (기본: 비활성화)")
```

**하이퍼파라미터 오버라이드:**
```python
# 기존 L58의 PRUNE_ATTN_HEADS = False 를 CLI로 제어
if args.prune_attn:
    PRUNE_ATTN_HEADS = True
```

---

### [MODIFY] [KD_Recovery.py](file:///c:/Users/정희승/Aimers-8/Aimers-8/Aimers-8/generate/KD_Recovery.py)

**CLI 인자 추가:**
```python
parser.add_argument("--train-attn", action="store_true",
                    help="어텐션 프로젝션(q,k,v,o_proj)도 학습 대상에 포함")
parser.add_argument("--optimizer", type=str, default="muon",
                    choices=["muon", "muon-clip", "adamw"],
                    help="Optimizer 선택 (기본: muon)")
```

**학습 대상 모듈 분기:**
```python
# Direct Training 모드에서 학습 대상 결정
if args.train_attn:
    DIRECT_TRAIN_MODULES = [
        "gate_proj", "down_proj",
        "q_proj", "k_proj", "v_proj", "o_proj",  # 어텐션 추가
    ]
else:
    DIRECT_TRAIN_MODULES = ["gate_proj", "down_proj"]  # 기존 설정
```

**Optimizer 분기:**
```python
# KDTrainer.create_optimizer() 오버라이드
if args.optimizer == "muon-clip":
    # 2D params → MuonClip, 나머지 → AdamW
elif args.optimizer == "muon":
    # 2D params → Muon, 나머지 → AdamW
else:  # adamw
    # 기존 HF Trainer 기본 AdamW 사용
```

---

### [NEW] [muon_optimizer.py](file:///c:/Users/정희승/Aimers-8/Aimers-8/Aimers-8/generate/muon_optimizer.py)

**구현 방식: GitHub 원본 코어 + 커스텀 래퍼**

1. **코어** — [Keller Jordan의 Muon 구현체](https://github.com/KellerJordan/Muon) (MIT 라이선스)에서 핵심 `Muon` 클래스 가져오기
   - Newton-Schulz orthogonalization (검증된 매직 넘버 계수 포함)
   - SGD momentum + orthogonalization 파이프라인
   - `state_dict()` / `load_state_dict()` 이미 지원
2. **래퍼(직접 작성)**
   - `QKClipWrapper`: QK-Clip 로직 추가 (attention head 파라미터 adaptive rescaling)
   - `HybridOptimizer`: 2D params → Muon/MuonClip, 나머지 → AdamW 자동 분배
   - HF Trainer의 `create_optimizer()` 호환 인터페이스

---

## 3. Progressive Layer-wise KD

`alpha_hidden` 예산을 softmax로 레이어별 경쟁 분배 (30 레이어 기준):
- **Softmax logit**: `sharpness × (progress - layer_ratio)` → 학습 초반에는 낮은 레이어에 집중, 후반에는 균등 분배
- **Threshold 클램핑**: softmax 출력 `< 0.01` → 0 강제 → 해당 레이어 **연산 스킵**

```python
def _get_layer_weights(self, num_layers):
    progress = min(self._current_step / self.total_train_steps, 1.0)
    logits = []
    for i in range(num_layers):
        layer_ratio = i / max(num_layers - 1, 1)
        logits.append(10.0 * (progress - layer_ratio))
    weights = F.softmax(torch.tensor(logits), dim=0).tolist()
    weights = [0.0 if w < 0.01 else w for w in weights]
    total = sum(weights)
    return [w / total for w in weights] if total > 0 else [1.0 / num_layers] * num_layers

# compute_loss: alpha_hidden 예산을 softmax 가중치로 분배
layer_weights = self._get_layer_weights(len(student_hidden))
hidden_loss = torch.tensor(0.0, device=student_logits.device)
for w, s, t in zip(layer_weights, student_hidden, teacher_hidden):
    if w > 0:
        hidden_loss = hidden_loss + w * (
            1.0 - F.cosine_similarity(s.float(), t.float(), dim=-1).mean()
        )
```

---

## 결정 완료 사항

- ✅ `--prune-attn` → 자동으로 `muon-clip`, 그 외 → `muon` 기본값
- ✅ `--optimizer`로 명시 시 자동 결정 무시
- ✅ `--train-attn` 시 `q_proj, k_proj, v_proj, o_proj` 전부 포함
- ✅ Muon 코어는 GitHub 원본에서 가져오고, HybridOptimizer 래퍼만 직접 작성

---

## Verification Plan

1. **CLI 인자 전달 테스트**: `run_pipeline.py --prune-attn --optimizer muon-clip`으로 실행하여 각 서브프로세스에 올바른 인자 전달되는지 `--help` 또는 dry-run으로 확인
2. **Optimizer 분기 테스트**: 소규모 데이터로 muon / muon-clip / adamw 각각 학습 시작 확인
3. **학습 재개 테스트**: 2 epoch → 추가 2 epoch, loss spike 없는지 확인
4. **Progressive 가중치 로깅**: 레이어별 가중치 변화 콘솔 출력

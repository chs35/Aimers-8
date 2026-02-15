## 하이퍼파라미터에 따른 모델 요약 결과

***
### Trial 1
Sparsity 0.25로 설정 시 14.8% 감소

벤치마크 성능

|태스크|TPOT(ms/token)|정답률(%)|
|-----|----|-----|
|gsm8k|2.53|0.1706, 0.1850|
|gsm8k(reasoning)|2.46|0.1622, 0.1729|

***

### Trial2
Sparsity 0.12로 설정 시 7.4% 감소(예상 목표치)

벤치마크 성능
|태스크|TPOT(ms/token)|정답률(%)|
|-----|----|-----|
|gsm8k|2.43|0.4117, 0.4412|

제출 시 점수 : 0.272

### KD Trial1(KD - LoRA)
Sparsity 0.12로 설정 시 파라미터 7.4% 감소(예상 목표치)


벤치마크 성능
|태스크|TPOT(ms/token)|정답률(%)|
|-----|----|-----|
|gsm8k|2.64|0.4678, 0.4754|

성능은 회복하였으나 크지 않음

### KD Trial2(KD - LoRA)
Sparsity 0.08로 설정 시 파라미터 추가 감소
intermediate 3744

벤치마크 성능
|태스크|TPOT(ms/token)|정답률(%)|
|-----|----|-----|
|gsm8k|1.50|0.5504, 5656|

성능 방어가 완벽하지는 않지만 효과적, 가속 효과 기대 가능.
제출 시 점수 : 0.548

### KD Trial3(KD - LoRA)
Sparsity 0.15로 설정
intermediate 3584와 3744 사이

성능은 손실이 크지만, KD의 효과를 관찰할 수 있었음
제출 시 점수 : 0.380

### KD Trial4(KD - LoRA)
Sparsity 0.12로 설정
intermediate 3584

성능은 손실이 크지만, KD의 효과를 관찰할 수 있었음
제출 시 점수 : 0.45

### KD Trial5(KD - LoRA)
Sparsity 0.08로 설정
intermediate 3744

벤치마크 성능
|태스크|TPOT(ms/token)|정답률(%)|
|-----|----|-----|
|gsm8k|1.50|0.5527, 0.5671|

더 강한 KD 진행
|LoRA Rank|LoRA Alpha|Epoch|
|---------|-----------|-----|
|128|256|7|

벤치 차이가 적어 차후 시도에서도 LoRA는 사용할 예정이며, 이 Sparsity에서 KD의 강도는 이번 Trial 정도가 적절하다고 판단.

### KD Trial 6(KD w/o LoRA)
확인을 위해 LoRA 없이 KD 진행, Sparsity는 Trial5와 동일, epoch도 7로 동일

벤치마크 성능
|태스크|TPOT(ms/token)|정답률(%)|
|-----|----|-----|
|gsm8k|2.41|0.5610,0.5754|

### KD Trial7(KD - LoRA)
Sparsity 0.08로 설정
intermediate 3744


|개선된 KD 기법|
|------------|
|Minitron 방식에 기반한 Temperature(T=1)|
|Adaptive KLD|
|CosineSimilarity for Hidden State|
|Dynamic Alpha|


|태스크|TPOT(ms/token)|정답률(%)|
|-----|----|-----|
|gsm8k|1.52|0.5406, 0.5512|


|LoRA Rank|LoRA Alpha|Epoch|
|---------|-----------|-----|
|64|128|5|

### KD Trial8(KD w/o LoRA)
Sparsity 0.08로 설정
intermediate 3744


epoch 12


|개선된 KD 기법|
|------------|
|Minitron 방식에 기반한 Temperature(T=1)|
|Adaptive KLD|
|CosineSimilarity for Hidden State|
|Dynamic Alpha|


|태스크|TPOT(ms/token)|정답률(%)|
|-----|----|-----|
|gsm8k|||

### 데이터셋 샘플링 후 시도
|파이프라인 순서|
|-------------|
|Vocabulary Pruining|
|Width Pruning with Intermediate Size 3584|
|KD recovery|
|FP8 static Quantization|

|태스크|Overall generate TPS(tokens/s)|정답률|
|-----|----|-----|
|gsm8k|503.72|0.5390, 0.5792|
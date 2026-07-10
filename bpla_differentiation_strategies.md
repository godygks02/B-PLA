# B-PLA Multiplier 차별화 방향 분석

## 질문에 대한 직접 답변

### energy_model.py의 sweep (terms/prefix/mant 최적 조합 탐색) 자체가 novelty가 되는가?

> [!WARNING]
> **솔직한 답: 그것만으로는 novelty가 약합니다.**

이유:
- **"설계 파라미터를 sweep해서 최적 조합을 찾는다"**는 것은 하드웨어 설계에서 보편적인 design space exploration (DSE)
- PAM 논문 자체가 이미 "configurable" 근사 곱셈기를 표방하며, 정확도-복잡도 트레이드오프를 다단계로 조절
- HAM도 기울기 PoT 비트 수를 변경하여 정확도-면적 트레이드오프를 제시
- 즉 "여러 파라미터를 조절해서 최적점을 찾을 수 있다" 자체는 이미 선행 연구들이 하고 있는 일

**하지만** — 이 sweep 능력을 **어떻게** 활용하느냐에 따라 강한 novelty가 될 수 있습니다.

---

## 기존 연구들의 실제 한계점 (Limitation)

기존 근사 곱셈기 연구들(HAM, PAM, ApproxLP)이 공통적으로 가진 구조적 한계:

| 한계점 | 설명 |
|--------|------|
| **① 고정 구성(Static Config)** | 하드웨어 설계 시점에 prefix bits, 계수 정밀도 등이 고정됨. 모든 레이어, 모든 연산에 동일한 근사 수준 적용 |
| **② 곱셈기 전용 설계** | 곱셈만 근사하고, 활성화 함수는 별도 유닛으로 처리. 두 유닛의 하드웨어 자원이 이중으로 소모됨 |
| **③ Generic 계수** | 타일 계수가 수학적 Taylor 전개 또는 최소자승법으로만 결정. 실제 NN 데이터 분포를 반영하지 않음 |
| **④ 단일 정밀도 대상** | 대부분 FP32 하나의 포맷만 대상. mixed-precision 환경 고려 없음 |
| **⑤ NN 맥락 부재** | 산술 레벨 에러만 보고. end-to-end 모델 정확도-에너지 Pareto는 미제공 |

---

## 차별화 방향 5가지 제안

### 방향 1: ⭐⭐⭐ Layer-Adaptive B-PLA Configuration (가장 유망)

**핵심 아이디어:**
```
기존: 모든 레이어에 동일한 (prefix=4, terms=1, mant=24) 적용
B-PLA: 레이어별 민감도에 따라 (prefix, terms, mant) 조합을 자동으로 다르게 할당
```

**구체적 방법:**
```python
# 레이어별 최적 B-PLA 구성 자동 할당 알고리즘
for layer in model.layers:
    sensitivity = measure_layer_sensitivity(layer, calibration_data)
    config = find_pareto_optimal_config(
        accuracy_budget=sensitivity,
        energy_model=bpla_energy_model,
        search_space={
            'prefix_bits': [2, 3, 4, 5, 6],
            'terms': [1, 2, 3],
            'mant_bits': [12, 16, 20, 24]
        }
    )
    layer.bpla_config = config  # 레이어마다 다른 구성!
```

**왜 novelty인가:**
- HAM/PAM: 설계 시점에 **하나의** 구성으로 고정 → 모든 레이어에 동일 적용
- B-PLA: 레이어별 **다른** (prefix, terms, mant) 조합을 할당하는 **자동 할당 프레임워크**
- 이것은 energy_model.py의 sweep을 **레이어별 민감도 분석과 결합**한 것
- mixed-precision quantization처럼 "mixed-approximation"이라는 새로운 축

**선행 연구 위험도:** 중간
- "layer-wise approximation"이라는 일반 개념은 이미 연구됨 (QoS-Nets 등)
- 하지만 **"prefix bits + dyadic terms + mantissa precision"이라는 3차원 구성 공간에서 레이어별 Pareto 최적 할당**은 매우 구체적이고 B-PLA 고유

> [!TIP]
> **이것이 energy_model.py sweep을 novelty로 바꾸는 방법입니다.** Sweep 자체가 기여가 아니라, "sweep 결과를 레이어별로 다르게 적용하는 자동 할당 프레임워크"가 기여입니다.

---

### 방향 2: ⭐⭐⭐ Data-Aware Calibrated Coefficients

**핵심 아이디어:**
```
기존 HAM/PAM: 수학적으로 결정된 계수 (Taylor center, 최소자승)
B-PLA: 실제 모델의 weight/activation 분포를 반영한 calibrated 계수
```

**구체적 방법:**
```python
# 현재 B-PLA: 수학적 center에서 Taylor 전개
a_ij = nu_j  # 타일 중심값 그대로

# 개선된 B-PLA: 실제 데이터 분포 기반 가중 최소자승
for tile (i, j):
    # 해당 타일에 실제로 떨어지는 (M1, M2) 샘플 수집
    samples = collect_mantissa_pairs_in_tile(calibration_data, i, j)
    weights = compute_frequency_weights(samples)  # 자주 나오는 영역에 가중치
    a_ij, b_ij, c_ij = weighted_least_squares(samples, weights)
    a_ij, b_ij, c_ij = quantize_to_dyadic(a_ij, b_ij, c_ij)  # PoT 제한 하에 최적화
```

**왜 novelty인가:**
- HAM: PoT 계수를 쓰지만 **수학적으로** 결정 (데이터 무관)
- PAM: 에러 분포를 분석하지만 **모델 데이터와 무관한** 일반적 피팅
- B-PLA: training-free이므로 모델 가중치는 안 건드리지만, **계수 테이블은 calibration data로 최적화** 가능
- 이것은 "training-free model deployment ≠ calibration-free coefficient generation"이라는 B-PLA의 기존 철학과 정확히 일치

**선행 연구 위험도:** 낮음
- 근사 곱셈기 분야에서 "NN 데이터 분포를 반영한 계수 캘리브레이션"은 거의 연구되지 않음
- 기존 연구들은 곱셈기를 **범용 산술 유닛**으로 설계 → 특정 모델에 맞추지 않음

---

### 방향 3: ⭐⭐ Non-Uniform Adaptive Prefix Tiling

**핵심 아이디어:**
```
기존: 균일한 2^k × 2^k 타일 분할 (모든 타일 크기 동일)
B-PLA: 오차가 큰 영역은 더 세밀하게, 오차가 작은 영역은 더 거칠게 분할
```

**구체적 방법:**
```
M1×M2 서피스에서:
- (0,0) 근처: M1*M2 ≈ 0 → 오차 작음 → 큰 타일 사용 (prefix 2-bit)
- (1,1) 근처: M1*M2 ≈ 1 → 오차 큼 → 작은 타일 사용 (prefix 6-bit)
→ 전체 LUT 크기는 동일하면서 오차 분포가 균등해짐
```

**왜 novelty인가:**
- HAM/PAM: 모두 **균일 분할** 사용
- Flex-SFU: 활성화에서만 비균일 분할 → 하드웨어 비교기 필요
- B-PLA: 비균일 분할이지만 **prefix routing의 재매핑**으로 구현 → 비교기 불필요

**선행 연구 위험도:** 중간-높음
- 비균일 세그먼트 자체는 활성화 쪽에서 많이 연구됨
- 곱셈기 쪽에서는 상대적으로 드물지만, 개념 자체가 "알려진 개선 방향"

---

### 방향 4: ⭐⭐⭐ Unified Datapath의 실질적 증명 강화 (가장 안전)

**핵심 아이디어:**
```
통합 평가기(Shared Evaluator)의 novelty를 더 이상 "아이디어" 수준이 아니라
정량적 하드웨어 이점으로 증명
```

**구체적 방법:**
```
비교 대상 3가지:
A) 별도 구성: PAM 곱셈기 + ML-PLAC 활성화기 (각각 독립)
B) B-PLA 별도 구성: B-PLA 곱셈 + B-PLA 활성화 (독립이지만 둘 다 shift-add)
C) B-PLA 공유: 단일 Shared Evaluator (시분할 멀티플렉싱)

측정 항목: 면적, 전력, 임계 경로, 유틸리티
결과 기대: C가 A 대비 면적 30-50% 절감 (evaluator core 공유로)
```

**왜 novelty인가:**
- 이것은 기존 통합 아키텍처 novelty를 **"주장"에서 "증거"로** 전환하는 것
- 선행 연구 중 곱셈기+활성화 공유 하드웨어를 제시한 논문이 없으므로, 이 비교 자체가 새로운 결과
- **가장 안전한 novelty**: 완전히 새로운 아이디어가 아니라, 이미 확인된 고유 영역의 증명 강화

**선행 연구 위험도:** 매우 낮음

---

### 방향 5: ⭐⭐ ANN-to-SNN Bridge (VLM_SNN_Research 연계)

**핵심 아이디어:**
```
B-PLA의 bit-prefix routing이 SNN의 spike 인코딩과 자연스럽게 결합
→ ANN-to-SNN 변환 파이프라인에서 B-PLA가 중간 표현으로 작동
```

**구체적 방법:**
```
기존 ANN-to-SNN 변환:
  ANN activation → rate coding → spike train → SNN accumulation
  
B-PLA 연계:
  ANN activation → B-PLA bit-prefix decomposition → deterministic bit-slice spikes
  → prefix bits가 곧 spike timing/priority를 결정
  → 높은 prefix = 높은 priority spike, 낮은 prefix = 낮은 priority spike
```

**왜 novelty인가:**
- 기존 ANN-to-SNN 연구와 근사 곱셈기 연구가 완전히 분리되어 있음
- B-PLA의 bit-prefix 구조가 spike encoding과 수학적으로 대응된다는 연결은 새로움
- 연구자님의 `Universal_Training_free_ANN2SNN_conversion` 프로젝트와 직접 연계 가능

**선행 연구 위험도:** 매우 낮음 (이 교차 영역을 다룬 논문 거의 없음)

**단점:** 증명이 어렵고, 스토리 구축에 시간 소요

---

## 종합 비교 및 권장 우선순위

| 방향 | Novelty 수준 | 선행 위험도 | 구현 난이도 | 논문 임팩트 | 권장 순위 |
|------|-------------|------------|------------|------------|----------|
| **1. Layer-Adaptive Config** | ⭐⭐⭐ | 중간 | 중간 | 높음 | **🥇 1순위** |
| **2. Data-Aware Calibration** | ⭐⭐⭐ | 낮음 | 낮음 | 중-높 | **🥇 1순위 (동시)** |
| **4. Unified Datapath 정량 증명** | ⭐⭐⭐ | 매우 낮음 | 높음 (RTL 필요) | 높음 | **🥈 2순위** |
| **3. Non-Uniform Tiling** | ⭐⭐ | 중-높 | 중간 | 중간 | 🥉 3순위 |
| **5. ANN-to-SNN Bridge** | ⭐⭐ | 매우 낮음 | 높음 | 높음 (if proven) | 🏅 장기 |

---

## 추천 전략: 방향 1+2를 결합한 "Compound Novelty"

> [!IMPORTANT]
> **가장 강력한 조합:** Layer-Adaptive Config + Data-Aware Calibration을 통합하면, B-PLA 곱셈기가 기존 HAM/PAM과 명확하게 분리됩니다.

```
HAM/PAM의 한계:
  "하나의 고정된 근사 곱셈기를 설계하여, 
   모든 레이어에 동일하게 적용"

B-PLA의 차별화:
  "실제 모델 데이터 분포를 반영하여 calibrated된 dyadic 계수 테이블을 생성하고,
   레이어별 민감도에 따라 (prefix, terms, mant) 조합을 자동 할당하여,
   에너지-정확도 Pareto 최적점에서 training-free 추론을 수행"
```

이 프레이밍에서 energy_model.py의 sweep은 더 이상 단순한 DSE가 아니라:
1. **Calibrated coefficient fitting** → 데이터 기반 계수 최적화
2. **Layer-wise sensitivity-guided allocation** → 레이어별 자동 구성 할당
3. **Energy-accuracy Pareto analysis** → 정량적 트레이드오프 곡선

이 **세 단계를 포함하는 자동화된 deployment pipeline**이 됩니다.

---

## 즉시 구현 가능한 다음 단계

```
1. [지금] modules/calibration.py 작성
   → 실제 모델의 weight/activation 통계 수집
   → 타일별 빈도 기반 가중 최소자승 계수 피팅
   → dyadic 제약 하 최적화

2. [지금] experiments/layer_sensitivity.py 작성
   → 소형 모델(MLP, LeNet)에서 레이어별 B-PLA 대체 시 정확도 민감도 측정
   → 레이어별 최적 (prefix, terms, mant) 조합 할당 알고리즘

3. [다음] experiments/pareto_sweep.py 작성
   → 에너지 모델 × 레이어 민감도 × 캘리브레이션 결합
   → 모델별 에너지-정확도 Pareto 곡선 생성
```

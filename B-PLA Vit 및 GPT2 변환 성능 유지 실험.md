# B-PLA ViT 및 GPT-2 변환 성능 유지 실험

## 1. 문서 목적

본 문서는 B-PLA(Bit-Prefix Piecewise Linear Approximation)를 pretrained ViT와 GPT-2에 적용했을 때, 별도 재학습 없이 원본 ANN의 추론 성능과 예측을 어느 정도 유지할 수 있는지 정리한 예비 실험 보고서다.

평가 대상은 다음 두 연산군이다.

1. Transformer MLP의 GELU 활성화 함수
2. ViT의 `nn.Linear` 및 GPT-2의 `Conv1D` projection

B-PLA는 부동소수점 곱셈과 비선형 활성화를 bit-prefix routing, coefficient LUT, dyadic affine evaluation으로 근사한다. 현재 실험 코드는 PyTorch 기반 정확도 민감도 proxy이며, 실제 multiplierless RTL이나 하드웨어 전력 측정 결과는 아니다.

## 2. 공통 실험 설정

### 2.1 B-PLA 설정

- Affine path: dyadic
- Prefix bits: 주로 2, 3, 4 비교
- Dyadic terms: 주로 2, 4, 8 비교
- 활성화 LUT: 모델 내 모든 GELU가 하나의 공유 LUT 사용
- Multiplier LUT: 동일 B-PLA configuration을 사용하는 Linear/Conv1D가 공유
- 재학습 및 weight update: 없음

### 2.2 평가 지표

ViT:

- Top-1 accuracy
- Top-5 accuracy
- ANN-B-PLA Top-1 prediction agreement
- Logit MAE/RMSE

GPT-2:

- Perplexity(PPL)
- ANN 대비 PPL 절대 및 상대 변화
- ANN-B-PLA next-token prediction agreement
- Logit MAE/RMSE
- 평가 target token 수

Prediction agreement는 ANN과 B-PLA가 같은 입력에서 동일한 최종 class 또는 next token을 선택한 비율이다. 정확도나 PPL이 비슷하더라도 agreement가 낮다면 개별 예측은 상당히 달라졌을 수 있으므로, 성능 유지 여부를 판단할 때 두 지표를 함께 보아야 한다.

## 3. ViT 실험

### 3.1 기본 설정

- 평가 표본: 100 images
- Batch size: 4
- ANN 기준 성능: Top-1 89.00%, Top-5 99.00%
- GELU 모듈 수: 12
- 전체 Linear 모듈 수: 73

### 3.2 고정 범위 GELU 근사

모든 GELU에 고정 범위 `[-4, 4]`를 사용하고 Linear는 변환하지 않았다.

| Prefix | Terms | GELU 수 | Top-1 | Top-5 | Top-1 agreement | Logit MAE | Logit RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 12 | 76.00% | 96.00% | 85.00% | 4.478645e-01 | 6.028519e-01 |
| 4 | 4 | 12 | 84.00% | 99.00% | 93.00% | 3.244059e-01 | 4.485949e-01 |
| 4 | 8 | 12 | 84.00% | 99.00% | 93.00% | 3.243938e-01 | 4.485864e-01 |

Terms를 2에서 4로 늘리면 Top-1과 prediction agreement가 개선되지만, 4에서 8로 늘려도 추가 개선이 거의 없다. 이는 해당 설정에서 dyadic coefficient term 수보다 잘못 설정된 activation fitting 범위가 주된 오차 원인일 가능성을 보여준다.

### 3.3 전체 모델 범위 기반 GELU calibration

원본 ViT에 calibration 데이터를 입력하여 모든 GELU 입력의 최대 절댓값을 측정하고, 하나의 대칭 범위에서 공유 GELU LUT를 생성했다.

Calibration 4 batch에서 측정된 범위:

```text
[-48.598358, +48.598358]
```

| Calibration batch | Prefix | Terms | Top-1 | Top-5 | Agreement | Logit MAE | Logit RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 2 | 88.00% | 99.00% | 99.00% | 2.353025e-02 | 3.291051e-02 |
| 4 | 4 | 4 | 89.00% | 99.00% | 100.00% | 6.588561e-03 | 9.068713e-03 |
| 4 | 3 | 4 | 89.00% | 99.00% | 100.00% | 4.993882e-03 | 6.924042e-03 |
| 4 | 2 | 4 | 89.00% | 99.00% | 100.00% | 8.581975e-03 | 1.177507e-02 |
| 4 | 2 | 3 | 89.00% | 99.00% | 100.00% | 9.248994e-03 | 1.283539e-02 |
| 4 | 2 | 2 | 88.00% | 99.00% | 99.00% | 3.358812e-02 | 4.643598e-02 |

Global calibration을 적용하면 고정 `[-4,4]` 대비 성능과 agreement가 크게 회복된다. 특히 `prefix=2, terms=3`에서도 ANN과 동일한 Top-1/Top-5 및 100% prediction agreement가 관측되었다. LUT 크기와 shift-add 복잡도를 고려하면 `prefix=2, terms=3` 또는 더 단순한 `prefix=2, terms=2`가 정확도-복잡도 절충 후보가 될 수 있다.

### 3.4 Calibration batch 수 비교

Calibration 16 batch에서 측정된 범위는 다음과 같다.

```text
[-48.689213, +48.689213]
```

| Calibration batch | Prefix | Terms | Top-1 | Top-5 | Agreement | Logit MAE | Logit RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4 | 4 | 88.00% | 99.00% | 99.00% | 6.403038e-03 | 8.960539e-03 |
| 16 | 2 | 2 | 88.00% | 99.00% | 99.00% | 3.324863e-02 | 4.576619e-02 |

4 batch와 16 batch의 측정 범위 차이는 약 0.09로 작았다. Logit 오차도 유사했으므로 activation range는 소수 calibration batch만으로 상당히 안정화되는 것으로 보인다. 다만 Top-1이 89%와 88%로 한 표본 차이를 보였으므로, 현재 100-image 평가에서는 1개 표본 변화가 1%p 변화가 된다는 점을 고려해야 한다.

### 3.5 Linear-only 변환

GELU는 exact로 유지하고 `prefix=2, terms=2`로 Linear만 변환했다.

| 변환 Linear 수 | Top-1 | Top-5 | Agreement | Logit MAE | Logit RMSE |
|---:|---:|---:|---:|---:|---:|
| 8 | 89.00% | 99.00% | 100.00% | 1.424536e-03 | 2.117769e-03 |
| 16 | 89.00% | 99.00% | 100.00% | 1.891036e-03 | 2.768960e-03 |
| 73 (전체) | 89.00% | 99.00% | 100.00% | 4.710019e-03 | 6.142535e-03 |

8개 변환에는 첫 번째 Transformer block의 Q/K/V/O projection과 MLP FC1/FC2, 다음 block의 일부 Q/K projection이 포함되었다. 전체 73개 Linear를 변환해도 100-image subset에서는 Top-1/Top-5와 prediction agreement가 유지되었다.

이는 B-PLA multiplier approximation이 현재 설정에서 ViT Linear projection에 매우 작은 perturbation만 발생시킨다는 예비 근거다. 다만 전체 Linear 변환은 100 images, batch size 4에서 약 25분이 소요되었다. 이 시간은 B-PLA 하드웨어 latency가 아니라 elementwise product를 생성하는 PyTorch proxy의 비효율을 반영한다.

### 3.6 GELU와 전체 Linear 결합

Calibration 4 batch의 공유 GELU LUT와 전체 73개 Linear 변환을 결합했다.

| Prefix | Terms | Linear 수 | GELU 수 | Top-1 | Top-5 | Agreement | Logit MAE | Logit RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 73 | 12 | 88.00% | 99.00% | 99.00% | 3.451621e-02 | 4.764691e-02 |

ANN 대비 Top-1은 1%p 낮고 Top-5는 동일했으며 prediction agreement는 99%였다. 따라서 현재 평가 범위에서는 모든 GELU와 모든 Linear projection을 동시에 근사해도 성능 저하가 제한적이었다.

## 4. GPT-2 실험

### 4.1 기본 설정

- Model: GPT-2 base
- Max sequence length: 256
- Stride: 256
- 기본 평가 window: 4
- 4-window target tokens: 1,020
- ANN 기준 PPL: 51.9030
- GELU 모듈 수: 12
- 전체 Conv1D projection 수: 48

### 4.2 고정 범위 GELU 근사

모든 GELU에 `[-4,4]` 고정 범위를 사용하고 Conv1D는 변환하지 않았다.

| Prefix | Terms | B-PLA PPL | PPL 변화 | Token agreement | Logit MAE | Logit RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 50.4662 | -1.4369 (-2.77%) | 75.69% | 1.575659e+01 | 3.761209e+01 |
| 4 | 4 | 50.2457 | -1.6573 (-3.19%) | 86.67% | 1.009062e+01 | 2.657019e+01 |
| 4 | 8 | 50.2489 | -1.6542 (-3.19%) | 86.86% | 1.008807e+01 | 2.656792e+01 |
| 3 | 2 | 50.5231 | -1.3800 (-2.66%) | 75.98% | 1.567439e+01 | 3.796482e+01 |
| 2 | 2 | 50.8110 | -1.0921 (-2.10%) | 75.69% | 1.607736e+01 | 3.851754e+01 |

고정 범위 실험에서는 B-PLA PPL이 ANN보다 낮아졌다. 그러나 token agreement는 75.69~86.86%에 불과하고 logit 오차도 매우 크다. 따라서 이 결과를 언어모델 성능 개선으로 해석하기 어렵다. 고정 범위 clipping으로 모델의 출력 분포가 크게 달라졌지만, 작은 1,020-token 평가에서 우연히 평균 loss가 낮아졌을 가능성이 있다.

### 4.3 전체 모델 범위 기반 GELU calibration

전체 GPT-2 GELU에서 측정된 공유 대칭 범위는 다음과 같다.

```text
[-62.850513, +62.850513]
```

| Prefix | Terms | B-PLA PPL | PPL 변화 | Token agreement | Logit MAE | Logit RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 52.7179 | +0.8149 (+1.57%) | 96.08% | 1.425324e+00 | 3.014069e+00 |
| 4 | 4 | 52.0508 | +0.1477 (+0.28%) | 97.75% | 8.189906e-01 | 2.104136e+00 |
| 4 | 8 | 52.0597 | +0.1567 (+0.30%) | 97.84% | 8.188994e-01 | 1.991293e+00 |
| 3 | 4 | 51.9599 | +0.0569 (+0.11%) | 98.43% | 8.450033e-01 | 1.651617e+00 |
| 2 | 4 | 52.0597 | +0.1566 (+0.30%) | 97.94% | 6.621424e-01 | 1.452943e+00 |
| 2 | 2 | 52.5805 | +0.6774 (+1.31%) | 95.39% | 1.453093e+00 | 3.220106e+00 |

Calibration 적용 후 PPL은 ANN과 유사한 범위로 돌아왔고 token agreement는 95.39~98.43%로 크게 증가했다. `prefix=3, terms=4`가 PPL 변화 +0.11%와 agreement 98.43%로 가장 안정적인 결과를 보였다. `prefix=2, terms=4`는 logit MAE/RMSE가 더 작지만 PPL과 agreement는 약간 낮아, 하나의 지표만으로 configuration을 선택해서는 안 된다는 점을 보여준다.

Terms 4에서 8로 늘렸을 때 PPL과 agreement 개선은 거의 없었다. ViT와 마찬가지로 4 terms 이후에는 coefficient term 수 외의 오차가 지배적인 것으로 보인다.

### 4.4 Conv1D-only 변환

GELU는 exact로 유지하고 GPT-2의 48개 Conv1D projection을 모두 변환했다.

| Prefix | Terms | Conv1D 수 | B-PLA PPL | PPL 변화 | Agreement | Logit MAE | Logit RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 48 | 51.9004 | -0.0027 (-0.01%) | 100.00% | 1.445352e-02 | 3.259348e-02 |
| 2 | 2 | 48 | 51.8570 | -0.0461 (-0.09%) | 99.22% | 2.769198e-01 | 7.643221e-01 |

`prefix=4, terms=4`에서는 PPL이 사실상 동일하고 token agreement가 100%였다. 더 작은 `prefix=2, terms=2`에서도 PPL 변화는 -0.09%로 작았지만 agreement와 logit 오차는 악화되었다. 따라서 GPT-2 projection도 B-PLA multiplier approximation에 상당히 강건한 것으로 보인다.

### 4.5 GELU와 전체 Conv1D 결합

| Windows | Prefix | Terms | Conv1D 수 | GELU 수 | ANN PPL | B-PLA PPL | PPL 변화 | Agreement | Logit MAE/RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 2 | 48 | 12 | 51.9030 | 52.7651 | +0.8620 (+1.66%) | 96.18% | 1.393622e+00 / 미기록 |
| 4 | 4 | 4 | 48 | 12 | 51.9030 | 52.1506 | +0.2476 (+0.48%) | 97.84% | 7.887990e-01 / 1.945788e+00 |
| 100 | 4 | 4 | 48 | 12 | 47.3864 | 47.4728 | +0.0863 (+0.18%) | 97.43% | 8.809959e-01 / 2.651705e+00 |

가장 의미 있는 결과는 100-window 실험이다. 총 25,500 target tokens에서 전체 48개 Conv1D와 12개 GELU를 함께 변환해도 PPL 증가는 0.18%였고 token agreement는 97.43%였다. 4-window 결과보다 평가 표본이 크기 때문에 현재 GPT-2 결과 중 신뢰도가 가장 높다.

## 5. 종합 해석 및 의의

### 5.1 Linear/Conv1D 근사는 상대적으로 안정적

ViT의 전체 73개 Linear와 GPT-2의 전체 48개 Conv1D projection은 B-PLA로 변환해도 성능 저하가 매우 작았다. 특히 ViT `prefix=2, terms=2`와 GPT-2 `prefix=4, terms=4`에서 원본과 동일하거나 거의 동일한 최종 예측이 관측되었다.

이는 B-PLA multiplier가 transformer projection 연산에 적용될 가능성을 보여주는 end-to-end model sensitivity 근거다.

### 5.2 GELU는 calibration 범위가 핵심

고정 `[-4,4]` GELU LUT는 ViT 정확도를 크게 낮추고 GPT-2 출력 분포를 크게 변화시켰다. 반면 전체 모델에서 관측한 activation 범위로 공유 LUT를 fitting하면 ViT prediction agreement가 최대 100%, GPT-2 token agreement가 최대 98.43%로 회복되었다.

따라서 현재 결과는 다음을 시사한다.

> B-PLA activation의 정확도는 dyadic terms나 prefix bits뿐 아니라, 실제 모델의 activation 분포를 반영한 calibration 범위에 강하게 의존한다.

### 5.3 낮은 prefix도 사용 가능

Calibration 이후 ViT에서 `prefix=2, terms=3`, GPT-2에서 `prefix=3, terms=4`가 높은 성능을 보였다. Prefix를 낮추면 multiplier LUT는 (2^{2k}), activation LUT는 대략 (2^k)에 비례해 작아지므로, 하드웨어 LUT 면적 및 접근 에너지 절감 가능성이 있다.

### 5.4 Training-free 변환 가능성

모든 실험은 pretrained weight를 수정하거나 재학습하지 않고 수행되었다. 모델별 calibration은 LUT 범위를 정하기 위한 forward observation일 뿐 backpropagation이나 optimizer update를 사용하지 않는다. 따라서 현재 결과는 B-PLA의 training-free deployment 가능성을 뒷받침한다.

## 6. 적용 범위와 한계

### 6.1 Full transformer conversion은 아님

현재 변환 범위는 다음과 같다.

- ViT: GELU, Q/K/V/O projection Linear, MLP Linear, classifier를 포함한 `nn.Linear`
- GPT-2: GELU, attention 및 MLP의 `Conv1D` projection

다음 연산은 아직 exact로 유지된다.

- Attention score (QK^T)
- Attention probability와 value의 곱 (PV)
- Softmax
- LayerNorm
- ViT patch embedding convolution
- GPT-2 embedding과 LM head 등 기타 미교체 경로

따라서 전체 Linear/Conv1D를 변환했다는 표현과 전체 attention 연산 또는 전체 모델 산술을 변환했다는 표현은 구분해야 한다.

### 6.2 작은 ViT 평가 표본

ViT 결과는 100 images에 기반하므로 한 표본의 변화가 정확도 1%p에 해당한다. 현재 결과는 configuration 탐색과 민감도 확인에는 유용하지만 publication-level 성능 유지 주장에는 부족하다.

### 6.3 GPT-2의 제한된 token 수

대부분의 GPT-2 sweep은 4 windows, 1,020 target tokens에 기반한다. PPL 변화가 음수인 고정 범위 결과도 표본 변동과 출력 분포 왜곡의 영향을 받을 수 있다. 100-window 결과는 더 신뢰할 만하지만 전체 WikiText test 평가에는 미치지 못한다.

### 6.4 대칭 global max 범위의 비효율

현재 calibration은 전체 GELU 입력의 최대 절댓값으로 대칭 범위를 만든다.

```text
ViT:   약 [-48.6, 48.6]
GPT-2: 약 [-62.9, 62.9]
```

실제 activation 분포가 비대칭이거나 극소수 outlier를 포함하면 사용 빈도가 낮은 구간에 LUT 표현력을 낭비할 수 있다. Raw min/max, 비대칭 범위, percentile clipping을 비교할 필요가 있다.

### 6.5 PyTorch runtime은 하드웨어 성능 근거가 아님

현재 B-PLA Linear/Conv1D proxy는 approximate scalar products를 큰 elementwise tensor로 확장한다. 따라서 full conversion이 매우 느리며 native GEMM이나 Tensor Core와 직접적인 runtime 비교가 불가능하다. LUT 사전 계산과 공유는 coefficient 생성 비용을 줄이지만 중간 tensor와 reduction 병목은 남는다.

### 6.6 전력 효율 미검증

현재 결과는 모델 성능 유지에 관한 근거다. Multiplier 제거, 전력 감소, 면적 감소, throughput 개선은 아직 입증하지 않았다. Dyadic term 수가 증가하면 shifter와 adder 비용도 증가하므로, 정확도가 높은 configuration이 항상 에너지 효율적인 것은 아니다.

## 7. 추후 실험 계획

### 7.1 평가 규모 확대

ViT:

- Imagenette 전체 validation set
- 가능하면 ImageNet-1k validation set
- ANN과 B-PLA의 paired bootstrap confidence interval
- 표본별 prediction change 분석

GPT-2:

- WikiText test 전체 또는 충분한 수만~수십만 target tokens
- Context length 128, 256, 512 비교
- Prefill과 autoregressive decode 분리 평가
- Greedy generation의 최초 token divergence 위치 분석

### 7.2 Activation calibration 개선

- 대칭 max-absolute 범위와 실제 global min/max 비교
- P0.1/P99.9, P0.01/P99.99 percentile 범위 비교
- GELU의 음수 tail은 0, 양수 tail은 identity로 처리하는 bypass 실험
- Calibration batch 수 및 calibration/evaluation split 분리
- Global LUT 1개와 2~4개 range cluster LUT 비교

### 7.3 Configuration Pareto sweep

- Prefix bits: 2~6
- Dyadic terms: 1~4 중심
- Slope, intercept, multiplier offset term budget 분리
- LUT 크기, shift-add 수, 정확도/PPL을 함께 표시
- Layer-sensitive exact fallback 및 mixed configuration

### 7.4 Attention 및 미변환 연산 확장

단계적으로 다음 연산을 추가 변환한다.

1. Q/K/V projection만 변환
2. Attention output projection 변환
3. (QK^T) batched matmul 변환
4. (PV) batched matmul 변환
5. Softmax 근사
6. LayerNorm 근사
7. 전체 transformer arithmetic coverage 평가

Attention matmul은 Softmax를 통해 작은 오차가 증폭될 수 있으므로 projection과 별도로 민감도를 측정해야 한다.

### 7.5 Runtime 및 하드웨어 평가

- Weight sign/exponent/prefix metadata 사전 계산
- Prefix별 multiplier LUT module 사전 생성 및 저장
- Fused CUDA/Triton B-PLA accumulation kernel
- Approximate product 중간 tensor 제거
- 연산별 theoretical energy model
- RTL/HLS shared affine evaluator 구현
- DSP/multiplier 사용량, LUT/FF/area, Fmax, dynamic/leakage power 측정
- 정확도-metric과 energy/inference의 Pareto curve 작성

## 8. 현재 단계의 결론

현재 실험은 B-PLA가 pretrained ViT와 GPT-2의 GELU 및 projection 연산에 training-free로 적용될 수 있으며, 적절한 global activation calibration을 사용하면 모델 성능과 예측을 높은 수준으로 유지할 수 있음을 보여준다.

가장 강한 결과는 다음과 같다.

- ViT: 전체 73개 Linear와 12개 GELU를 `prefix=2, terms=2`로 변환했을 때 Top-1 88%, Top-5 99%, agreement 99%
- GPT-2: 전체 48개 Conv1D와 12개 GELU를 `prefix=4, terms=4`로 변환하고 25,500 target tokens를 평가했을 때 PPL 증가 +0.18%, token agreement 97.43%

따라서 변환 성능 유지 가능성은 확인되었지만, 아직 full transformer conversion, 대규모 dataset 평가, 실제 multiplierless datapath 및 전력 효율 검증이 남아 있다. 현 단계에서는 결과를 다음과 같이 제한해 표현하는 것이 적절하다.

> B-PLA는 ViT와 GPT-2의 GELU 및 Linear/Conv1D projection에 적용했을 때, 모델별 global activation calibration과 적절한 prefix/term budget 아래에서 pretrained 모델의 성능을 거의 유지하는 예비 결과를 보였다.

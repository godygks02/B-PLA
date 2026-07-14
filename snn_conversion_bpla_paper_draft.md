# Prefix-PLA-SNN: Prefix-Routed Piecewise Linear Spiking Neurons for Training-Free ANN-to-SNN Conversion

## Abstract

Artificial neural networks (ANNs) have rapidly advanced modern computer vision, language modeling, and multimodal perception, achieving high accuracy through increasingly large and computation-intensive architectures. This progress, however, has made inference cost a central deployment bottleneck: dense multiply-accumulate operations, nonlinear activations, normalization layers, and attention-style arithmetic lead to substantial latency and power consumption. Spiking neural networks (SNNs) have emerged as a promising alternative because information can be represented with sparse binary events and processed through event-driven accumulation. To reuse the accuracy of pretrained ANNs without expensive SNN training, training-free ANN-to-SNN conversion methods have been actively studied. Existing conversion methods are effective for ReLU-based CNNs, but they remain limited when applied to modern vision and transformer architectures that contain spike-unfriendly operators such as GELU, Softmax, LayerNorm, and floating-point multiplication. To address this limitation, we propose **Prefix-PLA-SNN**, a prefix-routed piecewise linear spiking neuron architecture for training-free conversion. Prefix-PLA-SNN encodes fixed-point values as temporal bit-plane spike streams, selects one local piecewise-linear segment using the configured high-order prefix, and evaluates precompiled synaptic increments with event-gated additions. The proposed neuron is designed as a conversion module for replacing spike-unfriendly ANN operators while preserving pretrained weights. Preliminary implementation targets nonlinear activation and mantissa-interaction multiplication modules, and evaluates operator accuracy, spike activity, toy-model agreement, and operation-level energy proxy.

**Korean interpretation.** ANN은 현대 컴퓨터 비전, 언어 모델링, 멀티모달 인식에서 빠르게 발전하며 높은 정확도를 달성했지만, 그 발전은 더 크고 계산 집약적인 아키텍처에 의존해 왔다. 그 결과 dense MAC 연산, 비선형 activation, normalization layer, attention 계열 산술 연산으로 인해 추론 지연시간과 전력 소모가 중요한 배포 병목이 되었다. SNN은 정보를 sparse binary event로 표현하고 event-driven accumulation으로 처리할 수 있어 유망한 대안으로 등장했다. 본 초안은 fixed-point 값을 temporal bit-plane spike stream으로 인코딩하고, 설정된 상위 prefix로 local PLA segment를 한 번 선택하며, 사전 컴파일된 synaptic increment를 event-driven 덧셈으로 누적하는 **Prefix-PLA-SNN**을 제안한다. 런타임 coefficient-by-activation 곱셈과 dyadic term expansion은 사용하지 않는다.

## 1. Introduction

The computational cost of ANN inference has become a central bottleneck in deploying vision models, transformers, and multimodal architectures. Convolutional and linear layers require large numbers of multiplications and additions, while transformer-style models introduce additional nonlinear and normalization operators. Even when model weights are reused without retraining, deployment often requires a difficult trade-off between accuracy, latency, memory traffic, and energy consumption.

**Korean interpretation.** ANN 추론의 계산 비용은 비전 모델, Transformer, 멀티모달 모델을 배포할 때 중요한 병목이 된다. Conv/Linear layer는 많은 곱셈과 덧셈을 요구하고, Transformer 계열은 추가적인 비선형 및 정규화 연산을 포함한다. 학습된 weight를 그대로 재사용하더라도, 실제 배포에서는 정확도, 지연시간, 메모리 이동, 에너지 사이의 trade-off가 발생한다.

SNNs are promising because they replace continuous activations with discrete spike events. In converted SNNs, dense ANN activations are mapped to spike counts, rates, phases, or deterministic temporal codes. Ideally, this enables low-power inference through sparse accumulation and reduced floating-point computation. Classic ANN-to-SNN conversion methods showed that ReLU-based CNNs can be mapped to spiking neurons using threshold balancing, potential initialization, or calibration. Recent methods further improve low-latency conversion using QCFS-style activation replacement, adaptive calibration, signed or differential coding, and inference-scale threshold strategies.

**Korean interpretation.** SNN은 continuous activation을 discrete spike event로 바꾸기 때문에 저전력 추론 가능성이 있다. 변환된 SNN에서는 ANN activation이 spike count, rate, phase, deterministic temporal code 등으로 표현된다. 기존 CNN 중심 conversion 연구들은 ReLU 기반 ANN을 threshold balancing, potential initialization, calibration 등을 통해 SNN으로 변환할 수 있음을 보였다. 최근에는 QCFS, adaptive calibration, signed/differential coding, inference-scale threshold 전략 등으로 low-latency conversion을 개선하고 있다.

However, the conversion problem becomes much harder for transformer and vision-transformer models. Operators such as GELU, Softmax, LayerNorm, and floating-point multiplication are not simple ReLU-rate mappings. Recent operator-centric spiking transformer methods, including ECMT, SpikedAttention, and MBE-style conversion, show that these operators must be handled explicitly. Existing approaches often pay one of three costs: they retain floating-point paths, require extra fine-tuning, or use many timesteps to recover accuracy.

**Korean interpretation.** 하지만 Transformer와 ViT에서는 conversion 문제가 더 어렵다. GELU, Softmax, LayerNorm, 부동소수점 곱셈은 단순한 ReLU-rate mapping으로 처리하기 힘들다. ECMT, SpikedAttention, MBE 기반 conversion처럼 최근 연구들은 이러한 operator를 명시적으로 다루어야 함을 보여준다. 기존 방법은 floating-point path를 남기거나, 추가 fine-tuning을 요구하거나, 정확도 회복을 위해 많은 timestep을 사용하는 경우가 많다.

This draft proposes Prefix-PLA-SNN, a prefix-routed PLA spiking neuron for training-free ANN-to-SNN conversion. The key idea is to transform spike-unfriendly operators into local affine segments selected by bit-plane prefixes. The neuron receives a temporal bit-plane spike stream, selects one local segment from the configured prefix, accumulates its precompiled synaptic increments, and emits signed Few-Spikes outputs or decoded membrane values.

**Korean interpretation.** 본 초안은 training-free ANN-to-SNN conversion을 위한 prefix-routed PLA spiking neuron인 Prefix-PLA-SNN을 제안한다. 핵심은 SNN 친화적이지 않은 연산자를 bit-plane prefix로 선택되는 local affine segment로 바꾸는 것이다. 뉴런은 설정된 prefix로 local segment를 한 번 선택한 뒤, segment-specific synaptic increment를 누적하고 signed Few-Spikes 출력 또는 decoded membrane value를 생성한다.

## 2. Related Work

Classic ANN-to-SNN conversion methods focus primarily on CNNs with ReLU-like activations. Foundational work by Diehl et al. and Rueckauer et al. uses weight and threshold balancing to align ANN activations with spike firing rates. Later methods improve latency by optimizing membrane potentials, thresholds, or activation functions. QCFS-style conversion replaces ReLU with a quantization-aware clipped function to improve low-timestep accuracy, while adaptive calibration and inference-scale conversion reduce conversion error without fully retraining the source ANN.

**Korean interpretation.** 고전적인 ANN-to-SNN conversion은 주로 ReLU 기반 CNN에 초점을 둔다. Diehl, Rueckauer 계열의 연구는 weight/threshold balancing으로 ANN activation과 spike firing rate를 맞춘다. 이후 연구들은 membrane potential, threshold, activation function을 조정해 latency를 줄였다. QCFS는 ReLU를 quantization-aware clipped function으로 바꿔 low-timestep 정확도를 개선하고, adaptive calibration 및 inference-scale conversion은 source ANN 전체를 재학습하지 않고 conversion error를 줄인다.

Coding-centric conversion changes the spike representation rather than only tuning thresholds. Signed spikes, temporal coding, phase coding, burst coding, and differential coding reduce the number of timesteps or spike events needed to represent ANN activations. These methods are important for low-latency inference because rate coding often needs many timesteps to approximate real-valued activations accurately.

**Korean interpretation.** Coding-centric conversion은 threshold만 조정하는 것이 아니라 spike 표현 방식 자체를 바꾼다. signed spike, temporal coding, phase coding, burst coding, differential coding은 ANN activation을 표현하는 데 필요한 timestep 또는 spike event 수를 줄인다. rate coding은 실수 activation을 정확히 표현하려면 많은 timestep이 필요한 경우가 많기 때문에, 이러한 coding 방식은 low-latency inference에서 중요하다.

Transformer and ViT conversion introduces a different class of limitations. Self-attention and feed-forward blocks contain Softmax, GELU, LayerNorm, residual scaling, and matrix multiplication. Recent methods such as SpikedAttention, ECMT, SpikeZIP-TF, FAS, and MBE-based training-free spiking transformer conversion show that operator replacement is necessary. The survey motivating this draft identifies fully spike-driven normalization and attention, calibration-only LLM/VLM conversion, and multiplier-free operator replacement as open problems.

**Korean interpretation.** Transformer 및 ViT conversion에서는 다른 종류의 한계가 나타난다. Self-attention과 feed-forward block은 Softmax, GELU, LayerNorm, residual scaling, matrix multiplication을 포함한다. SpikedAttention, ECMT, SpikeZIP-TF, FAS, MBE 기반 training-free spiking transformer conversion 등은 operator replacement가 필요함을 보여준다. 본 초안의 survey는 fully spike-driven normalization/attention, calibration-only LLM/VLM conversion, multiplier-free operator replacement가 중요한 open problem임을 지적한다.

Prefix-PLA-SNN is positioned in this operator-centric conversion line. It does not claim that approximate multiplication alone is a new SNN method. Instead, it proposes a prefix-routed PLA spiking neuron that reduces nonlinear and arithmetic operators to prefix-local affine subproblems and realizes coefficient application as event-gated synaptic accumulation.

**Korean interpretation.** Prefix-PLA-SNN은 operator-centric conversion 흐름에 위치한다. 단순히 approximate multiplication이 새로운 SNN 방법이라고 주장하지 않는다. 대신 비선형 및 산술 연산자를 prefix-local affine subproblem으로 나누고, coefficient 적용을 event-gated synaptic accumulation으로 구현한다.

## 3. Problem Definition

Let a pretrained ANN contain an operator \(f\) that is difficult to convert directly into an SNN, such as GELU or a floating-point mantissa interaction. Standard conversion attempts to approximate \(f(x)\) with spike rate or spike count over \(T\) timesteps. For nonlinear and multi-variable operators, this creates conversion error:

```math
\epsilon_T = \left| f(x) - D(S_{1:T}) \right|,
```

where \(D\) decodes output spikes into a real-valued activation. The goal is to reduce this error without modifying the pretrained ANN weights and without retaining dense floating-point operator paths.

**Korean interpretation.** 사전학습된 ANN에 GELU나 부동소수점 mantissa interaction처럼 SNN으로 직접 변환하기 어려운 연산자 \(f\)가 있다고 하자. 표준 conversion은 \(T\) timestep 동안 spike rate 또는 spike count로 \(f(x)\)를 근사하려고 한다. 비선형 및 다변수 연산자에서는 이 과정에서 conversion error가 발생한다. 목표는 pretrained ANN weight를 수정하지 않고, dense floating-point operator path를 남기지 않으면서 이 error를 줄이는 것이다.

Prefix-PLA-SNN defines a conversion module \(g_\theta\) that first approximates the operator with a local affine segment and then realizes that segment as a spiking neuron:

```math
f(x) \approx a_i x + b_i, \quad i = \operatorname{prefix}(x),
```

```math
\hat{y} = D(\Phi_i(S_x[1:T]; a_i, b_i)).
```

Here, \(S_x[1:T]\) is a bit-plane spike stream and \(\Phi_i\) is the segment-specific spiking neuron selected once by the configured prefix. Its affine coefficients are compiled offline into per-timestep increments, so runtime evaluation uses conditional additions.

**Korean interpretation.** Prefix-PLA-SNN은 먼저 연산자 \(f\)를 local affine segment로 근사하고, 그 segment를 spiking neuron으로 구현하는 conversion module \(g_\theta\)를 정의한다. 입력 \(x\)는 bit-plane spike stream \(S_x[1:T]\)로 변환되고, 설정된 prefix routing이 segment \(i\)를 한 번 선택하며, \(\Phi_i\)가 사전 컴파일된 timestep별 increment를 조건부 누적한다.

## 4. Method

### 4.1 Bit-Plane Spike Encoding

Prefix-PLA-SNN encodes a clipped fixed-point input as a deterministic temporal bit-plane spike stream:

```math
x \approx \operatorname{sign}(x)\sum_{t=1}^{T} S_x[t] 2^{-p_t}, \quad S_x[t]\in\{0,1\}.
```

The most significant bit-planes arrive first and are used for prefix routing. This differs from conventional rate coding because the spike timing encodes binary significance rather than sampling a firing probability.

**Korean interpretation.** Prefix-PLA-SNN은 clipping된 fixed-point 입력을 deterministic temporal bit-plane spike stream으로 변환한다. 상위 bit-plane은 prefix routing에 사용된다. 이는 spike timing이 firing probability가 아니라 binary significance를 표현한다는 점에서 일반적인 rate coding과 다르다.

### 4.2 Prefix-Routed PLA Spiking Neuron

For each input, high-order bit-plane spikes determine the segment index:

```math
i = \operatorname{prefix}_k(S_x[1], \ldots, S_x[k]).
```

The selected segment stores affine parameters \(a_i\) and \(b_i\). The default FS-style neuron means Few-Spikes emission, not finite-step rate coding. It first computes a local affine membrane value:

```math
V_i = \sum_t a_i S_x[t]2^{-p_t} + b_i.
```

The output is emitted by a coarse-to-fine Few-Spikes decoder:

```math
\hat{y} = \theta(N^+ - N^-).
```

**Korean interpretation.** 각 입력에 대해 상위 bit-plane spike가 segment index를 결정한다. 선택된 segment는 affine parameter \(a_i, b_i\)를 가진다. 기본 FS-style neuron에서 FS는 finite-step이 아니라 Few-Spikes를 의미한다. 뉴런은 local affine membrane value를 계산한 뒤, coarse-to-fine 방식의 signed few-spike 출력으로 decode된다.

### 4.3 Fixed-Prefix Event-Driven Evaluation

Prefix-PLA-SNN selects one final PLA segment using the configured high-order prefix. The selected affine coefficients are compiled offline into per-bit-plane synaptic increments, and runtime spikes gate conditional additions into the membrane.

For each input bit-plane event \(S_x[t]\), the membrane adds a precompiled increment \(w_{i,t}=a_i2^{-p_t}\). Fixed-prefix routing and a single affine table evaluation are the only implemented mode.

**Korean interpretation.** 설정된 prefix로 최종 PLA segment를 한 번 선택한다. 각 bit-plane event \(S_x[t]\)가 발생하면 사전 컴파일된 increment \(w_{i,t}=a_i2^{-p_t}\)를 membrane에 더한다. 현재 구현은 fixed-prefix routing과 단일 affine table 평가만 사용한다.

### 4.4 IF Baseline

For ablation, the same PLA segment can be evaluated with an IF-style threshold-crossing neuron:

```math
V[t+1] = V[t] + a_i S_x[t]2^{-p_t} + b_i/T,
```

```math
S[t] = H(V[t]-\theta).
```

This baseline is useful because IF neurons are widely recognized in SNN literature, while the FS-style neuron better matches deterministic bit-plane PLA computation.

**Korean interpretation.** Ablation을 위해 동일한 PLA segment를 IF-style threshold-crossing neuron으로도 계산할 수 있다. IF neuron은 SNN 문헌에서 널리 알려져 있으므로 baseline으로 유용하고, FS-style neuron은 deterministic bit-plane PLA 계산과 더 잘 맞는 기본 구현이다.

### 4.5 Spiking Mantissa Interaction

For multiplication, Prefix-PLA-SNN does not claim to convert the entire FP32 multiplier into an SNN neuron. Instead, it converts the nonlinear mantissa interaction term:

```math
m_1m_2 \approx a_{ij}m_1 + b_{ij}m_2 + c_{ij}.
```

The prefixes of \(m_1\) and \(m_2\) select the tile \((i,j)\) once, and the selected affine interaction is evaluated through two bit-plane spike streams and event-gated accumulation. Sign and exponent handling remain deterministic conversion logic.

**Korean interpretation.** 곱셈의 경우 Prefix-PLA-SNN은 FP32 multiplier 전체를 SNN neuron으로 바꾼다고 주장하지 않는다. 대신 nonlinear mantissa interaction term \(m_1m_2\)만 PLA spiking operator로 변환한다. \(m_1, m_2\)의 prefix가 tile \((i,j)\)를 한 번 선택하고, 두 bit-plane spike stream이 해당 tile의 사전 컴파일된 increment를 membrane에 조건부 누적한다. sign과 exponent 처리는 deterministic conversion logic으로 유지된다.

## 5. Experimental Protocol

The preliminary implementation evaluates three paths. First, operator-level activation experiments compare exact ANN activations with Prefix-PLA-SNN outputs for ReLU, GELU, and QuickGELU. Second, multiplier experiments compare exact FP32 multiplication with Prefix-PLA-SNN mantissa interaction approximation. Finally, a NumPy-only toy MLP replaces GELU with Prefix-PLA-SNN and measures top-1 agreement with ANN logits. Energy is reported as an operation-level proxy using LUT reads, spike events, accumulation operations, and threshold comparisons.

**Korean interpretation.** 예비 구현은 세 경로에서 평가한다. 첫째, activation operator 실험에서 ReLU, GELU, QuickGELU에 대해 exact ANN activation과 Prefix-PLA-SNN 출력을 비교한다. 둘째, multiplier 실험에서 exact FP32 multiplication과 Prefix-PLA-SNN mantissa interaction approximation을 비교한다. 마지막으로 NumPy-only toy MLP에서 GELU를 Prefix-PLA-SNN으로 대체하고 ANN logits와 top-1 agreement를 측정한다. 에너지는 LUT read, spike event, accumulation operation, threshold comparison 기반의 operation-level proxy로 보고한다.

## 6. Preliminary Results Placeholder

The current repository provides executable modules and an experiment script for preliminary validation. The implementation uses FS to mean Few-Spikes and uses one configured final prefix for routing. In the current operator sweep, FS GELU reaches an MAE of about \(2.22\times10^{-3}\), while FS mantissa multiplication reaches a mean relative error of about \(1.55\times10^{-4}\). The operation-level energy proxy is about \(0.33\times\) an FP affine activation and \(0.95\times\) an FP32 multiplier for the tested configuration. Results should be treated as method-formulation evidence rather than publication-level energy claims until evaluated on real datasets, calibrated thresholds, and synthesized hardware.

**Korean interpretation.** 현재 repository는 예비 검증을 위한 실행 가능한 모듈과 실험 스크립트를 제공한다. 구현에서 FS는 Few-Spikes를 의미하며, 설정된 최종 prefix를 한 번 사용해 route한다. 현재 operator sweep에서 FS GELU의 MAE는 약 \(2.22\times10^{-3}\), FS mantissa multiplication의 mean relative error는 약 \(1.55\times10^{-4}\)다. 테스트한 구성의 operation-level energy proxy는 FP affine activation 대비 약 \(0.33\times\), FP32 multiplier 대비 약 \(0.95\times\)다. 이 결과는 method formulation 근거로 보아야 하며, 실제 dataset, calibrated threshold, 합성 하드웨어 평가 전까지 publication-level energy claim으로 사용해서는 안 된다.

## 7. Limitations and Future Work

Prefix-PLA-SNN is currently an operator-level and toy-model implementation. It does not yet include full ViT attention conversion, Softmax replacement, LayerNorm replacement, or dataset-scale evaluation. Future work should integrate the fixed-prefix event-driven PLA neuron with calibration-only conversion and fully spike-driven replacements for normalization and attention, then evaluate full conversion on CNN, ViT, and VLM workloads.

**Korean interpretation.** Prefix-PLA-SNN은 현재 operator-level 및 toy-model 구현이다. 아직 full ViT attention conversion, Softmax replacement, LayerNorm replacement, dataset-scale evaluation을 포함하지 않는다. 향후에는 fixed-prefix event-driven PLA neuron을 calibration-only conversion 및 fully spike-driven normalization/attention replacement와 결합하고, CNN/ViT/VLM workload에서 full conversion을 평가해야 한다.

## References Mentioned in the Survey

- Diehl et al., "Fast-classifying, high-accuracy spiking deep networks through weight and threshold balancing," IJCNN 2015.
- Rueckauer et al., "Conversion of continuous-valued deep networks to efficient event-driven networks," Frontiers in Neuroscience 2017.
- Bu et al., "Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks," 2023.
- Hu et al., "Fast-SNN: Fast Spiking Neural Network by Converting Quantized ANN," TPAMI 2023.
- Huang et al., "Towards High-performance Spiking Transformers from ANN to SNN Conversion," ACM MM 2024.
- Hwang et al., "SpikedAttention," NeurIPS 2024.
- Huang et al., "Differential Coding for Training-Free ANN-to-SNN Conversion," 2025.
- Chen et al., "FAS: Fast ANN-SNN Conversion for Spiking Large Language Models," 2025.
- Wang et al., "Training-Free ANN-to-SNN Conversion for High-Performance Spiking Transformers," AAAI 2026.

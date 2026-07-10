# B-PLA: A Training-Free Multiplierless Affine Arithmetic Framework for Energy-Efficient ANN Inference

## Abstract

B-PLA, or Bit-Prefix Piecewise Linear Approximation, is a training-free
arithmetic replacement framework for energy-efficient neural network inference.
Its central goal is not merely to design another approximate floating-point
multiplier. Instead, B-PLA aims to remove runtime multipliers from frequently
used ANN operations while preserving inference accuracy, and to unify
multiplication and nonlinear activation under a shared low-power datapath.

The proposed framework replaces expensive arithmetic operators with a common
execution pattern:

```text
bit-prefix routing -> coefficient lookup -> dyadic affine evaluation
```

In this structure, the input is routed to a local approximation region using
only high-order prefix information. The corresponding coefficients are fetched
from a compact table. Runtime evaluation is then performed with dyadic
coefficients, so multiplication by coefficients can be implemented using shifts,
adds, muxes, and small lookup tables instead of hardware multipliers.

This positioning is important. Prior work such as Mitchell-style logarithmic
multipliers, ApproxLP, and PAM has studied approximate multiplication, including
piecewise linear approximation of the mantissa interaction surface. Other prior
work has studied multiplierless or piecewise-linear activation units. B-PLA's
intended novelty is different: it treats multiplication and activation as two
instances of a single prefix-routed affine arithmetic primitive that can be
inserted into existing ANN inference graphs without retraining model weights.

The current repository contains the minimal arithmetic-level prototype:

- `modules/bpla_multiplier.py`: FP32 mantissa interaction approximation using
  2D bit-prefix affine tiles.
- `modules/bpla_activation.py`: activation approximation using FP32 bit-field
  prefix routing and 1D affine coefficient tables.

The next research stage is to turn this prototype into a genuinely
multiplierless, dyadic-quantized, model-evaluated, and hardware-costed
framework.

## 1. Research Motivation

Modern ANN inference repeatedly executes two expensive classes of operations.

```math
y = xw
```

```math
y = f(x)
```

The first operation appears in Linear layers, convolution, attention
projections, MLP blocks, and transformer projections. The second operation
appears in nonlinear functions such as GELU, Sigmoid, Tanh, ReLU, QuickGELU,
SiLU, and related functions.

These operations are traditionally implemented with floating-point
multipliers, MAC units, LUTs, polynomial units, or dedicated nonlinear function
units. Such units are accurate, but they are expensive in power, area, and
latency. This cost becomes especially important in edge inference, event-driven
inference, ANN-to-SNN conversion, and architectures where arithmetic energy is a
dominant part of the inference budget.

B-PLA starts from the following question:

> Can existing trained ANN models be executed with fewer or no runtime
> multipliers by replacing common arithmetic operators with a shared
> prefix-routed, shift-add affine approximation primitive?

This makes the research goal different from pure approximate multiplier design.
B-PLA is intended as a training-free inference framework: trained weights remain
unchanged, network topology remains unchanged, and only the implementation of
selected arithmetic operators is replaced.

## 2. Core Thesis

The central thesis of B-PLA is:

> Multiplication and nonlinear activation can be approximated as local affine
> functions selected by bit-prefix routing, and both can be executed by the same
> multiplierless shift-add datapath when their coefficients are dyadically
> quantized.

The general form is:

```math
\hat{y} = a_i x + b_i
```

or, for two-input arithmetic:

```math
\hat{y} = a_{ij}x_1 + b_{ij}x_2 + c_{ij}
```

The index is determined by prefix information:

```math
i = \text{prefix}(x)
```

```math
(i,j) = (\text{prefix}(x_1), \text{prefix}(x_2))
```

The coefficients are constrained toward dyadic or signed power-of-two-sum
forms:

```math
a_i \approx \sum_t s_t 2^{-p_t}, \quad s_t \in \{-1, +1\}
```

```math
a_{ij}, b_{ij} \approx \frac{n}{2^s}
```

Therefore, runtime coefficient multiplication can be realized as shifts and
adds.

## 3. Positioning Against Prior Work

### 3.1 Why B-PLA Is Not Just Another PLA Multiplier

The mathematical idea of approximating the FP mantissa interaction surface is
not sufficient novelty by itself. Prior work such as ApproxLP and PAM already
studies piecewise linear or affine approximation of the nonlinear multiplication
surface. Therefore, B-PLA should not claim novelty as:

```text
first method to approximate M1*M2 with piecewise affine planes
```

That claim is weak.

The stronger and more defensible claim is:

```text
B-PLA is a training-free, multiplierless, prefix-routed affine arithmetic
framework that unifies multiplication and nonlinear activation under a shared
datapath for ANN inference.
```

In this framing, ApproxLP and PAM are important baselines for the multiplier
component, but they do not define the whole B-PLA contribution. B-PLA's
contribution must be demonstrated at the framework and datapath level.

### 3.2 Difference From Approximate Multipliers

Approximate multiplier research primarily asks:

```text
How can a multiplier be approximated more cheaply?
```

B-PLA instead asks:

```text
Can the multiplier be removed from selected ANN inference arithmetic paths and
replaced by a shared shift-add affine primitive?
```

This distinction matters. A good approximate multiplier may still be a
multiplier-like unit. B-PLA aims to make the runtime computation multiplierless
after coefficient quantization.

### 3.3 Difference From Activation Accelerators

Activation accelerators often use PWL approximation, LUTs, non-uniform
breakpoints, polynomial approximation, or power-of-two coefficients. These
methods usually focus on nonlinear function units only.

B-PLA's activation module is not novel simply because it uses PWL fitting.
Its role is to show that activation approximation can share the same
prefix-routed affine evaluator used for multiplication. The novelty comes from
unification and hardware reuse, not from ordinary PWL activation approximation
alone.

### 3.4 Training-Free Deployment Advantage

B-PLA is designed to be inserted into existing ANN inference graphs without
updating model parameters.

```text
trained model -> operator replacement -> B-PLA inference
```

This is different from quantization-aware training, pruning with retraining, or
distillation-based compression. B-PLA may use offline arithmetic calibration to
build coefficient tables, but it should not require gradient-based training of
the original neural network weights.

The correct distinction is:

```text
training-free does not necessarily mean calibration-free
```

B-PLA can be training-free while still using offline coefficient calibration,
range selection, dyadic coefficient fitting, and layer-wise sensitivity
analysis.

## 4. B-PLA Multiplier Primitive

### 4.1 FP32 Decomposition

For a normal FP32 value:

```math
x = (-1)^S 2^E (1 + M)
```

where `S` is the sign bit, `E` is the unbiased exponent, and `M` is the mantissa
fraction in `[0, 1)`.

For two inputs:

```math
x_1x_2 =
(-1)^{S_1 \oplus S_2}
2^{E_1 + E_2}
(1 + M_1)(1 + M_2)
```

The mantissa product expands to:

```math
(1 + M_1)(1 + M_2)
= 1 + M_1 + M_2 + M_1M_2
```

Only the interaction term `M_1M_2` is nonlinear. B-PLA approximates this term
with local affine planes.

### 4.2 Prefix Tile Routing

When `prefix_bits = k`, the upper `k` bits of the 23-bit FP32 fraction select
the tile:

```math
i = \left\lfloor M_1 2^k \right\rfloor
```

```math
j = \left\lfloor M_2 2^k \right\rfloor
```

In implementation:

```text
index = fraction_q23 >> (23 - prefix_bits)
```

This is the routing advantage of B-PLA. It avoids complex breakpoint search for
the multiplier path and maps naturally to bit slicing and address generation.

### 4.3 Affine Approximation

Each tile approximates:

```math
M_1M_2 \approx a_{ij}M_1 + b_{ij}M_2 + c_{ij}
```

The current prototype uses the first-order approximation around the tile
center:

```math
\mu_i = \frac{i + 0.5}{2^k}
```

```math
\nu_j = \frac{j + 0.5}{2^k}
```

```math
a_{ij} = \nu_j,\quad b_{ij} = \mu_i,\quad c_{ij} = -\mu_i\nu_j
```

This gives:

```math
M_1M_2 \approx \nu_jM_1 + \mu_iM_2 - \mu_i\nu_j
```

This prototype is useful for arithmetic validation, but it is not yet the final
B-PLA hardware form. The final form should use calibrated and dyadically
quantized coefficients.

### 4.4 Dyadic Multiplierless Form

For multiplierless execution, each coefficient should be represented as a
dyadic value or as a short signed power-of-two expansion:

```math
a_{ij} \approx \sum_t s_t 2^{-p_t}
```

Then:

```math
a_{ij}M_1 \approx \sum_t s_t (M_1 >> p_t)
```

This turns coefficient multiplication into shift-add logic.

The multiplier primitive should therefore evolve from:

```text
float coefficient table + floating affine evaluation
```

to:

```text
integer/dyadic coefficient table + shift-add affine evaluation
```

## 5. B-PLA Activation Primitive

### 5.1 PWL Activation Form

For activation:

```math
y = f(x)
```

B-PLA approximates the function over a fixed range:

```math
x_c = \text{clip}(x, x_{\min}, x_{\max})
```

```math
f(x_c) \approx a_i x_c + b_i
```

Supported targets in the current prototype include:

- ReLU
- Sigmoid
- Tanh
- GELU
- QuickGELU

### 5.2 Prefix Segment Routing

The current B-PLA activation frontend uses direct FP32 bit-field extraction
rather than runtime range normalization. A clipped activation input is split
into sign, exponent, and mantissa prefix fields:

```math
x_c = (-1)^S 2^E (1 + M)
```

The segment address is generated from these fields:

```math
i = \text{addr}(S, \text{clamp}(E), \text{prefix}_k(M))
```

In implementation, the mantissa prefix is obtained by bit slicing:

```text
mantissa_prefix = fraction_q23 >> (23 - prefix_bits)
```

Small-magnitude values below a configurable exponent threshold are routed to a
single central segment. This inherits the useful S-PLA idea of FP32 bit
extraction, but the method is presented as B-PLA activation: bit-prefix routing
selects an affine approximation table entry. This avoids modeling activation
routing as:

```text
clip -> subtract x_min -> divide by range -> fixed-point conversion
```

which would introduce expensive runtime arithmetic and weaken the
multiplierless hardware claim.

The conceptual flow is still the same as the multiplier path:

```text
input bit fields -> coefficient address -> affine evaluation
```

### 5.3 Dyadic Activation Fitting

The current implementation uses unconstrained least-squares fitting:

```math
(a_i, b_i)
=
\arg\min_{a,b}
\sum_{x_n \in [l_i, r_i]}
(f(x_n) - (ax_n + b))^2
```

For the final B-PLA framework, this must be replaced or extended with
dyadic-constrained fitting:

```math
a_i \in \mathcal{D}, \quad b_i \in \mathcal{D}
```

where `\mathcal{D}` is a hardware-friendly coefficient set such as fixed-point
dyadic values or short signed power-of-two sums.

This is essential. Without dyadic or fixed-point constrained coefficients, the
activation module remains a PWL approximation module, but not a fully
multiplierless B-PLA primitive.

## 6. Unified Datapath Architecture

The architectural target is a shared arithmetic core:

```text
                    +------------------+
input bits/ranges ->| prefix router    |
                    +------------------+
                              |
                              v
                    +------------------+
                    | coefficient LUT  |
                    +------------------+
                              |
                              v
                    +------------------+
                    | shift-add affine |
                    | evaluator        |
                    +------------------+
                              |
                              v
                         output value
```

The multiplier path uses 2D routing:

```math
(i,j) = (\text{prefix}(M_1), \text{prefix}(M_2))
```

The activation path uses 1D routing:

```math
i = \text{addr}(S, \text{clamp}(E), \text{prefix}_k(M))
```

Both paths use the same class of operations:

```text
lookup coefficients -> shift inputs -> add shifted terms -> add offset
```

This is the intended source of B-PLA's hardware novelty. The same evaluator can
be time-multiplexed or spatially reused across arithmetic operations that would
otherwise require separate multiplier and nonlinear function units.

## 7. Novelty Claim

B-PLA's novelty should be stated carefully.

Weak claim:

```text
B-PLA approximates the mantissa interaction surface with piecewise affine
planes.
```

This overlaps strongly with prior approximate multiplier literature.

Stronger claim:

```text
B-PLA provides a training-free arithmetic replacement framework for ANN
inference that unifies multiplication and nonlinear activation as
prefix-routed affine approximations.
```

Strongest target claim:

```text
B-PLA provides a dyadic-quantized multiplierless hardware datapath that can
replace selected ANN multiplication and activation operations without retraining
model weights, improving energy efficiency while maintaining inference
accuracy.
```

The research should be evaluated against this strongest claim.

## 8. Current Implementation Status

The current codebase implements the smallest functional arithmetic prototype.

Implemented:

- FP32 decomposition for normal floating-point values.
- 2D prefix tile routing for mantissa interaction approximation.
- First-order affine tile coefficients for `M_1M_2`.
- 1D FP32 bit-field prefix routing for activation functions.
- Least-squares activation coefficient fitting.
- Basic arithmetic-level unit tests.

Not yet implemented:

- Dyadic coefficient quantization.
- Shift-add-only runtime evaluation.
- Fixed-point integer simulation.
- Coefficient memory compression.
- PyTorch operator replacement.
- Model-level inference benchmarks.
- Energy, area, and latency modeling.
- RTL or FPGA synthesis.
- ANN-to-SNN or event-driven integration.

This means the current prototype demonstrates the mathematical skeleton of
B-PLA, but not yet the full multiplierless framework claim.

## 9. Research Roadmap

### Stage 1: Arithmetic Ground Truth and Baselines

Goal: establish reliable arithmetic behavior before hardware claims.

Tasks:

1. Implement benchmark scripts for multiplier and activation error sweeps.
2. Sweep `prefix_bits = 2, 3, 4, 5, 6, 7, 8`.
3. Report MAE, RMSE, max absolute error, mean relative error, P99 relative
   error, signed error mean, and error bias.
4. Compare against exact FP32, Mitchell-style approximation, and a simple
   uniform PWL baseline for activation.
5. Add plots for accuracy vs table size.

Deliverables:

- `experiments/arithmetic_sweep.py`
- `results/arithmetic/` CSV files and plots
- documented error behavior in the overview or a separate technical note

### Stage 2: Dyadic Quantization

Goal: make B-PLA genuinely multiplierless.

Tasks:

1. Add coefficient quantizers for dyadic fixed-point coefficients.
2. Support signed power-of-two and two-term signed power-of-two coefficient
   expansions.
3. Compare unconstrained coefficients vs dyadic coefficients.
4. Measure the accuracy loss caused by coefficient quantization.
5. Implement shift-add-equivalent simulation, even if still written in Python.

Deliverables:

- `modules/dyadic.py`
- dyadic versions of multiplier and activation tables
- tests proving that runtime evaluation can be expressed as shifts and adds
- accuracy/complexity plots for dyadic bit width and number of PoT terms

### Stage 3: Calibrated Coefficient Fitting

Goal: improve accuracy under hardware constraints.

Tasks:

1. Replace pure Taylor coefficients with per-tile least-squares fitted planes.
2. Add optional unbiased offset calibration for each multiplier tile.
3. Add activation fitting under dyadic constraints.
4. Support calibration data drawn from synthetic distributions and real model
   activation/weight statistics.
5. Compare bit-field prefix calibration, uniform calibration baselines, and
   layer-aware calibration.

Deliverables:

- `modules/calibration.py`
- calibrated coefficient tables
- bias and error-distribution reports
- analysis of whether calibration is global, layer-wise, or model-specific

Important distinction:

```text
training-free model deployment is compatible with offline arithmetic
calibration.
```

The neural network weights should not be updated.

### Stage 4: PyTorch Operator Replacement

Goal: demonstrate training-free insertion into existing ANN inference.

Tasks:

1. Implement B-PLA versions of `Linear`, `Conv2d`, and selected activations.
2. Start with small models: MLP, LeNet-style CNN, small ResNet block.
3. Move to transformer-like blocks: MLP projection, attention projection, GELU.
4. Keep model weights fixed.
5. Measure accuracy drop as a function of prefix bits and dyadic precision.

Deliverables:

- `torch_bpla/` package
- model replacement utilities
- inference-only benchmark scripts
- accuracy tables for at least one vision dataset and one transformer block

### Stage 5: Energy and Hardware Cost Modeling

Goal: support the low-power claim quantitatively.

Tasks:

1. Build an operation-level cost model for FP multiplier, fixed-point
   multiplier, shift, add, mux, and LUT access.
2. Estimate energy per B-PLA multiplier operation and activation operation.
3. Compare against FP32, FP16, INT8, PAM-like approximate multiplier, and
   activation PWL baselines where possible.
4. Report energy-accuracy tradeoff curves.
5. Include memory access cost, not only arithmetic cost.

Deliverables:

- `experiments/energy_model.py`
- tables for operation counts and estimated energy
- energy vs accuracy plots
- clear assumptions for technology node and bit width

### Stage 6: Shared Datapath Design

Goal: demonstrate that multiplication and activation can share hardware.

Tasks:

1. Specify a common coefficient table format.
2. Define 1D and 2D address generation modes.
3. Define a shared shift-add affine evaluator.
4. Estimate scheduling and utilization for neural network layers.
5. Show area savings compared with separate approximate multiplier and
   activation units.

Deliverables:

- architecture diagram
- datapath specification
- cycle-level pseudo-scheduler
- comparison with separate-unit design

### Stage 7: RTL or FPGA Prototype

Goal: validate multiplierless implementation beyond Python simulation.

Tasks:

1. Implement a minimal Verilog/SystemVerilog shift-add affine evaluator.
2. Implement coefficient ROM and prefix address generation.
3. Synthesize on FPGA or with an open-source ASIC flow if available.
4. Report LUTs, registers, DSP usage, frequency, and estimated power.
5. Confirm that DSP/multiplier blocks are not used in the B-PLA core.

Deliverables:

- `rtl/` prototype
- synthesis reports
- hardware resource comparison
- proof that the runtime core is multiplierless

### Stage 8: ANN-to-SNN and Event-Driven Extension

Goal: connect B-PLA to the original low-power/event-driven motivation.

Tasks:

1. Identify where ANN-to-SNN conversion pipelines still use expensive
   multiply/activation operations.
2. Replace those arithmetic paths with B-PLA primitives.
3. Study whether prefix routing interacts naturally with spike sparsity,
   membrane integration, or event-triggered computation.
4. Measure accuracy, spike count, and estimated energy.

Deliverables:

- integration note with an ANN-to-SNN baseline
- event-driven operation count analysis
- accuracy and energy comparison

## 10. Study Plan

To strengthen the novelty argument, the following topics should be studied in
parallel with implementation.

Approximate arithmetic:

- Mitchell multiplier and improved logarithmic multipliers.
- MBM and bias-corrected approximate multiplication.
- ApproxLP and linearization-based approximate multiplication.
- PAM and unbiased/configurable floating-point approximation.

Activation approximation:

- PWL activation accelerators.
- Multiplierless activation units.
- Power-of-two coefficient approximation.
- Non-uniform segmentation and learned breakpoints.

Hardware implementation:

- Fixed-point arithmetic.
- Dyadic quantization and canonical signed digit representation.
- Shift-add multiplierless datapaths.
- LUT, mux, adder tree, and memory energy models.
- FPGA synthesis and DSP block avoidance.

Model-level deployment:

- Post-training quantization.
- Layer-wise sensitivity analysis.
- Training-free ANN-to-SNN conversion.
- Transformer MLP and attention projection arithmetic profiles.

## 11. Evaluation Metrics

B-PLA should be evaluated at four levels.

Arithmetic-level metrics:

- MAE
- RMSE
- max absolute error
- mean relative error
- P99 relative error
- signed error mean
- error bias by tile or segment

Model-level metrics:

- top-1 accuracy drop
- task metric drop
- layer-wise output error
- robustness across architectures

Hardware-level metrics:

- multiplier count
- shift count
- adder count
- LUT/ROM size
- estimated area
- estimated energy
- critical path or estimated latency

Framework-level metrics:

- number of replaceable operators
- retraining requirement
- calibration requirement
- accuracy-energy Pareto curve
- benefit of shared datapath vs separate units

## 12. Near-Term Implementation Checklist

The immediate next steps are:

1. Add dyadic coefficient quantization for both multiplier and activation.
2. Add shift-add-equivalent evaluation paths.
3. Add arithmetic sweep scripts and plots.
4. Add calibrated affine fitting for multiplier tiles.
5. Add PyTorch inference wrappers for Linear and GELU.
6. Run training-free inference on a small pretrained or fixed-weight model.
7. Build an energy model comparing FP multiplier, approximate multiplier, and
   B-PLA shift-add execution.
8. Update the paper narrative after real accuracy-energy results are available.

## 13. Conclusion

B-PLA should be framed as a training-free, multiplierless, prefix-routed affine
arithmetic framework for ANN inference. Its strongest novelty is not the use of
piecewise affine approximation alone. That idea already appears in approximate
multiplier and activation-accelerator literature. The stronger contribution is
the unification of multiplication and nonlinear activation into one
hardware-friendly arithmetic primitive that can be inserted into existing ANN
inference pipelines without retraining model weights.

To make this claim convincing, the project must move beyond the current
floating-coefficient prototype. The next decisive steps are dyadic coefficient
quantization, shift-add-only simulation, training-free model-level evaluation,
and hardware energy analysis. If these stages show that B-PLA preserves neural
network accuracy while reducing multiplier usage and energy, the framework can
claim a meaningful and defensible novelty position.

# ViT B-PLA Sensitivity Experiment Summary

This note summarizes the current ViT experiments for B-PLA. It is written as a
draft reference for a future paper experiment section. The results are
preliminary and should be treated as **model-sensitivity evidence**, not as
final hardware measurements.

## 1. Experiment Goal

The goal of this experiment is to answer a practical question:

> Can a pretrained Vision Transformer tolerate B-PLA-style approximation of
> nonlinear activations and selected Linear projections without retraining?

B-PLA is intended to replace expensive arithmetic paths with the following
common primitive:

```text
bit-prefix routing -> coefficient lookup -> dyadic affine evaluation
```

For this experiment, the target model is a pretrained ViT. The test focuses on
two operator classes:

1. GELU-like activation functions in the ViT MLP blocks.
2. `nn.Linear` projections in attention and MLP blocks.

The central research question is not whether this PyTorch proxy is fast. It is
not. The central question is whether the pretrained model's predictions remain
stable under the B-PLA approximation pattern.

## 2. Implementation Path

The experiment uses the CUDA-friendly PyTorch proxy in:

```text
modules/torch_bpla.py
```

and the ViT evaluation script:

```text
experiments/torch_bpla_vit_probe.py
```

This proxy avoids NumPy round trips and supports CUDA tensors, but it is still a
sensitivity probe. It expands Linear operations into elementwise approximate
products and reductions, so it is much slower than native GEMM. Therefore, full
Linear replacement is computationally expensive and should be used only on small
sample subsets.

Important distinction:

```text
This experiment tests accuracy tolerance to B-PLA-like approximation.
It does not prove RTL-level multiplierless hardware behavior.
```

Hardware claims still require fixed-point modeling, dyadic shift-add datapath
verification, and RTL or synthesis evidence.

## 3. Model and Dataset

Model:

```text
google/vit-base-patch16-224
```

Dataset:

```text
johnowhitaker/imagenette2-320
```

The Imagenette labels are mapped to their corresponding ImageNet-1k class
indices before accuracy computation.

Metrics:

- Top-1 accuracy
- Top-5 accuracy

Unless otherwise noted, the experiments used:

```text
prefix_bits = 4
affine_path = dyadic
max_shift = 16
```

The number of dyadic terms was swept for activation experiments.

## 4. B-PLA Approximation Modes

### 4.1 Float Affine Path

The float path uses B-PLA prefix routing and affine segment selection, but
keeps affine coefficients as floating-point values. It is useful as an upper
bound on the approximation quality before coefficient quantization.

Conceptually:

```text
y ~= a_i x + b_i
```

where `a_i` and `b_i` are stored and evaluated as floating-point values.

### 4.2 Dyadic Affine Path

The dyadic path constrains affine coefficients to signed power-of-two sums:

```text
a_i ~= sum_t s_t * 2^(-p_t)
```

where `s_t` is a sign and `p_t` is a shift amount. In the PyTorch proxy this is
implemented with tensor arithmetic, but it represents the shift-add structure
intended for hardware.

The key control parameter is:

```text
dyadic_terms
```

This controls the coefficient budget. More terms generally increase accuracy
but also increase shift-add complexity.

## 5. Experimental Commands

Baseline:

```bash
python experiments/torch_bpla_vit_probe.py \
  --num-samples 100 \
  --batch-size 16 \
  --no-linear \
  --no-gelu \
  --evaluate-ann
```

Activation-only:

```bash
python experiments/torch_bpla_vit_probe.py \
  --num-samples 100 \
  --batch-size 16 \
  --no-linear \
  --affine-path dyadic \
  --dyadic-terms 4 \
  --evaluate-ann
```

Limited Linear replacement:

```bash
python experiments/torch_bpla_vit_probe.py \
  --num-samples 100 \
  --batch-size 4 \
  --max-linear-modules 8 \
  --affine-path dyadic \
  --dyadic-terms 4
```

Linear-only:

```bash
python experiments/torch_bpla_vit_probe.py \
  --num-samples 100 \
  --batch-size 4 \
  --no-gelu \
  --max-linear-modules 8 \
  --affine-path dyadic \
  --dyadic-terms 2
```

## 6. Results

### 6.1 Baseline

| Replaced Linear modules | Replaced activations | Path | Terms | Top-1 | Top-5 |
|---:|---:|---|---:|---:|---:|
| 0 | 0 | none | - | 89.29 | 99.11 |

This is the reference accuracy for the tested Imagenette subset.

### 6.2 Activation-Only Results

| Replaced Linear modules | Replaced activations | Path | Terms | Top-1 | Top-5 | Top-1 drop | Top-5 drop |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 12 | float | - | 85.71 | 99.11 | -3.58 | 0.00 |
| 0 | 12 | dyadic | 1 | 73.00 | 93.00 | -16.29 | -6.11 |
| 0 | 12 | dyadic | 2 | 76.00 | 96.00 | -13.29 | -3.11 |
| 0 | 12 | dyadic | 3 | 81.00 | 98.00 | -8.29 | -1.11 |
| 0 | 12 | dyadic | 4 | 84.00 | 99.00 | -5.29 | -0.11 |
| 0 | 12 | dyadic | >=5 | 84.00 | 99.00 | -5.29 | -0.11 |

Main observation:

```text
Increasing dyadic terms improves activation accuracy until around 4 terms.
After 4 terms, the observed accuracy saturates on this subset.
```

This is a useful accuracy-complexity trade-off result. It suggests that ViT
activation replacement can be made reasonably accurate with a small dyadic
coefficient budget.

### 6.3 Linear-Only Results

| Replaced Linear modules | Replaced activations | Path | Terms | Top-1 | Top-5 | Top-1 drop | Top-5 drop |
|---:|---:|---|---:|---:|---:|---:|---:|
| 8 | 0 | dyadic | 2 | 89.00 | 99.00 | -0.29 | -0.11 |

Main observation:

```text
Replacing the first 8 Linear modules with dyadic B-PLA produced almost no
accuracy loss on the tested subset.
```

This is important because it indicates that the Linear approximation itself is
not uniformly destructive. Some ViT projection layers appear robust to B-PLA
approximation.

### 6.4 Combined Activation and Linear Replacement

| Replaced Linear modules | Replaced activations | Path | Terms | Top-1 | Top-5 | Top-1 drop | Top-5 drop |
|---:|---:|---|---:|---:|---:|---:|---:|
| 8 | 12 | dyadic | 2 | 76.00 | 96.00 | -13.29 | -3.11 |
| 16 | 12 | dyadic | 2 | 76.00 | 96.00 | -13.29 | -3.11 |
| 8 | 12 | dyadic | 4 | 84.00 | 99.00 | -5.29 | -0.11 |
| 16 | 12 | dyadic | 4 | 84.00 | 99.00 | -5.29 | -0.11 |
| 32 | 12 | dyadic | 4 | 84.00 | 99.00 | -5.29 | -0.11 |

Main observation:

```text
With 4-term dyadic activations, increasing replaced Linear modules from 8 to
32 did not introduce additional observed accuracy loss.
```

The combined results suggest that the dominant accuracy loss in the current
ViT experiments comes from activation approximation rather than from the first
several Linear replacements.

## 7. Interpretation

### 7.1 Activation Replacement Is Feasible but Coefficient Budget Matters

The float affine activation path gives a Top-1 score of 85.71, only 3.58 points
below the 89.29 baseline, while keeping Top-5 unchanged. This shows that the
prefix-routed affine activation approximation itself is fairly compatible with
ViT inference.

The dyadic path is more sensitive. With only 1 or 2 signed power-of-two terms,
the Top-1 drop is large. Increasing the budget to 4 terms recovers most of the
lost accuracy:

```text
dyadic terms 1 -> 2 -> 3 -> 4:
Top-1 73 -> 76 -> 81 -> 84
Top-5 93 -> 96 -> 98 -> 99
```

This supports the claim that dyadic term count provides a meaningful
accuracy-complexity knob.

### 7.2 Linear Replacement Appears Layer-Dependent

The first 8 Linear modules can be replaced with almost no accuracy loss when
GELU remains exact:

```text
Baseline:      89.29 / 99.11
Linear-only 8: 89.00 / 99.00
```

This suggests that at least some ViT Linear projections are tolerant to B-PLA
multiplier approximation. The result also supports a future layer-adaptive
deployment strategy rather than a uniform all-layer replacement policy.

### 7.3 Combined Results Point to Activation as the Current Bottleneck

When activation uses dyadic 2-term coefficients, the model remains around
76/96 even as the number of replaced Linear modules changes from 1 to 16 in
earlier probes. With dyadic 4-term activation, the model reaches 84/99 and
remains there as Linear replacement increases from 8 to 32.

This indicates that the current dominant bottleneck is the dyadic activation
approximation quality, not the early Linear replacements.

## 8. Paper-Level Insight

The current ViT results support the following preliminary claim:

> ViT inference tolerates B-PLA-style replacement of selected Linear projections
> and all GELU activations without retraining, but the accuracy depends strongly
> on the dyadic coefficient budget used for activation approximation.

More cautiously:

> On a ViT-Base/Imagenette sensitivity probe, replacing all 12 GELU activations
> with 4-term dyadic B-PLA and replacing up to 32 Linear modules preserved Top-5
> accuracy almost unchanged, while Top-1 accuracy dropped by about 5.29 points.

This is a useful result for a paper because it demonstrates three things:

1. B-PLA can be inserted into a pretrained transformer-style model without
   retraining.
2. Dyadic coefficient budget creates an observable accuracy-complexity
   trade-off.
3. Uniform replacement is not the only path; layer-sensitive or operator-aware
   replacement is likely the stronger deployment strategy.

## 9. Limitations

The current results should be presented carefully.

First, this is a PyTorch proxy experiment. It uses torch tensor operations to
simulate B-PLA-like approximation on CUDA. It is not an RTL implementation and
does not prove actual multiplier removal in hardware.

Second, the evaluation was run on a subset of Imagenette, not the full ImageNet
validation set. The results are suitable for early method validation but not
yet for final publication-level accuracy claims.

Third, full Linear replacement is computationally expensive in this proxy
implementation because native GEMM is expanded into elementwise approximate
products. Therefore, runtime should not be interpreted as hardware latency.

Fourth, dyadic coefficient generation currently uses a greedy signed
power-of-two approximation. More accurate coefficient fitting or
calibration-aware dyadic optimization may improve the activation results.

## 10. Recommended Next Experiments

### 10.1 Complete Linear-Only Sweep

Run Linear-only replacement for larger module counts:

```bash
python experiments/torch_bpla_vit_probe.py --num-samples 100 --batch-size 4 --no-gelu --max-linear-modules 16 --affine-path dyadic --dyadic-terms 2
python experiments/torch_bpla_vit_probe.py --num-samples 100 --batch-size 4 --no-gelu --max-linear-modules 32 --affine-path dyadic --dyadic-terms 2
```

This will clarify how far Linear replacement can go before accuracy degrades.

### 10.2 Activation Calibration

Improve dyadic activation fitting:

- compare greedy signed power-of-two fitting with error-aware fitting,
- fit slopes and intercepts under actual activation distributions,
- evaluate separate slope/intercept term budgets,
- test layer-wise activation ranges instead of a single global range.

### 10.3 Layer-Adaptive Deployment

Instead of replacing all modules uniformly, assign different B-PLA
configurations per layer:

```text
robust layers     -> lower dyadic terms or more replacements
sensitive layers  -> higher dyadic terms or exact fallback
```

This aligns with the emerging result that early Linear modules can be robust
while activation approximation dominates the current accuracy drop.

## 11. Suggested Figure/Table for Paper

Recommended table:

```text
Model: ViT-Base/16 on Imagenette
Columns: replaced Linear modules, replaced activations, affine path,
dyadic terms, Top-1, Top-5, Top-1 drop, Top-5 drop
```

Recommended plot:

```text
x-axis: dyadic terms for activation
y-axis: Top-1 and Top-5 accuracy
series: activation-only, activation + 8 Linear, activation + 32 Linear
```

Expected message:

```text
Dyadic coefficient budget controls the accuracy-complexity trade-off, and
selected Linear replacements add little extra degradation once activation
approximation is sufficiently accurate.
```

# B-PLA

Bit-Prefix Piecewise Linear Approximation (B-PLA) research code for
multiplierless neural network inference experiments.

B-PLA approximates selected floating-point multiplication and activation paths
with a shared pattern:

```text
bit-prefix routing -> coefficient lookup -> affine evaluation
```

The repository includes both hardware-oriented NumPy prototypes and fast
PyTorch proxy modules for large-model sensitivity tests.

## Contents

- `modules/bpla_multiplier.py`: FP32 mantissa-interaction B-PLA multiplier.
- `modules/bpla_activation.py`: FP32 bit-field-routed B-PLA activation.
- `modules/dyadic.py`: signed power-of-two dyadic coefficient utilities.
- `modules/pla_snn.py`: term-free event-driven PLA compiler and conditional-accumulation runtime.
- `modules/span_core.py`: SPAN spike packet, causal prefix, conditional-add, and threshold/reset primitives.
- `modules/span_synapse.py`: static ANN-weight SPAN synapse compiler and fused Linear/MAC prototype.
- `modules/span_ops.py`: causal unary activation and dynamic-dynamic mantissa-tile SPAN operators.
- `modules/span_normalization.py`: Softmax and LayerNorm composed from SPAN static, dynamic, and unary primitives.
- `modules/span_coverage.py`: conservative PyTorch graph coverage auditor for unmapped multiplication and transcendental modules.
- `PASN/`: self-contained PASN research folder with modules, experiments,
  results, tests, and lab-meeting Markdown.
- `modules/torch_bpla.py`: CUDA-friendly PyTorch B-PLA proxy layers.
- `modules/torch_pao.py`: PAO/PAM baseline (Kosson and Jaggi, NeurIPS 2023) forward primitives.
- `modules/torch_ptq.py`: W8A8 post-training-quantization baseline (per-token and per-tensor activation scales).
- `modules/compute_energy.py`: memory-free theoretical arithmetic energy model.
- `experiments/compute_energy_experiment.py`: primitive, MLP, ViT, and GPT-2 compute-energy comparison.
- `experiments/bpla_mlp_experiment.py`: hardware-style MNIST MLP probe.
- `experiments/torch_bpla_mlp_probe.py`: fast PyTorch MNIST MLP probe.
- `experiments/torch_bpla_gpt2_probe.py`: GPT-2 B-PLA sensitivity probe.
- `experiments/torch_bpla_vit_probe.py`: ViT B-PLA sensitivity probe.
- `experiments/pao_vs_bpla_primitive.py`: PAM vs. B-PLA multiplier fidelity, cost proxies, and accumulated bias.
- `experiments/pao_vs_bpla_model.py`: Exact / PAO / B-PLA training-free drop-in on the same checkpoint.
- `experiments/nonlinear_primitive_figure.py`: GELU three-panel figure, per-primitive fidelity, calibration-range policies.
- `experiments/replacement_coverage.py`: audit of converted sites and exact remaining paths.
- `tests/`: unit tests for NumPy, SNN, dyadic, torch proxy, and PAO baseline paths.

## PAO Baseline Comparison

`modules/torch_pao.py` reimplements the forward piecewise affine operations of
Kosson and Jaggi (NeurIPS 2023) so both methods can be inserted into the same
pretrained checkpoint with zero weight updates. `tests/test_pao.py` verifies it
against the Mogami int-addition trick used by the authors' released kernel, and
pins the published `-1/9` worst-case relative error.

```bash
python -m unittest tests.test_pao tests.test_bpla_table_forms
python experiments/pao_vs_bpla_primitive.py --num-samples 400000
python experiments/nonlinear_primitive_figure.py --model vit --num-images 8
python experiments/replacement_coverage.py --scopes multiplication --replace-lm-head --replace-conv2d
```

## W8A8 PTQ Baseline Comparison

`modules/torch_ptq.py` adds post-training quantization to the same harness.
PTQ, not PAO, is what a training-free method is normally measured against, so
this is the comparison that decides whether B-PLA is competitive.

Two recipes are reported as separate backends, because a single one would
misrepresent the baseline:

- `ptq-w8a8` uses dynamic **per-token** activation scales (ZeroQuant /
  LLM.int8() style). This is the strong form and the primary baseline.
- `ptq-w8a8-static` uses static **per-tensor** scales from percentile
  calibration, the conventional recipe.

Both use per-output-channel symmetric weight quantization and simulate int8
arithmetic by quantize-dequantize with a float32 accumulator, which is the same
accumulator every other backend uses. Calibration is forward-only over the same
batches B-PLA gets. A model that was replaced but never calibrated raises rather
than silently returning exact results.

The W8A8 backends run at the `multiplication` scope only: W8A8 leaves GELU,
Softmax and LayerNorm in floating point by convention, so it has no honest
`nonlinear` or `combined` row, and the harness refuses those combinations.

```bash
python -m unittest tests.test_ptq
bash run_ptq.sh          # matched 6-backend run, multiplication scope
```

## Table Construction and Replacement Scope

Two `TorchBPLAConfig` fields control how coefficient tables are built. Both
defaults are the better choice; the alternatives exist to reproduce older runs.

- `multiplier_form` (default `separable`): the tile-centre plane is separable,
  so `nu*m1 + mu*(m2-nu)` reproduces it from a single `2^k` array with no stored
  offset. Against the legacy `plane` form this is 5.7x to 20x more accurate at
  equal term budget and 48x smaller.
- `anchor_mode` (default `auto`): the point each 1-D segment is expanded around
  is chosen per table by measured error. Expanding about the `y`-intercept is a
  long extrapolation for tables far from the origin; `auto` keeps the intercept
  for `exp2` and moves it for the reciprocal and reciprocal square root.

Two module types are outside the default replacement scope and convert only when
asked, so that widening coverage stays an explicit and reported choice:

- `replace_lm_head=True` converts GPT-2's output projection, 31% of its weighted
  multiplies. The weight is shared rather than cloned so the tie to the token
  embedding survives.
- `replace_conv2d=True` converts ViT's patch embedding via `unfold`.

With both enabled, weighted-multiply coverage is 100% on ViT-Base and GPT-2;
without them it is 99.3% and 68.8%. Attention `QK`/`PV` products are converted in
either case and are not counted in those totals, having no weight module.

Two caveats are structural, not incidental. The PAO paper's models use ReLU and
it defines no piecewise affine GELU, so `pao_gelu_torch` is our composition from
its primitives. Section 2.7 of that paper sketches an `alpha` error-compensation
constant but reports no results for it; the primitive experiment fits and reports
it as the `pao-alpha` condition, because leaving it off would measure an
avoidable deficiency of the baseline rather than a property of the method.

## Install

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

For GPU runs, install a CUDA-enabled PyTorch build that matches your vast.ai
image before installing the remaining requirements.

## Quick Checks

```bash
python -m unittest discover -s tests
```

Run only the new full-arithmetic spiking prototypes:

```bash
python -m unittest tests.test_span tests.test_span_coverage -v
```

PASN (Prefix-Adaptive Spiking Neuron) is a separate SNN research
path, not a rename of SPAN or a spiking implementation of B-PLA arithmetic.
Its current prototype tests whether prefix-selected local basis banks can trade
total parameter memory for fewer active bases and threshold comparisons than
one global MBE-style neuron.
Run the GELU falsification benchmark in an environment with PyTorch:

```bash
python PASN/experiments/pasn_operator_benchmark.py --external-mbe --plot
```

The generated operator-level results under `PASN/experiments/results/` are
preliminary calibration evidence only.  They do not establish end-to-end SNN
accuracy, latency, energy, or hardware benefit.

SPAN (Spike-Prefix Affine Neuron) is the B-PLA + SNN conversion research path.
The SPAN modules are hardware-oriented operator prototypes. With 23 mantissa
events, the normal-FP32 static synapse and dynamic tile are regression-tested
against the existing float-affine B-PLA multiplier. Their zero operands use a
control path; subnormal, infinity, and NaN behavior is still a flagged software
fallback rather than a multiplier-free hardware implementation. The coverage
auditor establishes graph-replacement coverage only and does not replace RTL
synthesis evidence for zero inferred multipliers or DSP blocks.

Dry-run the large-model wrappers without downloading GPT-2 or ViT:

```bash
python experiments/torch_bpla_gpt2_probe.py --dry-run --affine-path dyadic --dyadic-terms 2
python experiments/torch_bpla_vit_probe.py --dry-run --affine-path dyadic --dyadic-terms 2
```

Run the standalone compute-only comparison without loading datasets or pretrained models:

```bash
python experiments/compute_energy_experiment.py --affine-path dyadic --dyadic-terms 2 --gpt2-sequence-length 256 --bpla-softmax --bpla-layernorm
python experiments/compute_energy_experiment.py --affine-path dyadic --dyadic-terms 2 --shift-energy-pj 0.05 --json-out results/compute_energy.json
```

This model excludes LUT, register, SRAM, DRAM, interconnect, and leakage energy.
It charges one common FP32 accumulation per scalar product and uses a zero-cost
`tanh` as a conservative lower bound for the conventional tanh-form GELU. The
MLP, ViT, and GPT-2 probes also print the same compute-only estimate for their
actual replacement settings.

The model also reports Softmax and LayerNorm row/element counts, their separate
ANN/B-PLA energy, and their contribution to the total. Conventional `exp`,
reciprocal, and reciprocal-square-root default to an optimistic one-FP32-
multiply energy (`3.7 pJ`) because the cited arithmetic table does not specify
those units. Override the assumptions with `--energy-exp-pj`,
`--energy-reciprocal-pj`, and `--energy-rsqrt-pj` in the ViT/GPT-2 probes, or
the corresponding `--exp-energy-pj`, `--reciprocal-energy-pj`, and
`--rsqrt-energy-pj` options in the standalone experiment.

## MNIST MLP Probe

Fast PyTorch proxy:

```bash
python experiments/torch_bpla_mlp_probe.py --max-test-samples 1000 --affine-path dyadic --dyadic-terms 2
```

Hardware-style NumPy bridge:

```bash
python experiments/bpla_mlp_experiment.py --max-test-samples 1000 --affine-path dyadic --dyadic-terms 2
```

## GPT-2 Probe

Check model loading, replacement, and one smoke forward, then stop before full
evaluation:

```bash
python experiments/torch_bpla_gpt2_probe.py --stop-after-conversion --affine-path dyadic --dyadic-terms 2
```

Run a short WikiText perplexity probe:

```bash
python experiments/torch_bpla_gpt2_probe.py --num-windows 4 --max-length 256 --stride 256 --affine-path dyadic --dyadic-terms 2
```

## ViT Probe

Check model loading, replacement, and one smoke forward:

```bash
python experiments/torch_bpla_vit_probe.py --stop-after-conversion --affine-path dyadic --dyadic-terms 2
```

Run a short Imagenette probe:

```bash
python experiments/torch_bpla_vit_probe.py --num-samples 100 --batch-size 4 --affine-path dyadic --dyadic-terms 2
```

The full ViT Linear replacement is intentionally expensive because every
matrix multiplication is expanded into elementwise B-PLA products. For quick
sensitivity checks, start with activation-only or a small number of Linear
modules:

```bash
python experiments/torch_bpla_vit_probe.py --num-samples 100 --batch-size 16 --no-linear --affine-path dyadic --dyadic-terms 2
python experiments/torch_bpla_vit_probe.py --num-samples 20 --batch-size 2 --max-linear-modules 4 --affine-path dyadic --dyadic-terms 2 --linear-chunk-out 128
```

For GPT-2, first sweep only a few Conv1D modules before trying full
replacement:

```bash
python experiments/torch_bpla_gpt2_probe.py --stop-after-conversion --max-conv1d-modules 4 --affine-path dyadic --dyadic-terms 2
python experiments/torch_bpla_gpt2_probe.py --num-windows 1 --max-conv1d-modules 4 --affine-path dyadic --dyadic-terms 2 --linear-chunk-out 128
```

Diagnose GPT-2 attention independently from Conv1D and GELU replacement:

```bash
python experiments/torch_bpla_gpt2_probe.py --num-windows 10 --max-length 32 --stride 32 --no-conv1d --no-gelu --evaluate-ann --attention-mode exact --attention-diagnostics
python experiments/torch_bpla_gpt2_probe.py --num-windows 10 --max-length 32 --stride 32 --no-conv1d --no-gelu --evaluate-ann --attention-mode bpla-qk --attention-diagnostics --affine-path float --prefix-bits 4
python experiments/torch_bpla_gpt2_probe.py --num-windows 10 --max-length 32 --stride 32 --no-conv1d --no-gelu --evaluate-ann --attention-mode bpla-pv --attention-diagnostics --affine-path float --prefix-bits 4
python experiments/torch_bpla_gpt2_probe.py --num-windows 10 --max-length 32 --stride 32 --no-conv1d --no-gelu --evaluate-ann --attention-mode bpla-full --attention-diagnostics --affine-path float --prefix-bits 4
```

`exact` validates the custom attention interface with native matmul. `bpla-qk`
approximates only the attention-score product, `bpla-pv` approximates only the
probability-value product, and `bpla-full` approximates both. Diagnostics record
the first attention call's QK-score, Softmax-probability, attention-output, and
masked-probability errors against exact matmul.

Approximate the remaining attention Softmax and model LayerNorm operations
with the composed B-PLA modules described in the accompanying design PDFs:

```bash
python experiments/torch_bpla_gpt2_probe.py --dry-run --no-conv1d --no-gelu --attention-mode exact --bpla-softmax --bpla-layernorm --attention-diagnostics --affine-path float
python experiments/torch_bpla_vit_probe.py --dry-run --no-linear --no-gelu --attention-mode exact --bpla-softmax --bpla-layernorm --attention-diagnostics --affine-path float
```

`--bpla-softmax` uses max subtraction followed by a prefix-routed affine
approximation of the fractional `exp2`, exponent shifts, a mantissa reciprocal,
and B-PLA multiplication. `--bpla-layernorm` composes B-PLA mean scaling,
squaring, variance scaling, mantissa reciprocal-square-root, exponent shifts,
and affine scaling. Both flags are opt-in so older probe configurations remain
comparable. The reductions, maximum, additions, and control paths are still
exact in this PyTorch sensitivity proxy.

Start the new operator isolation with `--affine-path float`. The default
two-term dyadic coefficients are deliberately coarse for reciprocal-based
normalization, so treat `--dyadic-terms` as an accuracy/cost sweep parameter
rather than assuming that two terms preserve probability normalization.

## Notes

The PyTorch B-PLA path is a CUDA-friendly sensitivity proxy. It is intended to
test whether pretrained models tolerate B-PLA-like approximation. Publication
claims about multiplierless hardware still require fixed-point modeling and RTL
or synthesis evidence.

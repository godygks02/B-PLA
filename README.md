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
- `modules/torch_bpla.py`: CUDA-friendly PyTorch B-PLA proxy layers.
- `modules/compute_energy.py`: memory-free theoretical arithmetic energy model.
- `experiments/compute_energy_experiment.py`: primitive, MLP, ViT, and GPT-2 compute-energy comparison.
- `experiments/bpla_mlp_experiment.py`: hardware-style MNIST MLP probe.
- `experiments/torch_bpla_mlp_probe.py`: fast PyTorch MNIST MLP probe.
- `experiments/torch_bpla_gpt2_probe.py`: GPT-2 B-PLA sensitivity probe.
- `experiments/torch_bpla_vit_probe.py`: ViT B-PLA sensitivity probe.
- `tests/`: unit tests for NumPy, SNN, dyadic, and torch proxy paths.

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

Dry-run the large-model wrappers without downloading GPT-2 or ViT:

```bash
python experiments/torch_bpla_gpt2_probe.py --dry-run --affine-path dyadic --dyadic-terms 2
python experiments/torch_bpla_vit_probe.py --dry-run --affine-path dyadic --dyadic-terms 2
```

Run the standalone compute-only comparison without loading datasets or pretrained models:

```bash
python experiments/compute_energy_experiment.py --affine-path dyadic --dyadic-terms 2 --gpt2-sequence-length 256
python experiments/compute_energy_experiment.py --affine-path dyadic --dyadic-terms 2 --shift-energy-pj 0.05 --json-out results/compute_energy.json
```

This model excludes LUT, register, SRAM, DRAM, interconnect, and leakage energy.
It charges one common FP32 accumulation per scalar product and uses a zero-cost
`tanh` as a conservative lower bound for the conventional tanh-form GELU. The
MLP, ViT, and GPT-2 probes also print the same compute-only estimate for their
actual replacement settings.

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

## Notes

The PyTorch B-PLA path is a CUDA-friendly sensitivity proxy. It is intended to
test whether pretrained models tolerate B-PLA-like approximation. Publication
claims about multiplierless hardware still require fixed-point modeling and RTL
or synthesis evidence.

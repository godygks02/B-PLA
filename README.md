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
- `modules/torch_bpla.py`: CUDA-friendly PyTorch B-PLA proxy layers.
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

## Notes

The PyTorch B-PLA path is a CUDA-friendly sensitivity proxy. It is intended to
test whether pretrained models tolerate B-PLA-like approximation. Publication
claims about multiplierless hardware still require fixed-point modeling and RTL
or synthesis evidence.

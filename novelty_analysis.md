# B-PLA Novelty Analysis and Positioning Strategy

## 1. Executive Summary

The current prior-art analysis suggests that **B-PLA does not clearly already exist as one complete method**, but many of its individual components are already present in related literature.

The safest interpretation is:

```text
B-PLA is not fully novel as a standalone approximate multiplier,
and it is not fully novel as a standalone PWL activation approximator.
Its strongest novelty is the unified arithmetic framework:
FP multiplication and nonlinear activation are both mapped to a
bit-prefix-routed, multiplierless affine evaluator.
```

Therefore, the paper should not be framed as merely:

- a new piecewise-linear approximate multiplier,
- a new piecewise-linear activation unit,
- a generic multiplierless approximation block,
- or a LUT-based approximation method.

Instead, the paper should be framed as:

```text
A training-free, multiplierless, bit-prefix-routed affine arithmetic framework
that treats FP multiplication and nonlinear activation as two instances of
one hardware-friendly primitive: coefficient lookup plus dyadic shift-add
affine evaluation.
```

## 2. Main Prior-Art Finding

The analysis found dense prior work in two separate areas:

1. **Approximate floating-point multipliers**
   - Piecewise-linear approximate multipliers already exist.
   - Sign and exponent separation with significand or mantissa approximation is already known.
   - Mantissa-product surface fitting is already studied.
   - Power-of-two or shift-friendly coefficients have also appeared in recent multiplier designs.

2. **Piecewise-linear activation hardware**
   - PWL activation approximation is well established.
   - LUT-based activation approximation is well established.
   - Multiplierless or shift-add implementation of PWL activation is also known.
   - Unified hardware for multiple activation functions already exists.

However, the analysis did **not** find a clear complete method that combines all of the following:

- FP mantissa interaction approximation using prefix-routed affine tiles,
- FP32 sign/exponent/mantissa-prefix routing for activation approximation,
- one shared prefix-routed affine evaluator for both multiplication and activation,
- training-free replacement of selected ANN inference arithmetic paths.

This combined architecture is the most promising novelty direction.

## 3. Closest Prior Work

### 3.1 ApproxLP

ApproxLP is a close conceptual predecessor for the B-PLA multiplier.

It:

- computes sign and exponent conventionally,
- approximates the significand part using fitted linear functions,
- replaces part of multiplication with weighted addition and normalization.

Overlap with B-PLA:

- high overlap on approximate FP multiplication,
- high overlap on separating cheap sign/exponent logic from expensive significand logic,
- high overlap on using linear approximation for multiplication.

Remaining B-PLA distinction:

- explicit mantissa-prefix routing,
- explicit approximation of only `M1*M2`,
- dyadic coefficient target for shift-add evaluation,
- intended reuse of the affine evaluator for activation.

### 3.2 PAM

PAM is a strong baseline for the B-PLA multiplier.

It:

- is a piecewise-linearly approximated floating-point multiplier,
- formulates approximation as an optimization problem,
- provides a configurable hardware architecture.

Overlap with B-PLA:

- very high overlap on PWL FP multiplier approximation,
- high overlap on configurable approximate multiplier architecture.

Remaining B-PLA distinction:

- B-PLA narrows the approximation to the mantissa interaction term,
- B-PLA uses bit-prefix tile indexing,
- B-PLA targets dyadic shift-add evaluation,
- B-PLA connects the multiplier primitive to activation approximation under one shared evaluator abstraction.

### 3.3 HAM

HAM is especially important because it is close to the dyadic-coefficient idea.

It:

- extracts exponent and mantissa,
- performs mixed linear fitting on the mantissa-product surface,
- constrains slopes to limited power-of-two elements for simpler logic.

Overlap with B-PLA:

- high overlap on mantissa-product surface approximation,
- high overlap on linear fitting,
- high overlap on power-of-two-friendly coefficient design.

Remaining B-PLA distinction:

- explicit 2D prefix-indexed affine tiles,
- multiplierless dyadic coefficient table as a central primitive,
- integration with activation routing and a shared evaluator.

### 3.4 ML-PLAC

ML-PLAC is a close activation-side baseline.

It:

- implements multiplierless PWL approximation for nonlinear functions,
- replaces slope multiplication with shift-add realization.

Overlap with B-PLA:

- high overlap on multiplierless PWL activation,
- high overlap on shift-add evaluation of affine segments.

Remaining B-PLA distinction:

- B-PLA uses FP32 bit-field prefix routing instead of generic segment routing,
- B-PLA is intended to share its affine evaluator with the multiplier path.

### 3.5 Flex-SFU

Flex-SFU is a strong activation front-end competitor.

It:

- supports activation-function acceleration,
- uses nonuniform piecewise approximation,
- includes floating-point support and address decoding.

Overlap with B-PLA:

- moderate to high overlap on activation approximation,
- overlap on floating-point-aware activation hardware.

Remaining B-PLA distinction:

- B-PLA should emphasize direct FP32 sign/exponent/mantissa-prefix routing,
- B-PLA should emphasize dyadic shift-add affine evaluation,
- B-PLA should emphasize unification with multiplication.

### 3.6 Other Relevant Baselines

Other relevant directions include:

- static and dynamic segmentation approximate FP multipliers,
- LUT-oriented approximate FP multipliers,
- generic hardware for multiple activation functions,
- unified smooth activation approximation hardware,
- multiplierless neural network inference,
- power-of-two quantized or shift-based neural networks,
- LUT-based multiplication for FPGA inference.

These do not necessarily invalidate B-PLA, but they limit which claims can be made safely.

## 4. Component-Level Novelty Judgment

### 4.1 B-PLA Multiplier

Judgment:

```text
Partially novel at best.
Possibly not novel if claimed broadly.
```

The multiplier claim is the weakest part because approximate FP multipliers using PWL or linearized mantissa/significand approximation are already crowded.

Unsafe broad claim:

```text
We propose a novel piecewise-linear approximate floating-point multiplier.
```

Safer narrow claim:

```text
We approximate only the FP mantissa interaction term using 2D mantissa-prefix
affine tiles whose coefficients are constrained for dyadic shift-add evaluation.
```

Even this should be presented carefully as an implementation-specific contribution, not as the entire novelty of the paper.

### 4.2 B-PLA Activation

Judgment:

```text
Partially novel.
```

PWL activation approximation and multiplierless activation hardware are already known. The possible novelty is the routing method:

```text
Direct FP32 sign/exponent/mantissa-prefix routing for activation segment
selection.
```

Unsafe broad claim:

```text
We propose a novel multiplierless PWL activation unit.
```

Safer narrow claim:

```text
We route FP32 activation inputs directly through their sign, exponent, and
mantissa-prefix fields, avoiding runtime range-normalization or interval-search
routing before selecting a dyadic affine segment.
```

### 4.3 Unified B-PLA Architecture

Judgment:

```text
Most promising novelty.
```

The prior-art analysis found unified hardware for multiple activation functions, and many approximate multiplier designs, but did not clearly identify a single primitive shared across both:

- approximate FP multiplication,
- nonlinear activation approximation.

This should become the main novelty axis.

Recommended claim:

```text
B-PLA unifies FP multiplication and nonlinear activation as prefix-routed
affine approximation problems and maps both to a common multiplierless
dyadic shift-add evaluator.
```

## 5. Claims to Avoid

The paper should avoid or heavily qualify the following claims:

- PWL approximation itself is novel.
- Approximate floating-point multiplication itself is novel.
- Mantissa/significand approximation itself is novel.
- LUT-based approximation itself is novel.
- Multiplierless PWL activation itself is novel.
- Training-free approximate inference itself is novel.
- Shift-add approximation itself is novel.
- Power-of-two coefficient approximation itself is novel.

These ideas are already represented in prior work.

## 6. Claims That Are More Defensible

The following claims are more defensible if supported by experiments:

1. **Unified arithmetic primitive**

   B-PLA maps multiplication and activation to the same prefix-routed affine evaluator pattern.

2. **Operator-specific routing with shared evaluator**

   Multiplication uses 2D mantissa-prefix routing, while activation uses 1D FP32 sign/exponent/mantissa-prefix routing. Both feed the same coefficient-LUT plus dyadic shift-add evaluator.

3. **Multiplier removal at the arithmetic-core level**

   B-PLA removes runtime multipliers from selected approximation cores by constraining coefficients to dyadic or signed power-of-two forms.

4. **Training-free operator replacement**

   B-PLA can be inserted into selected ANN inference operations without retraining the model, while offline coefficient calibration is allowed.

5. **Hardware area and energy advantage under realistic constraints**

   B-PLA may reduce energy and area if the added LUT, mux, routing, normalization, and packing overheads are smaller than the removed multipliers or DSP blocks.

## 7. Hardware Architecture Strategy

### 7.1 Do We Need One Physical Unified Hardware Block?

Not necessarily.

It may be inefficient to force every multiplier and every activation into one single physical block. In neural network accelerators, multiplication operations usually occur much more frequently than activation operations. A fully shared single unit could become a throughput bottleneck.

However, for novelty preservation, the paper should still present B-PLA as a **unified hardware primitive**.

The best framing is:

```text
One shared evaluator primitive with operator-specific routing front-ends.
```

This means:

- multiplier and activation may have different front-end routing logic,
- multiplier and activation may have different output reconstruction logic,
- but both use the same core pattern:

```text
prefix route -> coefficient LUT -> dyadic shift-add affine evaluator
```

### 7.2 Recommended Architecture Diagram

The main paper diagram should show a shared B-PLA core:

```text
                  mode
                   |
        +----------------------+
        |  B-PLA Front-End     |
        |  Mult / Activation   |
        +----------------------+
          |                  |
          |                  |
  +---------------+   +----------------+
  | Mult Routing  |   | Act Routing    |
  | M1,M2 prefix  |   | S,E,M prefix   |
  +---------------+   +----------------+
          |                  |
          +-------- mux -----+
                   |
        +----------------------+
        | Coefficient LUT      |
        | a,b,c or alpha,beta  |
        +----------------------+
                   |
        +----------------------+
        | Shared Dyadic        |
        | Shift-Add Affine     |
        | Evaluator            |
        +----------------------+
          |                  |
  +---------------+   +----------------+
  | Mantissa      |   | Activation     |
  | Reconstruct   |   | Output Path    |
  +---------------+   +----------------+
          |                  |
  +---------------+   +----------------+
  | Normalize     |   | Pack / Saturate|
  | Round / Pack  |   |                |
  +---------------+   +----------------+
```

This diagram preserves the novelty claim without forcing an unrealistic single physical unit for all operations.

### 7.3 Dedicated vs Shared Implementation

The paper can discuss two implementation options.

#### Option A: Dedicated B-PLA Blocks

Multiplier and activation use separate hardware blocks.

Advantages:

- higher throughput,
- easier scheduling,
- better for multiplier-dense workloads,
- practical for convolution and matrix multiplication pipelines.

Disadvantages:

- weaker visual novelty,
- less area sharing,
- easier for reviewers to say the method is just two known approximation blocks placed together.

#### Option B: Shared-Evaluator B-PLA Block

Multiplier and activation share the dyadic affine evaluator.

Advantages:

- stronger architectural novelty,
- lower area for low-throughput or edge designs,
- clearer evidence of unification,
- better supports the paper's central thesis.

Disadvantages:

- possible throughput bottleneck,
- requires scheduling and mode control,
- may need different precision settings for multiplier and activation paths.

#### Recommended Paper Position

The paper should present:

```text
A shared-evaluator architecture as the canonical B-PLA architecture,
with dedicated or replicated instances as implementation variants for
throughput scaling.
```

This keeps the novelty while remaining realistic.

## 8. Additional Method Elements to Add or Emphasize

### 8.1 Dyadic Coefficient Quantization

This is essential.

Without dyadic or signed power-of-two coefficients, B-PLA becomes a software-level affine approximation and may still require multipliers.

The paper should include:

- coefficient quantization algorithm,
- number of power-of-two terms per coefficient,
- coefficient bit width,
- energy and accuracy tradeoff by term count.

### 8.2 Hardware-Faithful Fixed-Point Datapath

The paper should explicitly define:

- mantissa fixed-point format,
- activation internal fixed-point format,
- adder width,
- rounding mode,
- normalization behavior,
- overflow and underflow handling,
- special-case behavior for zero, subnormal, infinity, and NaN.

This will make the hardware claim more credible.

### 8.3 Operation-Level Energy Model

Energy reporting should separate:

- arithmetic energy,
- LUT or ROM read energy,
- routing and mux energy,
- normalization and packing energy,
- memory access energy if included.

This matters because an FP multiplier energy number usually does not include full memory movement cost.

### 8.4 RTL or HLS Synthesis Evidence

The strongest evidence would be:

- RTL or HLS implementation,
- synthesis result showing no DSP or multiplier block in the B-PLA core,
- area comparison against FP multiplier and approximate multiplier baselines,
- critical path and frequency,
- power estimate.

### 8.5 End-to-End Model Accuracy

Arithmetic error alone is not enough.

The paper should evaluate:

- standalone multiplier error,
- standalone activation error,
- end-to-end ANN inference accuracy,
- layer sensitivity,
- replacement ratio,
- accuracy-energy Pareto curve.

## 9. Suggested Experimental Baselines

The paper should compare against:

- FP32 multiplier,
- FP16 multiplier,
- INT8 or fixed-point inference baseline,
- ApproxLP,
- PAM,
- HAM,
- static or dynamic segmentation approximate FP multipliers,
- LUT-oriented FP multiplier,
- ML-PLAC,
- Flex-SFU,
- generic multi-activation PWL hardware,
- ShiftCNN or other multiplierless neural network inference methods,
- LUTMUL-style FPGA multiplication if FPGA is targeted.

Not all baselines need RTL implementation, but the paper should clearly justify which comparisons are analytical, simulated, or synthesized.

## 10. Suggested Paper Framing

### Weak Framing

Avoid this:

```text
We propose a new approximate floating-point multiplier and a new activation
approximator.
```

This framing invites direct rejection because both areas are crowded.

### Stronger Framing

Use this:

```text
We propose B-PLA, a training-free arithmetic replacement framework that
unifies floating-point multiplication and nonlinear activation under a
bit-prefix-routed affine approximation primitive. B-PLA uses operator-specific
bit-field routing and a shared dyadic shift-add evaluator to remove runtime
multipliers from selected ANN inference paths.
```

### Most Important Sentence

The paper should repeatedly reinforce this idea:

```text
B-PLA is not merely an approximate multiplier or an activation approximator;
it is a shared arithmetic primitive that maps both operations to the same
prefix-routed multiplierless affine evaluation datapath.
```

## 11. Recommended Next Steps

1. Implement dyadic coefficient quantization for both multiplier and activation.
2. Update the activation module so its affine slopes can be evaluated by signed power-of-two terms.
3. Extend the energy model to activation, not only multiplier.
4. Add a hardware-style operation counter for both dedicated and shared-evaluator architectures.
5. Create two architecture diagrams:
   - dedicated multiplier and activation B-PLA blocks,
   - canonical shared-evaluator B-PLA block.
6. Run sweeps over:
   - prefix bits,
   - dyadic term count,
   - mantissa precision,
   - activation target,
   - coefficient bit width.
7. Evaluate end-to-end model accuracy.
8. If possible, build RTL or HLS for the B-PLA core.
9. Compare against the strongest multiplier and activation baselines.

## 12. Final Position

The current novelty position is promising but must be narrowed.

B-PLA should not rely on the multiplier alone for novelty. The multiplier component is too close to prior approximate FP multiplier literature. The activation component is also not novel if described only as PWL or multiplierless activation approximation.

The strongest paper contribution is:

```text
the unification of FP multiplication and nonlinear activation into one
bit-prefix-routed, dyadic shift-add affine arithmetic framework.
```

To preserve this novelty, the method should explicitly include:

- direct bit-field routing,
- dyadic coefficient quantization,
- multiplierless fixed-point affine evaluation,
- operator-specific front-ends,
- shared evaluator architecture,
- hardware evidence showing area and energy benefit.

If these are demonstrated experimentally, B-PLA can be positioned as a defensible framework-level contribution rather than a minor variant of existing approximate multiplier or PWL activation methods.


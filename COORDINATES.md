# Factor coordinates, not a new optimizer

LoRALab v0.2 adds an independent Python/browser inspector to the unchanged original online adaptation experiment. It asks a finite question: when two factor pairs represent the same model, do their next ordinary Euclidean SGD updates represent the same model too?

## Definitions and shapes

The frozen classifier has `W` with `m` output rows, `d` features and a frozen bias. The trainable factors have shapes `B: m x r` and `A: r x d`. Logits are `(W + BA)x + bias`. Unlike the original LoRA paper's common convention, this existing experiment has no `alpha/r` scale. No scale has been added to its original training.

`r` is factor width, not measured matrix rank. Always `rank(BA) <= min(m,d,r)`. The original classifier has `m=3,d=128`, so widths 4 and 8 are overcomplete. Their 524 and 1,048 trainable factor entries exceed the full baseline's 387 parameters. Width 2 has 262 entries, or 67.7% of 387. The full baseline trains three biases while adapters freeze them. Counts are not GPU memory, latency or quality measurements.

## Equivalent coordinates

For an invertible `r x r` matrix `C`, define

```
B' = B C
A' = C^-1 A
B'A' = BA
```

These are equivalent factors in real arithmetic, including rank-deficient factors such as the original `B=0` initialization. This algebra does not require a full-rank manifold chart. Floating-point inversion and reassociation can leave residuals. The report exposes both inverse residuals and model/logit residuals; it does not declare exact binary64 equivalence or certify predictions near a tie.

For a loss depending on `BA`, write `G = dL/d(BA)`. Its factor gradients are

```
grad_A = B^T G
grad_B = G A^T
```

Under the coordinate change, `grad_A' = C^T grad_A` and `grad_B' = grad_B C^-T`. Ordinary equal-rate Euclidean updates therefore depend on non-orthogonal factor coordinates. Orthogonal changes of basis do preserve ordinary SGD's merged update in real arithmetic. This inspector does not claim that ordinary SGD has the same invariance properties as Adam or LoRA-RITE.

## Full finite-step identity

Use simultaneous gradients at the same pre-update factors, with separate nonnegative rates:

```
A_next = A - eta_A B^T G
B_next = B - eta_B G A^T

B_next A_next - BA
  = -eta_A B B^T G - eta_B G A^T A
    + eta_A eta_B G (BA)^T G
```

Expanding the two factors proves the identity. The last term is second order in the step sizes and is retained. The native evaluator forms the expression using `G`, `BB^T`, `A^TA` and `(BA)^T`. The independent browser evaluator forms factor gradients and contracts `B grad_A`, `grad_B A` and `grad_B grad_A`. Both compare the expansion with direct multiplication of the updated factors. Returned maximum absolute residuals are measured binary64 diagnostics, not outward-rounded error bounds.

For a scalar gauge `C=cI`, the transformed first-order pieces scale as `c^2 BB^T G` and `c^-2 G A^T A`, while the merged model stays the same. Compensating with

```
eta_A' = eta / c^2
eta_B' = eta * c^2
```

makes each transformed next factor the corresponding transformed original next factor. This is a finite-step coordinate identity, not a new optimizer, a selected best learning rate, an implementation of LoRA+, or a reason for the original retention gap.

## Diagnostic loss and original training

The diagnostic uses mean softmax cross-entropy over the supplied examples, with shifted log-sum-exp and no probability logarithm floor. The original training computes the same softmax gradient but logs `-log(p_y+1e-12)`. That logged scalar differs slightly from the cross-entropy whose gradient is used. Original source and behavior remain unchanged.

A diagnostic batch can be the original first training example or all 120 original training examples. A simultaneous mean full-batch step is not the original sequential online epoch. Changing the diagnostic rate, scale, matrix gauge or epoch selection does not retrain the original model.

## Original archive

`coordinate_data/6-original-epochs.json.gz` includes:

- The entire trained base matrix and bias.
- All original 120-example A/B training and 45-example A/B test feature/label datasets.
- Four original factor widths 1, 2, 4 and 8, each with all 31 epoch boundaries, including initialization and epoch 30.
- Both full factor matrices, factor norms, completed online update counts, true mean training/test cross-entropies and test accuracies at every boundary.
- The unchanged original evaluation report, including the full fine-tuning baseline and merge residual.

There are 124 recorded states. This is not an archive of all 14,400 individual online updates. No best epoch replaces epoch 30. Regeneration uses the unchanged original fixture and online trainer. Tests independently replay the original trainer, compare every stored factor and accuracy, and recompute each loss. The archive includes adverse retention, not only shifted accuracy. Original reported numbers agree with the original README; no training is tuned to improve them.

`7-witnesses.json.gz` contains 48 complete native reports: four widths, epochs 0/1/10/30 and scalar gauges 0.25, 1 and 4. `8-controls.json.gz` adds identity, negative scale, shear, orthogonal, permutation and zero-rate cases. All matrices, gradients, logits, probabilities and update terms are retained, not only headline scores.

The browser's scale sweep shows all 31 positive gauges from 1/32 to 32, equally spaced in log2 scale, at the current fixed model, examples and diagnostic rate. A full matrix gauge in the selected report does not change this separate scalar sweep. No winning scale is selected or claimed.

## Input schema and operational bounds

`coordinates` accepts a strict JSON object with exactly these keys:

```
{
  "W": [[0,0],[0,0]],
  "bias": [0,0],
  "A": [[0.3,0.4]],
  "B": [[0.2],[-0.1]],
  "examples": [{"x":[1,0.5],"y":0}]
}
```

Bounds are 2 to 8 classes, 1 to 128 features, factor width 1 to 8, and 1 to 256 examples. Matrices must be rectangular and shape-compatible. All input numbers must be finite and bounded by absolute value 1,000,000; booleans are not numbers. Labels must be in-range integers. JSON input is at most 1 MB and 16 nested containers. Duplicate keys, nonfinite tokens, overflow such as `1e999`, trailing syntax and unknown schema keys are rejected.

The ordinary diagnostic rate is in `[0,2]`. Scalar gauges satisfy `1/32 <= abs(c) <= 32`, including negative sign changes. The browser and Python API additionally accept a full matrix gauge. Pivoted Gauss-Jordan rejects pivots below `1e-12` and computed infinity-condition estimates above `1e6`. These are operational rejection rules, not singular-value or numerical-rank certificates. All output numeric values are checked finite. The CLI currently exposes scalar gauges; use the Python API or browser for full matrices.

## Reproduction

```
python -m unittest discover -s tests -v
node --test tests/test_coordinates.cjs tests/test_coordinates_browser.cjs
python scripts/regenerate_coordinates.py
bash examples/verify_wheel.sh

python -m loralab evaluate --format json
python -m loralab gates --min-base-drop .3 --min-adapted-accuracy .95 \
  --max-rank2-param-share .7 --max-merge-diff 1e-12
python -m loralab trajectory --rank 2 --epoch 30 --gauge-scale 4 \
  --learning-rate .1 --batch first_train
python -m loralab coordinates --input problem.json --gauge-scale 4
```

31 Python tests include all 14 original tests unchanged. Four substantive Node/actual-Chrome tests compare 54 complete witnesses and all 124 retained states, with 591,615 numeric witness values at `1e-12`. Native controls include factor finite differences, random scalar gauges, an independent 80-digit Decimal finite-step expansion, and parity with an original online one-example update. Cross-runtime full retraining uses `1e-11` absolute/relative tolerance for libm and summation differences. Original test tolerances and gate thresholds are unchanged. Installed-wheel verification runs outside the checkout, imports all eight implementation modules, runs all four commands and reconstructs the archive.

## Limits and attribution

This is a synthetic three-class hashed bag-of-words experiment, not a transformer, LLM benchmark, general LoRA comparison, causal explanation, convergence result, robustness evaluation or deployment certificate. Token order is discarded, hashing can collide, and the original test examples are seeded synthetic examples. Model equality is separate from update equality; both are separate from held-out behavior. A small arithmetic residual proves none of those broader claims.

- [Hu et al., LoRA, section 4](https://arxiv.org/html/2106.09685v2): factorization, frozen base, initialization and merge context.
- [Mishra et al., fixed-rank matrix factorizations](https://arxiv.org/html/1209.0430v2): classical equivalent factor coordinates and quotient geometry. No manifold optimizer is implemented here.
- [Yen et al., LoRA-RITE](https://proceedings.iclr.cc/paper_files/paper/2025/file/bcbc0f660d2dde42f9d1d0ecb14a6f9a-Paper-Conference.pdf): transformation-invariant optimization context, not reproduced here.
- [Hayou et al., LoRA+](https://arxiv.org/html/2402.12354): unequal factor learning-rate context, not reproduced here.
- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/package_reference/lora): production implementation/configuration context, not a dependency or feature-parity claim.
- [Koroleva, the algebra of LoRA](https://xmarva.github.io/blog/2024/lora-algorithm-for-llms/): independent explanatory cross-check of shapes, rank bounds and counts.

These are standard methods and identities, not original algorithm claims.

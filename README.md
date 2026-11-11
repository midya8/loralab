# LoRALab

Low-rank adaptation from scratch: freeze a trained model, learn only a rank-`r` delta `W' = W + B @ A`, and measure what you actually bought. No ML frameworks - the forward pass, the gradients through the low-rank factors, and the evaluation harness are all hand-derived Python.

## What it does

- Trains a base linear classifier on distribution A (a synthetic NLP topic fixture).
- Shifts the distribution: B uses disjoint vocabulary, so the base model's accuracy on B collapses (1.00 → 0.44 on the checked-in fixture).
- Adapts to B with low-rank adapters at ranks 1-8 and with full fine-tuning, then measures, for each: accuracy on B, retained accuracy on A, and trainable-parameter share.
- Verifies that merging the adapter into the base weights (`W + B @ A`) reproduces adapter logits to within floating-point reassociation error, and enforces the whole story as CLI gates that fail in both directions.

## Measured results (checked-in fixture, pinned in tests)

| method | acc on B | retention on A | params | share |
|---|---|---|---|---|
| base (no adaptation) | 0.4444 | 1.0000 (on A) | 387 | - |
| lora r=1 | 0.9333 | 0.7333 | 131 | 33.9% |
| lora r=2 | 1.0000 | 0.7556 | 262 | 67.7% |
| lora r=4 | 1.0000 | 0.7778 | 524 | 135.4% |
| lora r=8 | 1.0000 | 0.7778 | 1048 | 270.8% |
| full fine-tune | 1.0000 | 0.9556 | 387 | 100.0% |

What the fixture shows, stated plainly:

- A rank-2 adapter matches full fine-tuning accuracy on the shifted distribution while training 67.7% of the parameters.
- On this fixture the adapters retain *less* accuracy on the original distribution than full fine-tuning does (0.76 vs 0.96). That is measured behavior of this fixture with this optimizer, not a general property of LoRA, and it is reported because hiding it would be cherry-picking.
- Merging `W + B @ A` reproduces adapter logits with max abs diff ~1e-14 on the fixture: the adapter computes `W x + B (A x)` and the merged model computes `(W + B @ A) x`, so the only difference is floating-point reassociation. The test pins the diff below 1e-12 and also pins that it is nonzero - the equivalence is exact algebra and approximate arithmetic.

## v0.2: same model, different step

The [factor-coordinate inspector](https://midya8.github.io/loralab/) keeps the original experiment and exposes a separate question: equivalent low-rank factors have the same merged model, but ordinary factor SGD can make different next updates. Independent Python and browser evaluators show full factor gradients, the finite-step first/second-order update identity, matrix gauges and scalar learning-rate compensation. Orthogonal basis changes preserve plain SGD in exact arithmetic; scaling and shear need not.

All 124 original epoch boundaries remain available, including the adverse retention results. The sweep shows every sampled scale, not a selected winner. Factor width is separate from matrix rank: the original three-output classifier has update rank at most three, and widths 4 and 8 use more parameters than full fine-tuning. The full baseline trains its bias; adapters freeze it. These facts are retained, not optimized away.

[Equations, schema, installed reproduction and limits](COORDINATES.md) explain the diagnostic, the unchanged online trainer, numerical bounds and classical attribution. This is not a new optimizer, a LoRA+/LoRA-RITE implementation, or a transformer benchmark.

## v0.3: how much rank changes a prediction?

The separate [exact predictive-rank inspector](https://midya8.github.io/loralab/rank.html) distinguishes supplied factor width, matrix rank, minimum all-input softmax-equivalent rank and finite-probe rank. It returns constructive rational factors plus a nonzero-minor lower witness, not a numerical-rank threshold. A separate native verifier checks products and determinants without running the rank producer. An independent browser BigInt engine decodes the archive's represented binary64 factors exactly.

All 124 archived states are retained. At every nonzero epoch, width-4/8 products have exact rank 3 but class-difference rank 2; width-1/2 ranks remain 1/2. The minimum all-input equivalent rank is the class-difference rank. Finite-probe rank can be smaller, with an explicit unseen-coordinate example and a nullspace construction showing why non-spanning probes do not prove global equality. These are classical linear-softmax identities, not a new algorithm or training result.

[Proof, exact schema, independent verifier, installed replay and limits](PREDICTIVE_RANK.md). Exported factors certify real probability functions with fixed base/bias, not rounded probabilities, training dynamics, accuracy, storage bytes or inference speed. Exact rank is not tolerance-based numerical rank, and fewer rational factor scalars need not use fewer bytes.

```bash
python -m loralab.predictive_rank --input input.json > report.json
python -m loralab.predictive_rank --verify --input report.json
bash examples/verify_rank_wheel.sh
```

## CLI

```bash
python -m loralab evaluate                 # table
python -m loralab evaluate --format json
python -m loralab gates --min-base-drop 0.3 \
    --min-adapted-accuracy 0.95 --max-rank2-param-share 0.7 \
    --max-merge-diff 1e-12
```

Gates exit 1 with named failures. Both directions are pinned by tests: the achievable set above passes, and `--max-rank2-param-share 0.5` fails (rank-2 costs 67.7%).

## Development

```bash
python -m unittest discover -s tests
./examples/demo.sh
```

39 Python tests, including all 31 previous tests unchanged. Seven substantive Node/actual-Chrome tests compare 54 complete witnesses (591,615 numeric values), all retained states, strict JSON and actual responsive UI behavior. The new exact-rank suite adds exhaustive small-matrix/minor checks, all124 full-report hash comparisons,11 exact controls and actual UI/export/native-replay checks. No runtime dependencies beyond Python 3.10+; web tests need Node 22+ and Chrome.

```bash
node --test tests/test_coordinates.cjs tests/test_coordinates_browser.cjs tests/test_rank.cjs tests/test_rank_browser.cjs
bash examples/verify_wheel.sh
python -m loralab trajectory --rank 2 --epoch 30 --gauge-scale 4
python -m loralab coordinates --input problem.json --gauge-scale 4
```

`trajectory` reads the packaged original archive and does not retrain. The scalar CLI and full-matrix Python/browser API are described in [COORDINATES.md](COORDINATES.md).

## Limitations

The base model is a linear classifier over hashed bag-of-words, not a transformer; the point is the adaptation mechanics (frozen base, low-rank factors, merge equivalence, cost/retention accounting), which apply to compatible linear layers, while parameter savings depend on shape. The fixture is synthetic and small so the suite runs deterministically in seconds; the retention behavior above is a property of this fixture, not a benchmark claim about LoRA in general.

## License

MIT

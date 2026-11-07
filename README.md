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

14 tests. No dependencies beyond Python 3.10+.

## Limitations

The base model is a linear classifier over hashed bag-of-words, not a transformer; the point is the adaptation mechanics (frozen base, low-rank factors, merge equivalence, cost/retention accounting), which are model-shape-independent. The fixture is synthetic and small so the suite runs deterministically in seconds; the retention behavior above is a property of this fixture, not a benchmark claim about LoRA in general.

## License

MIT

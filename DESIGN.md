# LoRALab - design (distinct from QuantLab/ScratchGPT/AttnLens)

QuantLab owns post-training quantization of a trained model. ScratchGPT/AttnLens
own full-model training from scratch. LoRALab owns PARAMETER-EFFICIENT
ADAPTATION: freeze a trained base model, adapt it to a shifted distribution
with a low-rank delta (B@A), and measure the three numbers that matter:
accuracy recovered, fraction of parameters trained, and how much of the
original distribution is retained (forgetting).

Artifacts:
1. data.py - deterministic synthetic text-ish classification: hashed bag-of-
   words features; distribution A (base) and shifted distribution B (new tokens and class-conditional vocabulary); one seed reproduces both bit for bit.
2. model.py - linear softmax classifier, trained with plain SGD on
   cross-entropy (gradients derived in code comments); forward through numpy-
   free pure-Python lists? NO - pure python matrices (lists), small dims.
3. lora.py - LoRA layer: frozen W, trainable A (r x in), B (out x r), B
   zero-init; forward W x + B (A x); hand-derived gradients dL/dA, dL/dB;
   merged_weights() returns W + B@A. Forward equality is exact algebra but
   approximate floating arithmetic, with nonzero reassociation error pinned
   below 1e-12.
4. eval.py contains the full fine-tune baseline, training both W and bias on B.
5. eval.py + cli.py - base acc on B, adapted acc vs rank (r in 1,2,4,8),
   full-FT acc, param ratios, retention on A after adaptation.
   Gates: --min-base-drop, --min-adapted-accuracy, --max-rank2-param-share,
   --max-merge-diff. Tests exercise passing and failing thresholds.
6. tests + demo.sh + CI + bounded README (toy linear model so measurements are
   finite seeded measurements, not exact arithmetic; the point is the adaptation machinery and the measured trade-offs).


## v0.2: additive coordinate diagnostics

The original data generator, classifier, factor trainer, evaluation harness, online update order, bias conventions and 14 tests stay unchanged. A separate evaluator diagnoses one fixed-state simultaneous gradient step, with its finite-step quadratic term. Its full-batch mode is not the original online epoch. See COORDINATES.md for exact identities, schema, sources and operational numerical bounds.

All four original widths retain all 31 epoch boundaries. This is 124 states, not all individual online updates. The browser uses independent sample-error/factor contractions while Python forms matrix-gradient identities; complete witnesses compare every returned numeric value. The archive is fully regenerated and independently replayed through the original trainer. Original retention and overcomplete widths are shown, not removed.

The original feature representation discards token order and can have hash collisions. Synthetic vocabulary remapping is an explicit finite fixture, not evidence of real deployment drift. Rank here is factor width; actual update rank is bounded by three classifier outputs. Scalar compensation is a coordinate identity, not a proposed optimizer or causal explanation of retention.

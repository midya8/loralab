# LoRALab - design (distinct from QuantLab/ScratchGPT/AttnLens)

QuantLab owns post-training quantization of a trained model. ScratchGPT/AttnLens
own full-model training from scratch. LoRALab owns PARAMETER-EFFICIENT
ADAPTATION: freeze a trained base model, adapt it to a shifted distribution
with a low-rank delta (B@A), and measure the three numbers that matter:
accuracy recovered, fraction of parameters trained, and how much of the
original distribution is retained (forgetting).

Artifacts:
1. data.py - deterministic synthetic text-ish classification: hashed bag-of-
   words features; distribution A (base) and shifted distribution B (feature
   mapping permuted + new tokens); one seed reproduces both bit for bit.
2. model.py - linear softmax classifier, trained with plain SGD on
   cross-entropy (gradients derived in code comments); forward through numpy-
   free pure-Python lists? NO - pure python matrices (lists), small dims.
3. lora.py - LoRA layer: frozen W, trainable A (r x in), B (out x r), B
   zero-init; forward W x + B (A x); hand-derived gradients dL/dA, dL/dB;
   merge() returns W + B@A and merged-forward == adapter-forward EXACTLY
   (same op order, pinned in tests).
4. finetune.py - full fine-tune baseline (train all of W on B).
5. eval.py + cli.py - base acc on B, adapted acc vs rank (r in 1,2,4,8),
   full-FT acc, param ratios, retention on A after adaptation.
   Gates: --min-adapted-accuracy --max-param-ratio --min-retention
   --min-full-ft-accuracy; negative control proves gates bite.
6. tests + demo.sh + CI + bounded README (toy linear model so measurements are
   exact; the point is the adaptation machinery and the measured trade-offs).

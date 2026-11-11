# Exact predictive rank, not numerical rank

LoRALab v0.3 adds a separate inspector. The original online trainer, synthetic data, four widths, all 124 archived boundaries, evaluation results, coordinate-SGD inspector and old tests remain intact. No model is retrained or replaced.

## Four different quantities

For a fixed linear classifier `softmax((W + M)x + b)`, with `M = B A`:

1. Supplied width is the inner dimension of B and A. It need not equal rank(M).
2. Matrix rank is the exact rank of M.
3. All-input predictive rank is the minimum rank of an update N giving the same real softmax probabilities for every real feature vector x, with W and b fixed.
4. Probe-only rank is the minimum rank matching the class-difference responses on the supplied finite probes. These are transparent algebraic probes, not an evaluation dataset.

These are classical linear-algebra consequences of the softmax equivalence class, not a proposed LoRA algorithm or a new compression result.

## All-input proof

Fix a reference class t. Let D have rows `D_j = M_j - M_t`, including a zero reference row. For finite real logits z and z', equality of softmax probabilities is equivalent to equality of every log probability ratio, hence `z_j-z_t = z'_j-z'_t`. Equivalently, their difference is a shared scalar times the all-ones class vector.

Thus N is equivalent on every x iff `N_j-N_t = D_j` as linear forms. Applying the row-difference linear map cannot raise rank, so `rank(N) >= rank(D)`. The update N=D itself satisfies all class differences and therefore attains this lower bound. Every equivalent update has form `D + 1 v^T` for some row vector v. The base and bias cancel because both are fixed.

The rank factorization of D supplies a concrete attaining pair. The reference choice changes that pair but not its rank. This is equality of real probability functions, not equality of logits, rounded probabilities, predicted labels after floating reassociation, losses after rounding, gradients through the new factors or future training.

## Finite probes and a constructive extension

Store probes as rows of X. The class-difference response is `Q = D X^T`. Every matching update N must have `rank(N) >= rank(Q)`: apply the row-difference map to N, then multiply by X^T. A minimum-rank update can be constructed as follows.

Take s independent rows I and s independent columns J of Q so that `K = Q[I,J]` is nonsingular. Set `B_p = Q[:,J]` and `A_p = K^-1 D[I,:]`. Because `Q = Q[:,J] K^-1 Q[I,:]`, the product `N_p = B_p A_p` satisfies `N_p X^T = Q`. Its rank is at most s and the lower bound is s. The zero-rank case uses N_p=0. Its reference row is zero.

If this particular extension differs from D, the report returns a class i, coordinate j and exact nonzero difference `(N_p-D)[i,j]`. The unseen vector e_j exhibits the changed class contrast. It is not a claim about accuracy on held-out natural inputs.

Even when N_p=D, non-spanning probes do not prove all-input equivalence for an arbitrary candidate. We construct a nonzero u in null(X), and add u^T to one non-reference class row of N_p. Every probe response remains unchanged. At x=u, that class contrast changes by `u^T u > 0`. Conversely, full feature-spanning probes imply `(N_p-D)X^T=0` only when N_p=D. The report certifies the probe-span rank too.

## Exact arithmetic semantics

Custom input is JSON with exactly these keys:

```json
{"matrix":[[1,0],[0,1],[0,0]],"reference":0,"probes":[[1,0]]}
```

Or replace `matrix` with `A` and `B`:

```json
{"A":[[1,2]],"B":[[1],[2],[0]],"reference":0,"probes":[[1,0]]}
```

All scalars must be safe JSON integers (absolute value at most 2^53-1), or strings matching `-?(0|[1-9][0-9]*)(/[1-9][0-9]*)?`. Integers of arbitrary supported size must be strings. Booleans, decimal JSON numbers, decimal strings, exponent strings, zero/negative denominators, duplicate keys, ragged arrays, unknown fields and mismatched dimensions are refused. `-0` is normalized to zero. JSON byte/text bounds are operational, not mathematical guarantees.

Bounds: 2-8 classes; 1-128 features; factor width 1-8; 1-16 probe rows; input scalar string length at most 256 characters; input/report text at most 4,000,000 characters. A generated rational longer than 8,192 characters is refused rather than reported as a certificate. Browser input nesting is capped at 128. Exact rational elimination can be expensive on adversarial inputs; these are small educational matrices, not an untrusted computational service or a scalable sparse solver.

The archive begins with binary64 factors. Python Fraction converts their exact represented values; the browser independently decodes sign/exponent/mantissa with DataView and BigInt. Both multiply the exact ratios. They do not read rank from a rounded merged matrix, use a tolerance, infer ideal optimizer rank, or repair small nonzero directions. For example, an arbitrarily tiny nonzero rational can raise the exact rank. Numerical rank is a different, tolerance-dependent question.

## Report and independent verifier

Schema: `loralab.predictive-rank.v1`. `problem` carries the complete input. Dimensions, reference and supplied width are explicit. `raw`, `global_certificate`, `probe_certificate`, and `probe_span` each contain:

- `rank`, exact rational factors `B`, `A`, and selected `rows`, `columns`;
- `minor`, the nonzero determinant of that square submatrix;
- at rank zero: B has m empty rows, A is empty, indices are empty and determinant is the conventional value 1.

The product is the upper witness. The nonzero minor is the lower witness. No assertion of vanishing larger minors is needed: the factorization already proves rank at most r. Factorization uses deterministic independent row/column skeletons and Gaussian elimination. `rank_verify.py` never calls the rank producer or its rref; it multiplies the supplied factors and uses a separate fraction-free Bareiss determinant for the selected minor. It checks every reported relation, rank count, shape, probe extension, unseen coordinate witness and nullspace counterexample. A different valid witness is allowed; certificates need not be identical to the producer's deterministic choice.

`probe_extension` contains the minimum-rank attaining factors on the finite probe set. `extension_unseen_coordinate` is null exactly when this extension equals D globally, or contains a concrete changed coordinate. `blind_direction` is null iff the probes span all features; otherwise it supplies the nonzero vector and changed-class construction above. `parameter_counts` counts algebraic scalar slots `r(m+d)`, not rational bit storage, trainable parameters after retraining, FLOPs or inference performance.

Native replay, outside the repository after installation:

```bash
python -m loralab.predictive_rank --input input.json > report.json
python -m loralab.predictive_rank --verify --input report.json
```

Both refuse malformed input with exit 2 and no partial JSON on stdout. The report verifier is a local scientific audit tool, not a security boundary against hostile resource exhaustion. Browser exports are the complete exact input, certificate and verification corpus. Editing any field clears the old result and disables exports until recertification; archive selection remains separate from unapplied custom edits. Exact matrix cells, including every last feature column, are selectable for their full rational values.

## Original archive findings

The unchanged archive contains 31 epoch boundaries at widths 1,2,4,8. Each original epoch has 120 sequential SGD updates. It does not contain every individual update. At epoch 0 all products are zero. At all 30 nonzero boundaries per width:

| Supplied width | Exact product rank | All-input predictive rank | Original trained scalar slots | Quotient factor scalar slots |
| --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 131 | 131 |
| 2 | 2 | 2 | 262 | 262 |
| 4 | 3 | 2 | 524 | 262 |
| 8 | 3 | 2 | 1048 | 262 |

These are exact results over the represented factors, not a claim that the mathematical idealization of SGD gained rank. Rounding can create tiny exact directions. The original trainer has three output classes, so the raw product rank is at most 3, and class-difference rank at most 2. Counts do not change the historical training cost. The full fine-tuning baseline trains its bias; adapters freeze it. Original retained accuracy remains worse than full fine-tuning on this synthetic fixture. No retention improvement is claimed.

## Reproduction and checks

```bash
python -m unittest discover -s tests -v
node --test tests/test_coordinates.cjs tests/test_coordinates_browser.cjs tests/test_rank.cjs tests/test_rank_browser.cjs
bash examples/verify_wheel.sh
bash examples/verify_rank_wheel.sh
python -m loralab.rank_fixtures  # regenerates only the new exact corpus and browser corpus
```

The new native suite compares all 729 ternary 3x2 matrices at each of three reference classes with independent Leibniz-minor rank enumeration, checks reference/common-shift/factor-gauge identities, rank-zero cases, tiny rational directions, duplicate/zero and full-span probes, all packaged corpus bytes, altered factor/minor leaves, strict refusals and CLI replay. The browser's BigInt engine matches every exact control leaf and SHA-256 of every complete native report at all 124 archived states. Hashes cover the full report, not just rank integers.

Actual Chrome checks every archived selection through the real UI, all 11 controls, responsive layout, every available exact factor cell including rightmost columns, trusted input invalidation, strict refusals, all three exports and independent native certificate replay. Existing Python/Chrome tests remain unchanged and run alongside the new suite. The new isolated installed-wheel gate regenerates the corpus, verifies every packaged control, audits/verifies all 124 archived states, and replays native JSON commands without the checkout on the import path.

## Limits and attribution

This inspector changes the representation of a fixed linear output probability function. It does not certify binary64 probabilities, CE, argmax stability, saved bytes, throughput, conditioning, optimizer-state equivalence, future factor training, or hidden-layer interchangeability. Casting exact rational factors to floats discards the certificate's arithmetic semantics. The fixture is synthetic hashed bag-of-words, not a transformer. Fewer factor slots can require more bytes when entries are long exact ratios. No data-driven truncation threshold, SVD approximation, optimizer, new benchmark or commercial API is introduced.

- Yang et al., [Breaking the Softmax Bottleneck](https://arxiv.org/abs/1711.03953), properties 1-2: equality modulo common shifts and rank-family context. The output-expressivity paper is background, not a claim that it implements this LoRALab inspector.
- Hu et al., [LoRA](https://arxiv.org/abs/2106.09685), section 4.1: frozen base plus BA factors. The original LoRALab trainer has no alpha/r scale; no external benchmark results are transferred.
- Nick Higham, [Matrix Rank Relations](https://nhigham.com/2021/02/23/matrix-rank-relations/): rank factorization, products, minors and the numerical-rank distinction.
- [Rank factorization](https://en.wikipedia.org/wiki/Rank_factorization): independent row/column skeleton background.
- [SciPy softmax documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html): definition and numerical shift stabilization; numerical outputs are not used as exact witnesses.

The quotient and finite-probe proofs are stated explicitly above rather than attributed as a new discovery.

# LightGBM Hyperparameter Tuning Report — DelayCast

**Date:** 2026-06-22 · **Experiment:** `delaycast-tune` · **Model family:** LightGBM
**Split:** time-based — train (2022–2023, 13.28M) / validation (2024 H1, 3.40M) / test (2024 H2, 3.56M)
**Selection metric:** validation PR-AUC · **Imbalance handling:** `class_weight="balanced"` (~20% positives)

---

## TL;DR (the one-paragraph verdict)

Tuning **did not improve the model** in any meaningful way. All four configurations landed
within **0.001 of each other on validation PR-AUC**, none beat the simple baseline, and every
one **early-stopped within 12–83 trees** — the signature of a problem that is **feature-limited,
not hyperparameter-limited**. The Stage-2 champion (registered `delaycast-champion`) actually
edges out every tuned config on the test set, because it trained on **~25% more data**. The clear
lesson: **marginal data beat marginal tuning**, and the road to a better model runs through
**new features (weather), not hyperparameters.** The most actionable output is the **threshold
sweep**, which shows recall is a dial you set, not a number you're stuck with.

---

## 1. What each configuration changed, and why

Every config is LightGBM with `n_estimators=3000` (an upper bound — validation **early stopping**
picks the real count) and `class_weight="balanced"`. They differ on the levers that matter:

| Config | learning_rate | num_leaves | min_child_samples | reg_lambda | The hypothesis being tested |
|---|---|---|---|---|---|
| **baseline** | 0.05 | 64 | 20 | 0.0 | Reproduce the Stage-2 LightGBM — the bar to beat. |
| **slow-deep** | 0.03 | 128 | 50 | 1.0 | *Slower learning + wider trees + light L2.* Smaller steps often generalize better; more leaves add capacity; bigger leaves + L2 keep it honest. |
| **reg-heavy** | 0.05 | 64 | 200 | 5.0 | *Aggressive regularization.* Force big leaves (≥200 samples) and strong L2 — does constraining the model help on noisy data? |
| **wide** | 0.05 | 256 | 100 | 2.0 | *High capacity.* Very wide trees (256 leaves) to see if extra expressiveness captures more signal before overfitting. |

**The levers, in plain terms:**
- **`learning_rate`** — how big a step each tree takes. Lower = more careful, usually needs more trees.
- **`num_leaves`** — how complex each tree can get (more leaves = more interactions captured, more overfit risk).
- **`min_child_samples`** — minimum rows in a leaf; higher = smoother, more conservative trees.
- **`reg_lambda`** — L2 penalty on leaf weights; higher = stronger pull toward simplicity.

---

## 2. Results — every config, validation and test

| Config | best_iter | valid PR-AUC | valid recall | **test PR-AUC** | **test recall** | test precision |
|---|---|---|---|---|---|---|
| **baseline** | 71 | **0.3348** | 0.5964 | **0.3467** | 0.5680 | 0.3090 |
| slow-deep | 83 | 0.3348 | 0.5941 | 0.3447 | 0.5675 | 0.3077 |
| reg-heavy | 67 | 0.3339 | 0.5949 | 0.3458 | 0.5684 | 0.3081 |
| wide | 12 | 0.3342 | 0.5976 | 0.3395 | 0.5706 | 0.3032 |

**Winner on the selection metric (validation PR-AUC): `baseline`** (0.3348) — by a margin so small
(0.0009 over `reg-heavy`) it's **statistical noise**, not a real difference.

---

## 3. Config-vs-config analysis

**Everything is tied.** The spread across all four is **0.0009 in validation PR-AUC** and **0.0072
in test PR-AUC**. When four genuinely different hyperparameter regimes — from heavily regularized
to high-capacity — all produce the same score, the model has **saturated what these features can
explain.** Knobs only matter when the model is capacity-constrained; here it isn't.

**`best_iteration` is the tell.** Early stopping halted every config in **12–83 trees** (the
defaults assume hundreds-to-thousands). The model extracts essentially all available signal in a
few dozen trees, then validation stops improving. That's a **low-signal regime**:
- **`wide` stopped at just 12 trees** — 256 leaves is so expressive it fits the learnable signal
  almost instantly, then would only start memorizing noise, so early stopping cuts it off. Its
  slightly *worse* test PR-AUC (0.3395) is the faint smell of that over-capacity.
- **`reg-heavy`** (the opposite extreme) lands in the same place — proof that regularization wasn't
  the bottleneck either. You can't regularize your way to signal that isn't in the data.

**Net:** none of the deliberate changes moved the needle. This is a **null result, and a useful
one** — it rules out "we just didn't tune enough" as the explanation for the modest performance.

---

## 4. Tuned vs. the Stage-2 champion (the important comparison)

| Model | train rows | test PR-AUC | test recall | test precision | test ROC-AUC |
|---|---|---|---|---|---|
| **Stage-2 champion** (registered `delaycast-champion`) | **16.68M** | **0.3499** | **0.5726** | 0.3111 | 0.6907 |
| Best tuned (`baseline`) | 13.28M | 0.3467 | 0.5680 | 0.3090 | — |

**The Stage-2 champion is still the best model** — it beats every tuned config on test PR-AUC and
recall. Why would the *untuned* model win? **Data.** The Stage-2 champion trained on **everything
before 2024-07 (16.68M rows)**; the tuning run had to **carve out 2024 H1 as a validation set**,
leaving only **13.28M** rows to train on. Those ~3.4M extra rows (≈25% more data) bought more than
any hyperparameter change did.

> **Headline lesson for the README/interview:** *I tuned LightGBM across capacity and regularization
> regimes and measured a null result — hyperparameters were not the bottleneck. The untuned champion,
> trained on 25% more data, remained best. The lever that matters here is **data and features, not
> tuning.*** This is a more sophisticated finding than "tuning improved my model by 0.3%."

⚠️ **Caveat (stated honestly):** this isn't a perfectly controlled comparison — the champion trained
on more rows *and* used a fixed 600 trees rather than early stopping. But the direction is
unambiguous: tuning on less data did not catch up to the baseline on more data.

---

## 5. Are we overfitting? (loss-curve read)

From the per-iteration train-vs-validation curves at `best_iteration`:

| Config | train logloss | valid logloss | gap | train AUC | valid AUC | gap |
|---|---|---|---|---|---|---|
| baseline | 0.6320 | 0.6552 | +0.023 | 0.6964 | 0.6567 | +0.040 |
| slow-deep | 0.6324 | 0.6545 | +0.022 | 0.6965 | 0.6571 | +0.039 |
| reg-heavy | 0.6325 | 0.6552 | +0.023 | 0.6956 | 0.6564 | +0.039 |
| wide | 0.6319 | 0.6544 | +0.023 | 0.6982 | 0.6582 | +0.040 |

**We are NOT overfitting.** The train→validation gaps are small and nearly identical across
configs (~0.04 AUC). An overfit model shows a *large* gap (train AUC ≫ valid AUC); here they're
close. Combined with the tiny `best_iteration`, the diagnosis is the opposite of overfitting:

> The model is **underfitting the *problem*** (the features simply don't contain enough signal to
> push validation AUC past ~0.657), **not overfitting the *data*.** More trees, more leaves, or
> less regularization can't fix that — only better features can.

---

## 6. The threshold sweep — recall is a dial, not a verdict

Validation recall/precision for `baseline` as the decision threshold moves (the model is identical;
only the cut point changes):

| Threshold | Recall (delays caught) | Precision (flags that are real) | Read |
|---|---|---|---|
| 0.20 | **0.999** | 0.214 | Flags almost everything — useless (precision ≈ base rate) |
| 0.30 | 0.966 | 0.227 | Very wide net |
| 0.40 | **0.836** | 0.257 | **Catch 84% of delays; ~1 in 4 flags real** |
| **0.50** (default) | 0.596 | 0.305 | The "recall 0.57" you saw — a conservative cut |
| 0.60 | 0.340 | 0.361 | Precision-leaning |
| 0.70 | 0.138 | 0.429 | Only the surest calls |
| 0.80 | 0.012 | 0.505 | Almost nothing flagged |

**This is the answer to "why is recall only 0.57?"** — because the default threshold is 0.5. For an
IOC that would rather investigate a fine flight than miss a cascading delay, an operating point
around **threshold ≈ 0.35–0.40** is far more appropriate: **~84% of real delays caught**, at the
cost of precision dropping to ~0.26 (about 1 in 4 alerts is a true delay). That's a **business
decision about the cost of a miss vs. a false alarm**, and the model supports whichever point you
choose. The Streamlit app should expose this (or hard-code a recall-leaning threshold), not blindly
use 0.5.

---

## 7. Recommendations

1. **Keep the Stage-2 champion** (`delaycast-champion` v1, LightGBM) as the registered model. Tuning
   confirmed it is at/near the ceiling for these features; no tuned config justifies replacing it.
2. **Do not promote any tuned config** — they're statistically tied with, and on test slightly
   behind, the champion. Promoting one would be noise-chasing.
3. **Set the serving threshold below 0.5** (≈0.35–0.40 for a recall-first IOC tool), and surface the
   recall/precision trade in the README and app. This is the single biggest "improvement" available
   right now and costs nothing.
4. **The real next lever is features, not hyperparameters** — `weather` (origin & dest forecast at
   scheduled departure/arrival) is the #1 candidate, since same-day weather is the causal driver the
   current schedule-only features can't see. Expect a real PR-AUC lift there, unlike from tuning.
5. **(Optional) Cross-validation:** a sampled time-series CV would put error bars on these numbers
   and confirm the ties are within noise — but given the unanimous null result, it would almost
   certainly just confirm "no meaningful difference." Low priority.

---

## Appendix — reproducing this

- Sweep submitted via `azureml/submit_tune.py` (configs in the `CONFIGS` dict), training in
  `azureml/train_tune.py` (3-way time split, early stopping, per-iteration loss logging, threshold sweep).
- All runs + curves live in AzureML Studio → experiment **`delaycast-tune`** (run names
  `lgbm-{config}-06220356`). The loss curves and threshold sweep are under each run's **Metrics** tab.
- Stage-2 champion metrics: experiment `delaycast`, run `train-lgbm` (see also CLAUDE.md Stage 2 table).

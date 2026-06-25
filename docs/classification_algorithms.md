# Classification Algorithms — A Deep Reference (with the math intuition)

A working data scientist's guide to the classification algorithms that matter, the math that
makes each one tick, **when/why/how/where** to use it, and the hyperparameters that actually
move the needle. Written against the DelayCast problem (predict `delayed = 1 if ArrDelay > 15`),
but the content is general.

> **How to read the math:** every model is just two choices — (1) a **hypothesis** (what shape
> of decision boundary it can draw) and (2) a **loss** (how it's penalized for being wrong) —
> plus an **optimizer** that searches for the hypothesis minimizing the loss. Keep that frame and
> every algorithm below is a variation on it.

---

## Part 0 — Foundations you need first

### 0.1 What classification actually computes

Given features $\mathbf{x} \in \mathbb{R}^d$, we want $P(y=1 \mid \mathbf{x})$ — a probability in
$[0,1]$. We then threshold it (default 0.5) into a class. Two philosophies:

- **Discriminative** models learn $P(y \mid \mathbf{x})$ directly (Logistic Regression, SVM, trees, NNs).
- **Generative** models learn $P(\mathbf{x} \mid y)$ and $P(y)$, then flip with Bayes' rule
  $P(y\mid\mathbf{x}) \propto P(\mathbf{x}\mid y)P(y)$ (Naive Bayes, LDA/QDA).

### 0.2 The decision boundary

The boundary is the surface where $P(y=1\mid\mathbf{x}) = 0.5$. Its **shape** is what
distinguishes models:
- **Linear** boundary (a hyperplane): Logistic Regression, linear SVM, LDA.
- **Piecewise-axis-aligned** (staircase): decision trees and their ensembles.
- **Smooth non-linear**: kernel SVM, neural networks, QDA.

Picking a model = picking the boundary family that matches your data's true structure. Too simple
→ **underfit (high bias)**; too flexible → **overfit (high variance)**.

### 0.3 Loss functions (how "wrong" is measured)

Most classifiers minimize one of these over the training set:

- **Log loss / cross-entropy** (LR, NNs, boosting): for true label $y\in\{0,1\}$ and predicted
  $\hat p = P(y=1\mid\mathbf{x})$,
$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\Big[\, y_i \log \hat p_i + (1-y_i)\log(1-\hat p_i)\,\Big].
$$
  It explodes ($\to\infty$) when you're confidently wrong, so it pushes for calibrated probabilities.
- **Hinge loss** (SVM): $\max(0,\,1 - y_i f(\mathbf{x}_i))$ with $y\in\{-1,+1\}$ — zero penalty
  once a point is correctly classified *with margin*, linear penalty inside the margin.
- **Gini / entropy** (trees): impurity of a node, see §5.

### 0.4 Optimization: gradient descent in one breath

For differentiable losses we move parameters $\boldsymbol\theta$ downhill:
$$
\boldsymbol\theta \leftarrow \boldsymbol\theta - \eta \,\nabla_{\boldsymbol\theta}\mathcal{L},
$$
where $\eta$ is the **learning rate**. Trees don't use this (they search splits greedily); boosting
uses a functional version of it (each tree fits the negative gradient of the loss).

### 0.5 Regularization (the bias–variance dial)

Add a penalty on parameter size to fight overfitting:
- **L2 (ridge):** $\lambda\sum_j \theta_j^2$ — shrinks weights smoothly.
- **L1 (lasso):** $\lambda\sum_j |\theta_j|$ — drives weights to exactly 0 → feature selection.

$\lambda$ (often exposed as `C = 1/λ`) is the master knob: high regularization = simpler, higher
bias; low = more flexible, higher variance.

---

## Part 1 — The algorithms

For each: **What · Why · How (math) · When/Where · Hyperparameters · Pros/Cons.**

---

## 1. Logistic Regression (LR)

**What.** The linear, probabilistic baseline. Despite "regression" in the name, it's a classifier.

**Why.** Fast, interpretable (coefficients = log-odds effects), well-calibrated probabilities, hard
to overfit with regularization. It's the floor every other model must beat.

**How (math).** Take a linear score $z = \mathbf{w}^\top\mathbf{x} + b$ and squash it to a
probability with the **sigmoid**:
$$
\hat p = \sigma(z) = \frac{1}{1 + e^{-z}}.
$$
The model is **linear in the log-odds** (logit):
$$
\log\frac{\hat p}{1-\hat p} = \mathbf{w}^\top\mathbf{x} + b.
$$
So each weight $w_j$ is the change in log-odds per unit of $x_j$ — directly interpretable
("$e^{w_j}$ = odds multiplier"). Fit by minimizing log loss (§0.3) via gradient descent. The loss
is **convex**, so there's a single global optimum — no local-minimum worries.

**When/Where.** Linearly (in log-odds) separable problems; when you need interpretability or
calibrated probabilities; as a baseline always; high-dimensional sparse data (text, one-hot) with
L1/L2. In DelayCast it's model #1 and sets the recall/PR-AUC floor.

**Key hyperparameters.**
| Param (sklearn) | Meaning | Guidance |
|---|---|---|
| `penalty` | L1 / L2 / elasticnet | L2 default; L1 for feature selection on sparse one-hot |
| `C` | inverse reg. strength ($1/\lambda$) | small C = more reg.; tune on log scale `[0.01 … 100]` |
| `class_weight` | reweight classes | `'balanced'` for our ~20% positives |
| `solver` | optimizer | `lbfgs` (L2), `liblinear`/`saga` (L1, large/sparse) |
| `max_iter` | optimization steps | bump if it warns "did not converge" |

**Pros:** fast, interpretable, calibrated, convex. **Cons:** only linear boundaries (needs manual
interaction/polynomial features for non-linearity).

---

## 2. k-Nearest Neighbors (k-NN)

**What.** "You are like your neighbors." Classify a point by majority vote of its $k$ closest
training points. No training — it memorizes the data (lazy learner).

**Why.** Zero modeling assumptions; naturally non-linear, even multi-modal boundaries.

**How (math).** Define a distance, usually Euclidean
$d(\mathbf{x},\mathbf{x}') = \sqrt{\sum_j (x_j - x'_j)^2}$. For a query point, find the $k$ nearest
training points and predict
$$
\hat p(y=1) = \frac{1}{k}\sum_{i \in N_k(\mathbf{x})} \mathbb{1}[y_i = 1].
$$
Because distance dominates, you **must scale features** (else large-range columns swamp the metric)
and it suffers the **curse of dimensionality** — in high $d$ everything is roughly equidistant.

**When/Where.** Small/medium, low-dimensional datasets with a meaningful distance; quick non-linear
baseline. **Not** DelayCast (20M rows × 700 one-hot dims — prediction is $O(N)$ per query and
distance is meaningless in that sparse high-dim space).

**Key hyperparameters.** `n_neighbors` (k — small = low bias/high variance, large = smoother);
`weights` (`uniform` vs `distance`); `metric` (`minkowski`/`manhattan`); `p` (1=Manhattan, 2=Euclid).

**Pros:** dead simple, non-linear, no training. **Cons:** slow at predict time, memory-hungry,
needs scaling, dies in high dimensions.

---

## 3. Naive Bayes (NB)

**What.** A generative classifier using Bayes' rule with a "naive" independence assumption.

**Why.** Extremely fast, great with tiny data and very high dimensions (classic for text/spam).

**How (math).** Bayes' rule: $P(y\mid\mathbf{x}) \propto P(y)\prod_j P(x_j \mid y)$, where the
**naive assumption** is that features are conditionally independent given the class (the product).
That assumption is usually false, yet the *argmax* often lands on the right class anyway. Variants
differ in $P(x_j\mid y)$: **Gaussian** (continuous), **Multinomial** (counts), **Bernoulli** (binary).

**When/Where.** Text classification, spam, high-dim sparse counts, very small training sets, a
lightning-fast baseline. Rarely the best on rich tabular data like DelayCast (the independence
assumption ignores feature interactions that matter — e.g. carrier×origin).

**Key hyperparameters.** `alpha` (Laplace smoothing — prevents zero probabilities for unseen
feature/class combos); `var_smoothing` (GaussianNB stability).

**Pros:** blazing fast, tiny-data friendly, no tuning. **Cons:** independence assumption,
poorly-calibrated probabilities, can't model interactions.

---

## 4. Decision Tree

**What.** A flowchart of yes/no splits that partitions feature space into axis-aligned boxes,
predicting a constant in each box.

**Why.** Interpretable, non-linear, handles mixed feature types, no scaling needed. The building
block of Random Forests and boosting.

**How (math).** Greedily choose the split (feature + threshold) that most reduces **impurity**.
For a node with class-1 fraction $p$:
- **Gini impurity:** $G = 1 - \sum_c p_c^2 = 2p(1-p)$ for binary.
- **Entropy:** $H = -\sum_c p_c \log_2 p_c$.

A split into children $L,R$ is scored by the impurity *drop* (information gain):
$$
\Delta = I(\text{parent}) - \frac{N_L}{N}I(L) - \frac{N_R}{N}I(R).
$$
Recurse until a stopping rule (depth, min samples, no gain). A lone deep tree **overfits** — it can
carve a box around every noisy point — which is why we regularize it or ensemble it.

**When/Where.** When you need a human-readable rule set; as a base learner; quick non-linear model.
Single trees rarely win on accuracy — ensembles (§5, §6) do.

**Key hyperparameters.** `max_depth`, `min_samples_split`, `min_samples_leaf`,
`max_features`, `criterion` (`gini`/`entropy`), `ccp_alpha` (cost-complexity pruning). All are
ways to **limit flexibility** and curb overfitting.

**Pros:** interpretable, non-linear, no scaling, handles interactions. **Cons:** high variance
(unstable to small data changes), overfits alone, can't extrapolate.

---

## 5. Random Forest (RF) — bagging of trees

**What.** Many decorrelated decision trees; average their votes.

**Why.** Big variance reduction over a single tree, strong off-the-shelf accuracy, little tuning.

**How (math).** Averaging $T$ identically-distributed estimators each with variance $\sigma^2$ and
pairwise correlation $\rho$ gives ensemble variance
$$
\rho\sigma^2 + \frac{1-\rho}{T}\sigma^2.
$$
So you win by **(a) more trees** (shrinks the second term) and **(b) lower correlation $\rho$**.
RF lowers $\rho$ two ways: **bagging** (each tree sees a bootstrap sample of rows) and **feature
subsampling** (each split considers a random subset of columns). Trees are grown deep (low bias);
averaging kills the variance.

**When/Where.** Strong default for tabular data; robust when you can't tune much; good feature
importances. In DelayCast it's the non-linear baseline *between* LR and boosting — but it's slow
on 20M × ~700 sparse one-hot columns (see CLAUDE.md / the modeling notes).

**Key hyperparameters.**
| Param | Effect |
|---|---|
| `n_estimators` | more trees = lower variance, diminishing returns, linear cost |
| `max_depth` / `min_samples_leaf` | regularize each tree (we cap these for speed at 20M rows) |
| `max_features` | columns considered per split — *the* knob for decorrelation (`sqrt` is classic) |
| `class_weight` | `'balanced'` for imbalance |
| `n_jobs` | parallelism (trees are embarrassingly parallel) |

**Pros:** accurate, low-tuning, parallel, robust, importances. **Cons:** memory/compute heavy,
large model, slower inference, weaker than boosting on many problems, weak on high-cardinality
one-hot.

---

## 6. Gradient Boosting (GBM → XGBoost / LightGBM / CatBoost)

**What.** Build trees **sequentially**, each new tree correcting the *errors* of the running
ensemble. Usually the top performer on tabular data.

**Why.** State-of-the-art accuracy on structured/tabular problems; efficient modern implementations
scale to tens of millions of rows.

**How (math).** Model the prediction as an additive sum of trees
$F_M(\mathbf{x}) = \sum_{m=1}^{M} \nu\, h_m(\mathbf{x})$. At each step, fit the new tree $h_m$ to the
**negative gradient** of the loss w.r.t. the current prediction (gradient descent *in function
space*):
$$
r_{im} = -\left[\frac{\partial \mathcal{L}(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F=F_{m-1}}.
$$
For log loss these "pseudo-residuals" are simply $r_i = y_i - \hat p_i$ — each tree learns where
the model is currently wrong. The **learning rate** $\nu$ (shrinkage) scales each tree's
contribution so no single tree dominates. XGBoost adds a second-order (Newton) term and an explicit
regularization penalty on tree complexity:
$$
\Omega(h) = \gamma T + \tfrac{1}{2}\lambda\sum_j w_j^2,
$$
($T$ = #leaves, $w_j$ = leaf weights) — this is why it generalizes well.

**The flavors:**
- **XGBoost** — level-wise tree growth, second-order optimization, heavy regularization. Robust default.
- **LightGBM** — **histogram** binning (bucket each feature into ~255 bins → split on bins, not
  raw sorts) + **leaf-wise** growth (split the highest-loss leaf). Much faster on large data; the
  workhorse for DelayCast's 20M rows.
- **CatBoost** — native categorical handling (ordered target encoding) + symmetric trees; least
  tuning, great when you have many high-cardinality categoricals (carrier/origin/dest!).

**When/Where.** The go-to when accuracy matters on tabular data and you can afford tuning. DelayCast's
likely champion.

**Key hyperparameters (the ones that matter, with intuition).**
| Param | What it controls | Direction |
|---|---|---|
| `n_estimators` | number of trees | more = stronger, but overfits late; pair with early stopping |
| `learning_rate` (`eta`/`nu`) | shrinkage per tree | **lower = better generalization but needs more trees** (classic trade) |
| `max_depth` / `num_leaves` | tree complexity | the main overfitting knob (LightGBM: `num_leaves < 2^depth`) |
| `subsample` | row sampling per tree | <1 adds stochasticity → regularizes |
| `colsample_bytree` | column sampling | decorrelates trees, regularizes |
| `min_child_weight` / `min_data_in_leaf` | min evidence per leaf | higher = smoother, less overfit |
| `reg_lambda` / `reg_alpha` | L2 / L1 on leaf weights | crank up if overfitting |
| `scale_pos_weight` / `class_weight` | imbalance | set ≈ neg/pos (~4 for us) |
| `early_stopping_rounds` | stop when val metric stalls | the cleanest way to set `n_estimators` |

**The cardinal rule:** **low `learning_rate` + many trees + early stopping** beats high learning
rate almost always. Tune depth/leaves and regularization next; sampling params last.

**Pros:** best-in-class tabular accuracy, efficient at scale, flexible losses. **Cons:** more
hyperparameters, sequential (less trivially parallel than RF), can overfit if untuned, less
interpretable (mitigate with SHAP).

---

## 7. Support Vector Machine (SVM)

**What.** Find the hyperplane that separates classes with the **maximum margin** (widest buffer).

**Why.** Strong in high-dimensional spaces; the **kernel trick** gives non-linear boundaries
elegantly; effective when $d > N$.

**How (math).** Maximize the margin = minimize $\tfrac{1}{2}\|\mathbf{w}\|^2$ subject to every point
being on the correct side with margin: $y_i(\mathbf{w}^\top\mathbf{x}_i + b)\ge 1$. Real data isn't
clean, so **soft-margin** allows violations via slack, controlled by `C`:
$$
\min_{\mathbf{w},b}\ \tfrac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i.
$$
The **kernel trick**: replace dot products $\mathbf{x}_i^\top\mathbf{x}_j$ with a kernel
$K(\mathbf{x}_i,\mathbf{x}_j)$ (e.g. RBF $K=\exp(-\gamma\|\mathbf{x}_i-\mathbf{x}_j\|^2)$) to draw
non-linear boundaries **without ever computing the high-dim coordinates**. Only the **support
vectors** (points on/inside the margin) determine the boundary.

**When/Where.** Small-to-medium datasets, high-dimensional (text, genomics), clear margins. **Not**
20M rows — kernel SVM training is roughly $O(N^2)$–$O(N^3)$; it doesn't scale to DelayCast. (Linear
SVM via `LinearSVC`/SGD does scale, but then you're close to LR.)

**Key hyperparameters.** `C` (margin softness — low = wide margin/more bias); `kernel`
(`linear`/`rbf`/`poly`); `gamma` (RBF reach — high = wiggly/overfit); `degree` (poly).

**Pros:** effective in high-d, kernel flexibility, robust margins. **Cons:** scales badly with N,
no native probabilities (needs Platt scaling), sensitive to `C`/`gamma`, needs scaling.

---

## 8. Linear / Quadratic Discriminant Analysis (LDA / QDA)

**What.** Generative models assuming each class is Gaussian; classify to the most probable class.

**How (math).** Assume $P(\mathbf{x}\mid y=c) = \mathcal{N}(\boldsymbol\mu_c, \Sigma_c)$. **LDA**
assumes a *shared* covariance $\Sigma$ → **linear** boundary; **QDA** allows per-class $\Sigma_c$ →
**quadratic** boundary. Apply Bayes' rule to get $P(y\mid\mathbf{x})$.

**When/Where.** Smallish data where the Gaussian assumption roughly holds; LDA also doubles as a
supervised dimensionality-reduction technique. Niche for DelayCast (mixed categorical features
break the Gaussian assumption).

**Key hyperparameters.** `solver` (`svd`/`lsqr`/`eigen`); `shrinkage` (regularizes the covariance
estimate — useful when $d$ is large vs $N$).

**Pros:** fast, closed-form, no tuning, good when assumptions hold. **Cons:** strong distributional
assumptions, linear (LDA), struggles with categoricals.

---

## 9. Neural Networks (MLP)

**What.** Stacked layers of weighted sums + non-linear activations; a universal function
approximator.

**How (math).** Each layer computes $\mathbf{a}^{(l)} = g(W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)})$
with a non-linearity $g$ (ReLU $\max(0,z)$, etc.); the final layer uses sigmoid/softmax for
probabilities. Trained by **backpropagation** — the chain rule computing $\nabla\mathcal{L}$ through
every layer — with SGD/Adam. Depth + non-linearity let it learn arbitrary boundaries and feature
interactions automatically.

**When/Where.** Huge datasets, **unstructured** data (images, text, audio, sequences), or when
complex interactions defeat simpler models. On **plain tabular** data like DelayCast, gradient
boosting usually **beats** MLPs while being faster and easier — so an NN is overkill here.

**Key hyperparameters.** Architecture (`hidden_layer_sizes` / width & depth), `activation`,
`learning_rate`, `batch_size`, `alpha` (L2), `dropout`, epochs + early stopping, optimizer.

**Pros:** maximal flexibility, dominates unstructured data, learns features. **Cons:** data- and
compute-hungry, many knobs, opaque, easy to overfit, usually loses to boosting on tabular data.

---

## Part 2 — When to use which (decision guide)

| Situation | Reach for |
|---|---|
| Need a baseline / interpretability / calibrated probs | **Logistic Regression** |
| Tabular data, want the best accuracy | **Gradient boosting** (LightGBM/XGBoost/CatBoost) |
| Tabular, want strong accuracy with near-zero tuning | **Random Forest** |
| Many high-cardinality categoricals | **CatBoost** (or boosting + target encoding) |
| Very high-dim sparse text, tiny data | **Naive Bayes** or **linear SVM/LR** |
| Small data, clear margin, high-dim | **SVM (RBF)** |
| Small low-dim data, meaningful distance | **k-NN** |
| Unstructured data (image/text/audio), lots of it | **Neural network** |
| Data too big for one machine | distributed (**Spark ML**) version of the above |

**Rules of thumb that hold up:**
1. **Always train an LR baseline first** — if the fancy model barely beats it, the signal is linear.
2. On **tabular** data, **boosting > RF > single tree > LR** in accuracy, but in the reverse order
   for speed/interpretability. Pick where on that curve you need to sit.
3. **More data favors lower-bias models** (boosting, NNs); **less data favors higher-bias models**
   (LR, NB, LDA) that won't overfit.
4. **Scale-sensitive** models (LR, SVM, k-NN, NN) need standardized features; **tree-based models
   don't** (they split on thresholds, scale-invariant).

### DelayCast mapping
- LR = honest baseline (sets recall/PR-AUC floor, interpretable coefficients).
- RF = non-linear sanity check (slow on 20M × one-hot — capped).
- XGBoost / LightGBM = expected champions; **LightGBM** for speed at 20M rows.
- Optimize **recall + PR-AUC** (~20% positives) → use `class_weight`/`scale_pos_weight`. See §10.

---

## Part 3 — Cross-cutting: imbalance, metrics, tuning

### 10. Handling class imbalance (~20% delayed)
- **Reweighting:** `class_weight='balanced'` (LR/RF/SVM) or `scale_pos_weight ≈ neg/pos` (XGB) makes
  each minority example count more in the loss. *Preferred* — no data thrown away or duplicated.
- **Resampling:** undersample majority or oversample minority (**SMOTE** synthesizes minority points).
  Use only on the *training* fold, never the test fold.
- **Threshold tuning:** don't accept 0.5 — pick the threshold from the precision–recall curve that
  hits your operating point (for an IOC, a recall target).
- **Metrics:** report **Recall** (catch real delays), **Precision**, **PR-AUC** (imbalance-robust),
  **F1**; treat **accuracy** as context only (a "never delayed" model already scores ~80%).

### 11. The bias–variance tradeoff (the lens behind every hyperparameter)
$$
\mathbb{E}[\text{error}] = \underbrace{\text{bias}^2}_{\text{too simple}} + \underbrace{\text{variance}}_{\text{too flexible}} + \underbrace{\sigma^2}_{\text{irreducible noise}}.
$$
Every regularization knob (tree depth, `C`, `lambda`, `k`, learning rate) trades these. Diagnose
with **learning/validation curves**: high train *and* val error = underfit (add capacity); low train
but high val error = overfit (regularize / get more data).

### 12. Tuning workflow (practical)
1. **Validation that matches the task** — for a *forecasting* problem like DelayCast, a **time-based
   split** (train on earlier months, test on later), never a random split, which leaks the future.
2. **Search smart:** start with sensible defaults → **random search** (beats grid for the same
   budget) → optionally **Bayesian** (Optuna/Hyperopt) for the final squeeze.
3. **Boosting order:** fix a low `learning_rate` + `early_stopping` → tune `num_leaves`/`max_depth`
   → tune `min_data_in_leaf` + regularization (`reg_lambda`/`reg_alpha`) → tune sampling
   (`subsample`/`colsample`) last.
4. **Always tune inside cross-validation** and judge on the **same metric you'll report** (here,
   recall + PR-AUC), not accuracy.

### 13. Interpretability (because an IOC must trust the call)
- **Global:** coefficients (LR), feature importances (trees/RF), gain (boosting).
- **Local + reliable:** **SHAP** values — game-theoretic per-prediction attributions; the standard
  for explaining boosting models ("this flight is high-risk because origin_delay_rate=0.34 and
  dep_hour=19").

---

## One-line summaries (the cheat sheet)

- **Logistic Regression** — linear log-odds; the baseline you must beat.
- **k-NN** — vote of neighbors; simple, dies in high dimensions.
- **Naive Bayes** — Bayes + independence; fast, great for text.
- **Decision Tree** — flowchart of splits; interpretable, overfits alone.
- **Random Forest** — bag of decorrelated trees; strong default, low tuning.
- **Gradient Boosting** — sequential error-correcting trees; tabular champion.
- **SVM** — max-margin (+ kernels); high-d, doesn't scale to millions of rows.
- **LDA/QDA** — Gaussian-per-class generative; fast, assumption-heavy.
- **Neural Net** — stacked non-linear layers; king of unstructured data, overkill for tabular.

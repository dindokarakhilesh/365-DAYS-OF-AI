# Day 99 — Perceptron Loss Function | Hinge Loss | Binary Cross Entropy | Sigmoid Function

---

## 1. Why do we need a loss function?

On Day 98 we trained the perceptron with a *trick*. A more general way to train **any** model is:

1. Pick a **model** (how scores or probabilities are computed)
2. Pick a **loss function** (how wrong a prediction is)
3. Use **gradient descent** to minimize the average loss over the data

Changing the **activation** and the **loss** gives different algorithms:

| Algorithm | Activation | Loss | Output |
|---|---|---|---|
| **Perceptron** | Step | Perceptron loss | Class (0/1) |
| **Linear SVM** | None (score) | Hinge loss (+ L2) | Class with a margin |
| **Logistic Regression** | **Sigmoid** | **Binary Cross Entropy** | Probability |
| **Softmax Regression** | Softmax | Categorical Cross Entropy | Class probabilities |
| **Linear Regression** | Identity | MSE | Number |

This "model + loss + optimizer" recipe is the base of all of deep learning.

---

## 2. Notation: margin

For a point `x` with label `t ∈ {−1, +1}` and score `s = w·x + b`, the **margin** is

```
m = t · s
```

| Margin | Meaning |
|---|---|
| `m > 0` | Correct side of the boundary |
| `m < 0` | Wrong side (misclassified) |
| `|m|` large | Far from the boundary |

(For labels in `{0, 1}` use `t = 2y − 1`.)

---

## 3. 0-1 loss (what we really want, but cannot optimize)

```
L₀₋₁ = 1 if m ≤ 0 else 0
```

It counts mistakes. But its gradient is **zero almost everywhere**, so gradient descent cannot use it. We use **smooth or convex surrogates** instead.

---

## 4. Perceptron loss

```
L = max(0, −m) = max(0, −t · s)
```

- Correct point (`m > 0`) → loss `0`
- Misclassified point → loss `−m`, growing linearly with how wrong it is

**Gradient (misclassified point):** `∂L/∂w = −t·x`, so gradient descent gives `w ← w + η·t·x`, which is exactly the **perceptron trick**.

**Limitations**
- Loss is `0` as soon as a point is correct, so there is **no margin** (the line can sit right next to a point)
- Many weight vectors give zero loss (not unique)
- Not differentiable at `m = 0`
- Only hard predictions, no probability

---

## 5. Hinge loss

```
L = max(0, 1 − m) = max(0, 1 − t · s)
```

- Zero only if the point is correct **and** at least **1 unit** beyond the boundary (`m ≥ 1`)
- Points with `0 < m < 1` (correct but too close) are **still penalized**
- Pushes the model toward a **margin**

**Subgradient:** `−t·x` if `m < 1`, otherwise `0`.

**Soft-margin linear SVM objective:**

```
minimize   (λ/2)·‖w‖²  +  (1/n) · Σ max(0, 1 − tᵢ·sᵢ)
```

The `‖w‖²` term keeps the margin wide (a smaller `‖w‖` means a wider margin `1/‖w‖`).

**Perceptron loss vs hinge loss:** hinge is the perceptron loss **shifted by 1**. That small shift is what creates the margin.

---

## 6. The Sigmoid function

```
σ(z) = 1 / (1 + e^(−z))
```

Turns any real score `z` into a number in `(0, 1)`, interpreted as `P(y = 1 | x)`.

### Properties

| Property | Value |
|---|---|
| Range | `(0, 1)` |
| `σ(0)` | `0.5` |
| Symmetry | `σ(−z) = 1 − σ(z)` |
| Derivative | `σ'(z) = σ(z)·(1 − σ(z))`, maximum `0.25` at `z = 0` |
| Monotonic | Always increasing |
| Inverse | **Logit:** `z = log(p / (1 − p))` (log-odds) |

### Logistic regression model

```
z = w·x + b
p = σ(z) = P(y = 1 | x)
ŷ = 1 if p ≥ 0.5 else 0          (the boundary is still w·x + b = 0)
```

The score `z` is the **log-odds** of class 1, so the model is *linear in log-odds*.

### Pros and cons

- Smooth and differentiable, and gives probabilities
- **Saturates:** for large `|z|`, `σ'(z) ≈ 0`, so gradients **vanish** (an issue in deep networks, which is why ReLU became popular for hidden layers)
- Not zero-centered
- Needs a **numerically stable** implementation (avoid `e^z` overflow for large positive `z`)

---

## 7. Binary Cross Entropy (BCE) / Log loss

For `y ∈ {0, 1}` and `p = σ(z)`:

```
BCE = −[ y·log(p) + (1 − y)·log(1 − p) ]
```

Per case:
- `y = 1` → `−log(p)` (small when `p` is near 1)
- `y = 0` → `−log(1 − p)` (small when `p` is near 0)

Dataset loss: average over all samples.

### Intuition: penalty for being confidently wrong

| True y | Predicted p | BCE |
|---|---|---|
| 1 | 0.99 | 0.010 |
| 1 | 0.80 | 0.223 |
| 1 | 0.50 | 0.693 |
| 1 | 0.20 | 1.609 |
| 1 | 0.01 | **4.605** |

The loss grows without bound as a confident prediction becomes wrong (`−log p → ∞` as `p → 0`).

### Where it comes from (maximum likelihood)

Assume `y ~ Bernoulli(p)`. Likelihood of one sample: `pʸ · (1 − p)^(1−y)`. Maximizing the likelihood of all data is the same as minimizing the **negative log-likelihood**, which is exactly BCE.

### The clean gradient

With `p = σ(z)` and `z = w·x + b`:

```
∂BCE/∂z = p − y
∂BCE/∂w = (p − y) · x
∂BCE/∂b = (p − y)
```

The sigmoid derivative **cancels** with the derivative of the log. The gradient is simply **prediction minus target**.

**Gradient descent update:** `w ← w − η · (1/n) Σ (pᵢ − yᵢ)·xᵢ`

### Margin form
With `t ∈ {−1, +1}`: `BCE = log(1 + e^(−t·s))`, which makes it easy to compare with hinge and perceptron loss.

### Properties
- **Convex** in `w` for logistic regression (a single global minimum)
- Strictly positive (never exactly 0), so it always rewards **more confident correct** predictions
- Needs `p` clipped to `[ε, 1−ε]` to avoid `log(0)`

---

## 8. Why not MSE with a sigmoid?

MSE: `(p − y)² / 2`. Its gradient wrt `z`:

```
∂MSE/∂z = (p − y) · p · (1 − p)
```

- The extra factor `p(1 − p)` is the sigmoid derivative, which **vanishes when the sigmoid saturates**
- If the model is **confidently wrong** (for example `p = 0.0025`, `y = 1`), the MSE gradient is about `0.0025` while the BCE gradient is about `1`: learning is very slow exactly when it should be fast
- The MSE loss with a sigmoid is also **not convex** in `w`

So: **sigmoid + BCE** for binary classification, and **MSE** for regression.

---

## 9. Comparing the losses (by margin `m`)

| Margin m | 0-1 | Perceptron `max(0,−m)` | Hinge `max(0,1−m)` | Log loss `log(1+e^−m)` |
|---|---|---|---|---|
| −1 | 1 | 1 | 2 | 1.313 |
| 0 | 1 | 0 | 1 | 0.693 |
| 0.5 | 0 | 0 | 0.5 | 0.474 |
| 1 | 0 | 0 | 0 | 0.313 |
| 2 | 0 | 0 | 0 | 0.127 |

| Loss | Margin? | Smooth? | Probabilities? | Used in |
|---|---|---|---|---|
| Perceptron | No | No (kink at 0) | No | Perceptron |
| Hinge | Yes | No (kink at 1) | No | SVM |
| Log loss (BCE) | Soft (always rewards larger m) | Yes | **Yes** | Logistic regression, neural networks |

---

## 10. Which loss, which activation?

| Task | Output activation | Loss |
|---|---|---|
| Binary classification | **Sigmoid** (1 unit) | **Binary Cross Entropy** |
| Multi-class (one label) | **Softmax** (K units) | Categorical Cross Entropy |
| Multi-label | Sigmoid (K units) | BCE per label |
| Regression | Identity / linear | MSE or MAE |
| Margin-based classification (SVM-style) | None (raw score) | Hinge |

---

## 11. Key takeaways

- A model is defined by **activation + loss**; training = minimize the loss with gradient descent.
- **Perceptron loss** `max(0, −m)` reproduces the perceptron trick but has no margin.
- **Hinge loss** `max(0, 1 − m)` adds a margin (SVM).
- **Sigmoid** maps scores to probabilities, with `σ' = σ(1 − σ)`.
- **BCE** is the maximum likelihood loss for binary outputs, and with a sigmoid its gradient is just **`p − y`**.
- Use **BCE, not MSE**, for sigmoid classifiers.

## 12. Revision questions

1. Write the perceptron loss, the hinge loss and the BCE in terms of the margin.
2. Why does the hinge loss produce a margin while the perceptron loss does not?
3. State four properties of the sigmoid and derive its derivative.
4. What is the logit function and how is it related to the sigmoid?
5. Derive `∂BCE/∂z = p − y`.
6. Why is the BCE penalty so large for a confident wrong prediction?
7. Why does sigmoid + MSE learn poorly compared with sigmoid + BCE?
8. Which output activation and loss would you use for binary, multi-class and regression tasks?

---
*Next topic: continue the Deep Learning course roadmap.*

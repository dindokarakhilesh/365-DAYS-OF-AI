# Day 98 — Perceptron Trick | How to Train a Perceptron | Perceptron Part 2

**Goal of today:** given labelled data, how do we *find* the weights `w` and bias `b` of a perceptron?

---

## 1. Recap and setup

A perceptron predicts

```
z = w1·x1 + w2·x2 + ... + wn·xn + b
ŷ = 1 if z ≥ 0 else 0
```

**Bias trick (notation):** add a constant input `x0 = 1` and treat the bias as a weight `w0 = b`:

```
x_aug = [1, x1, x2, ..., xn]
w     = [w0, w1, w2, ..., wn]       (w0 is the bias)
z     = w · x_aug
```

Now the decision boundary is simply `w · x_aug = 0`, and one weight vector describes the entire line, plane or hyperplane.

---

## 2. The Perceptron Trick

### Idea
Start with any line. Whenever a point is on the **wrong side**, **move the line toward that point** so it becomes correctly classified (or at least closer to being correct). Repeat.

### Rule for a misclassified point `x`

| Case | Where the point is | Fix |
|---|---|---|
| True class **1**, predicted **0** | On the negative side, should be positive | `w ← w + η · x_aug` |
| True class **0**, predicted **1** | On the positive side, should be negative | `w ← w − η · x_aug` |
| Correctly classified | | **No change** |

`η` (eta) is the **learning rate**.

### Unified update rule
Since `(y − ŷ)` equals `+1`, `−1` or `0` in these three cases:

```
w ← w + η · (y − ŷ) · x_aug
```

### Algorithm

```
1. Initialize w (zeros or small random numbers)
2. Repeat until no mistakes (or max iterations):
       pick a random training point (x, y)
       compute ŷ = step(w · x_aug)
       if ŷ ≠ y:
           w ← w + η (y − ŷ) x_aug
3. Return w
```

---

## 3. Why does the trick work?

**Algebraic reason.** Take a class-1 point with `z = w·x < 0`. After the update `w_new = w + η x`:

```
w_new · x = (w + η x) · x = w·x + η ‖x‖²
```

Since `η ‖x‖² > 0`, the score for this point **increases**, moving toward the positive side. Similarly, subtracting `η x` for a class-0 point **decreases** its score.

**Geometric reason.** Adding `x` to `w` rotates the weight vector (the normal of the line) toward `x`, and the boundary turns so that `x` ends up on the positive side. Subtracting rotates it away.

> One update may not fix the point completely (if `η` is small), and it may break other points. That is why we repeat many times. The convergence theorem (below) says this works out for separable data.

---

## 4. Worked example

Weights `w = [w0, w1, w2] = [−5, 1, 1]`, learning rate `η = 1`.

**Case A: true class 1, point (2, 1)**
- `x_aug = [1, 2, 1]`, `z = −5 + 2 + 1 = −2` → predicted 0, true 1 → misclassified
- `w_new = w + x_aug = [−4, 3, 2]`
- New score: `−4 + 3·2 + 2·1 = 4 ≥ 0` → now predicted 1, correct

**Case B: true class 0, point (4, 3)**
- `x_aug = [1, 4, 3]`, `z = −5 + 4 + 3 = 2` → predicted 1, true 0 → misclassified
- `w_new = w − x_aug = [−6, −3, −2]`
- New score: `−6 − 12 − 6 = −24 < 0` → now predicted 0, correct (a large jump, since η = 1 is big)

---

## 5. Perceptron Convergence Theorem

If the training data is **linearly separable**, the perceptron trick **converges** to a separating boundary after a **finite number of updates**.

The number of updates is bounded by roughly `(R / γ)²`, where
- `R` = largest distance of any point from the origin
- `γ` = margin of the best separating boundary (how far the closest points are from it)

So **well-separated data** (large margin) converges **faster**.

If the data is **not** separable, the algorithm **never stops**: some point is always misclassified and keeps pushing the line around.

---

## 6. Role of the learning rate η

- With `w` **initialized to zeros**, `η` only **scales** the weights, so the final boundary direction does not change.
- With a **non-zero initial** `w`, `η` controls how large each correction is compared with the current weights:
  - **Too small:** many updates to get anywhere
  - **Too large:** big jumps (overshooting), possibly more updates
- For the basic perceptron, `η` is **less sensitive** than in gradient-descent methods, but it still matters for non-zero initialization and for non-separable data.

---

## 7. Perceptron Part 2: training with a loss function

The trick is a neat heuristic. A more general way to see it is: **define a loss and minimize it with gradient descent.**

### Notation
Use labels `t ∈ {−1, +1}` (that is, `t = 2y − 1`), and score `s = w · x_aug`.

- Correct classification means `t · s > 0`.
- Misclassification means `t · s ≤ 0`.

### Perceptron loss

```
L(w) = (1/n) · Σ max(0, −t_i · s_i)
```

- If the point is correct (`t·s > 0`): loss for that point = **0**
- If misclassified: loss = `−t·s` (**positive, grows with how wrong the point is**)

### Gradient

For a misclassified point: `∂L/∂w = −t · x_aug`. For a correct point: `0`.

### Gradient descent step

```
w ← w − η · ∂L/∂w = w + η · t · x_aug      (for misclassified points)
```

This is **exactly the perceptron trick**. So:

> **Perceptron trick = (stochastic) gradient descent on the perceptron loss.**

### Variants

| Variant | Update uses |
|---|---|
| **Stochastic (online)** | One random point at a time |
| **Mini-batch** | A small group of points |
| **Batch** | Sum over all misclassified points per step |

### Loss comparison (why this view matters)

| Loss | Formula (per point) | Output model |
|---|---|---|
| **Perceptron loss** | `max(0, −t·s)` | Hard 0/1 |
| **Hinge loss** (SVM) | `max(0, 1 − t·s)` | Hard 0/1, with a margin |
| **Log loss** (logistic regression) | `log(1 + e^(−t·s))` | Probability |
| **Squared error** | `(t − s)²` | Regression-style |

Changing the loss changes the algorithm: hinge → SVM, log loss → logistic regression. This is the bridge toward the rest of deep learning, where we pick a loss and minimize it with gradient descent.

---

## 8. Problems with the perceptron approach

1. **Needs linearly separable data** to converge.
2. **Non-separable data:** mistakes keep fluctuating. A common fix is the **pocket algorithm**: keep the best weights seen so far and return them.
3. **Not the best boundary:** it stops at the first separating line, which may pass very close to some points (small **margin**). A linear **SVM** picks the max-margin line.
4. **Hard output:** no probabilities, no confidence.
5. **Not differentiable activation (step):** this is why modern networks use smooth activations such as sigmoid/ReLU, so that **backpropagation** works.

---

## 9. Summary table

| Concept | Key point |
|---|---|
| Bias trick | Add `x0 = 1`, so the bias becomes `w0` |
| Perceptron trick | If wrong: add `x` (class 1) or subtract `x` (class 0) |
| Unified rule | `w ← w + η (y − ŷ) x` |
| Why it works | `w_new · x = w·x ± η‖x‖²` |
| Convergence | Guaranteed only for linearly separable data |
| Loss view | `max(0, −t·s)`; gradient descent gives the same update |
| Limitations | Non-separable data, no margin, no probabilities |
| Next step | Smooth activations, log loss, and multi-layer networks |

---

## 10. Revision questions

1. What is the bias trick, and why is it convenient?
2. State the perceptron trick for both types of misclassification.
3. Show algebraically that adding `η·x` to `w` increases the score of that point.
4. What does the convergence theorem guarantee, and what does it require?
5. What happens if the data is not linearly separable? Name one fix.
6. Define the perceptron loss and derive its gradient.
7. How is the perceptron trick related to gradient descent?
8. Why does the perceptron not necessarily find the best separating line?
9. How do hinge loss and log loss change the algorithm?

---
*Next topic: continue the Deep Learning course roadmap.*

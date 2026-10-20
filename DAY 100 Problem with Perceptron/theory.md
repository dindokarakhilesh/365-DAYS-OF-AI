# Day 100 — Problems with the Perceptron

A single perceptron is simple and historically important, but it has serious limitations. Understanding them explains **why we need multi-layer neural networks, smooth activations and backpropagation**.

---

## 1. Recap

```
z = w · x + b
ŷ = 1 if z ≥ 0 else 0
```

- It draws **one straight line (hyperplane)**: `w · x + b = 0`
- Trained with the **perceptron trick** (or gradient descent on the perceptron loss)
- Works when the data is **linearly separable**

---

## 2. Problem 1: Only linear decision boundaries

A perceptron can only separate classes with a **straight line / plane / hyperplane**.

| Dataset shape | Single perceptron |
|---|---|
| Two well-separated blobs | Works |
| **XOR** pattern | **Fails** |
| **Concentric circles** | **Fails** |
| **Two moons** | **Fails** (cannot reach perfect accuracy) |

### The XOR problem

| x1 | x2 | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Proof that one perceptron cannot solve XOR.** We would need weights `w1, w2` and bias `b` with:

```
(0,0) → 0 :   b < 0
(0,1) → 1 :   w2 + b ≥ 0
(1,0) → 1 :   w1 + b ≥ 0
(1,1) → 0 :   w1 + w2 + b < 0
```

Adding the middle two inequalities: `w1 + w2 + 2b ≥ 0`, so `w1 + w2 + b ≥ −b`. Since `b < 0`, `−b > 0`, hence `w1 + w2 + b > 0`. This **contradicts** the last inequality. No such weights exist.

**Geometric view:** the points (0,1) and (1,0) are on one diagonal and (0,0) and (1,1) are on the other. No single line can put each diagonal on its own side.

This limitation was highlighted by **Minsky and Papert (1969)** and contributed to the **first AI winter**.

*(In the notebook, a brute-force search over about 227,000 weight combinations finds 100% accuracy for AND and OR, but at most 75% for XOR.)*

---

## 3. Problem 2: No convergence on non-separable data

- The convergence theorem guarantees success **only** for linearly separable data.
- If the classes overlap or have a non-linear shape, some point is **always** misclassified, so the updates **never stop**.
- The number of mistakes per epoch keeps **fluctuating** instead of decreasing to zero.

**Workaround: the pocket algorithm.** Keep a copy of the best weights seen so far and return them. This gives the best line found, but a line can still be the wrong shape for the data.

---

## 4. Problem 3: No unique solution, and no margin

- On separable data there are **infinitely many** separating lines.
- Which one the perceptron returns depends on the **initial weights**, the **order** of the data and the **learning rate**.
- It stops at the **first** line that separates the data, which can be **very close** to some points (a tiny margin) and may generalize poorly.
- A **linear SVM** chooses the line with the **maximum margin**, which is generally more robust.

*(In the notebook, 20 different runs all reach 100% training accuracy, but the closest point to the boundary ranges from 0.001 to 0.045 units, while the max-margin SVM gets 0.064.)*

---

## 5. Problem 4: Sensitivity to feature scaling

- The update `w ← w + η(y − ŷ)x` is **proportional to the input values**.
- If one feature has a much larger scale than the others, it dominates the updates and the boundary swings wildly.
- In the notebook, multiplying one feature by 1000 on separable data made the perceptron fail to converge within 1000 epochs, while the original and standardized versions converged in about 10 epochs.
- **Fix:** standardize or normalize the inputs (a good habit for neural networks too).

---

## 6. Problem 5: Hard output, no probability

- The step function outputs only **0 or 1**.
- A point barely across the boundary and a point far away get the **same answer**, with no sense of confidence.
- Many applications need probabilities (risk scores, medical decisions, ranking).
- **Fix:** a smooth output activation such as **sigmoid** or **softmax**, trained with **cross entropy** (Day 99).

---

## 7. Problem 6: Not differentiable, so no gradient descent through layers

- The step function has **zero gradient everywhere** (and is undefined at 0).
- Gradient-based learning, and especially **backpropagation** through several layers, needs **smooth, differentiable** activations.
- That is why modern networks use **sigmoid, tanh, ReLU** and variants.

---

## 8. Problem 7: Binary classification only

- One perceptron has **one output** (two classes).
- For `K` classes you need tricks such as **one-vs-rest** (train `K` perceptrons and take the highest score), or better, a **softmax output layer**.

---

## 9. Problem 8: No feature learning (single layer)

- A single perceptron works on the **raw inputs** only. It cannot build intermediate features.
- Complex patterns (images, language) need **hierarchies of features**, which needs **hidden layers**.

---

## 10. How do we fix these problems?

### Fix A: Feature engineering (manual)
Add non-linear features so the data becomes linearly separable in the new space.

| Data | Extra feature | Result |
|---|---|---|
| XOR | `x1 · x2` (XOR = `x1 + x2 − 2·x1·x2`) | Linearly separable pattern |
| Concentric circles | `x1² + x2²` (squared radius) | Perfectly separable |

**Drawback:** a human must guess the right features. Deep learning aims to **learn** them automatically.

### Fix B: Multi-Layer Perceptron (MLP)
Stack layers of neurons with **non-linear activations**. Hidden layers learn new features, and the output layer separates the data in that learned space.

XOR with two layers (from Day 97):

```
Hidden unit 1 = OR(x1, x2)
Hidden unit 2 = NAND(x1, x2)
Output        = AND(hidden 1, hidden 2)
```

**Universal Approximation Theorem (informal):** a network with at least one hidden layer and a non-linear activation can approximate any continuous function on a bounded domain arbitrarily well (given enough hidden units).

### Fix C: Better activation and loss
- Smooth activations (sigmoid, tanh, ReLU) make gradient descent possible
- Cross entropy loss gives probabilities and clean gradients

### Fix D: Backpropagation
Computes gradients for **all weights in all layers** efficiently with the chain rule, which makes training multi-layer networks practical.

### Fix E: Other approaches
- **Kernel methods / SVM**: implicit non-linear features
- **Decision trees / ensembles**: non-linear boundaries by splitting the space

---

## 11. Summary table

| Problem | Why it happens | Fix |
|---|---|---|
| Only linear boundaries (XOR fails) | One hyperplane | Hidden layers, non-linear features |
| Never converges on non-separable data | Always some mistakes | Pocket, smooth loss, better model |
| Many solutions, no margin | Stops at first separating line | SVM, regularization |
| Sensitive to scale | Updates proportional to inputs | Standardize features |
| Hard 0/1 output | Step activation | Sigmoid / softmax |
| Cannot use gradient descent through layers | Step has zero gradient | Smooth activations, backpropagation |
| Binary only | One output | One-vs-rest, softmax |
| No learned features | Single layer | Deep networks |

---

## 12. Key takeaways

- A single perceptron is a **linear binary classifier with a hard output**. That makes it fast and simple, but limited.
- **XOR** is the classic example, and the algebra above shows it can never be solved by one perceptron.
- The cure is **multiple layers + non-linear, differentiable activations + backpropagation**, which leads to **Multi-Layer Perceptrons** and deep learning.

## 13. Revision questions

1. Why can't a single perceptron solve XOR? Give the algebraic or geometric argument.
2. What happens to the perceptron trick on non-separable data, and what is the pocket algorithm?
3. Why does the perceptron not give the best separating line?
4. Why is the perceptron sensitive to feature scaling?
5. Why can't we run gradient descent through step activations?
6. Name two ways to make XOR solvable.
7. What does the Universal Approximation Theorem say (informally)?
8. How would you use perceptrons for a 3-class problem?

---
*Next topic: Multi-Layer Perceptron (MLP) notation and intuition.*

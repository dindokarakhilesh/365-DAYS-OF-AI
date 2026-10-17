# Day 97 — What is a Perceptron? Perceptron vs Neuron | Perceptron Geometric Intuition

---

## 1. What is a Perceptron?

The **Perceptron** (Frank Rosenblatt, 1958) is the simplest neural network: **a single artificial neuron that makes a binary decision**.

It takes several inputs, multiplies each by a weight, adds a bias, and passes the result through a **step function**.

```
 x1 ──w1──┐
 x2 ──w2──┤
 x3 ──w3──┼──►  z = w·x + b  ──►  step(z)  ──►  ŷ ∈ {0, 1}
  ⋮       │
 xn ──wn──┘          ▲
                     b (bias)
```

### Mathematical definition

```
z = w1·x1 + w2·x2 + ... + wn·xn + b = w · x + b

ŷ = 1   if z ≥ 0
ŷ = 0   if z < 0
```

| Symbol | Meaning |
|---|---|
| **x** | Input features |
| **w** | Weights: importance of each input (learned) |
| **b** | Bias: shifts the decision threshold (learned) |
| **z** | Weighted sum (pre-activation) |
| **step** | Activation function that outputs 0 or 1 |
| **ŷ** | Predicted class |

**Bias as a threshold:** `w·x + b ≥ 0` is the same as `w·x ≥ −b`. The neuron "fires" only when the weighted evidence passes a threshold of `−b`.

---

## 2. Perceptron as a logic gate

The perceptron can represent **AND**, **OR** and **NOT** gates:

| Gate | Weights | Bias | Check |
|---|---|---|---|
| **AND** | w = [1, 1] | b = −1.5 | fires only when both inputs are 1 |
| **OR** | w = [1, 1] | b = −0.5 | fires when at least one input is 1 |
| **NOT** (1 input) | w = [−1] | b = 0.5 | flips the input |

But it **cannot** represent **XOR** (see section 7).

---

## 3. Geometric intuition (the key idea)

### 3.1 The decision boundary is a line / plane / hyperplane

The perceptron separates the space into two regions using the equation:

```
w · x + b = 0
```

| Number of inputs | Boundary shape |
|---|---|
| 1 | a point |
| 2 | a **line** |
| 3 | a **plane** |
| n | an **(n−1)-dimensional hyperplane** |

For 2 inputs: `w1·x1 + w2·x2 + b = 0`, which is the line

```
x2 = −(w1/w2)·x1 − b/w2     (slope = −w1/w2, intercept = −b/w2)
```

### 3.2 The weight vector is perpendicular to the boundary

- **w** is the **normal vector** of the boundary. It points toward the region classified as **1**.
- Changing the **direction** of w **rotates** the boundary.
- Scaling w (while keeping direction) does not move the boundary, it only changes how large z is.

### 3.3 The bias moves the boundary

- The boundary's distance from the origin is **|b| / ‖w‖**.
- Increasing/decreasing **b** **shifts** the line parallel to itself without rotating it.
- With **b = 0** the boundary always **passes through the origin**, which limits flexibility. That is why the bias is needed.

### 3.4 Sign and distance of any point

For a point **x**:

```
signed distance to boundary = (w · x + b) / ‖w‖
```

- **Positive** → class 1 side
- **Negative** → class 0 side
- **Zero** → exactly on the boundary
- Larger magnitude → farther from the boundary (more "confident" in the geometric sense)

### 3.5 Linear separability

A dataset is **linearly separable** if one straight line (hyperplane) can separate the two classes perfectly.

- Perceptron **can** learn it.
- If not separable (XOR, circles, moons), a single perceptron **cannot** reach perfect accuracy.

---

## 4. The Perceptron learning rule

Start with random (or zero) weights, then loop over the training data:

```
for each sample (x, y):
    ŷ = step(w·x + b)
    error = y − ŷ                 # −1, 0 or +1
    w ← w + η · error · x
    b ← b + η · error
```

η (eta) is the **learning rate**.

### Intuition for the update
| Situation | error | Effect |
|---|---|---|
| Correct prediction | 0 | No change |
| Predicted 0, true is 1 (missed a positive) | +1 | **Add** x to w: boundary rotates toward x |
| Predicted 1, true is 0 (false alarm) | −1 | **Subtract** x from w: boundary rotates away from x |

Each mistake nudges the boundary toward classifying that point correctly.

### Perceptron Convergence Theorem
If the data is **linearly separable**, the perceptron learning rule is **guaranteed to find a separating boundary in a finite number of updates**. If not separable, it **never settles** (it keeps updating).

---

## 5. Perceptron vs Neuron

### 5.1 Biological neuron vs artificial neuron

| Biological neuron | Artificial neuron |
|---|---|
| Dendrites receive signals | Inputs **x** |
| Synapse strength | Weights **w** |
| Cell body sums signals | Weighted sum + bias |
| Fires if threshold is crossed | Activation function |
| Axon sends output | Output **a** |

The artificial neuron is only **loosely inspired** by the biological one.

### 5.2 Perceptron vs modern artificial neuron

| Aspect | **Perceptron** | **Modern artificial neuron** |
|---|---|---|
| Activation | **Step** (hard threshold) | Sigmoid, tanh, ReLU, softmax, etc. |
| Output | Binary (0 or 1) | Continuous value (e.g. 0 to 1, or any positive number) |
| Learning rule | Perceptron rule (updates only on mistakes) | **Gradient descent + backpropagation** |
| Differentiable? | **No** (step has zero or undefined gradient) | **Yes** (needed for gradient descent) |
| Tasks | Binary classification, linearly separable | Regression, classification, deep networks |
| Layers | Single neuron / single layer | Stacked into deep networks |
| Probabilistic output | No | Possible (sigmoid / softmax) |

> **Simple view:** a perceptron is **one type of artificial neuron** (step activation + perceptron rule). Today, "neuron" usually means a unit with a smooth activation trained by backpropagation.

### 5.3 Perceptron vs Logistic Regression
- Both compute `w·x + b` and give a **linear** boundary.
- Logistic regression applies a **sigmoid** and is trained by **minimizing log-loss**, giving a **probability**.
- The perceptron applies a **step** and updates **only on mistakes**, giving a hard 0/1.

---

## 6. Worked example (AND gate)

Inputs `(x1, x2)`, weights `w = [1, 1]`, bias `b = −1.5`:

| x1 | x2 | z = x1 + x2 − 1.5 | ŷ |
|---|---|---|---|
| 0 | 0 | −1.5 | 0 |
| 0 | 1 | −0.5 | 0 |
| 1 | 0 | −0.5 | 0 |
| 1 | 1 | **+0.5** | **1** |

The boundary `x1 + x2 = 1.5` is a line that separates (1,1) from the other three points.

---

## 7. Limitations: the XOR problem

XOR outputs 1 only when the inputs **differ**:

| x1 | x2 | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

No single straight line can separate {(0,1), (1,0)} from {(0,0), (1,1)}. A single perceptron fails.

Minsky and Papert pointed this out in 1969, which contributed to the **first AI winter**.

**The fix:** use **multiple layers** with non-linear activations (a **Multi-Layer Perceptron**). For example, XOR = `(x1 OR x2) AND NOT (x1 AND x2)`, which can be built from three perceptrons arranged in two layers.

### Other limitations
- Only **linear** boundaries
- Only **binary** output (without extensions)
- Does not converge if data is not linearly separable
- No probabilities, and not trainable by gradient descent

---

## 8. Key takeaways

- A perceptron computes `ŷ = step(w·x + b)`, a **binary linear classifier**.
- Geometrically it draws the boundary **w·x + b = 0**, where **w is perpendicular to the boundary** and **b shifts it**.
- The learning rule `w ← w + η(y − ŷ)x` converges **only if data is linearly separable**.
- A perceptron is a special case of the artificial neuron (step activation); modern neurons use smooth activations and **gradient descent**.
- XOR shows why we need **multi-layer** networks.

## 9. Revision questions

1. Write the perceptron equation and explain each term.
2. What does the bias do geometrically?
3. Why is the weight vector perpendicular to the decision boundary?
4. State the perceptron update rule and explain it for each type of mistake.
5. What is linear separability? Give one separable and one non-separable example.
6. Give three differences between a perceptron and a modern neuron.
7. Why can't a single perceptron solve XOR, and how is it fixed?
8. How does the perceptron differ from logistic regression?

---
*Next topic: continue the Deep Learning course roadmap.*

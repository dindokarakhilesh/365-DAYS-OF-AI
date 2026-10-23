# Day 103 — Forward Propagation | How a Neural Network Predicts Output

---

## 1. What is forward propagation?

**Forward propagation** (the *forward pass*) is the process of passing an input through the network, **layer by layer, from input to output**, to produce a prediction.

```
x  →  [ Layer 1 ]  →  [ Layer 2 ]  →  ...  →  [ Layer L ]  →  ŷ
```

- It **only computes**, it does **not learn** (no weights change).
- It is used for **prediction (inference)**, and as the **first half of every training step** (the second half is backpropagation).

---

## 2. Notation recap (from Day 101)

| Symbol | Meaning |
|---|---|
| `a⁰ = x` | Input (layer 0) |
| `W^l`, `b^l` | Weights `(n_{l−1}, n_l)` and biases `(n_l,)` of layer `l` |
| `z^l` | Pre-activation: `z^l = a^{l−1} W^l + b^l` |
| `f^l` | Activation function of layer `l` |
| `a^l` | Activation (output) of layer `l`: `a^l = f^l(z^l)` |
| `ŷ = a^L` | Network output |

---

## 3. The algorithm

```
a⁰ = x
for l = 1, 2, ..., L:
    z^l = a^{l−1} · W^l + b^l        # linear step
    a^l = f^l( z^l )                  # non-linear step
ŷ = a^L
```

Each layer does two things: **a linear transformation** (matrix multiplication plus bias) and **a non-linear activation**.

### As one big function (composition)

```
ŷ = f^L( f^{L−1}( … f^1( x W^1 + b^1 ) … ) W^L + b^L )
```

### Batch version (samples as rows)

```
Z^l = A^{l−1} · W^l + b^l       shape (m, n_l)    (b is broadcast across rows)
A^l = f^l( Z^l )                shape (m, n_l)
```

---

## 4. Worked example: 3 → 2 → 2 → 1

**Story:** predict if a student is placed. Scaled inputs: `x = [CGPA, IQ, 12th marks] = [0.8, 0.6, 0.9]`.

Hidden layers use **ReLU**, the output layer uses **sigmoid**.

```
W¹ = [[ 0.2,  0.5],      b¹ = [ 0.1, -0.1]
      [ 0.4, -0.3],
      [ 0.1,  0.6]]

W² = [[ 0.3, -0.2],      b² = [ 0.05, 0.1]
      [ 0.5,  0.4]]

W³ = [[ 0.7],            b³ = [ 0.1]
      [-0.4]]
```

### Layer 1 (3 → 2, ReLU)

```
z¹₁ = 0.8·0.2 + 0.6·0.4 + 0.9·0.1 + 0.1  = 0.16 + 0.24 + 0.09 + 0.1 = 0.59
z¹₂ = 0.8·0.5 + 0.6·(−0.3) + 0.9·0.6 − 0.1 = 0.40 − 0.18 + 0.54 − 0.1 = 0.66

a¹ = ReLU([0.59, 0.66]) = [0.59, 0.66]
```

### Layer 2 (2 → 2, ReLU)

```
z²₁ = 0.59·0.3 + 0.66·0.5 + 0.05 = 0.177 + 0.33 + 0.05  = 0.557
z²₂ = 0.59·(−0.2) + 0.66·0.4 + 0.1 = −0.118 + 0.264 + 0.1 = 0.246

a² = ReLU([0.557, 0.246]) = [0.557, 0.246]
```

### Output layer (2 → 1, sigmoid)

```
z³₁ = 0.557·0.7 + 0.246·(−0.4) + 0.1 = 0.3899 − 0.0984 + 0.1 = 0.3915

ŷ = σ(0.3915) = 1 / (1 + e^(−0.3915)) ≈ 0.5966
```

**Prediction:** probability of placement ≈ **0.597**, so with a 0.5 threshold the predicted class is **1 (placed)**.

(The same numbers are reproduced in the notebook.)

---

## 5. Choosing activations by layer

### Hidden layers
| Activation | Typical use |
|---|---|
| **ReLU** | Default choice for most hidden layers |
| **tanh** | Zero-centred, used in smaller or recurrent nets |
| **Sigmoid** | Rare in hidden layers (saturates, vanishing gradients) |

### Output layer (depends on the task)

| Task | Output layer | Prediction | Loss |
|---|---|---|---|
| Binary classification | 1 node, **sigmoid** | `ŷ ≥ 0.5` → class 1 | Binary cross entropy |
| Multi-class (one label) | `K` nodes, **softmax** | `argmax` | Categorical cross entropy |
| Multi-label | `K` nodes, **sigmoid** each | Threshold per label | BCE per label |
| Regression | 1 (or more) nodes, **linear** (no activation) | The raw value | MSE / MAE |

**Softmax:** `softmax(z)_k = e^{z_k} / Σ_j e^{z_j}`: outputs are positive and sum to 1.

---

## 6. Shapes cheat sheet

| Quantity | Shape |
|---|---|
| `X` | `(m, n_0)` |
| `W^l` | `(n_{l−1}, n_l)` |
| `b^l` | `(n_l,)` |
| `Z^l`, `A^l` | `(m, n_l)` |
| `ŷ` | `(m, n_L)` |

**Quick check:** inner dimensions in `A^{l−1} · W^l` must match: `(m, n_{l−1}) · (n_{l−1}, n_l)`.

---

## 7. What to store for backpropagation (the cache)

During the forward pass, **keep** `a^0, z^1, a^1, …, z^L, a^L`.

Backpropagation reuses them to compute gradients, so recomputing them would waste time. The cost: **memory** grows with depth, width and batch size.

---

## 8. Cost of a forward pass

- A dense layer from `n_{l−1}` to `n_l` nodes for `m` samples does about `m · n_{l−1} · n_l` multiply-adds.
- Total forward cost ≈ `m · Σ_l n_{l−1} · n_l` multiply-adds, so cost grows with **width²**-like terms and with **depth**.
- Forward propagation is mostly **matrix multiplication**, which is why **GPUs/TPUs** accelerate it so well.

### Vectorize and batch
*Notebook comparison, network `100 → 256 → 128 → 1`:* plain Python loops took about 15 ms per sample, vectorized NumPy on one row about 0.02 ms, and vectorized on a whole batch of 1000 about 0.008 ms per sample (roughly 1,800 times faster than loops in our run; exact numbers depend on your machine). **Never loop over neurons or samples in Python** when matrix operations can do it.

---

## 9. Numerical stability

- **Softmax:** `e^z` overflows for large `z`. Use `softmax(z − max(z))`. In the notebook, the naive version of `[1000, 1001, 1002]` returned `nan`, while the stable version gave `[0.09, 0.245, 0.665]`.
- **Sigmoid:** compute with `1/(1+e^{−z})` for `z ≥ 0` and `e^z/(1+e^z)` for `z < 0`.
- **Log of probabilities:** clip probabilities (for example to `[1e-12, 1 − 1e-12]`) before `log`, or use the loss functions that work on raw scores (logits).

---

## 10. Initial weights matter

- With **all-zero weights** every neuron in a layer computes the same thing (symmetry), so they would also learn the same thing.
- Use **small random** values, scaled to the layer size:
  - **He initialization** (for ReLU): `std = sqrt(2 / n_in)`
  - **Xavier / Glorot** (for sigmoid/tanh): `std = sqrt(1 / n_in)` (or `sqrt(2/(n_in + n_out))`)
- Biases are usually started at 0.

---

## 11. Common bugs

1. **Shape mismatch** (using `W · x` with the row-vector convention, or swapping `n_in` and `n_out`).
2. **Forgetting the bias**, or adding it with the wrong shape.
3. **Applying an activation on the output layer** when the task needs raw values (regression), or applying softmax before a loss that already includes it.
4. **Using ReLU on the output** for a probability.
5. **Naive softmax/sigmoid** causing `nan` or `inf`.
6. **Forgetting to scale inputs**, which makes the first layer's pre-activations huge or tiny.

---

## 12. Forward propagation vs training

| | Forward propagation | Backpropagation |
|---|---|---|
| Direction | Input → output | Output → input |
| Computes | Predictions, loss | Gradients of the loss |
| Changes weights? | **No** | Gradients are used to **update** them |
| Needed for | Prediction and training | Training only |

A training step = **forward pass → loss → backward pass → weight update**.

---

## 13. Key takeaways

- Forward propagation computes `a^l = f^l(a^{l−1} W^l + b^l)` for each layer, starting from `a⁰ = x`.
- The output layer depends on the task: **sigmoid** (binary), **softmax** (multi-class), **linear** (regression).
- Store `z^l` and `a^l` for backpropagation.
- It is mostly matrix multiplication: **vectorize**, **batch**, and use **numerically stable** functions.
- It only computes the prediction, learning happens in the backward pass and weight update.

## 14. Revision questions

1. Write the forward-propagation equations for layer `l`, for one sample and for a batch.
2. Why do we need an activation function after each linear step?
3. Which output activation and loss would you use for binary, multi-class and regression problems?
4. What shapes do `W^l`, `b^l` and `Z^l` have?
5. What is cached during the forward pass and why?
6. Why is the naive softmax unstable and how do you fix it?
7. Why shouldn't all weights start at zero?
8. Compute the forward pass by hand for a `2 → 2 → 1` network of your own.

---
*Next topic: Backpropagation intuition (how a network learns from its mistakes).*

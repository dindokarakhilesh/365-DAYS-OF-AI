# Day 101 — MLP Notation

A **Multi-Layer Perceptron (MLP)** stacks several layers of neurons. Before studying how it is trained (backpropagation), we need a clean, consistent **notation** so that every formula can be written precisely.

---

## 1. What is an MLP?

An MLP is a feed-forward neural network with:

- an **input layer** (just the features, no computation)
- one or more **hidden layers** (neurons with non-linear activations)
- an **output layer**

Every node in a layer is connected to **every node in the next layer** (a *fully connected* or *dense* network). Information flows in one direction: input → hidden → output (**forward propagation**).

```
Input layer      Hidden layer 1     Hidden layer 2     Output layer
 (layer 0)         (layer 1)          (layer 2)         (layer 3)

   x1 ───────┐      a¹₁ ──────┐       a²₁ ──────┐
   x2 ───────┼───►  a¹₂ ──────┼───►   a²₂ ──────┼───►   a³₁ = ŷ
   x3 ───────┤      a¹₃ ──────┘
   x4 ───────┘
```

Example above: **4 → 3 → 2 → 1**.

---

## 2. Notation table

| Symbol | Meaning |
|---|---|
| `L` | Number of layers **excluding** the input layer (hidden layers + output layer) |
| `l` | Layer index, `l = 0` is the input layer, `l = L` is the output layer |
| `n_l` | Number of nodes (neurons) in layer `l`; `n_0` = number of input features |
| `m` | Number of training samples |
| `x` | Input vector, `x = a⁰` |
| `x_i` | `i`-th input feature |
| `w^l_{ij}` | Weight connecting **node `i` of layer `l−1`** to **node `j` of layer `l`** |
| `W^l` | Weight matrix of layer `l`, shape `(n_{l−1}, n_l)` |
| `b^l_j` | Bias of node `j` in layer `l` |
| `b^l` | Bias vector of layer `l`, shape `(n_l,)` |
| `z^l_j` | Pre-activation (weighted sum) of node `j` in layer `l` |
| `f^l` | Activation function used in layer `l` |
| `a^l_j` | Output (activation) of node `j` in layer `l`, `a^l_j = f(z^l_j)` |
| `ŷ` | Network output, `ŷ = a^L` |
| `y` | True label |

### Conventions to remember
- **Layer 0 is the input layer** and it has no weights, biases or activation.
- **Depth** of the network = `L` (some authors say "a 3-layer network" for 4 → 3 → 2 → 1, since the input layer is not counted). Number of **hidden layers = L − 1**.
- The **superscript** is the **layer**, the **subscript** is the **node** (or the pair of nodes for a weight). Do not confuse the superscript with a power. Some books write `[l]` instead, like `W^{[l]}`.
- **Index order for weights is `ij` = from `i` to `j`** in these notes. Some books use the opposite order (`w_{ji}`: to `j`, from `i`), so always check the convention of the source you are reading.

---

## 3. Computation at a single node

For node `j` in layer `l`:

```
z^l_j = Σ_{i=1}^{n_{l−1}}  w^l_{ij} · a^{l−1}_i  +  b^l_j
a^l_j = f( z^l_j )
```

where `a^0_i = x_i`.

Example (first hidden node of the 4 → 3 → 2 → 1 network):

```
z¹₁ = w¹₁₁·x1 + w¹₂₁·x2 + w¹₃₁·x3 + w¹₄₁·x4 + b¹₁
a¹₁ = f(z¹₁)
```

Each node is exactly the neuron from Day 94, and an MLP is just many of them arranged in layers.

---

## 4. Computation for a whole layer (vectorized)

Stack the weights of layer `l` in a matrix `W^l` where **column `j` holds the weights going into node `j`**:

```
W^l = [ w^l_{ij} ]   shape (n_{l−1}, n_l)
```

For one sample (row vector `a^{l−1}`):

```
z^l = a^{l−1} · W^l + b^l          shape (n_l,)
a^l = f^l( z^l )
```

For a batch of `m` samples stacked as rows, `A^{l−1}` has shape `(m, n_{l−1})` and:

```
Z^l = A^{l−1} · W^l + b^l          shape (m, n_l)      (b^l is broadcast over rows)
A^l = f^l( Z^l )                   shape (m, n_l)
```

### Whole network as function composition

```
ŷ = f^L( f^{L−1}( ... f^1( x W^1 + b^1 ) ... W^{L−1} + b^{L−1} ) W^L + b^L )
```

The network is a **composition of simple functions**, each being "linear map, then non-linearity".

---

## 5. Shapes cheat sheet (batch of `m` samples)

| Quantity | Shape |
|---|---|
| `X = A⁰` | `(m, n_0)` |
| `W^l` | `(n_{l−1}, n_l)` |
| `b^l` | `(n_l,)` |
| `Z^l`, `A^l` | `(m, n_l)` |
| `ŷ = A^L` | `(m, n_L)` |

**Quick check:** in `A^{l−1} · W^l`, the inner dimensions must match: `(m, n_{l−1}) · (n_{l−1}, n_l) → (m, n_l)`.

---

## 6. Counting parameters

For layer `l`:
- weights: `n_{l−1} × n_l`
- biases: `n_l`

```
Total parameters = Σ_{l=1}^{L} ( n_{l−1} · n_l + n_l )
```

### Example: 4 → 3 → 2 → 1

| Layer | Weights | Biases | Parameters |
|---|---|---|---|
| 1 (4 → 3) | 4 × 3 = 12 | 3 | 15 |
| 2 (3 → 2) | 3 × 2 = 6 | 2 | 8 |
| 3 (2 → 1) | 2 × 1 = 2 | 1 | 3 |
| **Total** | 20 | 6 | **26** |

Input-layer nodes have **no parameters**.

---

## 7. Worked numerical example (2 → 2 → 1, sigmoid everywhere)

```
x  = [1, 2]
W¹ = [[0.1, 0.3],      b¹ = [0.1, 0.1]
      [0.2, 0.4]]
W² = [[0.5],           b² = [0.2]
      [0.6]]
```

**Layer 1**

```
z¹₁ = 1·0.1 + 2·0.2 + 0.1 = 0.6        a¹₁ = σ(0.6) ≈ 0.6457
z¹₂ = 1·0.3 + 2·0.4 + 0.1 = 1.2        a¹₂ = σ(1.2) ≈ 0.7685
```

**Layer 2 (output)**

```
z²₁ = 0.6457·0.5 + 0.7685·0.6 + 0.2 ≈ 0.9839
ŷ = a²₁ = σ(0.9839) ≈ 0.7279
```

---

## 8. Two weight-matrix conventions

Different sources lay the matrices out differently. They describe the **same network**.

| | Convention A (these notes) | Convention B |
|---|---|---|
| Samples are | **rows** | **columns** |
| `W^l` shape | `(n_{l−1}, n_l)` | `(n_l, n_{l−1})` |
| Forward step | `Z = A_prev · W + b` | `Z = W · A_prev + b` |
| Seen in | scikit-learn (`coefs_`), Keras `Dense` | Many textbooks and courses, PyTorch `nn.Linear.weight` |

Conversion: `W_B = (W_A)ᵀ`.

---

## 9. What gets stored for backpropagation?

During the forward pass we keep ("cache") the values `z^l` and `a^l` for every layer. The backward pass uses them to compute gradients of the loss with respect to every `W^l` and `b^l`. That is the topic of the next days.

---

## 10. Common mistakes

1. **Counting the input layer** in `L`, or counting layers inconsistently ("3-layer" vs "2 hidden layers").
2. **Swapping the weight index order** (`w_{ij}` vs `w_{ji}`) and then getting the wrong matrix shape.
3. **Mixing the two conventions** (using `W · x` with a `(n_{l−1}, n_l)` matrix).
4. Forgetting that **biases are per node**, not per input.
5. Reading the superscript `l` as an exponent.

---

## 11. Key takeaways

- An MLP is layers of neurons, fully connected, with information flowing forward.
- Notation: `w^l_{ij}` (from node `i` of layer `l−1` to node `j` of layer `l`), `b^l_j`, `z^l_j`, `a^l_j = f(z^l_j)`, with `a⁰ = x` and `ŷ = a^L`.
- Vectorized step: `Z^l = A^{l−1} W^l + b^l`, `A^l = f(Z^l)`.
- Parameters in layer `l`: `n_{l−1} · n_l + n_l`.
- Always track **shapes**, and check which weight convention a source uses.

## 12. Revision questions

1. What does `w^2_{31}` represent?
2. For a network `5 → 4 → 4 → 2`, what are `L`, `n_0 … n_L`, and the shape of each `W^l`?
3. Count the parameters of a `10 → 8 → 8 → 4 → 1` network.
4. Write the vectorized forward equations for one layer, for a batch of `m` samples.
5. What is the difference between `z^l` and `a^l`?
6. Convert a weight matrix from convention A to convention B.
7. Why do we store `z^l` and `a^l` during the forward pass?

---
*Next topic: Forward propagation in an MLP (the next video in the course).*

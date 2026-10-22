# Day 102 — Multi Layer Perceptron | MLP Intuition

---

## 1. The starting point: one perceptron is not enough

A single perceptron draws **one straight line**. It cannot solve XOR, concentric circles or moons.

**Core question:** how can we use many simple linear units to build a **non-linear** decision boundary?

**Answer:** connect them in **layers**. This is the **Multi Layer Perceptron (MLP)**.

---

## 2. Big picture intuition

> A hidden layer **transforms the data into a new space** (a new set of features). The output layer then solves a **simple linear problem** in that new space.

```
Original space              Hidden layer               Output layer
(not linearly separable) →  (new features h1, h2...) →  (a linear classifier in
                                                          the new feature space)
```

Equivalent ways to think about it:

1. **Feature learning:** each hidden neuron learns a useful feature (a "detector").
2. **Space transformation:** layer by layer, the data is bent and stretched until classes can be separated by a line.
3. **Combining half-planes:** each neuron splits the space with a line, and the next layer combines these splits into complex regions.

---

## 3. Intuition 1: Combining perceptrons

### One neuron = one half-plane
A neuron `σ(w·x + b)` is close to 1 on one side of the line `w·x + b = 0` and close to 0 on the other side.

### Output neuron = a combination
Take two hidden neurons `h1`, `h2` and let the output neuron compute `σ(h1 + h2 − c)`:

| Combination | Effect |
|---|---|
| **AND-like** (`c` close to the sum of both) | Region where both neurons fire (an intersection, for example a wedge or strip) |
| **OR-like** (`c` small) | Region where at least one neuron fires (a union) |
| **Weighted sum** | A smooth blend of both |

Adding the (soft) outputs of linear units gives a **curved, non-linear** boundary, even though each piece was linear.

---

## 4. Intuition 2: XOR solved by a hidden layer

```
h1 = OR(x1, x2)
h2 = NAND(x1, x2)
output = AND(h1, h2)
```

| x1 | x2 | h1 (OR) | h2 (NAND) | Output (AND) = XOR |
|---|---|---|---|---|
| 0 | 0 | 0 | 1 | 0 |
| 0 | 1 | 1 | 1 | 1 |
| 1 | 0 | 1 | 1 | 1 |
| 1 | 1 | 1 | 0 | 0 |

In the **`(h1, h2)` space** the XOR classes **are linearly separable**: the two positive points (1,1) are on one side and the negatives (0,1), (1,0) on the other. The hidden layer changed the geometry so that the output neuron only needs a line.

---

## 5. Intuition 3: Width (more neurons in a layer)

With step-like neurons, `k` half-planes combined with an **AND** form a **convex polygon**:

| k half-planes | Shape |
|---|---|
| 3 | Triangle |
| 4 | Quadrilateral |
| 6 | Hexagon |
| large k | Approaches a **circle** |

So a network with enough hidden neurons can represent curved regions like circles.

*Notebook result (hand-built network, no training):* on a concentric-circles dataset, accuracy was 0.83 with 3 hidden neurons, 0.95 with 4 and 1.00 from 5 neurons onward.

*Trained networks (one hidden layer, tanh):* test accuracy on circles went 0.38 → 0.52 → 1.00 → 0.99 for 1, 2, 4, 16 hidden neurons, and on moons 0.50 → 0.88 → 0.87 → 0.87. A few neurons are enough for simple shapes, and extra neurons then add little.

---

## 6. Intuition 4: Depth (more layers)

- Layer 1 detects **simple features** (lines, edges).
- Layer 2 combines them into **more complex features** (corners, shapes).
- Later layers build **abstract concepts** (faces, objects, meaning).

This **hierarchical feature learning** is why *deep* networks excel on images, speech and language.

**Width vs depth**
- A single wide hidden layer can approximate any function (see section 7), but may need a very large number of neurons.
- Deeper networks can often represent complicated functions **more efficiently**, by reusing features across layers.

*Notebook result (two spirals dataset):* a 2-neuron hidden layer reached about 0.63 test accuracy, 8 neurons about 0.82, while larger or deeper networks (a 64-neuron layer, or 16-16 and 16-16-16) reached 0.98 to 1.00. This is one random split with one optimizer, so treat it as an illustration, not a benchmark.

---

## 7. Intuition 5: Universal approximation (building curves from bumps)

- A very **steep sigmoid** acts like a **step** function.
- The **difference of two steps** is a **bump** (one hidden neuron minus another).
- A **sum of bumps**, each scaled to the target height, approximates **any continuous curve**. More bumps give a finer approximation.

*Notebook result (approximating `sin(x)` with bumps):* maximum error 0.97 with 4 hidden neurons (2 bumps), 0.36 with 16 neurons, and 0.08 with 64 neurons.

### Universal Approximation Theorem (informal)
A feed-forward network with **one hidden layer**, a **non-linear activation** and **enough hidden neurons** can approximate **any continuous function** on a bounded domain to any desired accuracy.

**Important caveats**
- It says such a network **exists**, not that gradient descent will **find** it.
- The needed width may be **huge**.
- It says nothing about **generalization** to new data.

---

## 8. Ways to make an MLP more powerful (and more flexible)

| Change | Effect |
|---|---|
| **Add nodes to a hidden layer** (width) | More half-planes / bumps, so more complex regions |
| **Add hidden layers** (depth) | Hierarchical features, more twisty boundaries |
| **Multiple output nodes** | Multi-class (softmax), multi-label (sigmoid per label), multi-target regression |
| **Skip connections** | Feed early layers directly to later layers (for example `a² = f(W a¹ + W_skip x + b)`), which helps training of very deep networks (ResNets) |
| **Different activations** | Control how non-linearity is introduced (ReLU, tanh, sigmoid) |

---

## 9. Why non-linear activations are essential

Without non-linearity, stacking layers gives:

```
W2 (W1 x) = (W2 W1) x     (just another linear map)
```

So the whole network would still be **one perceptron-like linear model**, and depth would add nothing. The activation function is what lets layers *bend* the space.

---

## 10. The price of power

- More parameters means more **data**, more **compute** and more **risk of overfitting**
- Harder to **interpret**
- Needs good **initialization**, **learning rates** and **regularization**
- Training requires **backpropagation** (coming next)

---

## 11. Key takeaways

- An MLP = layers of perceptrons with **non-linear activations**.
- Hidden layers **re-represent the data** so the output layer can separate it linearly.
- Each hidden neuron is a **soft half-plane**; the next layer **combines** them into complex regions.
- **Width** gives more pieces, **depth** gives hierarchical features.
- **Universal approximation:** one hidden layer with enough neurons can approximate any continuous function, but finding the weights is a training problem.

## 12. Revision questions

1. Explain, in one sentence, what a hidden layer does.
2. Show how XOR is solved by `OR`, `NAND` and `AND` neurons, and describe the hidden space.
3. How does an AND of `k` half-planes relate to a circle?
4. What is the difference between adding width and adding depth?
5. Why would an MLP with only linear activations be useless?
6. State the Universal Approximation Theorem and name two caveats.
7. What is a skip connection and why is it useful?

---
*Next topic: Forward propagation (how an MLP computes its prediction step by step).*

# Day 107 — Loss Functions in Deep Learning

---

## 1. What is a loss function?

A **loss function** measures **how wrong the model's prediction is** compared with the true value. It turns "how bad is this prediction?" into a single number that gradient descent can reduce.

```
prediction ŷ ──┐
               ├──► Loss L(y, ŷ) ──► backpropagation ──► update weights
true value y ──┘
```

Training loop: **forward pass → loss → backward pass (gradients of the loss) → weight update**.

> The loss is what the model **optimizes**. Metrics (accuracy, R², F1) are what **humans** read. They are related but not the same thing.

---

## 2. Loss function vs cost function

| Term | Meaning |
|---|---|
| **Loss function** | Error on a **single** sample, `L(yᵢ, ŷᵢ)` |
| **Cost function** | **Average** loss over a batch or the dataset, plus any regularization: `J = (1/n) Σ L(yᵢ, ŷᵢ) + λ·R(w)` |
| **Objective function** | The general thing being minimized (often the cost) |

In practice the words are often used interchangeably, and Keras reports the **averaged** value as "loss".

---

## 3. What makes a good loss function?

1. **Matches the goal** of the task (predict numbers, classes, probabilities, rankings).
2. **Differentiable** (or at least almost everywhere) so gradients exist. Accuracy itself cannot be used as a loss because it is a step function with zero gradient.
3. **Good gradient behaviour**: not vanishing when the model is badly wrong, not exploding.
4. **Ideally convex** in the model's output (so a single minimum), although deep networks are non-convex overall.
5. **Robust to the data issues you have** (outliers, class imbalance).

---

## 4. Regression losses

Let `e = y − ŷ` be the error.

### 4.1 Mean Squared Error (MSE / L2 loss)

```
MSE = (1/n) Σ (yᵢ − ŷᵢ)²          gradient wrt ŷ:  −2e
```

| Pros | Cons |
|---|---|
| Smooth and differentiable everywhere | **Sensitive to outliers** (errors are squared) |
| Gradient shrinks as the error shrinks, so it settles smoothly | Units are squared, hard to interpret |
| Convex for linear models | Large errors can dominate training |

The constant prediction that minimizes MSE is the **mean** of the targets.

### 4.2 Mean Absolute Error (MAE / L1 loss)

```
MAE = (1/n) Σ |yᵢ − ŷᵢ|            gradient wrt ŷ:  −sign(e)
```

| Pros | Cons |
|---|---|
| **Robust to outliers** | Not differentiable at `e = 0` |
| Same units as the target | Constant gradient size, so it can overshoot near the optimum (use a decaying learning rate) |

The constant prediction that minimizes MAE is the **median**.

*(Notebook example: for `[1, 2, 3, 4, 100]` the best constant for MSE is 22 (the mean) and for MAE is 3 (the median).)*

### 4.3 Root Mean Squared Error (RMSE)

`RMSE = √MSE`: same units as the target, but still outlier-sensitive. Mostly used as a **metric**.

### 4.4 Huber loss

```
L_δ(e) = ½ e²                  if |e| ≤ δ
       = δ (|e| − ½ δ)         otherwise
```

- **MSE-like** for small errors (smooth), **MAE-like** for large errors (robust).
- `δ` is a hyperparameter: the error size at which the loss switches behaviour.

### 4.5 Others (brief)
| Loss | Note |
|---|---|
| **Log-cosh** | Smooth, behaves like MSE for small and MAE for large errors |
| **MSLE** (mean squared log error) | For targets spanning orders of magnitude, penalizes relative error |
| **MAPE** | Percentage error, unstable when targets are near 0 |

### Summary: regression

| Loss | Outliers | Smooth | Typical use |
|---|---|---|---|
| MSE | Sensitive | Yes | Default for regression |
| MAE | Robust | No (kink at 0) | Data with outliers |
| Huber | Robust | Yes | Compromise, noisy data |

*Notebook demonstration:* fitting `y = 2x + 1` with four outliers, the MSE line's slope was pulled to about 3.8, while MAE gave about 1.99 and Huber about 2.03 (true slope 2).

---

## 5. Classification losses

### 5.1 Binary Cross Entropy (BCE / log loss)

For `y ∈ {0, 1}` and `p = σ(z)` = predicted probability of class 1:

```
BCE = −[ y·log(p) + (1 − y)·log(1 − p) ]
```

- `y = 1` → `−log(p)`; `y = 0` → `−log(1 − p)`
- **Gradient with respect to the score `z`:** `p − y` (prediction minus target)
- Comes from **maximum likelihood** for a Bernoulli model
- Punishes **confident wrong** predictions very heavily

Use with: **sigmoid** output (1 node).

### 5.2 Categorical Cross Entropy (CCE)

For `K` classes, softmax probabilities `p` and a **one-hot** target `y`:

```
CCE = − Σₖ yₖ · log(pₖ)  =  − log(p_true class)
```

- **Gradient with respect to the scores:** `p − y` (same clean form)
- Use with: **softmax** output (`K` nodes), **one-hot** labels

### 5.3 Sparse Categorical Cross Entropy (SCCE)

The **same loss**, but the label is an **integer** class index (for example `3`) instead of a one-hot vector. No need to one-hot encode, which saves memory when `K` is large.

| Label format | Keras loss |
|---|---|
| `[0, 0, 0, 1, 0]` (one-hot) | `categorical_crossentropy` |
| `3` (integer) | `sparse_categorical_crossentropy` |
| `0` or `1` (binary) | `binary_crossentropy` |

### 5.4 Cross entropy and KL divergence (intuition)

```
CE(p, q) = H(p) + KL(p ‖ q)
```

`p` = true distribution, `q` = predicted distribution. For one-hot targets the entropy `H(p) = 0`, so **minimizing cross entropy = minimizing the KL divergence** between truth and prediction.

### 5.5 Hinge loss (SVM style)

For labels `t ∈ {−1, +1}` and raw score `s`:

```
hinge = max(0, 1 − t·s)          squared hinge = max(0, 1 − t·s)²
```

- Zero only when the point is correct **with a margin of at least 1**
- No probabilities, but encourages a margin (see Day 99)

### 5.6 Loss values to remember (cross entropy)

| Probability given to the true class | Loss `−log p` |
|---|---|
| 0.99 | 0.010 |
| 0.90 | 0.105 |
| 0.50 | 0.693 |
| 0.10 | 2.303 |
| 0.01 | 4.605 |

---

## 6. Dealing with imbalanced classes

| Technique | Idea |
|---|---|
| **Class weights / weighted CE** | Multiply the loss of rare-class samples by a larger weight |
| **Focal loss** | `FL = −α (1 − pₜ)^γ log(pₜ)`: down-weights **easy** examples so training focuses on hard ones |
| **Resampling** | Oversample the rare class or undersample the common class |

*Notebook table for focal loss (γ = 2):* at `pₜ = 0.9` the loss shrinks to about 1% of the cross-entropy value (0.0011 vs 0.1054), while at `pₜ = 0.1` it keeps about 81% (1.865 vs 2.303).

---

## 7. Label smoothing

Instead of hard targets `1` and `0`, use `1 − ε` and `ε/(K − 1)`. This stops the model from becoming over-confident and often improves generalization. In Keras: `label_smoothing=0.1` in the cross-entropy loss.

---

## 8. Logits and numerical stability

- **Logits** are the raw scores before sigmoid or softmax.
- Computing `softmax` then `log` can overflow or underflow for extreme scores.
- The stable approach computes the loss **from the logits** using the **log-sum-exp trick**:

```
log softmax(z)ₖ = zₖ − log Σⱼ e^{zⱼ}      (computed after subtracting max(z))
```

In Keras this is `from_logits=True`:

```python
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)   # last layer has NO softmax
```

> **Important:** use *either* a softmax/sigmoid layer *and* `from_logits=False`, *or* a linear last layer *and* `from_logits=True`. Doing both applies the activation twice and hurts training.

*(In the notebook, for logits `[1000, 0, −1000]` with the true class last, the naive version breaks, the stable version returns 2000, and clipping probabilities at `1e-12` would cap the loss near 27.6 and hide how wrong the model is.)*

---

## 9. Why cross entropy instead of MSE for classification?

With a sigmoid output `p = σ(z)`, the gradient of the loss with respect to the score `z` is:

| Loss | Gradient wrt `z` |
|---|---|
| **BCE** | `p − y` |
| **MSE** | `(p − y) · p · (1 − p)` |

The extra factor `p(1 − p)` in MSE **vanishes when the sigmoid saturates**. If the model is confidently wrong (for example `z = −6` when `y = 1`), the gradient sizes are about **0.9975 (BCE)** against **0.0025 (MSE)**: BCE learns fast exactly when it should, MSE barely moves.

Also, MSE with a sigmoid is **not convex** in the weights, while BCE with a sigmoid (logistic regression) is.

---

## 10. Pairing output layer, activation and loss

| Task | Output layer | Loss |
|---|---|---|
| Regression | 1+ nodes, **linear** | MSE / MAE / Huber |
| Binary classification | 1 node, **sigmoid** | Binary cross entropy |
| Multi-class (one label) | `K` nodes, **softmax** | (Sparse) categorical cross entropy |
| Multi-label (several labels can be true) | `K` nodes, **sigmoid** | Binary cross entropy per label |
| Imbalanced classes | Sigmoid / Softmax | Weighted CE or focal loss |
| Margin-based | Raw score | Hinge |

---

## 11. Regularization in the cost

```
J = (1/n) Σ L(yᵢ, ŷᵢ) + λ · R(w)
```

| Penalty | `R(w)` | Effect |
|---|---|---|
| **L2** (weight decay / ridge) | `Σ w²` | Shrinks weights toward 0 smoothly |
| **L1** (lasso) | `Σ |w|` | Pushes many weights to exactly 0 (sparsity) |
| **Elastic net** | Both | Combination |

`λ` controls the strength: too small → overfitting, too large → underfitting. Biases are usually **not** penalized.

*Notebook demonstration (degree-12 polynomial on 15 noisy points):* without a penalty the training MSE was 0.023 but the test MSE was 0.68 with enormous weights; with a small L2 penalty (`λ = 1e-4`) the training MSE rose to 0.042 while the test MSE dropped to about 0.009.

---

## 12. Custom loss functions in Keras

A loss takes `(y_true, y_pred)` and returns a value (per sample or averaged). Use TensorFlow operations so gradients can flow.

```python
import tensorflow as tf

def my_huber(y_true, y_pred, delta=1.0):
    e = y_true - y_pred
    small = tf.abs(e) <= delta
    loss = tf.where(small, 0.5 * tf.square(e), delta * (tf.abs(e) - 0.5 * delta))
    return tf.reduce_mean(loss)

model.compile(optimizer="adam", loss=my_huber)
```

---

## 13. Sanity checks with loss values

1. **Initial loss:** for a `K`-class softmax model with small random weights, the first loss should be close to `ln(K)` (for example `ln(10) ≈ 2.303`, `ln(2) ≈ 0.693`). If it is far off, suspect a bug (wrong labels, wrong activation or loss).
   *(In the notebook: K = 10 gave 2.3032, K = 100 gave 4.6045, against 2.3026 and 4.6052.)*
2. **Overfit one tiny batch:** a working model should drive the loss near 0 on 10-20 samples.
3. **Loss must decrease** (on average). If it is `nan`, check the learning rate, inputs, and `log(0)` issues.
4. **Compare training and validation loss** to detect overfitting.

---

## 14. Common mistakes

1. **MSE for classification** (slow, vanishing gradients).
2. **Mismatched label format and loss** (integer labels with `categorical_crossentropy`, one-hot labels with `sparse_...`).
3. **Applying softmax and also using `from_logits=True`** (or neither).
4. **Sigmoid with a multi-class single-label problem**, or **softmax for multi-label**.
5. **Using accuracy as a loss** (not differentiable).
6. **Comparing raw loss values across different tasks or scales** (they are not comparable).
7. **Ignoring outliers** when using MSE, or ignoring imbalance when using plain cross entropy.
8. **Forgetting to clip or stabilize** `log` inputs in hand-written losses.

---

## 15. Keras quick reference

| Loss | Class / string |
|---|---|
| MSE | `keras.losses.MeanSquaredError()` / `"mean_squared_error"` |
| MAE | `keras.losses.MeanAbsoluteError()` / `"mean_absolute_error"` |
| Huber | `keras.losses.Huber(delta=1.0)` |
| Binary CE | `keras.losses.BinaryCrossentropy(from_logits=False)` |
| Categorical CE | `keras.losses.CategoricalCrossentropy()` |
| Sparse categorical CE | `keras.losses.SparseCategoricalCrossentropy(from_logits=False)` |
| Hinge | `keras.losses.Hinge()` |
| KL divergence | `keras.losses.KLDivergence()` |

---

## 16. Key takeaways

- The **loss** is what gradient descent minimizes: **per sample** (loss) or **averaged plus regularization** (cost).
- **Regression:** MSE (default), MAE (robust), Huber (both). **Classification:** (sparse) categorical cross entropy for multi-class, binary cross entropy for binary and multi-label.
- Cross entropy gives the clean gradient **`p − y`** and avoids the vanishing gradients that MSE has with sigmoid/softmax.
- Always **match output activation, loss and label format**, and use **logits-based** losses for stability.
- Sanity-check: initial loss ≈ **ln(K)**, and the model should overfit a tiny batch.
- Handle **outliers** (MAE/Huber), **imbalance** (class weights/focal loss) and **overfitting** (L1/L2 in the cost) by adjusting the loss or cost.

## 17. Revision questions

1. What is the difference between a loss function and a cost function?
2. Why can't accuracy be used as a loss function?
3. Compare MSE, MAE and Huber: which statistic does each of MSE and MAE estimate, and how does each treat outliers?
4. Write BCE and CCE, and explain what "sparse" means.
5. Derive or state the gradient of BCE through a sigmoid, and explain why MSE has a weaker gradient.
6. What does `from_logits=True` do, and what is the log-sum-exp trick?
7. Which output layer and loss do you use for binary, multi-class, multi-label and regression tasks?
8. What should the initial loss of a 10-class softmax model be, and why?
9. How does L2 regularization change the cost function, and what does `λ` control?
10. How does focal loss help with imbalanced data?

---
*Next topic: Backpropagation (how gradients of the loss are computed through the network).*

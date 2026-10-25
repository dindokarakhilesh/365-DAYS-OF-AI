# Day 105 — Handwritten Digit Classification using ANN (MNIST Dataset)

Our first **image classification** project, and our first **multi-class** problem: given a picture of a handwritten digit, predict which digit (0-9) it shows.

---

## 1. The MNIST dataset

| Property | Value |
|---|---|
| Content | Handwritten digits 0-9 |
| Size | **70,000** images: **60,000** training + **10,000** test |
| Image size | **28 × 28** pixels |
| Channels | 1 (**grayscale**) |
| Pixel values | Integers **0 to 255** (0 = black background, 255 = brightest stroke) |
| Classes | 10 (digits 0 to 9), roughly balanced |

MNIST (derived from NIST handwriting data) is the "hello world" of deep learning: small, clean and quick to train on a laptop.

---

## 2. How a computer sees an image

An image is a **grid of numbers**. For MNIST, each image is a `28 × 28` matrix of pixel intensities.

| Object | Shape |
|---|---|
| One image | `(28, 28)` |
| Training set `X_train` | `(60000, 28, 28)` |
| Training labels `y_train` | `(60000,)` (integers 0-9) |
| Test set `X_test` | `(10000, 28, 28)` |
| Flattened image | `(784,)` since `28 × 28 = 784` |

(A color image would have 3 channels, shape `(height, width, 3)`.)

---

## 3. Preprocessing

### 3.1 Scale the pixels
```
X = X / 255.0          # values now between 0 and 1
```
Why: large inputs (0-255) make pre-activations large, push activations into extreme ranges and make gradient descent slower and less stable. Small, consistent inputs train faster and better.

### 3.2 Flatten
An ANN (dense network) takes a **vector** as input, so the `28 × 28` image is unrolled into **784 numbers**. In Keras this is a `Flatten()` layer at the start of the model.

### 3.3 Labels

| Label format | Loss to use |
|---|---|
| **Integer** labels (0 to 9) | `sparse_categorical_crossentropy` |
| **One-hot** labels (for example digit 3 → `[0,0,0,1,0,0,0,0,0,0]`) | `categorical_crossentropy` |

Both give the same result. The sparse version just skips the one-hot step.

---

## 4. The architecture

```
Input image (28×28) → Flatten (784) → Dense(128, ReLU) → Dense(32, ReLU) → Dense(10, softmax)
```

```python
model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(32,  activation="relu"),
    layers.Dense(10,  activation="softmax"),
])
```

### Parameter count (Day 101 formula)

| Layer | Weights | Biases | Total |
|---|---|---|---|
| 784 → 128 | 784 × 128 = 100,352 | 128 | 100,480 |
| 128 → 32 | 128 × 32 = 4,096 | 32 | 4,128 |
| 32 → 10 | 32 × 10 = 320 | 10 | 330 |
| **Total** | | | **104,938** |

`Flatten` has no parameters. Most parameters sit in the **first** layer, because the input is large.

### Why 10 output nodes?
One node per class. Node `k` produces the **score (logit)** for digit `k`, and softmax turns the 10 scores into probabilities.

---

## 5. Softmax

```
softmax(z)_k = e^{z_k} / Σ_j e^{z_j}
```

| Property | Meaning |
|---|---|
| Every output is in (0, 1) | Can be read as a probability |
| Outputs **sum to 1** | A distribution over the 10 classes |
| Monotonic in `z_k` | The largest score gets the largest probability |
| Amplifies differences | `e^z` makes the biggest score dominate |

**Example:** scores `[2.0, 1.0, 0.1]` give probabilities of about `[0.659, 0.242, 0.099]`.

**Relation to sigmoid:** for 2 classes, softmax reduces to a sigmoid. Use **sigmoid** for binary problems (one output node) and **softmax** for multi-class problems where exactly one class is right.

**Prediction:** `predicted_digit = argmax(probabilities)`.

---

## 6. Loss: categorical cross entropy

For one image with true class `c` and predicted probabilities `p`:

```
loss = − log( p_c )
```

(Only the probability given to the **true class** matters.)

| Probability given to the true class | Loss |
|---|---|
| 0.99 | 0.010 |
| 0.90 | 0.105 |
| 0.50 | 0.693 |
| 0.10 | 2.303 |
| 0.01 | 4.605 |

Being confidently wrong is punished heavily. This is the multi-class version of binary cross entropy (Day 99).

**Gradient:** for softmax + cross entropy, the gradient with respect to the scores is simply `p − y_onehot` (prediction minus target), the same clean form as sigmoid + BCE.

---

## 7. Compile and train

```python
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)
```

| Concept | Meaning |
|---|---|
| `validation_split=0.2` | Keras holds out the **last 20%** of the training data to monitor generalization (it takes them before shuffling) |
| Training samples used | 60,000 × 0.8 = 48,000 |
| Batch size 32 | A weight update every 32 images |
| Steps per epoch | 48,000 / 32 = **1,500** |
| Epoch | One full pass over the 48,000 training images |

---

## 8. Reading the training curves

| Pattern | Meaning |
|---|---|
| Training and validation loss both fall | Learning |
| Training loss keeps falling, **validation loss flattens or rises** | **Overfitting** (memorizing training images) |
| Training accuracy ≫ validation accuracy | Overfitting |
| Both flat and poor | Underfitting (model too small, learning rate wrong, or inputs not scaled) |

For an ANN of this size on MNIST, you should typically see test accuracy in the high 90s (roughly 97 to 98%), with a visible gap between training and validation accuracy after several epochs. Your exact numbers vary with the random seed.

---

## 9. Evaluation

### 9.1 Test accuracy
`model.evaluate(X_test, y_test)` returns the loss and accuracy on the **held-out** test set. Use the test set once, at the end.

### 9.2 Predictions
```python
proba = model.predict(X_test)           # shape (10000, 10)
y_pred = proba.argmax(axis=1)           # shape (10000,)
```
(`predict_classes` is no longer available. Use `argmax`.)

### 9.3 Confusion matrix (10 × 10)
Row = true digit, column = predicted digit. The diagonal holds the correct predictions; off-diagonal cells show **which digits get confused**.

Typical confusions for handwritten digits: **4 ↔ 9**, **3 ↔ 5 ↔ 8**, **7 ↔ 1**, **2 ↔ 7**.

### 9.4 Per-class precision, recall and F1
`classification_report` shows if some digits are much harder than others.

### 9.5 Look at the mistakes
Plot misclassified images with their predicted probability. Many mistakes are ambiguous even for humans, and seeing them tells you whether the model needs more data, a better architecture or cleaner labels.

---

## 10. Predicting a single image

1. The model expects a **batch**, so add a batch dimension: `image[np.newaxis]` gives shape `(1, 28, 28)`.
2. Apply the **same preprocessing** (divide by 255).
3. `model.predict(...)` gives 10 probabilities, and `argmax` gives the digit.

### For your own handwriting
Match the training format exactly:
- **Grayscale**, resized to **28 × 28**
- **Light digit on a dark background** (MNIST style). If your image is dark ink on white paper, **invert** it.
- **Centered** with some margin
- **Scaled** to 0-1

Most "my digit is predicted wrong" problems come from a mismatch in these steps, not from the model.

---

## 11. Improving the model

| Technique | Effect |
|---|---|
| **Keep inputs scaled** | Much faster, more stable training (notebook experiment compares raw 0-255 vs scaled 0-1) |
| **Dropout** | Randomly switches off neurons while training: less overfitting |
| **EarlyStopping** | Stops at the best validation loss and restores the best weights |
| **More/wider layers** | More capacity (watch for overfitting) |
| **Learning-rate tuning** | Faster, smoother convergence |
| **Batch normalization** | Stabilizes training of deeper networks |
| **Data augmentation** (small shifts, rotations, zoom) | More variety, better generalization |
| **Convolutional Neural Network (CNN)** | The real step up for images (see below) |

---

## 12. Limitations of an ANN on images

1. **Flattening destroys 2D structure.** The network does not know pixel (5, 5) is next to pixel (5, 6). It must learn spatial relationships from scratch.
2. **No translation invariance.** A digit shifted a few pixels looks like a very different vector, so the ANN struggles unless it saw shifted examples.
3. **Parameter explosion.** A `224 × 224 × 3` color image has 150,528 inputs. A first dense layer of just 128 neurons would need `150,528 × 128 ≈ 19.3 million` weights.
4. **Weak feature reuse.** Every location learns its own separate weights, even if the same stroke pattern appears everywhere.

**Solution:** **CNNs** use small shared filters that slide over the image, keep the 2D structure, have far fewer parameters and detect patterns anywhere in the image. That is the next major topic after the ANN foundations.

---

## 13. Common mistakes

1. **Not scaling** the pixels (or scaling the training images but not the images you predict on).
2. **Wrong loss for the label format** (`sparse_categorical_crossentropy` for integer labels, `categorical_crossentropy` for one-hot).
3. **Forgetting `Flatten`**, or flattening in the wrong place, giving shape errors.
4. **Using `sigmoid` with 10 outputs** for a single-label problem instead of `softmax`.
5. **Forgetting the batch dimension** when predicting one image.
6. **Tuning on the test set** instead of the validation set.
7. **Feeding images in a different format** (colors, size, background) than the training data.

---

## 14. Keras quick reference (new today)

| Call | Purpose |
|---|---|
| `keras.datasets.mnist.load_data()` | Downloads MNIST and returns `(X_train, y_train), (X_test, y_test)` |
| `layers.Flatten()` | Turns `(28, 28)` into `(784,)` |
| `layers.Dense(10, activation="softmax")` | Multi-class output layer |
| `loss="sparse_categorical_crossentropy"` | Loss for integer class labels |
| `model.fit(..., validation_split=0.2)` | Hold out part of the training data for validation |
| `model.predict(X).argmax(axis=1)` | Predicted class for each image |
| `model.layers[i].get_weights()` | Read the learned weights and biases |

---

## 15. Key takeaways

- An image is a **grid of pixel numbers**. Scale it to 0-1 and **flatten** it to feed an ANN.
- Multi-class output: **10 nodes + softmax**, loss **(sparse) categorical cross entropy**, prediction **argmax**.
- The network here has **104,938 parameters**, most of them in the first layer.
- Evaluate with **accuracy, confusion matrix, per-class metrics, and by looking at mistakes**.
- Use **scaling, dropout and early stopping** to train well and fight overfitting.
- ANNs on images **lose the 2D structure and need many parameters**, which is exactly what **CNNs** fix.

## 16. Revision questions

1. What is the shape of `X_train` for MNIST, and what does each number mean?
2. Why do we divide pixel values by 255?
3. Why do we flatten the images, and what is the input size of the first dense layer?
4. Count the parameters of the `784 → 128 → 32 → 10` network.
5. Why softmax in the output layer, and how do we get the predicted digit from it?
6. When do you use `sparse_categorical_crossentropy` and when `categorical_crossentropy`?
7. What does the loss equal if the model gives probability 0.5 to the true class?
8. Name two ways to recognize overfitting from the training curves and two ways to reduce it.
9. What does the confusion matrix tell you that accuracy does not?
10. Give two reasons why a plain ANN is a poor fit for large images.

---
*Next topic: continue the Deep Learning course roadmap (Loss functions in deep learning).*

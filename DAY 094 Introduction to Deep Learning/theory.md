# Day 94 — Introduction to Deep Learning

## 1. What is Deep Learning?

**Deep Learning (DL)** is a subfield of Machine Learning that uses **artificial neural networks with many layers** (hence "deep") to learn patterns directly from data.

```
Artificial Intelligence  ⊃  Machine Learning  ⊃  Deep Learning
```

- **AI**: any technique that makes machines act intelligently.
- **ML**: algorithms that learn from data instead of being explicitly programmed.
- **DL**: ML models built from stacked layers of neurons that learn their own features.

---

## 2. Machine Learning vs Deep Learning

| Aspect | Machine Learning | Deep Learning |
|---|---|---|
| Feature engineering | Done **manually** by humans | Learned **automatically** by the network |
| Data needed | Works with small/medium data | Needs large data to shine |
| Hardware | CPU is usually enough | GPU/TPU strongly preferred |
| Training time | Minutes to hours | Hours to weeks |
| Interpretability | Often easier (trees, linear models) | Harder ("black box") |
| Best at | Tabular/structured data | Images, audio, text, video |
| Typical models | Linear/Logistic Regression, SVM, Random Forest, XGBoost | CNN, RNN, LSTM, Transformer, GAN |

**Key idea:** in classical ML you design the features; in DL the network **learns the features** layer by layer
(edges → shapes → objects, for images).

---

## 3. Why Deep Learning now? (Why did it take off?)

1. **Big Data**: internet, phones and sensors produce huge labelled datasets (ImageNet, Common Crawl).
2. **Compute power**: GPUs/TPUs make matrix math very fast.
3. **Better algorithms**: ReLU, dropout, batch normalization, Adam, better initialization.
4. **Open-source frameworks**: TensorFlow, Keras, PyTorch lower the barrier.
5. **Transfer learning & pretrained models**: reuse knowledge instead of training from scratch.

### Short history
- 1943: McCulloch–Pitts neuron
- 1958: **Perceptron** (Rosenblatt)
- 1986: **Backpropagation** popularized (Rumelhart, Hinton, Williams)
- 1998: LeNet (CNN for digits)
- 2012: **AlexNet** wins ImageNet, the deep learning boom begins
- 2017: **Transformer** ("Attention Is All You Need")
- 2020s: Large Language Models, diffusion models, multimodal AI

---

## 4. Biological inspiration → Artificial neuron

A biological neuron receives signals (dendrites), processes them (cell body) and fires an output (axon) if the signal is strong enough.

An **artificial neuron** copies this idea mathematically:

```
inputs x1, x2, ..., xn
        │
   weighted sum:  z = w1·x1 + w2·x2 + ... + wn·xn + b
        │
   activation:    a = f(z)
        │
     output a
```

- **x**: inputs (features)
- **w**: weights (importance of each input, *learned*)
- **b**: bias (shifts the decision threshold, *learned*)
- **f**: activation function (adds non-linearity)

In vector form: **z = w · x + b**, **a = f(z)**

---

## 5. Neural Network structure

```
Input Layer  →  Hidden Layer(s)  →  Output Layer
```

- **Input layer**: raw features (pixels, words, numbers). No computation.
- **Hidden layers**: where learning happens. More hidden layers = deeper network.
- **Output layer**: final prediction (class probabilities, a number, etc.).

A network is called **"deep"** when it has **more than one hidden layer** (modern ones have tens to hundreds).

### Layer math (matrix form)
For layer *l*:

```
Z[l] = W[l] · A[l-1] + b[l]
A[l] = f(Z[l])
```

Shapes (for a layer with `n_out` neurons and `n_in` inputs):
- `W`: (n_out, n_in)
- `b`: (n_out, 1)

Number of parameters in a dense layer = `n_in × n_out + n_out`.

---

## 6. Why non-linear activation functions?

If we stack layers **without** non-linearity:

```
W2 (W1 x) = (W2 W1) x   →   still just ONE linear transformation
```

So depth would be useless. Activation functions let the network model **curved, complex** decision boundaries.

| Activation | Formula | Range | Notes |
|---|---|---|---|
| **Sigmoid** | 1 / (1 + e^(−z)) | (0, 1) | Binary output; suffers vanishing gradient |
| **Tanh** | (e^z − e^−z)/(e^z + e^−z) | (−1, 1) | Zero-centred; still vanishing gradient |
| **ReLU** | max(0, z) | [0, ∞) | Default for hidden layers; fast; can "die" |
| **Leaky ReLU** | max(0.01z, z) | (−∞, ∞) | Fixes dying ReLU |
| **Softmax** | e^zi / Σ e^zj | (0, 1), sums to 1 | Multi-class output layer |

---

## 7. How does a network learn? (Big picture)

Training loop, repeated many times:

1. **Forward propagation**: pass inputs through the network → get prediction ŷ.
2. **Loss function**: measure the error between ŷ and the true y.
   - Regression: MSE
   - Binary classification: Binary Cross-Entropy
   - Multi-class: Categorical Cross-Entropy
3. **Backpropagation**: compute gradients of the loss w.r.t. every weight (chain rule).
4. **Optimizer step** (Gradient Descent / Adam): update weights
   `w ← w − η · ∂L/∂w`  (η = learning rate)

*(Each of these steps gets its own detailed day later in the roadmap.)*

---

## 8. Types of Neural Networks (preview)

| Network | Best for | Examples |
|---|---|---|
| **ANN / MLP** (Dense) | Tabular data, basics | Price prediction, churn |
| **CNN** | Images / spatial data | Face recognition, medical imaging |
| **RNN / LSTM / GRU** | Sequences / time series | Speech, stock forecasting |
| **Transformer** | Text, and now almost everything | ChatGPT, BERT, translation |
| **Autoencoder** | Compression, denoising, anomaly detection | Fraud detection |
| **GAN / Diffusion** | Generating new data | Image generation, deepfakes |

---

## 9. Real-world applications

- **Computer Vision**: face unlock, self-driving cars, X-ray diagnosis
- **NLP**: chatbots, translation, summarization
- **Speech**: voice assistants, speech-to-text
- **Recommendation**: YouTube, Netflix, Amazon
- **Healthcare**: drug discovery, disease detection
- **Generative AI**: text, image, music, code generation

---

## 10. Limitations / Challenges

- Needs **lots of data** and **compute**
- **Overfitting** on small datasets
- **Black box**: hard to explain decisions
- Sensitive to **data bias** and **adversarial examples**
- Hyperparameter tuning is hard (layers, neurons, learning rate...)

---

## 11. Key Takeaways

- DL = neural networks with multiple layers that **learn features automatically**.
- Each neuron: `a = f(w·x + b)`.
- **Non-linear activations** are what make depth useful.
- Training = forward pass → loss → backprop → weight update.
- DL beats classical ML on **unstructured data** (images, text, audio), while classical ML often wins on small tabular data.

## 12. Quick revision questions

1. Difference between ML and DL in terms of feature engineering?
2. What are weights and bias in a neuron?
3. Why can't we use only linear activation in all layers?
4. What makes a neural network "deep"?
5. Name the four steps of the training loop.

---
*Next topic: Perceptron.*

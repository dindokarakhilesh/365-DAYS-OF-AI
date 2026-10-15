# Day 95 — What is Deep Learning? Deep Learning vs Machine Learning

## 1. What is Deep Learning?

**Deep Learning** is a branch of Machine Learning that uses **multi-layer artificial neural networks** to learn
patterns from data. The key idea is **representation learning**: the model discovers the useful features by itself
instead of a human designing them.

```
Raw data  →  Layer 1  →  Layer 2  →  ...  →  Layer N  →  Prediction
              (simple      (combinations     (high-level
               patterns)    of patterns)      concepts)
```

Example for an image of a face:
- Early layers detect **edges and corners**
- Middle layers detect **eyes, nose, ears**
- Deep layers detect **a whole face**

This **hierarchical feature learning** is the main reason deep learning works so well on images, audio and text.

---

## 2. The relationship between AI, ML and DL

```
Artificial Intelligence      (machines that act intelligently)
   └── Machine Learning      (learn patterns from data)
          └── Deep Learning  (learn using deep neural networks)
```

---

## 3. Traditional Machine Learning workflow

```
Raw Data → Manual Feature Engineering → ML Algorithm → Output
              (done by a human)
```

Example: to classify **cat vs dog** images, an engineer would hand-craft features such as ear shape, fur texture,
colour histograms, edge counts, and then feed these numbers into an SVM or Random Forest.

**Problem:** finding good features needs **domain expertise**, is **time-consuming**, and often does not generalize.

## 4. Deep Learning workflow

```
Raw Data → [ Feature Extraction + Classification ] → Output
              (both learned together by the network)
```

The network is trained **end to end**: raw pixels go in, labels come out, and the network figures out the features.

---

## 5. Deep Learning vs Machine Learning (detailed)

| Aspect | Machine Learning | Deep Learning |
|---|---|---|
| **Feature extraction** | Manual (human engineered) | Automatic (learned) |
| **Data requirement** | Works with small to medium data | Usually needs large data |
| **Data type** | Best on structured / tabular data | Best on unstructured data (image, text, audio, video) |
| **Hardware** | CPU is usually enough | GPU / TPU recommended |
| **Training time** | Short | Long |
| **Prediction (inference) time** | Usually fast | Can be slow for big models |
| **Interpretability** | Often high (trees, linear models) | Low ("black box") |
| **Hyperparameter tuning** | Fewer hyperparameters | Many (layers, units, learning rate, batch size...) |
| **Performance as data grows** | Plateaus after some point | Keeps improving with more data |
| **Typical algorithms** | Linear/Logistic Regression, SVM, KNN, Decision Tree, Random Forest, XGBoost | ANN, CNN, RNN, LSTM, Transformer, GAN |

### Performance vs amount of data (concept)

```
Performance
   │                       ____________  Deep Learning (large network)
   │                  ____/
   │             ____/    ----------------  Traditional ML (plateaus)
   │        ____/---------
   │   ___/---
   └──────────────────────────────────────→ Amount of data
```

Traditional ML tends to **saturate**, while deep networks can keep **improving** as data and model size grow.
(This is a general trend, not a rule for every dataset.)

---

## 6. Why Deep Learning is popular now

1. **Data**: massive labelled datasets (ImageNet, web text, video)
2. **Hardware**: GPUs/TPUs make large matrix operations fast
3. **Algorithms**: ReLU, dropout, batch normalization, Adam optimizer, residual connections, attention
4. **Software**: TensorFlow, Keras, PyTorch make building models easy
5. **Pretrained models and transfer learning**: you can reuse models trained by others

---

## 7. Core building blocks of Deep Learning

| Component | Role |
|---|---|
| **Neuron** | Computes `a = f(w·x + b)` |
| **Layer** | A group of neurons (input, hidden, output) |
| **Weights & biases** | Learnable parameters |
| **Activation function** | Adds non-linearity (ReLU, Sigmoid, Tanh, Softmax) |
| **Loss function** | Measures prediction error (MSE, Cross-Entropy) |
| **Optimizer** | Updates weights (SGD, Adam, RMSprop) |
| **Backpropagation** | Computes gradients using the chain rule |
| **Epoch / Batch** | One full pass over the data / a chunk of data per update |

---

## 8. When should you use which?

**Prefer Machine Learning when:**
- Data is small or tabular (spreadsheet-like)
- You need an explainable model (finance, healthcare rules, compliance)
- Compute and time are limited
- A quick, strong baseline is needed

**Prefer Deep Learning when:**
- Data is images, audio, text or video
- You have large amounts of data (or a good pretrained model)
- Features are hard to design by hand
- Highest possible accuracy matters and you have GPUs

> **Good practice:** always start with a simple ML baseline. Move to deep learning only if the baseline is not enough.

---

## 9. Common deep learning architectures (overview)

| Architecture | Used for |
|---|---|
| **ANN / MLP** | Tabular data, basic prediction |
| **CNN** | Image classification, detection, segmentation |
| **RNN / LSTM / GRU** | Sequences, time series, older NLP |
| **Transformer** | Language models, translation, vision, multimodal |
| **Autoencoder** | Compression, denoising, anomaly detection |
| **GAN / Diffusion** | Image and media generation |

---

## 10. Applications

- **Vision:** face recognition, medical imaging, self-driving cars
- **Language:** chatbots, translation, summarization, search
- **Speech:** voice assistants, transcription
- **Recommendation:** video, music and shopping suggestions
- **Science:** protein structure prediction, drug discovery
- **Generative AI:** text, image, audio and code generation

---

## 11. Common misconceptions

1. *"Deep learning always beats machine learning."* No. On small tabular data, gradient boosting often wins.
2. *"Deep learning works like the human brain."* Neural networks are only loosely inspired by the brain.
3. *"More layers always means better."* Not always. Deeper models can overfit or be hard to train.
4. *"Deep learning needs no data preparation."* Cleaning, normalizing and splitting data still matter.

---

## 12. Key takeaways

- Deep learning = neural networks with many layers that **learn features automatically**.
- ML needs **manual feature engineering**, DL does **representation learning**.
- DL shines on **unstructured data** and **large datasets**, but needs **more compute** and is **harder to interpret**.
- Start with a simple baseline, then increase complexity only when needed.

## 13. Revision questions

1. Define deep learning in one sentence.
2. What is representation learning?
3. Give three differences between ML and DL.
4. Why does DL need GPUs?
5. When would you choose Random Forest over a neural network?
6. What does "end to end" learning mean?

---
*Next topic: continue the Deep Learning course roadmap.*

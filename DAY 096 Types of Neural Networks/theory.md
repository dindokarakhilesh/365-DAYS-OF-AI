# Day 96 — Types of Neural Networks | History of Deep Learning | Applications of Deep Learning

---

# Part 1: Types of Neural Networks

Different data needs different architectures. The main families are:

| Type | Full name | Data it handles | Core idea |
|---|---|---|---|
| **ANN / MLP** | Artificial Neural Network / Multi-Layer Perceptron | Tabular data | Fully connected layers |
| **CNN** | Convolutional Neural Network | Images, grids | Sliding filters detect local patterns |
| **RNN** | Recurrent Neural Network | Sequences | Hidden state carries memory across time steps |
| **LSTM / GRU** | Long Short-Term Memory / Gated Recurrent Unit | Long sequences | Gates control what to remember or forget |
| **Transformer** | Attention-based network | Text, images, audio, multimodal | Self-attention looks at all positions at once |
| **Autoencoder** | Encoder-decoder network | Compression, denoising, anomalies | Learn to reconstruct the input through a bottleneck |
| **GAN** | Generative Adversarial Network | Image/media generation | Generator vs discriminator game |
| **Diffusion model** | Denoising generative model | Image/audio/video generation | Learn to remove noise step by step |
| **GNN** | Graph Neural Network | Graphs, molecules, social networks | Nodes exchange messages with neighbours |

## 1. ANN / MLP (Multi-Layer Perceptron)
- Every neuron in one layer connects to every neuron in the next layer (**dense / fully connected**).
- Formula per layer: `a = f(W·x + b)`
- **Use for:** tabular data, simple regression and classification.
- **Limitation:** ignores spatial/sequential structure, and has too many parameters for raw images.

## 2. CNN (Convolutional Neural Network)
- Uses small **filters (kernels)** that slide over the input and detect local patterns (edges, textures, shapes).
- Key ideas: **local connectivity**, **weight sharing**, **pooling** (downsampling).
- Typical pipeline: `Image → Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Dense → Output`
- **Use for:** image classification, object detection, segmentation, medical imaging.
- **Famous models:** LeNet, AlexNet, VGG, ResNet, EfficientNet.

## 3. RNN (Recurrent Neural Network)
- Processes a sequence one step at a time and keeps a **hidden state** as memory:
  `h_t = f(W_x · x_t + W_h · h_(t-1) + b)`
- **Use for:** time series, text, speech.
- **Limitation:** **vanishing/exploding gradients** make long-range memory hard, and processing is sequential (slow).

## 4. LSTM and GRU
- Improved RNNs with **gates** (input, forget, output in LSTM) that control information flow.
- Handle **longer dependencies** than a basic RNN.
- **GRU** is a lighter version with fewer gates.
- **Use for:** machine translation (older systems), speech recognition, forecasting.

## 5. Transformer
- Replaces recurrence with **self-attention**: every token can look directly at every other token.
- Core formula: `Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V`
- Processes all positions **in parallel**, so training scales very well on GPUs.
- **Use for:** language models, translation, summarization, vision transformers, speech, multimodal models.
- **Famous models:** BERT, GPT family, T5, ViT.

## 6. Autoencoder
```
Input → Encoder → Latent code (bottleneck) → Decoder → Reconstruction
```
- Trained to reproduce its own input, which forces it to learn a compact representation.
- **Use for:** dimensionality reduction, denoising, anomaly detection (high reconstruction error = anomaly).
- **Variants:** denoising AE, sparse AE, **VAE** (variational autoencoder, which can also generate data).

## 7. GAN (Generative Adversarial Network)
- Two networks compete:
  - **Generator** creates fake data from noise
  - **Discriminator** tries to tell real from fake
- Both improve through the competition.
- **Use for:** image synthesis, style transfer, data augmentation.
- **Challenge:** training can be unstable (mode collapse).

## 8. Diffusion models
- Add noise to data step by step, then train a network to **reverse** the process.
- Generate new samples by starting from pure noise and denoising.
- **Use for:** modern image, audio and video generation.

## 9. Graph Neural Networks (GNN)
- Operate on graphs: each node updates its representation using its neighbours.
- **Use for:** molecules and drug discovery, recommendation, social networks, road networks.

### Which architecture for which task?

| Task | Typical choice |
|---|---|
| Predict house price from a table | MLP (or gradient boosting) |
| Classify images | CNN / Vision Transformer |
| Detect objects in video | CNN-based detectors / Transformers |
| Forecast time series | LSTM / GRU / Transformer |
| Translate or summarize text | Transformer |
| Detect fraud or anomalies | Autoencoder |
| Generate images | Diffusion / GAN |
| Predict molecule properties | GNN |

---

# Part 2: History of Deep Learning

| Year | Milestone |
|---|---|
| **1943** | McCulloch and Pitts propose the first mathematical model of a neuron |
| **1958** | Frank Rosenblatt invents the **Perceptron** |
| **1969** | Minsky and Papert show single-layer perceptrons cannot solve XOR, interest drops (**first AI winter**) |
| **1974 / 1986** | **Backpropagation** is described (Werbos) and popularized (Rumelhart, Hinton, Williams) |
| **1989** | Yann LeCun applies a CNN with backprop to handwritten zip codes |
| **1997** | Hochreiter and Schmidhuber introduce the **LSTM** |
| **1998** | **LeNet-5** for digit recognition |
| **Late 1980s-1990s** | Limited data and compute, other methods (SVMs) dominate (**second AI winter**) |
| **2006** | Hinton and colleagues revive deep networks with layer-wise pretraining ("deep learning" term spreads) |
| **2009** | **ImageNet** dataset released |
| **2012** | **AlexNet** wins ImageNet by a large margin using GPUs, the modern boom begins |
| **2014** | **GANs** (Goodfellow) and attention for translation (Bahdanau et al.) |
| **2015** | **ResNet** enables very deep networks (skip connections), Batch Normalization, Adam optimizer |
| **2016** | AlphaGo defeats a world champion Go player |
| **2017** | **Transformer** architecture ("Attention Is All You Need") |
| **2018** | **BERT** and early GPT models, pretrained language models become standard |
| **2020** | GPT-3 shows few-shot abilities at large scale |
| **2021** | AlphaFold 2 solves protein structure prediction at high accuracy |
| **2022 onwards** | Diffusion image models and ChatGPT-style assistants reach the public, generative AI boom |

### Why did deep learning succeed after 2012?
1. Large datasets (ImageNet, web-scale text)
2. GPU computing
3. Algorithmic fixes (ReLU, dropout, better initialization, batch norm, Adam)
4. Open-source frameworks (Theano, TensorFlow, PyTorch)

### Lessons from history
- Ideas existed decades before they worked at scale. **Data + compute + algorithms** all had to mature.
- Progress came in waves (hype, disappointment, revival).

---

# Part 3: Applications of Deep Learning

| Domain | Applications | Typical architecture |
|---|---|---|
| **Computer Vision** | Face recognition, object detection, self-driving perception, medical image diagnosis, OCR | CNN, ViT |
| **Natural Language Processing** | Chatbots, translation, summarization, sentiment analysis, search | Transformer |
| **Speech and Audio** | Voice assistants, speech-to-text, text-to-speech, music generation | CNN/RNN/Transformer |
| **Recommendation systems** | Video, music, shopping, news feeds | MLP, embeddings, GNN |
| **Healthcare** | Disease detection from scans, drug discovery, protein folding | CNN, GNN, Transformer |
| **Finance** | Fraud detection, risk scoring, algorithmic trading signals | MLP, LSTM, autoencoder |
| **Robotics and Autonomous systems** | Navigation, grasping, drone control, reinforcement learning | CNN + RL |
| **Generative AI** | Text, image, audio, video and code generation | Transformer, Diffusion, GAN |
| **Science and Engineering** | Weather forecasting, materials discovery, physics simulation | GNN, Transformer |
| **Agriculture and Industry** | Crop disease detection, defect inspection, predictive maintenance | CNN, LSTM |

### Challenges and ethics
- **Data bias** can lead to unfair predictions
- **Privacy** concerns with personal data
- **Explainability** of black-box models
- **Misuse** such as deepfakes and misinformation
- **Energy cost** of training very large models

---

## Key takeaways

- Choose the architecture by the **structure of the data**: tables (MLP), grids (CNN), sequences (RNN/LSTM/Transformer), graphs (GNN).
- **Transformers** dominate modern language and many other tasks, **CNNs** remain strong for vision.
- Deep learning grew from the 1940s perceptron to today's generative AI, with big jumps in **2012 (AlexNet)** and **2017 (Transformer)**.
- Applications span vision, language, speech, healthcare, finance, science and generative AI.

## Revision questions

1. Which architecture would you pick for images, for time series and for text, and why?
2. What problem do LSTM gates solve compared to a basic RNN?
3. Write the self-attention formula and explain each term.
4. Why did the first AI winter happen?
5. Name three factors behind the deep learning boom after 2012.
6. How does an autoencoder detect anomalies?
7. What are the generator and discriminator in a GAN?

---
*Next topic: continue the Deep Learning course roadmap.*

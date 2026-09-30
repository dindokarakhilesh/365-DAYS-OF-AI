# Introduction to Natural Language Processing (NLP)
### NLP Engineering Bootcamp: From Basics to Advanced

---

## 1. What is NLP?

**Natural Language Processing (NLP)** is the field of AI that enables computers to **understand, interpret, generate, and respond to human language** (text and speech). It sits at the intersection of **linguistics**, **computer science**, and **machine learning**.

Human language is hard for machines because it is:
- **Ambiguous**: "I saw a man with a telescope" (who has the telescope?)
- **Context-dependent**: "bank" (river bank vs. money bank)
- **Noisy and irregular**: slang, typos, emojis, code-mixing ("Kal meeting hai, don't be late")
- **Creative**: sarcasm, idioms, metaphors

### Two sides of NLP
| Side | Meaning | Examples |
|---|---|---|
| **NLU** (Understanding) | Extract meaning from text | Sentiment analysis, intent detection, NER |
| **NLG** (Generation) | Produce human-like text | Chatbots, summarization, translation |

---

## 2. Real-World Applications

- **Search engines** and question answering
- **Machine translation** (Google Translate)
- **Chatbots and virtual assistants** (customer support, Siri, Alexa)
- **Sentiment analysis** (product reviews, social media monitoring)
- **Spam / fake-news detection**
- **Text summarization** (news, legal and medical documents)
- **Information extraction** (resumes, invoices, contracts)
- **Speech recognition and text-to-speech**
- **Code generation and assistants**

---

## 3. Levels of Language Analysis (The NLP Stack)

| Level | Question it answers | Example task |
|---|---|---|
| **Lexical / Morphological** | What are the words and their forms? | Tokenization, stemming, lemmatization |
| **Syntactic** | How are words arranged? | POS tagging, parsing |
| **Semantic** | What does it mean? | Word sense disambiguation, NER, embeddings |
| **Discourse** | How do sentences relate? | Coreference resolution, summarization |
| **Pragmatic** | What is the intent in context? | Dialogue, sarcasm detection |

---

## 4. The Classical NLP Pipeline

```
Raw Text -> Cleaning -> Tokenization -> Normalization -> Feature Extraction -> Model -> Evaluation -> Deployment
```

### 4.1 Text Cleaning
- Lowercasing, removing HTML tags, URLs, extra whitespace
- Handling punctuation, numbers, emojis (remove or convert, depending on the task)
- Fixing encodings and contractions ("don't" -> "do not")

### 4.2 Tokenization
Splitting text into smaller units called **tokens**.
- **Word tokenization**: "I love NLP" -> `["I", "love", "NLP"]`
- **Sentence tokenization**: splitting a paragraph into sentences
- **Subword tokenization** (BPE, WordPiece, SentencePiece): "unhappiness" -> `["un", "happi", "ness"]`. Used by modern models (BERT, GPT) to handle rare words and reduce vocabulary size.
- **Character tokenization**: each character is a token.

### 4.3 Stop-Word Removal
Removing very common words ("the", "is", "and") that carry little meaning for many tasks. **Caution**: for tasks like sentiment analysis, words like "not" are critical, so removing them can flip the meaning.

### 4.4 Stemming vs. Lemmatization
| | Stemming | Lemmatization |
|---|---|---|
| Method | Chops suffixes with rules | Uses a dictionary and morphology |
| Output | May not be a real word ("studies" -> "studi") | Valid base word ("studies" -> "study") |
| Speed | Fast | Slower |
| Examples | Porter, Snowball | WordNet, spaCy |

### 4.5 POS Tagging and NER
- **Part-of-Speech (POS) tagging**: label each word as noun, verb, adjective, etc.
- **Named Entity Recognition (NER)**: find entities such as PERSON, ORG, LOCATION, DATE.

---

## 5. Text Representation (Turning Words into Numbers)

Machine learning models need numbers, not strings. This is the heart of NLP engineering.

### 5.1 Bag of Words (BoW)
Represent each document as a vector of word counts over a fixed vocabulary.
- Simple and effective baseline.
- **Drawbacks**: ignores word order and meaning, sparse and high-dimensional.

### 5.2 N-grams
Sequences of `n` consecutive tokens (unigram, bigram, trigram). Bigrams like "not good" partly capture local word order and negation.

### 5.3 TF-IDF (Term Frequency, Inverse Document Frequency)
Weights words by how important they are to a document relative to the whole corpus:

```
TF(t, d)  = (count of t in d) / (total terms in d)
IDF(t)    = log( N / (1 + df(t)) )        # N = number of documents, df = documents containing t
TF-IDF    = TF * IDF
```

- Common words (appearing in many documents) get **low** weight.
- Distinctive words get **high** weight.

### 5.4 Word Embeddings
Dense, low-dimensional vectors where **similar words have similar vectors**.
- **Word2Vec** (CBOW, Skip-gram), **GloVe**, **FastText**.
- Famous example: `king - man + woman ~ queen`.
- Based on the **distributional hypothesis**: "You shall know a word by the company it keeps."
- **Limitation**: one vector per word, regardless of context ("bank" always has the same vector).

### 5.5 Contextual Embeddings
Modern models (ELMo, BERT, GPT) produce a **different vector for a word depending on its sentence**, solving the "bank" problem.

### 5.6 Similarity Measures
- **Cosine similarity**: `cos(A, B) = (A . B) / (||A|| ||B||)`. Most common for text vectors.
- Euclidean distance, Jaccard similarity (set overlap).

---

## 6. Classical NLP Tasks and Models

### 6.1 Text Classification
Assign a label to a document (spam/ham, positive/negative, topic).
- Models: **Naive Bayes**, **Logistic Regression**, **SVM** on BoW / TF-IDF features.
- Metrics: accuracy, precision, recall, F1, confusion matrix.

### 6.2 Language Modeling
Predict the probability of a sequence of words / the next word.
- **N-gram language models**: `P(w_n | w_{n-1}, ..., w_{n-k+1})` estimated from counts.
- **Perplexity** measures how well a model predicts text (lower is better).

### 6.3 Sequence Labeling
Assign a label to each token (POS tagging, NER). Classical models: **HMM, CRF**.

### 6.4 Topic Modeling
Discover hidden topics in a corpus: **LDA**, **NMF**, **LSA (SVD on TF-IDF)**.

---

## 7. Deep Learning for NLP

| Model | Key idea | Limitation |
|---|---|---|
| **RNN** | Processes tokens sequentially with a hidden state | Vanishing gradients, slow, forgets long context |
| **LSTM / GRU** | Gates control what to remember/forget | Still sequential, hard to parallelize |
| **Seq2Seq + Attention** | Encoder-decoder; decoder "attends" to relevant input words | Still built on RNNs |
| **Transformer** | Self-attention over all tokens in parallel | Quadratic cost in sequence length |

### The Transformer (the foundation of modern NLP)
Introduced in *"Attention Is All You Need"* (2017).
- **Self-attention**: every token looks at every other token to decide what matters. `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`
- **Multi-head attention**, **positional encodings** (to inject word order), **feed-forward layers**, **residual connections and layer norm**.
- Highly parallelizable, so it scales to huge datasets.

---

## 8. Pretrained Language Models and Transfer Learning

Instead of training from scratch, we **pretrain** on massive text, then **fine-tune** on a specific task.

| Family | Architecture | Pretraining objective | Good for |
|---|---|---|---|
| **BERT** (and RoBERTa, DistilBERT) | Encoder-only | Masked Language Modeling | Classification, NER, QA, embeddings |
| **GPT** family | Decoder-only | Next-token prediction | Text generation, chat, reasoning |
| **T5 / BART** | Encoder-decoder | Text-to-text / denoising | Translation, summarization |

### Large Language Models (LLMs)
- Very large decoder-based Transformers trained on internet-scale text.
- Adapted with **instruction tuning** and **RLHF / preference optimization** to follow instructions.
- Techniques: **prompt engineering**, **few-shot learning**, **fine-tuning (LoRA / PEFT)**, **RAG (Retrieval-Augmented Generation)**, **agents and tool use**.

---

## 9. Evaluation Metrics

| Task | Common metrics |
|---|---|
| Classification | Accuracy, Precision, Recall, F1, ROC-AUC |
| Sequence labeling (NER) | Entity-level F1 |
| Machine translation | BLEU, chrF, COMET |
| Summarization | ROUGE, BERTScore |
| Language modeling | Perplexity |
| Generation / LLMs | Human evaluation, LLM-as-judge, task benchmarks |

---

## 10. NLP Engineering: From Notebook to Production

1. **Define the problem and success metric** (business goal, not just accuracy).
2. **Collect and label data**; check class imbalance and data quality.
3. **Build a baseline first** (TF-IDF + Logistic Regression), then try heavier models only if needed.
4. **Error analysis**: look at what the model gets wrong.
5. **Deploy**: REST API (FastAPI), batching, caching, model quantization for latency and cost.
6. **Monitor**: data drift, latency, failure cases; retrain periodically.
7. **Responsible AI**: bias, fairness, privacy, hallucinations, safety.

### Common Tools and Libraries
`NLTK`, `spaCy`, `scikit-learn`, `Gensim`, `Hugging Face Transformers / Datasets / Tokenizers`, `PyTorch`, `TensorFlow`, `LangChain / LlamaIndex`, vector databases (FAISS, Chroma, Pinecone).

---

## 11. Suggested Bootcamp Roadmap

| Stage | Topics |
|---|---|
| **1. Foundations** | Python, regex, text cleaning, tokenization, stemming/lemmatization |
| **2. Classical NLP** | BoW, TF-IDF, n-grams, Naive Bayes, text classification, topic modeling |
| **3. Embeddings** | Word2Vec, GloVe, FastText, similarity search |
| **4. Deep Learning** | RNN, LSTM, Seq2Seq, attention |
| **5. Transformers** | Self-attention, BERT, GPT, Hugging Face fine-tuning |
| **6. LLM Engineering** | Prompting, RAG, fine-tuning (LoRA), evaluation, agents |
| **7. Production** | APIs, optimization, monitoring, MLOps, responsible AI |

---

## 12. Challenges in NLP
- Ambiguity and context understanding
- Low-resource languages (including many Indian languages) and code-mixing
- Bias and toxicity inherited from training data
- Hallucinations in generative models
- Evaluation of open-ended generation
- Compute cost and latency

# Word Embeddings in NLP: Frequency-Based vs Prediction-Based

## 1. Why word embeddings?

Computers cannot work with raw words, so we need to turn them into numbers. The simplest idea is to give each word an integer ID or a one-hot vector. This has two big problems:

- **No notion of similarity.** "king" and "queen" are exactly as far apart as "king" and "banana".
- **Huge, sparse vectors.** A vocabulary of 100,000 words gives 100,000-dimensional vectors that are almost entirely zeros.

A **word embedding** is a **dense, low-dimensional vector** (typically 50 to 300 dimensions) in which **similar words lie close together**.

### The idea behind all of them: the Distributional Hypothesis

> "You shall know a word by the company it keeps." (J. R. Firth, 1957)

Words that appear in similar contexts have similar meanings. *Every* embedding method below is a different way of turning "context" into numbers.

| | Frequency-based (count-based) | Prediction-based (neural) |
|---|---|---|
| **Core idea** | Count co-occurrences, reweight, then compress | Train a model to predict words from context |
| **Statistics used** | Global (whole corpus at once) | Local (one window at a time) |
| **Training** | Matrix building plus factorisation (deterministic) | Gradient descent (stochastic) |
| **Examples** | BoW, TF-IDF, Co-occurrence, PPMI, LSA/SVD | Word2Vec (Skip-gram, CBOW), FastText |
| **Hybrid** | GloVe: global counts, optimised with a gradient-based loss | |

---

## 2. Frequency-Based Embeddings

### 2.1 One-Hot Encoding

For a vocabulary of size `V`, word *i* becomes a vector with a 1 in position *i* and 0 elsewhere.

```
cat   = [1, 0, 0, 0]
dog   = [0, 1, 0, 0]
apple = [0, 0, 1, 0]
```

- ✅ Simple, no training
- ❌ Dimension equals `V`
- ❌ Every pair of words has cosine similarity 0, so there is no semantics

### 2.2 Bag of Words (BoW)

Represents a **document** as a vector of word counts and ignores word order.

| Document | the | cat | sat | dog | ran |
|---|---|---|---|---|---|
| "the cat sat" | 1 | 1 | 1 | 0 | 0 |
| "the dog ran" | 1 | 0 | 0 | 1 | 1 |

- ✅ Strong baseline for text classification
- ❌ Loses word order ("dog bites man" is the same as "man bites dog")
- ❌ Common words ("the") dominate
- ❌ Vectors are sparse and high-dimensional

> Transposing the term-document matrix gives a crude *word* vector, with each word described by the documents it appears in.

### 2.3 TF-IDF (Term Frequency - Inverse Document Frequency)

Fixes BoW's "common words dominate" problem by weighting each word by how **informative** it is.

$$\text{tf}(t,d) = \text{count of } t \text{ in } d \qquad
\text{idf}(t) = \log\frac{N}{1+\text{df}(t)}$$

$$\text{tfidf}(t,d) = \text{tf}(t,d)\times \text{idf}(t)$$

- `N`: number of documents
- `df(t)`: number of documents containing `t`

**Intuition:** a word that appears in *every* document ("the") gets an idf near 0. A word that appears in only a few documents ("palace") gets a high weight.

- ✅ Great for search and retrieval, keyword extraction and classification
- ❌ Still sparse, still no real semantic similarity ("car" and "automobile" remain unrelated)

### 2.4 Word-Word Co-occurrence Matrix

Pick a **window size** `w`. Build a `V x V` matrix `C` where `C[i][j]` is how many times word *j* appears within `w` words of word *i*.

Example with window = 1 on *"I like deep learning. I like NLP."*

| | I | like | deep | learning | NLP |
|---|---|---|---|---|---|
| I | 0 | 2 | 0 | 0 | 0 |
| like | 2 | 0 | 1 | 0 | 1 |
| deep | 0 | 1 | 0 | 1 | 0 |
| learning | 0 | 0 | 1 | 0 | 0 |
| NLP | 0 | 1 | 0 | 0 | 0 |

Each **row** is a word vector. Words with similar rows (similar contexts) are similar in meaning. This is the first representation that really encodes semantics.

Variants: weight neighbours by distance (e.g. `1/distance`), or use asymmetric windows (left vs right context).

- ✅ Captures meaning through shared context
- ❌ `V x V` is huge, sparse and memory-hungry
- ❌ Raw counts are dominated by frequent words ("the", "of")

### 2.5 PPMI: Reweighting the Counts

**Pointwise Mutual Information** asks: *do words `w` and `c` co-occur more often than they would by chance?*

$$\text{PMI}(w,c) = \log\frac{P(w,c)}{P(w)\,P(c)}$$

Negative PMI values are unreliable, because they require huge corpora to estimate, so we clip them:

$$\text{PPMI}(w,c) = \max\big(\text{PMI}(w,c),\,0\big)$$

**Context distribution smoothing:** raise context counts to the power α = 0.75 before normalising. This reduces PMI's bias toward rare words.

- ✅ Much better than raw counts, because it removes the frequency bias
- ❌ Still `V x V` and sparse

### 2.6 Dimensionality Reduction: SVD and LSA

Compress the (PPMI) matrix `M` of shape `V x V` into short dense vectors with **Truncated Singular Value Decomposition**:

$$M \approx U_k\,\Sigma_k\,V_k^{\top}$$

Keep only the top-`k` singular values (e.g. `k = 100`). The embedding of word *i* is row *i* of

$$E = U_k\,\Sigma_k^{1/2} \quad(\text{or simply } U_k)$$

When applied to the **term-document** matrix, this is **Latent Semantic Analysis (LSA)**. When applied to a **word-word PPMI** matrix, it is the strong count-based baseline described by Levy & Goldberg (2014).

**Why does it work?** The small singular values mostly capture noise, while the large ones capture the dominant patterns. Words that share patterns end up close together, even if they never co-occurred directly (like synonyms).

- ✅ Dense, low-dimensional, deterministic
- ✅ Uses global statistics efficiently
- ❌ SVD is `O(V^3)`-ish, which is expensive for huge vocabularies
- ❌ Hard to add new words or new data (you must recompute)

---

## 3. Prediction-Based Embeddings

Instead of counting and then compressing, we **train a shallow neural network on a prediction task** and keep its weights as the embeddings. The prediction task itself is a throwaway. The *by-product* (the weights) is what we want.

### 3.1 Word2Vec (Mikolov et al., 2013)

Two architectures, both with **one hidden layer and no non-linearity**:

```
Skip-gram:  center word  --->  predict the surrounding words
CBOW:       surrounding words  --->  predict the center word
```

For the sentence *"the quick brown **fox** jumps over"* with window = 2:

- **Skip-gram** creates pairs: (fox → quick), (fox → brown), (fox → jumps), (fox → over)
- **CBOW** creates one example: [quick, brown, jumps, over] → fox

#### Architecture

Two embedding matrices, each `V x d`:

- `W` (input / "center" vectors `v`)
- `W'` (output / "context" vectors `u`)

The probability of context word `o` given center word `c` is a softmax:

$$P(o\mid c) = \frac{\exp(u_o^{\top} v_c)}{\sum_{w=1}^{V}\exp(u_w^{\top} v_c)}$$

Training maximises the log-likelihood of true (center, context) pairs.

#### Skip-gram vs CBOW

| | Skip-gram | CBOW |
|---|---|---|
| Input to model | 1 center word | Average of context vectors |
| Predicts | Each context word | The center word |
| Training examples per window | Many (one per context word) | One |
| Rare words | ✅ Handled better | ❌ Smoothed away |
| Speed | Slower | Faster |
| Best for | Smaller datasets, rare words | Large datasets, frequent words |

#### The softmax problem and Negative Sampling

The denominator sums over the *entire vocabulary* (100k+ words) at every step, which is far too slow. Two fixes exist:

1. **Hierarchical softmax:** replace the flat softmax with a binary tree and use `O(log V)` operations.
2. **Negative sampling (most popular):** turn the problem into **binary classification**. Is this (center, context) pair real, or was the context word picked at random?

$$\mathcal{L} = -\log\sigma(u_o^{\top}v_c) \;-\; \sum_{k=1}^{K}\log\sigma(-u_{n_k}^{\top}v_c)$$

- `σ` is the sigmoid function
- `K` is the number of negative samples (5 to 20 for small data, 2 to 5 for large)
- Negatives `n_k` are sampled from the unigram distribution to the power 0.75:

$$P_n(w) \propto \text{count}(w)^{3/4}$$

This lowers the chance of picking very frequent words and raises the chance of picking rare ones.

**Gradients (for SGNS)** with `p = σ(u·v)` and label `y ∈ {0,1}`:

$$\frac{\partial \mathcal{L}}{\partial v_c} = \sum (p - y)\,u,\qquad
\frac{\partial \mathcal{L}}{\partial u} = (p - y)\,v_c$$

#### Other tricks

- **Subsampling of frequent words:** drop word `w` with probability `1 - sqrt(t / f(w))` (`t ≈ 1e-5`), so "the" appears less often in training.
- **Dynamic window:** sample the window size randomly from `1..w`, which weights nearby words more.

#### The famous analogy property

$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$

Directions in the space encode relations (gender, tense, capital-of). Note that this needs a *large* training corpus to emerge.

#### Neat theoretical result

Levy & Goldberg (2014) proved that **Skip-gram with negative sampling implicitly factorises a shifted PMI matrix**:

$$u_o^{\top} v_c \approx \text{PMI}(o,c) - \log K$$

So the two families are closer than they first appear: Word2Vec is, in effect, implicit matrix factorisation.

### 3.2 GloVe (Pennington et al., 2014): The Hybrid

GloVe keeps Word2Vec's "learn by gradient descent" but trains on **global co-occurrence counts** `X`. The goal is for the dot product of two vectors to match the log of their co-occurrence count:

$$J = \sum_{i,j} f(X_{ij})\,\big(w_i^{\top}\tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\big)^2$$

with a weighting function that caps the influence of very frequent pairs and ignores very rare ones:

$$f(x)=\begin{cases}(x/x_{max})^{\alpha} & x<x_{max}\\ 1 & \text{otherwise}\end{cases}\qquad x_{max}=100,\ \alpha=0.75$$

The final embedding is usually `w + w~`.

### 3.3 FastText (Bojanowski et al., 2017): Subword Embeddings

Word2Vec treats each word as an atomic unit, so it cannot handle **unseen words** or morphology. FastText represents a word as the sum of its **character n-gram** vectors.

For `where` with n = 3, the n-grams are `<wh, whe, her, ere, re>`, plus the whole word `<where>`.

- ✅ Handles out-of-vocabulary words and misspellings
- ✅ Great for morphologically rich languages (Hindi, Turkish, Finnish, German)
- ❌ Bigger model, slower to train

---

## 4. Head-to-Head Comparison

| Aspect | One-Hot / BoW / TF-IDF | Co-occurrence + PPMI + SVD | Word2Vec | GloVe | FastText |
|---|---|---|---|---|---|
| Type | Frequency | Frequency | Prediction | Hybrid | Prediction |
| Dense? | ❌ | ✅ | ✅ | ✅ | ✅ |
| Dimension | `V` | 50 to 500 | 50 to 300 | 50 to 300 | 50 to 300 |
| Semantic similarity | ❌ | ✅ | ✅ | ✅ | ✅ |
| Uses global statistics | ✅ | ✅ | ❌ (local) | ✅ | ❌ (local) |
| Handles OOV words | ❌ | ❌ | ❌ | ❌ | ✅ |
| Incremental training | ✅ | ❌ | ✅ | ❌ | ✅ |
| Scales to huge corpora | ✅ | ❌ (SVD cost) | ✅✅ | ✅ | ✅ |
| Typical use | Search, classification | Small corpora, baselines | General NLP features | General NLP features | Rare words, morphology |

### Which should I use?

- **Document classification or search** → TF-IDF (hard to beat as a baseline)
- **Small corpus, need fast and deterministic results** → PPMI + SVD
- **General-purpose word vectors** → Word2Vec skip-gram, or pre-trained GloVe/FastText
- **Many rare words, typos or morphology** → FastText
- **State-of-the-art NLP** → contextual embeddings (BERT, GPT)

---

## 5. Limitations of Static Embeddings

All methods here give **one vector per word**, regardless of context:

- **Polysemy:** "bank" (river) and "bank" (money) share a single vector.
- **Bias:** vectors absorb stereotypes from training text (e.g. gender-occupation associations).
- **No sentence-level meaning:** averaging word vectors is a weak sentence representation.

**Contextual embeddings** (ELMo, BERT, GPT) solve the first problem. The vector for "bank" is computed *from the whole sentence*, so it changes with context.

---

## 6. Evaluating Embeddings

- **Intrinsic**
  - *Word similarity:* correlate cosine similarity with human ratings (WordSim-353, SimLex-999)
  - *Analogy tasks:* Google analogy dataset (`king:queen :: man:woman`)
  - *Clustering and visualisation:* t-SNE, PCA, UMAP
- **Extrinsic**
  - Plug the embeddings into a downstream task (NER, sentiment, QA) and measure accuracy

**Cosine similarity** is the standard way to compare two vectors:

$$\cos(a,b) = \frac{a\cdot b}{\lVert a\rVert\,\lVert b\rVert}$$

---

## 7. Practical Tips

1. **Preprocess consistently:** lowercase, tokenise, optionally remove rare words (`min_count`).
2. **Window size matters:** small windows (2 to 5) capture *syntactic/functional* similarity (e.g. "dog" with "cat"). Large windows (5 to 10+) capture *topical* similarity (e.g. "dog" with "leash").
3. **More data beats fancier algorithms.** Word2Vec on 1M sentences usually beats a clever method on 10k.
4. **Embedding dimension:** 100 to 300 is a good default. Gains flatten beyond that.
5. **Mean-centre or normalise** vectors before cosine comparisons, since a shared "frequency direction" can inflate all similarities.
6. **Use pre-trained vectors** (GloVe, FastText, word2vec-google-news) when you have little data.

---

## 8. Quick Reference: Formulas

| Concept | Formula |
|---|---|
| TF-IDF | $\text{tf}(t,d)\cdot\log\frac{N}{1+\text{df}(t)}$ |
| PMI | $\log\frac{P(w,c)}{P(w)P(c)}$ |
| PPMI | $\max(\text{PMI},0)$ |
| SVD embedding | $E=U_k\Sigma_k^{1/2}$ |
| Skip-gram softmax | $P(o\mid c)=\frac{e^{u_o^\top v_c}}{\sum_w e^{u_w^\top v_c}}$ |
| SGNS loss | $-\log\sigma(u_o^\top v_c)-\sum_k\log\sigma(-u_{n_k}^\top v_c)$ |
| Negative sampling dist. | $P_n(w)\propto\text{count}(w)^{0.75}$ |
| GloVe loss | $\sum f(X_{ij})(w_i^\top\tilde w_j+b_i+\tilde b_j-\log X_{ij})^2$ |
| Cosine similarity | $\frac{a\cdot b}{\lVert a\rVert\lVert b\rVert}$ |

---

## 9. References

- Firth, J. R. (1957). *A synopsis of linguistic theory.*
- Deerwester et al. (1990). *Indexing by Latent Semantic Analysis.*
- Mikolov et al. (2013). *Efficient Estimation of Word Representations in Vector Space.* (Word2Vec)
- Mikolov et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* (Negative sampling)
- Pennington, Socher & Manning (2014). *GloVe: Global Vectors for Word Representation.*
- Levy & Goldberg (2014). *Neural Word Embedding as Implicit Matrix Factorization.*
- Levy, Goldberg & Dagan (2015). *Improving Distributional Similarity with Lessons Learned from Word Embeddings.*
- Bojanowski et al. (2017). *Enriching Word Vectors with Subword Information.* (FastText)

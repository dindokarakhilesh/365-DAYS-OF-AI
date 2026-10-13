# Word2Vec: CBOW vs Skip-Gram

*Explanation, implementation notes and NLP-engineering guidance, with a real-life e-commerce example.*

---

## 1. The real-life problem

You are an NLP engineer at **ShopSmart**, an online store. A customer searches:

> `cheap notebook for college`

The catalogue contains:

> `budget laptop with fast processor and long battery for students`

A keyword search finds **zero overlap** between the two: *cheap ≠ budget*, *notebook ≠ laptop*. Customers see "no results" and leave.

You have no labelled data and no dictionary of synonyms, but you do have **thousands of product reviews and descriptions**. Can the machine discover on its own that *cheap, affordable* and *budget* mean the same thing?

**Yes. That is what Word2Vec does.**

---

## 2. The core idea

> *"You shall know a word by the company it keeps."* (J. R. Firth)

Look at how customers write:

```
this budget laptop has a fast processor
this cheap laptop has a fast processor
this affordable laptop has a fast processor
```

*budget, cheap* and *affordable* appear in **identical contexts**. Word2Vec turns that observation into a **dense vector per word** so that words used in similar contexts get **nearby vectors**.

Word2Vec is a *shallow neural network* trained on a **fake prediction task**. We don't care about its predictions. The **weights it learns are the embeddings**.

| Without embeddings (one-hot / keywords) | With Word2Vec |
|---|---|
| `cheap` and `budget` are unrelated | `cheap` and `budget` have cosine similarity ≈ 0.95 |
| Vectors have 100,000+ dimensions | Vectors have 100 to 300 dimensions |
| No generalisation | Generalises to unseen combinations |

---

## 3. The two architectures

Take the sentence **"the budget laptop has a fast processor"**, centre word = **laptop**, window = 2.

```
          context            centre           context
   [the] [budget]  ──────►  LAPTOP  ◄──────  [has] [a] [fast] [processor]
```

### 3.1 Skip-Gram: *centre → context*

> "Given **laptop**, predict the words around it."

Training pairs from one window:

```
(laptop → the)   (laptop → budget)   (laptop → has)   (laptop → a)
```

### 3.2 CBOW (Continuous Bag of Words): *context → centre*

> "Given [the, budget, has, a], predict the missing word."

One training example per window:

```
[the, budget, has, a]  →  laptop
```

The context vectors are **averaged**, so word order inside the window is ignored (hence "bag of words").

### Side by side

```
 CBOW                                   SKIP-GRAM

 the  ─┐                                         ┌─► the
 budget─┤                                        ├─► budget
        ├─► average ─► [hidden] ─► laptop laptop ─┤
 has  ──┤                                        ├─► has
 a    ──┘                                        └─► a
```

---

## 4. The mathematics

### 4.1 Parameters

Two matrices of size `V × d` (`V` = vocabulary size, `d` = embedding dimension):

- **`W_in`**: row `v_w` is the vector of word `w` **as a centre/input** word. *This is usually the embedding we keep.*
- **`W_out`**: row `u_w` is the vector of word `w` **as a context/output** word.

Using two vectors per word makes the maths simpler. A common trick is to use `W_in + W_out` (or just `W_in`) as the final embedding.

### 4.2 Skip-Gram with the full softmax

The probability that word `o` appears in the context of centre word `c`:

$$P(o\mid c)=\frac{\exp(u_o^\top v_c)}{\sum_{w=1}^{V}\exp(u_w^\top v_c)}$$

Training maximises the log-likelihood over all (centre, context) pairs:

$$\mathcal{J}=-\frac1T\sum_{t=1}^{T}\;\sum_{-m\le j\le m,\;j\ne0}\log P(w_{t+j}\mid w_t)$$

### 4.3 CBOW with the full softmax

Average the context vectors into a hidden vector, then score every candidate centre word:

$$h=\frac{1}{2m}\sum_{-m\le j\le m,\;j\ne0}v_{w_{t+j}},\qquad P(w_t\mid \text{ctx})=\frac{\exp(u_{w_t}^\top h)}{\sum_{w}\exp(u_w^\top h)}$$

### 4.4 The problem: the softmax is too slow

The denominator sums over **every word in the vocabulary**. With `V = 1,000,000`, each training step needs 1,000,000 dot products. Two solutions:

| Technique | Idea | Cost per step |
|---|---|---|
| **Hierarchical softmax** | Put words in a Huffman tree; predict the path | `O(log V)` |
| **Negative sampling** (most popular) | Binary classification: real pair or fake pair? | `O(K)`, with `K` ≈ 5 to 20 |

### 4.5 Negative sampling

For a **real** pair `(c, o)` we want the model to say "yes". We also draw `K` **random "noise" words** `n_1 … n_K` and want the model to say "no" for `(c, n_k)`:

$$\mathcal{L}=-\log\sigma(u_o^\top v_c)\;-\;\sum_{k=1}^{K}\log\sigma(-u_{n_k}^\top v_c),\qquad \sigma(x)=\frac1{1+e^{-x}}$$

**Where do noise words come from?** The unigram distribution raised to the power ¾:

$$P_n(w)\propto \text{count}(w)^{3/4}$$

The exponent flattens the distribution: very frequent words ("the") get sampled a bit less, and rare words a bit more than their raw frequency suggests.

### 4.6 Gradients (what the code actually does)

Let `p = σ(u·v)` and `y = 1` for the real pair, `y = 0` for noise pairs. Then

$$\frac{\partial\mathcal L}{\partial v_c}=\sum_{j}(p_j-y_j)\,u_j\qquad\qquad \frac{\partial\mathcal L}{\partial u_j}=(p_j-y_j)\,v_c$$

Update by stochastic gradient descent: `θ ← θ − η · ∂L/∂θ`.

For **CBOW** the same formulas hold with `v_c` replaced by `h`. The gradient with respect to `h` is then **shared equally** among the context words:

$$\frac{\partial\mathcal L}{\partial v_{ctx}}=\frac{1}{|C|}\frac{\partial\mathcal L}{\partial h}$$

### 4.7 Why does this learn synonyms?

*cheap* and *budget* are always surrounded by the same words (`laptop`, `for`, `students`, …). To predict those neighbours well, the model must give `cheap` and `budget` **nearly identical vectors**. Words that are never interchangeable (`cheap` vs `television`) end up far apart.

---

## 5. Worked mini-example (one Skip-Gram step)

Sentence: `budget laptop`, window = 1, `K = 2`.

1. **Real pair:** (`laptop` → `budget`), label **1**
2. **Noise pairs** (sampled randomly): (`laptop` → `milk`) and (`laptop` → `sneakers`), label **0**
3. Compute scores `s = u · v_laptop` for the three context words, apply the sigmoid, giving `p₁, p₂, p₃`.
4. Errors: `(p₁ − 1)`, `p₂`, `p₃`.
5. Move `v_laptop` toward `u_budget` and away from `u_milk` and `u_sneakers`. Move `u_budget` toward `v_laptop`, and the others away.

Repeat millions of times. Words that share contexts get pulled together.

---

## 6. CBOW vs Skip-Gram: the full comparison

| Aspect | **Skip-Gram** | **CBOW** |
|---|---|---|
| **Prediction task** | centre → each context word | averaged context → centre |
| **Examples per window** | up to `2m` (one per context word) | 1 |
| **Updates per epoch** | many (≈ 3× more in our demo) | fewer |
| **Training speed** | slower | **faster** (≈ 2× in our demo) |
| **Word order in window** | ignored | ignored |
| **Rare words** | *Expected* to be better: each rare word is a centre for many examples | *Expected* to be weaker: its signal is averaged with neighbours |
| **Frequent words** | good | **very good**, smoother representations |
| **Small corpora** | **preferred** | can work, needs more epochs |
| **Huge corpora** | works, costly | **preferred** for speed |
| **Typical learning rate** | 0.025 | 0.05 |
| **Typical window** | 5 | 5 to 10 |
| **Semantics vs syntax** | slightly better on semantic analogies | slightly better on syntactic tasks |

> ⚠️ **Honest note from our experiments.** On the small, very regular ShopSmart corpus, both models reached 100% synonym Hit@3, and for two rare words (8 occurrences each) CBOW ranked the true synonym *as high as or higher than* Skip-Gram. The "Skip-Gram is better for rare words" rule is a tendency observed on large real-world corpora, **not a law**. Always compare both on your own data.

### Rule of thumb

```
Do you have a small or medium corpus, or care about rare terms (product names, jargon)?
        → Skip-Gram

Is your corpus huge and training time matters, or do you mostly care about common words?
        → CBOW

Not sure?
        → Train both (it's cheap), evaluate on a task that reflects your product.
```

---

## 7. The practical details that matter

### 7.1 Preprocessing

1. **Tokenise & normalise**: lowercase, strip punctuation, handle numbers and unicode, and apply the *same* rules at training and serving time.
2. **`min_count`**: drop words seen fewer than ~5 times on real data. Their vectors are noise.
3. **Subsampling of frequent words**: discard token `w` with probability

$$P_{\text{discard}}(w)=1-\sqrt{\frac{t}{f(w)}}\qquad(t\approx10^{-3}\text{ to }10^{-5})$$

where `f(w)` is the word's relative frequency. This speeds up training and improves vectors for rarer words.

4. **Dynamic window**: sample the effective window size uniformly from `1..m` for each centre word. Closer words are then used more often.

### 7.2 Hyper-parameters

| Parameter | Typical | Effect |
|---|---|---|
| `vector_size` (d) | 100 to 300 | Higher = more capacity, more data needed |
| `window` (m) | 5 to 10 | **Small:** words that can *replace* each other (synonyms, same part of speech). **Large:** words from the same *topic* |
| `negative` (K) | 5 to 20 (small data), 2 to 5 (big data) | More negatives = better but slower |
| `sample` (t) | 1e-3 to 1e-5 | Subsampling strength |
| `min_count` | 5 to 20 | Vocabulary cut-off |
| `epochs` | 5 to 30 | More for small corpora |
| `alpha` (learning rate) | 0.025 (SG), 0.05 (CBOW) | Decayed linearly to ~0 |

### 7.3 Turning word vectors into sentence/product vectors

Word2Vec gives *word* vectors. For search we need *document* vectors. The simplest strong baseline is the **IDF-weighted average**:

$$\vec{d}=\frac{\sum_{w\in d}\text{idf}(w)\,\vec w}{\left\lVert\sum_{w\in d}\text{idf}(w)\,\vec w\right\rVert}$$

IDF stops common words from dominating the average. Then rank documents by cosine similarity:

$$\cos(a,b)=\frac{a\cdot b}{\lVert a\rVert\,\lVert b\rVert}$$

---

## 8. Back to ShopSmart: what we built

1. **Corpus:** ~4,000 synthetic reviews and descriptions across 9 product families (laptops, phones, headphones, TVs, shirts, jeans, shoes, milk, rice), with synonyms like *laptop/notebook*, *cheap/budget/affordable*, *sneakers/shoes/footwear*.
2. **Models:** Skip-Gram and CBOW with negative sampling, written from scratch in NumPy.
3. **Results (from our run):**
   - Both models placed true synonyms in each other's top-3 neighbours (Hit@3 = 100%).
   - Neighbours made sense: `notebook → laptop`, `cheap → affordable, budget`, `sneakers → shoes, footwear`.
   - **Query expansion** turned `earphones for the gym` into `earphones + headphones + earbuds + …`
4. **Semantic search vs keyword search** on six labelled queries (top-1 accuracy):

| Method | Recall@1 |
|---|---|
| Keyword overlap | 50% |
| Averaged Skip-Gram vectors | 83% |
| Averaged CBOW vectors | 67% |

   Queries such as `earphones for the gym` and `trousers for work` returned **nothing** with keywords but the right product with embeddings.

5. **A failure we kept on purpose:** `tv for family movies` returned a rice pack. The generic word *family* appears in food reviews as often as in TV reviews, and averaging let it pull the query toward groceries. This is a real weakness of averaged embeddings.

---

## 9. Limitations

| Limitation | Example | Remedy |
|---|---|---|
| **One vector per word** (polysemy) | *apple* (fruit) vs *apple* (phone) | Contextual models (BERT and relatives) |
| **Out-of-vocabulary words** | New product name, typo `lapto` | **FastText** (subword n-grams), spell correction |
| **Needs lots of in-domain text** | Few reviews → noisy vectors | Pre-trained vectors, more data |
| **Bias** | Stereotypes in training text show up in vectors | Audit and debias before using in people-affecting decisions |
| **Averaging loses structure** | "not good" ≈ "good" | Sentence encoders, supervised rankers |
| **Static** | New slang/products need retraining | Scheduled retraining |

---

## 10. NLP-engineering checklist

**Data**
- [ ] Train on **in-domain** text (your reviews, tickets, queries), not just Wikipedia.
- [ ] Same normalisation at train and serve time.
- [ ] Keep a held-out set of synonym pairs / labelled queries for evaluation.

**Training**
- [ ] Try both `sg=1` (Skip-Gram) and `sg=0` (CBOW).
- [ ] Sweep `window` ∈ {2, 5, 10} and `vector_size` ∈ {100, 200, 300}.
- [ ] Fix random seeds; run several seeds and compare. A single run can mislead.

**Evaluation**
- [ ] **Intrinsic:** word-similarity correlation (SimLex-999, WordSim-353), analogy accuracy, synonym Hit@k.
- [ ] **Extrinsic (more important):** search Recall@k / NDCG, classification F1, click-through rate.

**Serving**
- [ ] Store the embedding matrix and vocabulary together; version them.
- [ ] Pre-compute document vectors offline; use an approximate-NN index (FAISS, HNSW, ScaNN).
- [ ] Define the OOV policy (skip, fall back to keywords, or use FastText).
- [ ] Monitor zero-result rate, query reformulations, and drift; schedule retraining.

---

## 11. Quick-reference formulas

| Concept | Formula |
|---|---|
| Skip-Gram softmax | $P(o\mid c)=\dfrac{e^{u_o^\top v_c}}{\sum_w e^{u_w^\top v_c}}$ |
| CBOW hidden vector | $h=\dfrac1{\lvert C\rvert}\sum_{c\in C}v_c$ |
| Negative-sampling loss | $-\log\sigma(u_o^\top v_c)-\sum_{k}\log\sigma(-u_{n_k}^\top v_c)$ |
| Noise distribution | $P_n(w)\propto\text{count}(w)^{0.75}$ |
| Gradient wrt score | $p-y$ |
| Subsampling | $P_{\text{discard}}=1-\sqrt{t/f(w)}$ |
| Cosine similarity | $\dfrac{a\cdot b}{\lVert a\rVert\lVert b\rVert}$ |
| Doc vector | IDF-weighted mean of word vectors, then normalise |

---

## 12. Key takeaways

1. Word2Vec learns meaning from **context alone**, with no labels, by training a tiny network on a prediction task and keeping its weights.
2. **Skip-Gram:** centre → context. More updates per word, the usual choice for small corpora and rare terms, slower.
3. **CBOW:** averaged context → centre. Faster, smoother, a good fit for big corpora and frequent words.
4. **Negative sampling** replaces an expensive `V`-way softmax with `K + 1` binary decisions.
5. Embeddings make search **synonym-aware**, but they are one signal, not a complete search engine. Measure on your own data and combine them with other features.

---

## References

- Mikolov et al. (2013). *Efficient Estimation of Word Representations in Vector Space.*
- Mikolov et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality.*
- Levy & Goldberg (2014). *Neural Word Embedding as Implicit Matrix Factorization.*
- Levy, Goldberg & Dagan (2015). *Improving Distributional Similarity with Lessons Learned from Word Embeddings.*
- Rong (2014). *word2vec Parameter Learning Explained.*
- Bojanowski et al. (2017). *Enriching Word Vectors with Subword Information.* (FastText)

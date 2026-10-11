# Hashing Vectorization in NLP

## 1. The problem: turning text into numbers

Machine-learning models need numeric input. The classic **Bag-of-Words (BoW)** approach converts each document into a vector with one dimension per word in the vocabulary.

Standard vectorizers such as `CountVectorizer` and `TfidfVectorizer` work in two passes:

1. **Fit**: scan the entire corpus and build a `word -> column index` dictionary (the vocabulary).
2. **Transform**: look each word up in that dictionary and fill in counts.

This has three practical problems:

| Problem | Why it hurts |
|---|---|
| **Memory** | The vocabulary dictionary grows with the corpus (millions of n-grams for large data). |
| **Needs a full pass over the data** | You cannot start vectorizing until the vocabulary is built, which rules out streaming. |
| **Unseen words** | New words at inference time are silently dropped; the vocabulary is frozen. |

Hashing vectorization removes the vocabulary entirely.

---

## 2. The idea: the hashing trick

Instead of looking up a word's column in a dictionary, **compute** it with a hash function:

```
column_index = hash(token) mod n_features
```

- `hash(...)` is a deterministic function that maps any string to a large integer.
- `n_features` is a fixed number you choose up front (e.g. 2^18 = 262,144).
- The output vector always has exactly `n_features` dimensions, no matter how many distinct words exist.

### Worked example

Let `n_features = 8` and suppose a hash function gives:

| token | hash(token) | hash mod 8 |
|---|---|---|
| `cat` | 4021 | 5 |
| `sat` | 982 | 6 |
| `mat` | 1573 | 5 |

Document: `"cat sat mat"` becomes a vector of length 8 where index 5 has value 2 (`cat` and `mat` collided) and index 6 has value 1.

That clash is a **hash collision**, the central trade-off of the method.

---

## 3. Collisions and the signed hash

Because infinitely many strings map into `n_features` buckets, different words will sometimes land in the same column. Two consequences:

- Information is lost: you cannot tell which word contributed to a bucket.
- Collisions *add up* and bias the counts.

### Alternate sign (signed hashing)

Scikit-learn uses a **second hash bit** to pick a sign (+1 or -1) for each token:

```
index = |h(token)| mod n_features
sign  = +1 if h(token) >= 0 else -1
vector[index] += sign * count
```

If two colliding words get opposite signs, their contributions partially **cancel** instead of accumulating. The expected value of the inner product between two hashed vectors then equals the inner product of the original vectors, so the hashed space approximately preserves similarity. This is controlled by `alternate_sign=True` (the default) in `HashingVectorizer`.

Caveat: with signed values you can get negative entries, which breaks models that need non-negative input, such as `MultinomialNB`. Set `alternate_sign=False` for those.

---

## 4. Choosing `n_features`

- Larger `n_features` means fewer collisions but a wider vector (more memory for dense models, bigger coefficient arrays).
- Smaller `n_features` means more collisions and more noise.
- Output is a **sparse** matrix, so the cost of a large `n_features` is mostly in the downstream model's weights, not in storing documents.
- Scikit-learn's default is `2**20` (about 1 million). A power of two is conventional.
- Rule of thumb: pick `n_features` well above the number of *important* distinct tokens, and tune it like any hyperparameter by checking validation accuracy as you grow it.

---

## 5. Pros and cons

### Advantages

- **Stateless**: no `fit` needed; `fit` is effectively a no-op.
- **Constant memory**: no vocabulary dictionary is stored.
- **Streaming / online learning**: documents can be vectorized one batch at a time, ideal with `partial_fit`.
- **Handles unseen words**: any new token still maps to some bucket.
- **Parallel friendly**: separate workers get identical results with no shared state.
- **Fast**: one hash per token.

### Disadvantages

- **Not invertible**: you cannot map a column back to the word, so no feature names and limited interpretability.
- **Collisions** introduce noise.
- **No built-in IDF**: use `TfidfTransformer` afterward if you want it (which does require a `fit` over the matrix).
- **No vocabulary pruning** such as `min_df`, `max_df` or `max_features`.
- Negative values when `alternate_sign=True`.

---

## 6. `HashingVectorizer` in scikit-learn

```python
from sklearn.feature_extraction.text import HashingVectorizer

vec = HashingVectorizer(n_features=2**18, alternate_sign=True, norm="l2")
X = vec.transform(docs)      # sparse CSR matrix, shape (n_docs, 2**18)
```

Key parameters:

| Parameter | Meaning | Default |
|---|---|---|
| `n_features` | Number of output columns (hash buckets) | `2**20` |
| `alternate_sign` | Use signed hashing to reduce collision bias | `True` |
| `norm` | `'l1'`, `'l2'` or `None` row normalization | `'l2'` |
| `binary` | Presence/absence instead of counts | `False` |
| `ngram_range` | Include word n-grams, e.g. `(1, 2)` | `(1, 1)` |
| `analyzer` | `'word'`, `'char'`, `'char_wb'` | `'word'` |
| `lowercase`, `stop_words`, `token_pattern` | Same preprocessing options as `CountVectorizer` | |

Related class: `FeatureHasher` hashes generic feature dictionaries or lists of strings, which is useful for non-text categorical data.

### Hash function used

Scikit-learn uses **MurmurHash3 (32-bit)**. Do **not** use Python's built-in `hash()` for this: it is randomized per process for strings (`PYTHONHASHSEED`), so results are not reproducible across runs.

---

## 7. Comparison with other vectorizers

| Feature | CountVectorizer | TfidfVectorizer | HashingVectorizer |
|---|---|---|---|
| Needs `fit` on corpus | Yes | Yes | No |
| Stores vocabulary | Yes | Yes | No |
| Memory grows with vocabulary | Yes | Yes | No |
| Inverse mapping (feature names) | Yes | Yes | No |
| Online / streaming | Hard | Hard | Yes |
| IDF weighting | No | Yes | Via `TfidfTransformer` |
| Collisions | None | None | Possible |

---

## 8. When to use it

**Good fit**
- Very large corpora that do not fit in memory (out-of-core learning).
- Streaming text where new vocabulary keeps appearing.
- Distributed or parallel preprocessing.
- Production systems where a bounded memory footprint matters.
- Large n-gram or character n-gram feature spaces.

**Poor fit**
- Small or medium datasets where a vocabulary easily fits in memory.
- Tasks where interpretability matters (inspecting top words per class).
- Models that need non-negative features unless `alternate_sign=False`.

---

## 9. Typical workflow

```
raw text  ->  HashingVectorizer (stateless)  ->  [optional TfidfTransformer]
          ->  linear model with partial_fit (SGDClassifier, PassiveAggressive, MultinomialNB)
```

Because the vectorizer is stateless, the same instance can transform training batches, validation data and live production traffic identically.

---

## 10. Key takeaways

1. The hashing trick replaces a vocabulary lookup with `hash(token) mod n_features`.
2. Memory is fixed and no vocabulary is stored, which makes it great for big and streaming data.
3. Collisions are the price; signed hashing and a large `n_features` keep their effect small.
4. You lose interpretability; if you need feature names, use `CountVectorizer` or `TfidfVectorizer`.
5. In scikit-learn, combine `HashingVectorizer` with `partial_fit` classifiers for out-of-core learning.

---

## References

- Weinberger, K. et al. (2009). *Feature Hashing for Large Scale Multitask Learning.* ICML.
- Scikit-learn documentation: `sklearn.feature_extraction.text.HashingVectorizer`, `sklearn.feature_extraction.FeatureHasher`.
- Scikit-learn user guide: *Text feature extraction, Vectorizing a large text corpus with the hashing trick*.

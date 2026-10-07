# Corpus, Document, Vocabulary and Word in NLP

These four terms are the basic vocabulary of Natural Language Processing (NLP). Almost every text-based ML pipeline (Bag of Words, TF-IDF, Word2Vec, Transformers) starts by organising text in terms of them.

```
Corpus
 ├── Document 1  →  "I love NLP"
 ├── Document 2  →  "NLP loves data"
 └── Document 3  →  "I love data"

Vocabulary = { I, love, loves, NLP, data }   (unique words across the whole corpus)
```

---

## 1. Word (Token)

A **word** is the smallest meaningful unit of text we work with. In practice NLP uses the more general term **token**, because a token may be a word, a subword (`un`, `##happy`), a punctuation mark, a number or even a character.

- Sentence: `"I love NLP!"`
- Tokens: `["I", "love", "NLP", "!"]`

**Important distinctions**

| Term | Meaning | Example |
|------|---------|---------|
| Word / Token occurrence | Every single appearance of a word in text | `"the cat saw the dog"` has 5 tokens |
| Word type | A unique word form | the same sentence has 4 types (`the`, `cat`, `saw`, `dog`) |

The process of splitting text into tokens is called **tokenization**.

---

## 2. Document

A **document** is a single unit of text in your dataset. What counts as a "document" depends on your problem, you decide the granularity.

| Task | One document is... |
|------|--------------------|
| Spam detection | one email |
| Sentiment analysis | one movie review or one tweet |
| News classification | one news article |
| Search engine | one web page |
| Chatbot intent detection | one user message |

A document is a sequence of tokens. It is the row of your dataset: each document gets a label (in supervised learning) and a numeric representation (vector).

---

## 3. Corpus

A **corpus** (plural: **corpora**) is the entire collection of documents you use for NLP. It is your whole text dataset.

- 10,000 movie reviews → a corpus of reviews
- All Wikipedia articles → the Wikipedia corpus
- Famous examples: Brown Corpus, Penn Treebank, Common Crawl, IMDB reviews, Wikipedia dumps

Corpus properties that matter in ML:

- **Size**: number of documents and total tokens
- **Domain**: medical, legal, social media, news... (a model trained on tweets behaves badly on legal text)
- **Language(s)**
- **Labelled vs unlabelled**: labelled corpora train supervised models, unlabelled corpora are used for pretraining (e.g. language models)
- **Quality / noise**: spelling errors, duplicates, bias

---

## 4. Vocabulary

The **vocabulary** (often written `V`) is the **set of all unique tokens** found in the corpus (after preprocessing). Its size is written `|V|`.

Each vocabulary item is mapped to an integer ID, because models work with numbers, not strings:

```
word2idx = {"I": 0, "love": 1, "loves": 2, "NLP": 3, "data": 4}
idx2word = {0: "I", 1: "love", 2: "loves", 3: "NLP", 4: "data"}
```

Why the vocabulary matters:

- It defines the **length of vectors** in Bag of Words / TF-IDF (one dimension per vocabulary word).
- It defines the **size of the embedding matrix** (`|V| x embedding_dim`) and the output layer of language models.
- **Out-of-vocabulary (OOV) words**: words not seen in the training corpus. They are usually mapped to a special `<UNK>` token.
- Common special tokens: `<PAD>` (padding), `<UNK>` (unknown), `<BOS>`/`<EOS>` (begin/end of sentence).

Ways of controlling vocabulary size:

- lowercasing
- removing stop words and rare words (e.g. keep only words appearing at least 5 times)
- stemming / lemmatization (`loves`, `loving` → `love`)
- subword tokenization (BPE, WordPiece), which nearly removes the OOV problem

---

## How they relate

```
Corpus  ⊃  Documents  ⊃  Tokens (words)
Vocabulary = set(all tokens in Corpus)       →   |V| ≤ total number of tokens
```

Example corpus of 3 documents:

| Doc | Text | Tokens in doc |
|-----|------|---------------|
| D1 | `I love NLP` | 3 |
| D2 | `NLP loves data` | 3 |
| D3 | `I love data` | 3 |

- Corpus size: **3 documents**
- Total tokens: **9**
- Vocabulary: `{I, love, loves, NLP, data}` → **|V| = 5**

---

## Quick comparison

| Concept | What it is | Level | Example |
|---------|-----------|-------|---------|
| Word / Token | smallest text unit | lowest | `love` |
| Document | one piece of text | middle | `"I love NLP"` |
| Corpus | collection of all documents | highest | all 3 sentences |
| Vocabulary | set of *unique* words in corpus | derived from corpus | `{I, love, NLP, ...}` |

---

## Related metrics

- **Term Frequency (TF)**: how many times a word appears in one document
- **Document Frequency (DF)**: how many documents contain the word
- **TF-IDF**: `TF × log(N / DF)`, where `N` is the number of documents in the corpus. It shows why corpus, document and vocabulary are all needed to compute even a simple feature.
- **Type-Token Ratio** = `|V| / total tokens`, a rough measure of lexical diversity

---

## Common mistakes

1. **Confusing corpus with vocabulary.** The corpus holds *all text including repeats*, the vocabulary holds *only unique words*.
2. **Building the vocabulary from test data.** Build it from the **training corpus only**, otherwise you leak information.
3. **Forgetting OOV handling.** New words at inference time must map to `<UNK>` (or use subword tokenization).
4. **Treating "document" as always meaning a file.** It is whatever unit your task defines.

---

## Summary

- **Word/Token**: one unit of text.
- **Document**: one text sample, a sequence of tokens.
- **Corpus**: the full collection of documents.
- **Vocabulary**: the unique tokens of the corpus, each mapped to an ID. This defines the input/output space of your NLP model.

See `code.ipynb` for hands-on examples of all four concepts.

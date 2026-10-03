# Text Preprocessing in NLP
## Cleaning NLP Text — Sarcasm Detection Project

### 1. What is Text Preprocessing?

Text preprocessing is the process of converting raw, messy text into a cleaner and more consistent form before using it in an NLP model.

Raw text can contain:

- Uppercase and lowercase variations
- URLs
- User mentions
- Hashtags
- Punctuation
- Numbers
- Emojis
- Extra spaces
- Spelling variations
- Stopwords
- Slang and informal language

The objective is not simply to delete everything that looks unnecessary. The objective is to preserve useful information while reducing irrelevant noise.

---

## 2. Why Preprocessing is Important

Machine-learning algorithms work with numerical representations rather than raw sentences. Preprocessing helps us create consistent text that can be converted into useful numerical features.

A typical pipeline is:

```text
Raw Text
   ↓
Cleaning
   ↓
Normalization
   ↓
Tokenization
   ↓
Stopword Handling
   ↓
Stemming / Lemmatization
   ↓
Feature Extraction
   ↓
Machine Learning Model
```

---

## 3. Text Cleaning

### Lowercasing

Converts text to lowercase.

Example:

```text
"HELLO World" → "hello world"
```

This reduces duplicate vocabulary entries.

However, capitalization can sometimes be useful in sarcasm or sentiment analysis, so it should be removed only when appropriate.

### Removing URLs

Example:

```text
"Visit https://example.com now"
→ "Visit now"
```

URLs usually do not provide useful information for a basic text classifier, although the domain or URL pattern can sometimes be useful as a feature.

### Removing Mentions

Example:

```text
"@user this is great"
→ "this is great"
```

Mentions can be removed when the identity of the user is irrelevant.

### Hashtags

A hashtag can be handled in different ways.

Example:

```text
"#amazing"
```

Instead of deleting it blindly, a real project may keep the word `amazing` while removing only the `#` symbol.

Hashtags may be especially useful in social-media NLP.

### Numbers

Numbers can be removed when they do not contribute to the task.

Example:

```text
"I waited 100 minutes"
```

For some applications, however, the number itself is important. Therefore, number removal should depend on the dataset.

### Punctuation

Punctuation such as `.`, `,`, and `?` can be removed for some traditional NLP pipelines.

But sarcasm detection requires caution:

```text
"Great."
"Great!!!"
"Great..."
```

These sentences may communicate different tones. Therefore, punctuation can be a useful sarcasm feature.

---

## 4. Tokenization

Tokenization breaks text into smaller units called tokens.

Example:

```text
"I love NLP"
```

becomes:

```text
["I", "love", "NLP"]
```

Common tokenization methods include:

- Word tokenization
- Sentence tokenization
- Subword tokenization

Modern transformer models often use subword tokenization.

---

## 5. Stopwords

Stopwords are very common words such as:

```text
the, is, a, an, to, of, in
```

Removing them can reduce the number of features.

However, stopword removal is **task-dependent**.

For sarcasm detection, grammar and context can matter. Therefore, removing every stopword is not always the best choice.

---

## 6. Stemming

Stemming reduces words to a root-like form by applying simple rules.

Example:

```text
playing
played
plays
```

may be reduced to something similar to:

```text
play
```

A common algorithm is the Porter Stemmer.

### Advantages

- Fast
- Simple
- Reduces vocabulary size

### Disadvantages

- May produce non-dictionary words
- Can remove too much information

---

## 7. Lemmatization

Lemmatization converts a word into its meaningful dictionary form.

Example:

```text
better → good
running → run
```

depending on the lemmatizer and linguistic information available.

### Advantages

- More linguistically meaningful
- Usually cleaner than basic stemming

### Disadvantages

- Slower
- Requires linguistic resources

---

# 8. Sarcasm Detection

Sarcasm detection is the task of identifying whether a piece of text expresses sarcasm.

Example:

```text
"Fantastic! My computer crashed again."
```

The literal word `Fantastic` sounds positive, but the overall context can indicate a negative or sarcastic meaning.

This makes sarcasm detection harder than ordinary sentiment classification.

---

## 9. Why Sarcasm is Difficult

Sarcasm often depends on:

- Context
- Tone
- Previous conversation
- Punctuation
- Emojis
- Capitalization
- Word combinations
- Contradiction between literal words and intended meaning
- Cultural or situational knowledge

For example:

```text
"Wonderful, another three-hour meeting."
```

The word `Wonderful` is positive, but the complete sentence can be sarcastic.

---

# 10. Preprocessing for Sarcasm Detection

A common beginner pipeline is:

```text
Raw Text
→ Lowercase
→ URL / mention handling
→ Basic normalization
→ Tokenization
→ Optional stopword removal
→ Optional stemming/lemmatization
→ TF-IDF
→ Classifier
```

For sarcasm detection, preprocessing should be designed carefully.

### Information that may be useful

- `!!!`
- `...`
- Emojis
- Hashtags
- Capitalization
- Repeated characters
- Negation
- Contrast words
- Sentence context

Therefore, aggressive cleaning can sometimes reduce model performance.

---

# 11. Feature Extraction

Machine-learning algorithms need numerical input.

### Bag of Words

Represents text using word counts.

### TF-IDF

TF-IDF stands for:

**Term Frequency–Inverse Document Frequency**

It gives importance to words based on their frequency within a document and across the collection.

A simplified idea is:

```text
TF-IDF = Term Frequency × Inverse Document Frequency
```

TF-IDF is a strong baseline for many traditional text-classification tasks.

### N-grams

N-grams capture groups of consecutive words.

Examples:

```text
Unigram:
"very"

Bigram:
"very good"

Trigram:
"not very good"
```

Using unigrams and bigrams can capture more context than single words alone.

---

# 12. Machine Learning Pipeline

After preprocessing:

```text
Text
 ↓
Preprocessing
 ↓
TF-IDF / Other Vectorization
 ↓
Train-Test Split
 ↓
Model Training
 ↓
Prediction
 ↓
Evaluation
```

Common baseline classifiers include:

- Logistic Regression
- Naive Bayes
- Support Vector Machine
- Random Forest

For larger projects, deep-learning and transformer-based approaches can also be used.

---

# 13. Data Leakage

Data leakage occurs when information from the test set incorrectly influences training.

For example, do not fit a TF-IDF vectorizer on the complete dataset before splitting.

Incorrect:

```text
Complete Dataset
     ↓
TF-IDF fit_transform
     ↓
Train/Test Split
```

Better:

```text
Dataset
 ↓
Train/Test Split
 ↓
Fit TF-IDF on Training Data
 ↓
Transform Training Data
 ↓
Transform Test Data
```

This keeps evaluation more reliable.

---

# 14. Evaluation

Accuracy alone is not always enough.

Useful metrics include:

### Accuracy

Percentage of correct predictions.

### Precision

Of the texts predicted as sarcastic, how many were actually sarcastic?

### Recall

Of the truly sarcastic texts, how many did the model identify?

### F1-score

A combined measure based on precision and recall.

For imbalanced datasets, precision, recall, F1-score, and a confusion matrix should be examined along with accuracy.

---

# 15. Important Project Decisions

For a real sarcasm detection project, decide carefully:

1. Which dataset will be used?
2. How are sarcastic and non-sarcastic labels defined?
3. How will duplicates be handled?
4. Will emojis be preserved?
5. Will hashtags be preserved?
6. Will punctuation be preserved?
7. Should stopwords be removed?
8. Should stemming or lemmatization be used?
9. Which vectorizer will be used?
10. Which evaluation metrics will be reported?

These choices should be tested experimentally rather than assumed to be universally correct.

---

# 16. Recommended Project Structure

```text
Sarcasm-Detection/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── code.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── theory.md
└── README.md
```

---

# 17. Key Takeaways

- Text preprocessing converts raw text into a form suitable for NLP.
- Cleaning should remove irrelevant noise without destroying useful signals.
- Tokenization divides text into manageable units.
- Stopword removal is optional and task-dependent.
- Stemming is simpler and faster; lemmatization is more linguistically meaningful.
- Sarcasm is difficult because literal words can differ from intended meaning.
- Emojis, punctuation, capitalization, hashtags, and context can contain useful sarcasm signals.
- TF-IDF is a useful traditional baseline for text classification.
- Avoid data leakage when building the feature-extraction pipeline.
- Evaluate sarcasm classifiers using more than accuracy when the dataset is imbalanced.

---

## Final NLP Engineer Workflow

```text
Collect Dataset
      ↓
Explore Dataset
      ↓
Clean / Normalize Text
      ↓
Tokenize
      ↓
Feature Engineering
      ↓
Train / Validation / Test Split
      ↓
TF-IDF / Embeddings
      ↓
Train Classifier
      ↓
Evaluate
      ↓
Error Analysis
      ↓
Improve Preprocessing + Model
      ↓
Deploy
```

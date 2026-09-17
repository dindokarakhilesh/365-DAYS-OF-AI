# Scam Detection Using Machine Learning

## Fraud Detection Python | NLP with Scikit-Learn

This project demonstrates a basic **NLP text-classification** workflow to classify messages as **Scam** or **Legitimate**.

> The dataset in the notebook is a small educational example. A real fraud-detection system needs a large, representative and properly labelled dataset.

## Workflow

```text
Text Data
   ↓
Train/Test Split
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
Prediction
   ↓
Evaluation
```

## 1. NLP

**NLP (Natural Language Processing)** allows computers to process and analyse human language.

Example:

```text
"Congratulations! You won a prize. Click this link now."
```

Text must be converted into numerical features before it can be used by many Machine Learning models.

## 2. TF-IDF

**TF-IDF (Term Frequency–Inverse Document Frequency)** converts text into numerical values based on word importance.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

**Important:** Fit the vectorizer only on training data, then use `transform()` on test/new data.

## 3. Train-Test Split

Training data is used to learn patterns, while testing data is used to evaluate the model on unseen examples.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
```

## 4. Logistic Regression

Logistic Regression can be used for binary classification.

```text
0 → Legitimate
1 → Scam
```

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
```

## 5. Prediction

```python
y_pred = model.predict(X_test_tfidf)
```

For a new message:

```python
new_message = ["Send your OTP now to claim your reward."]

new_tfidf = vectorizer.transform(new_message)
prediction = model.predict(new_tfidf)

print("SCAM" if prediction[0] == 1 else "LEGITIMATE")
```

## 6. Model Evaluation

### Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
```

### Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

It reports:
- Precision
- Recall
- F1-score
- Support

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
```

## 7. Scikit-Learn Pipeline

TF-IDF and the ML model can be combined:

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
```

## 8. Important Learning Point

The model learns statistical patterns from the training data. It does not understand a scam exactly like a human.

Performance depends on:
- Dataset size
- Data quality
- Correct labels
- Variety of scam examples
- Legitimate examples
- Feature engineering
- Model selection
- Evaluation method

## Requirements

```bash
pip install pandas scikit-learn
```

## Conclusion

This project demonstrates:

**NLP → TF-IDF → Classification Model → Prediction → Evaluation**

Scikit-Learn provides the tools needed to build this basic text-classification workflow.

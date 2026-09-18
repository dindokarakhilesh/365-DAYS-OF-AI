# Scam Detection Using Machine Learning — Part 2

## TF-IDF Vectorizer | NLP with Scikit-Learn

This project focuses on using **TF-IDF Vectorizer** to convert scam-related text into numerical features for Machine Learning.

> The notebook uses a small educational dataset. A production fraud-detection system requires a large, representative and properly labelled dataset.

## Workflow

```text
Text → Train/Test Split → TF-IDF → Logistic Regression → Prediction → Evaluation
```

## 1. NLP

**NLP (Natural Language Processing)** helps computers process human language. Text must be represented numerically before many Machine Learning algorithms can use it.

## 2. TF-IDF

**TF-IDF = Term Frequency–Inverse Document Frequency**

It assigns numerical importance to words based on their occurrence in documents.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)
```

## 3. Fit and Transform

Training data:

```python
X_train_tfidf = tfidf.fit_transform(X_train)
```

Test data:

```python
X_test_tfidf = tfidf.transform(X_test)
```

**Important:** Fit the vectorizer only on training data. Use `transform()` for test and new data to avoid data leakage.

## 4. Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
```

In this project:

```text
0 → Legitimate
1 → Scam
```

## 5. Prediction

```python
y_pred = model.predict(X_test_tfidf)
```

New message:

```python
new_message = ["Send your bank details to claim your reward."]

new_tfidf = tfidf.transform(new_message)
prediction = model.predict(new_tfidf)
```

## 6. Evaluation

### Accuracy

```python
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
```

### Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

Provides:
- Precision
- Recall
- F1-score
- Support

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))
```

## 7. Inspect Features

```python
features = tfidf.get_feature_names_out()
coefficients = model.coef_[0]
```

Model coefficients can help inspect which learned words are associated more strongly with each class.

## 8. Pipeline

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

## 9. Important Learning Points

1. NLP converts language into useful machine-readable representations.
2. TF-IDF converts text into numerical features.
3. Fit TF-IDF on training data only.
4. Use `transform()` for test and new messages.
5. Logistic Regression can perform binary text classification.
6. Evaluate using accuracy, classification report and confusion matrix.

## Requirements

```bash
pip install pandas scikit-learn
```

## Conclusion

Part 2 demonstrates the practical use of **TF-IDF Vectorizer** in scam detection:

**Text → TF-IDF → Classification Model → Prediction → Evaluation**

The quality and representativeness of the dataset strongly affect model performance.

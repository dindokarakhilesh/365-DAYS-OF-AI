# Confusion Matrix in Machine Learning

## Classification Report | Precision | Recall | Scikit-Learn

This project explains important evaluation metrics used for **classification models** in Machine Learning.

## 1. Confusion Matrix

A confusion matrix shows the number of correct and incorrect predictions made by a classification model.

|                | Predicted 0 | Predicted 1 |
|----------------|-------------:|-------------:|
| **Actual 0**   | TN | FP |
| **Actual 1**   | FN | TP |

### Terms

- **TP — True Positive:** Actual positive and predicted positive.
- **TN — True Negative:** Actual negative and predicted negative.
- **FP — False Positive:** Actual negative but predicted positive.
- **FN — False Negative:** Actual positive but predicted negative.

## 2. Accuracy

Accuracy measures the overall percentage of correct predictions.

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

## 3. Precision

Precision tells us how many predicted positive samples were actually positive.

```text
Precision = TP / (TP + FP)
```

High precision means fewer false positives.

## 4. Recall

Recall tells us how many actual positive samples were correctly identified.

```text
Recall = TP / (TP + FN)
```

High recall means fewer false negatives.

## 5. F1-Score

F1-Score is the harmonic mean of Precision and Recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

It is useful when we need a balance between precision and recall.

## 6. Classification Report

Scikit-Learn provides `classification_report()` to display:

- Precision
- Recall
- F1-Score
- Support

Example:

```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
```

## 7. Scikit-Learn Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
print(cm)
```

To visualize it:

```python
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[0, 1]
)

disp.plot()
```

## 8. Main Metrics at a Glance

| Metric | Meaning |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Correct positive predictions / all predicted positives |
| Recall | Correct positive predictions / all actual positives |
| F1-Score | Balance between precision and recall |
| Confusion Matrix | TP, TN, FP and FN counts |
| Classification Report | Detailed metrics for each class |

## Conclusion

A classification model should not always be judged by accuracy alone. The **Confusion Matrix, Precision, Recall, F1-Score, and Classification Report** help us understand exactly how well the model is performing and what types of errors it is making.

### Requirements

```bash
pip install scikit-learn matplotlib
```

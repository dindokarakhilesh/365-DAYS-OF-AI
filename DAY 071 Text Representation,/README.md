# Mastering Unsupervised Learning with Scikit-Learn

## K-Means Clustering, PCA & More

This project covers important concepts of **Unsupervised Learning** using Scikit-Learn.

### Topics Covered

- Unsupervised Learning
- Clustering
- K-Means Clustering
- Centroids
- Inertia
- Elbow Method
- Feature Scaling
- Principal Component Analysis (PCA)
- Principal Components
- Explained Variance
- PCA + K-Means workflow
- Other Unsupervised Learning algorithms

---

## 1. What is Unsupervised Learning?

Unsupervised Learning is a type of Machine Learning where the model learns patterns or structure from data without using predefined target labels.

The main goal is to discover hidden structure in the dataset.

Common tasks include:

- Clustering
- Dimensionality Reduction
- Anomaly Detection
- Association Rule Learning

---

## 2. K-Means Clustering

K-Means is a clustering algorithm that divides data into a specified number of groups called clusters.

The algorithm uses **centroids** to represent the center of each cluster.

Basic workflow:

```text
Data
  ↓
Choose K
  ↓
Initialize Centroids
  ↓
Assign Samples to Nearest Centroid
  ↓
Update Centroids
  ↓
Repeat
  ↓
Final Clusters
```

---

## 3. Inertia

Inertia measures the total squared distance between each sample and its assigned cluster centroid.

Scikit-Learn provides it through:

```python
kmeans.inertia_
```

Lower inertia generally means samples are closer to their assigned centroids, but inertia should be considered together with other information when choosing `k`.

---

## 4. Elbow Method

The Elbow Method runs K-Means with different values of `k` and plots the corresponding inertia.

The point where the improvement starts to slow down can be used as a candidate for the number of clusters.

---

## 5. PCA

**PCA = Principal Component Analysis**

PCA is a dimensionality-reduction technique.

It transforms the original features into a smaller number of new features called **principal components**.

PCA is useful for:

- Reducing dimensions
- Visualization
- Removing redundant information
- Making high-dimensional data easier to analyze

---

## 6. Project Workflow

```text
Dataset
   ↓
Feature Scaling
   ↓
K-Means Clustering
   ↓
Cluster Labels
   ↓
PCA
   ↓
2D Visualization
```

PCA and K-Means perform different tasks, but they can be used together in an unsupervised learning workflow.

---

## 7. Requirements

Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## 8. Files

- `Code.ipynb` — Practical implementation
- `README.md` — Project overview
- `theory.md` — Detailed theory and revision notes

---

## Conclusion

This project provides a practical introduction to unsupervised learning with Scikit-Learn, focusing on **K-Means Clustering and PCA**.

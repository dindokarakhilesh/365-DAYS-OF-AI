# Unsupervised Learning with Scikit-Learn — Theory

## Topic

**Mastering Unsupervised Learning with Scikit-Learn: K-Means, PCA & More**

---

## 1. Introduction to Unsupervised Learning

Unsupervised Learning is a branch of Machine Learning in which the model learns patterns, structures, or relationships from data without being given predefined target labels.

In supervised learning, we normally have:

```text
Input Features → Target
```

In unsupervised learning, we mainly have:

```text
Input Data → Hidden Structure / Pattern
```

The model tries to discover useful information from the data itself.

### Common applications

- Customer segmentation
- Document grouping
- Image grouping
- Anomaly detection
- Data visualization
- Dimensionality reduction
- Pattern discovery

---

## 2. Main Types of Unsupervised Learning

Important unsupervised learning tasks include:

### 2.1 Clustering

Clustering groups similar observations together.

Examples:

- K-Means
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Models

### 2.2 Dimensionality Reduction

Dimensionality reduction reduces the number of features while trying to preserve important information.

Examples:

- PCA
- Kernel PCA
- Truncated SVD
- t-SNE
- Other manifold-learning methods

---

# K-MEANS CLUSTERING

## 3. What is K-Means?

K-Means is an unsupervised clustering algorithm used to divide data into **K clusters**.

Here, `K` represents the number of clusters that we want the algorithm to create.

Each cluster is represented by a **centroid**.

The centroid is the center point of a cluster in feature space.

---

## 4. How K-Means Works

The basic K-Means process is:

### Step 1 — Choose K

Select the number of clusters.

Example:

```python
K = 4
```

### Step 2 — Initialize Centroids

The algorithm initializes the centroids.

### Step 3 — Assign Samples

Each sample is assigned to the nearest centroid according to a distance measure.

### Step 4 — Update Centroids

The centroid of each cluster is recalculated using the samples assigned to that cluster.

### Step 5 — Repeat

The assignment and centroid-update steps continue until the algorithm converges or reaches its iteration limit.

### Final Result

Each observation receives a cluster label.

---

## 5. Centroid

A centroid represents the center of a cluster.

For a cluster containing points, the centroid is calculated from the mean of the feature values.

The centroid changes during the K-Means optimization process until the algorithm reaches a stable solution.

---

## 6. Distance in K-Means

K-Means commonly uses Euclidean distance.

For two points:

```text
A = (x1, y1)
B = (x2, y2)
```

Euclidean distance is:

```text
Distance = √[(x2 - x1)² + (y2 - y1)²]
```

The nearest centroid is used when assigning a sample to a cluster.

---

## 7. Why Feature Scaling Matters

K-Means is based on distances.

If one feature has a much larger numerical scale than another feature, it can have a stronger effect on the distance calculation.

Example:

```text
Age        → 18 to 70
Income     → 10,000 to 1,000,000
```

Income may dominate the distance calculation.

Feature scaling helps place features on comparable scales.

A common Scikit-Learn tool is:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 8. K-Means in Scikit-Learn

Scikit-Learn provides:

```python
from sklearn.cluster import KMeans
```

Example:

```python
kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)
```

### Important parameters

#### `n_clusters`

Specifies the number of clusters.

```python
n_clusters=4
```

#### `n_init`

Controls how many initial centroid configurations are tried.

Using multiple initializations can help obtain a better solution.

#### `max_iter`

Maximum number of iterations for one run.

#### `tol`

Controls the convergence tolerance.

#### `random_state`

Makes the initialization reproducible when randomness is involved.

---

## 9. Cluster Labels

After fitting K-Means, each sample receives a cluster label.

Example:

```python
clusters = kmeans.fit_predict(X_scaled)
```

The resulting values can look like:

```text
0
1
2
3
```

These numbers are simply identifiers for clusters. A label such as `0` is not inherently better or more important than `1`.

---

## 10. Cluster Centers in Scikit-Learn

K-Means stores the final centroids in:

```python
kmeans.cluster_centers_
```

Example:

```python
print(kmeans.cluster_centers_)
```

These centers are represented in the feature space used to train the model.

---

# INERTIA

## 11. What is Inertia?

Inertia is the sum of squared distances between each observation and the centroid of its assigned cluster.

Scikit-Learn provides:

```python
kmeans.inertia_
```

Conceptually:

```text
Inertia
=
Sum of squared distances
from samples to their assigned centroids
```

A smaller inertia indicates tighter clusters for a particular value of K.

However, increasing the number of clusters generally decreases inertia, so inertia by itself should not be used as the only criterion for selecting K.

---

# ELBOW METHOD

## 12. What is the Elbow Method?

The Elbow Method is a practical technique for examining different values of K.

Typical process:

1. Run K-Means for several K values.
2. Record the inertia for each K.
3. Plot K against inertia.
4. Look for a point where the improvement begins to decrease substantially.

This point is often called the **elbow**.

The elbow is a candidate rather than a mathematical guarantee of the correct number of clusters.

---

## 13. Elbow Method Code

```python
inertias = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )
    model.fit(X_scaled)
    inertias.append(model.inertia_)
```

Then plot the values:

```python
plt.plot(range(1, 11), inertias)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()
```

---

# LIMITATIONS OF K-MEANS

## 14. Important Limitations

K-Means has several limitations:

1. The number of clusters must be specified.
2. It is sensitive to feature scale.
3. It can be affected by outliers.
4. Different initializations can produce different solutions.
5. It works best for certain cluster shapes and structures.
6. It may not represent irregular or density-based clusters well.

Therefore, K-Means should be selected according to the structure of the data and the purpose of the analysis.

---

# PCA

## 15. What is PCA?

**PCA = Principal Component Analysis**

PCA is a dimensionality-reduction technique.

It transforms the original features into a new set of features called **principal components**.

The principal components are ordered according to the amount of variance they explain.

---

## 16. Why PCA is Used

High-dimensional datasets may contain many features.

Working with many dimensions can make:

- Visualization difficult
- Computation more expensive
- Redundant information harder to identify
- Data analysis more complex

PCA can represent the data using fewer dimensions while preserving as much variance as possible in the selected components.

---

## 17. Principal Components

A principal component is a new direction in the feature space.

The first principal component captures the largest amount of variance possible under the PCA objective.

The next component captures the largest remaining variance subject to being orthogonal to the previous component.

Therefore:

```text
PC1 → largest explained variance
PC2 → next largest explained variance
PC3 → next largest explained variance
...
```

---

## 18. PCA in Scikit-Learn

Scikit-Learn provides:

```python
from sklearn.decomposition import PCA
```

Example:

```python
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)
```

This converts the original feature representation into two principal components.

---

## 19. `n_components`

The parameter:

```python
n_components
```

controls the number of principal components to keep.

Example:

```python
PCA(n_components=2)
```

means that the transformed dataset will contain two principal components.

---

## 20. Explained Variance

PCA provides information about how much variance each component explains.

Scikit-Learn provides:

```python
pca.explained_variance_ratio_
```

Example:

```python
print(pca.explained_variance_ratio_)
```

If the values are:

```text
[0.60, 0.25]
```

then the first component explains approximately 60% of the variance and the second explains approximately 25%.

Together:

```text
0.60 + 0.25 = 0.85
```

So the two selected components explain approximately 85% of the variance.

---

## 21. PCA Components

Scikit-Learn stores the principal-component directions in:

```python
pca.components_
```

These values describe the directions of the principal components relative to the original features.

---

## 22. PCA Transformation

The basic workflow is:

```text
Original Dataset
      ↓
Feature Scaling
      ↓
PCA
      ↓
Reduced Dataset
```

Example:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

---

# PCA + K-MEANS

## 23. Using PCA and K-Means Together

PCA and K-Means solve different problems.

### K-Means

Used for:

```text
Clustering
```

### PCA

Used for:

```text
Dimensionality Reduction
```

They can be combined.

A common workflow is:

```text
Raw Data
   ↓
Feature Scaling
   ↓
PCA
   ↓
K-Means
   ↓
Cluster Labels
```

However, PCA is not a mandatory requirement for K-Means.

---

## 24. PCA for Visualization

One important use of PCA is visualization.

Suppose a dataset has four features:

```text
Feature 1
Feature 2
Feature 3
Feature 4
```

A normal graph cannot directly show all four dimensions.

PCA can reduce the representation to two components:

```text
PC1
PC2
```

These two components can then be plotted on a 2D graph.

---

# OTHER UNSUPERVISED LEARNING METHODS

## 25. Hierarchical Clustering

Hierarchical clustering creates a hierarchy of groups.

It can be represented using a tree-like structure called a **dendrogram**.

It can be useful when we want to study relationships between groups at different levels.

---

## 26. DBSCAN

**DBSCAN = Density-Based Spatial Clustering of Applications with Noise**

DBSCAN groups points based on density.

It can identify:

- Dense regions
- Noise points
- Clusters with shapes that are not necessarily spherical

Unlike standard K-Means, DBSCAN does not require the number of clusters to be specified in advance in the same way.

---

## 27. Gaussian Mixture Models

A Gaussian Mixture Model represents data using a mixture of probability distributions.

It provides a probabilistic approach to clustering.

In contrast, K-Means assigns each sample to one cluster based on the nearest centroid.

---

## 28. t-SNE

t-SNE is mainly used for nonlinear dimensionality reduction and visualization.

It is often used to visualize high-dimensional data in two or three dimensions.

It should not be interpreted as simply another replacement for PCA; the methods have different objectives and behavior.

---

# PRACTICAL WORKFLOW

## 29. General Unsupervised Learning Workflow

A practical workflow can be:

```text
1. Understand the Dataset
        ↓
2. Clean the Data
        ↓
3. Select Relevant Features
        ↓
4. Handle Missing Values
        ↓
5. Scale Features When Appropriate
        ↓
6. Apply an Unsupervised Algorithm
        ↓
7. Analyze the Discovered Structure
        ↓
8. Visualize Results
        ↓
9. Validate the Results
```

---

## 30. Important Scikit-Learn Classes

| Class | Purpose |
|---|---|
| `StandardScaler` | Feature scaling |
| `KMeans` | K-Means clustering |
| `PCA` | Principal Component Analysis |
| `DBSCAN` | Density-based clustering |
| `AgglomerativeClustering` | Hierarchical clustering |
| `GaussianMixture` | Gaussian mixture modeling |

---

# KEY TERMS

## 31. Revision Table

| Term | Meaning |
|---|---|
| Unsupervised Learning | Learning patterns without predefined target labels |
| Clustering | Grouping similar observations |
| K-Means | Centroid-based clustering algorithm |
| Cluster | Group of observations |
| Centroid | Center representation of a cluster |
| K | Number of clusters |
| Inertia | Sum of squared distances to assigned centroids |
| Elbow Method | Method for examining candidate values of K |
| PCA | Principal Component Analysis |
| Principal Component | New feature direction created by PCA |
| Explained Variance | Amount of variance represented by components |
| Dimensionality Reduction | Reducing the number of features/dimensions |
| DBSCAN | Density-based clustering algorithm |
| Hierarchical Clustering | Clustering based on a hierarchy of groups |

---

# IMPORTANT REVISION POINTS

## 32. Quick Revision

1. Unsupervised Learning works without predefined target labels.
2. Clustering is one of the major unsupervised learning tasks.
3. K-Means divides observations into K clusters.
4. A centroid represents the center of a K-Means cluster.
5. K-Means repeatedly assigns points and updates centroids.
6. K-Means is distance-based.
7. Feature scaling can be important before K-Means.
8. `kmeans.inertia_` gives the model's inertia.
9. The Elbow Method can help examine candidate values of K.
10. PCA means Principal Component Analysis.
11. PCA is used for dimensionality reduction.
12. PCA creates new features called principal components.
13. `n_components` controls the number of components retained.
14. `explained_variance_ratio_` shows the proportion of variance explained by each component.
15. `components_` stores principal-component directions.
16. PCA can make high-dimensional data easier to visualize.
17. PCA and K-Means have different purposes.
18. PCA can be combined with K-Means in a practical workflow.
19. DBSCAN is density-based clustering.
20. Hierarchical clustering builds a hierarchy of groups.

---

# CONCLUSION

Unsupervised Learning helps discover structure in data when predefined target labels are not available.

**K-Means** is a widely used clustering algorithm that groups observations around centroids.

**PCA** is a dimensionality-reduction technique that transforms the original feature space into principal components and can make high-dimensional data easier to analyze and visualize.

Together with other methods such as DBSCAN, hierarchical clustering, Gaussian mixture models, and visualization techniques, these tools form an important part of the Scikit-Learn unsupervised learning toolkit.

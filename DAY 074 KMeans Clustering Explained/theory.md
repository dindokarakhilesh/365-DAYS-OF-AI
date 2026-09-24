# DAY 73 — K-Means Clustering & Elbow Method

## 1. Introduction

K-Means Clustering is an **unsupervised machine learning algorithm** used to divide unlabeled data into a fixed number of groups called **clusters**.

The main idea is:

> Put similar data points into the same cluster and keep different clusters separated.

K-Means means:
- **K** = number of clusters
- **Means** = the mean/average position of points in a cluster, called its **centroid**

K-Means does not require a target/label column during training.

---

## 2. What Is Clustering?

Clustering means grouping data points according to similarity.

For example, a customer dataset can contain:
- annual income
- spending score

Without giving the model customer categories, K-Means can group customers according to their feature values.

The resulting cluster numbers are not automatically meaningful business labels; they are groups formed according to the data and distance objective.

---

## 3. Why K-Means Is Unsupervised Learning

Supervised learning uses labeled data:

`Features → Known Target`

K-Means uses unlabeled data:

`Features → Discovered Clusters`

Therefore, K-Means belongs to **Unsupervised Learning**.

---

## 4. Important Terms

### K
The number of clusters requested.

If `K = 3`, K-Means creates three clusters.

### Centroid
The center of a cluster. It is calculated from the mean of the points assigned to that cluster.

### Cluster
A group of data points assigned to the same centroid.

### Distance
K-Means commonly uses Euclidean distance to determine the nearest centroid.

### WCSS
**Within-Cluster Sum of Squares** measures the sum of squared distances between points and their assigned centroids.

### Inertia
Scikit-Learn's `inertia_` is the sum of squared distances of samples to their closest cluster center. It corresponds to the K-Means within-cluster squared-distance objective.

---

## 5. How K-Means Works

### Step 1 — Choose K
Decide how many clusters you want.

Example:

`K = 3`

K-Means itself does not automatically know the correct K.

### Step 2 — Initialize Centroids
K initial cluster centers are created. Scikit-Learn uses `k-means++` by default for initialization.

### Step 3 — Assign Points
Each point is assigned to the nearest centroid.

### Step 4 — Recalculate Centroids
For each cluster, the centroid is recalculated as the mean of its assigned points.

### Step 5 — Repeat
The algorithm repeats:

`Assign → Recalculate → Assign → Recalculate`

until convergence or the iteration limit is reached.

---

## 6. Mathematical Objective

K-Means tries to minimize the total squared distance between each point and its assigned centroid:

`WCSS = Σ(distance between point and centroid)²`

Lower WCSS means the clusters are more compact according to this objective.

However, lower WCSS alone cannot determine the best K because WCSS generally decreases when more clusters are added.

---

## 7. What Is WCSS?

**WCSS = Within-Cluster Sum of Squares**

It measures how far points inside clusters are from their cluster centroids.

Conceptually:

`WCSS = Sum of squared distances within all clusters`

Tightly packed clusters have lower WCSS.

Widely spread clusters have higher WCSS.

In Scikit-Learn, the `inertia_` value represents this within-cluster squared-distance objective.

---

## 8. Why We Cannot Just Choose the Lowest WCSS

Suppose we calculate WCSS for K = 1 through K = 10.

As K increases, the model has more centroids and can generally make clusters more compact. Therefore, WCSS decreases.

If we simply selected the smallest WCSS, we could keep increasing K and create too many clusters.

We need a way to identify where additional clusters stop giving large improvements.

---

# 9. Elbow Method

The **Elbow Method** is a graphical technique used to help choose a reasonable number of clusters for K-Means.

### Procedure

1. Train K-Means for several K values.
2. Calculate inertia/WCSS for each K.
3. Plot K on the X-axis.
4. Plot inertia/WCSS on the Y-axis.
5. Look for a noticeable bend, called the **elbow**.
6. Use the K around that bend as a candidate value.

The idea is that after the elbow, adding more clusters produces progressively smaller improvements.

The elbow method is a **heuristic**, not a guarantee of one mathematically correct K.

---

## 10. Example

Suppose:

| K | Inertia |
|---|---:|
| 1 | 500 |
| 2 | 280 |
| 3 | 150 |
| 4 | 120 |
| 5 | 105 |
| 6 | 98 |

The reductions are large at first and then become smaller. The bend in the curve can indicate a reasonable K.

If no clear bend exists, another validation method such as silhouette analysis can be considered.

---

## 11. Elbow Method Code Pattern

```python
wcss = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    wcss.append(model.inertia_)
```

Plot:

```python
plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS / Inertia")
plt.title("Elbow Method")
plt.show()
```

---

## 12. Why Feature Scaling Matters

K-Means is distance-based.

If one feature ranges from `0–100` and another from `0–100000`, the larger-scale feature can dominate distance calculations.

Therefore, numerical features often need scaling before K-Means.

A common approach:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Then run K-Means on `X_scaled`.

---

## 13. K-Means in Scikit-Learn

```python
from sklearn.cluster import KMeans

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

model.fit(X)
```

### Important methods and attributes

- `fit(X)` → trains the model
- `predict(X)` → assigns new samples to clusters
- `fit_predict(X)` → fits and returns cluster labels
- `cluster_centers_` → learned centroids
- `labels_` → training-sample cluster assignments
- `inertia_` → final within-cluster squared-distance objective

---

## 14. Important Parameters

### `n_clusters`
Number of clusters K.

### `init`
Centroid initialization method. Scikit-Learn commonly defaults to `k-means++`.

### `n_init`
Number of initializations. The best result according to the K-Means objective is retained.

### `max_iter`
Maximum iterations for one initialization.

### `tol`
Convergence tolerance.

### `random_state`
Makes the random component reproducible when applicable.

---

## 15. Advantages

- Simple to understand
- Easy to implement
- Relatively fast
- Useful for exploratory data analysis
- Works well for compact, approximately spherical groups
- Available directly in Scikit-Learn

---

## 16. Limitations

### 1. K must be chosen
The number of clusters must be supplied.

### 2. Feature scale matters
Unscaled features can distort distance calculations.

### 3. Sensitive to outliers
Extreme observations can affect centroids.

### 4. Shape assumptions
K-Means is generally more suitable for compact, roughly spherical and similarly scaled clusters than highly curved or irregular groups.

### 5. Initialization matters
Different starting points can lead to different local solutions. Multiple initializations reduce this problem.

### 6. Elbow may be unclear
Some datasets do not produce a clear elbow.

---

## 17. Elbow Method vs Silhouette Score

**Elbow Method**
- Uses inertia/WCSS
- Focuses on improvement in cluster compactness
- Uses a graph of K vs inertia

**Silhouette Score**
- Considers cohesion within a cluster and separation from other clusters
- Can provide additional evidence when choosing K

They can be used together when the elbow curve is ambiguous.

---

## 18. K-Means Workflow

```text
Raw Data
   ↓
Select Features
   ↓
Clean Data
   ↓
Scale Features
   ↓
Try Multiple K Values
   ↓
Calculate Inertia / WCSS
   ↓
Plot Elbow Curve
   ↓
Choose a Reasonable K
   ↓
Train Final K-Means Model
   ↓
Get Cluster Labels
   ↓
Analyze / Visualize Clusters
```

---

## 19. Applications

K-Means can be used for:
- customer segmentation
- market segmentation
- grouping similar products
- image color quantization
- grouping users by behavior
- exploratory pattern discovery

The interpretation of clusters depends on the selected features and the application.

---

## 20. K-Means vs Classification

### Classification
- Supervised learning
- Uses labeled data
- Predicts known classes

### K-Means
- Unsupervised learning
- Uses unlabeled data
- Creates groups based on feature similarity

Cluster numbers such as `0`, `1`, and `2` do not automatically represent meaningful class names.

---

## 21. Interview Questions

### What is K-Means?
K-Means is an unsupervised clustering algorithm that partitions data into K groups using cluster centroids.

### What is a centroid?
The mean position of the points assigned to a cluster.

### What is WCSS?
The sum of squared distances between points and their assigned cluster centroids.

### What is inertia?
In Scikit-Learn, inertia is the sum of squared distances from each sample to its closest cluster center.

### What is the Elbow Method?
A graphical method that helps choose K by examining how inertia/WCSS decreases as K increases.

### Why is scaling important?
Because K-Means uses distances, features with larger numerical scales can dominate.

### Is K-Means supervised?
No. It is unsupervised.

### Does K-Means automatically find K?
No. K must be supplied. Methods such as the Elbow Method can help select a reasonable value.

---

## 22. Quick Revision

```text
K-Means
↓
Unsupervised Learning
↓
Choose K
↓
Initialize Centroids
↓
Assign Points to Nearest Centroid
↓
Recalculate Centroids
↓
Repeat Until Convergence
↓
Measure Inertia / WCSS
↓
Use Elbow Method to Help Select K
```

### Key Terms

- K → number of clusters
- Cluster → group of similar points
- Centroid → mean center
- WCSS → Within-Cluster Sum of Squares
- Inertia → within-cluster squared-distance objective in Scikit-Learn
- Elbow Method → graphical heuristic for selecting K
- `cluster_centers_` → learned centroids
- `labels_` → cluster assignments
- `inertia_` → final inertia
- `fit_predict()` → fit and return labels

## Conclusion

K-Means is a fundamental unsupervised clustering algorithm.

Remember these core points:

1. K-Means works without target labels.
2. K is the number of clusters.
3. Each cluster has a centroid.
4. Points are assigned to the nearest centroid.
5. Centroids are repeatedly updated using the mean.
6. WCSS/inertia measures within-cluster squared distance.
7. Inertia generally decreases as K increases.
8. The Elbow Method helps identify a reasonable K by looking for diminishing improvement.
9. Feature scaling is important for distance-based clustering.
10. The Elbow Method is a heuristic and should be interpreted with the data context and, when useful, additional validation.

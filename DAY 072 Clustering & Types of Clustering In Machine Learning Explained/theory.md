# Clustering & Types of Clustering Algorithms — Theory

## 1. What is Clustering?

Clustering is an **unsupervised machine learning technique** used to organize unlabeled observations into groups called **clusters**.

Basic idea:

```text
Similar observations → Same cluster
Different observations → Different clusters
```

Unlike classification, clustering does not begin with predefined class labels. Different algorithms can define a cluster in different ways, such as distance from a centroid, density, hierarchy, or probability distribution.

---

## 2. Why Clustering is Used

Common applications include:

- Customer segmentation
- Document grouping
- Image analysis
- Pattern discovery
- Market segmentation
- Anomaly detection
- Exploratory data analysis
- Scientific data analysis

---

## 3. Similarity, Distance and Density

Clustering algorithms need a way to determine relationships between observations.

### Distance
Nearby observations can be considered similar. K-Means commonly uses distance to assign observations to centroids.

### Similarity
Some approaches work with similarity relationships between observations.

### Density
Density-based methods identify areas where observations are concentrated. DBSCAN is a major example.

---

# TYPES OF CLUSTERING

## 4. Major Types

Important clustering families include:

1. **Partitioning / Centroid-based Clustering**
2. **Hierarchical Clustering**
3. **Density-based Clustering**
4. **Distribution-based / Model-based Clustering**
5. **Mean / Mode-seeking Clustering**
6. **Graph-based Clustering**

These categories can overlap.

---

# 5. K-MEANS CLUSTERING

K-Means is a **centroid-based partitioning algorithm**.

It divides observations into a specified number of clusters represented by centroids.

### Main process

```text
Choose K
   ↓
Initialize centroids
   ↓
Assign points to nearest centroid
   ↓
Update centroids
   ↓
Repeat
   ↓
Final clusters
```

The number of clusters, `K`, is specified by the user.

### Scikit-Learn

```python
from sklearn.cluster import KMeans

model = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

labels = model.fit_predict(X)
```

Final centroids:

```python
model.cluster_centers_
```

---

## 6. K-Means Inertia

Inertia is the sum of squared distances between observations and their assigned cluster centroids.

```python
model.inertia_
```

Lower inertia means tighter clusters for that particular model, but increasing the number of clusters generally reduces inertia. Therefore, inertia alone does not determine the correct `K`.

---

# 7. HIERARCHICAL CLUSTERING

Hierarchical clustering creates a hierarchy of groups, commonly represented by a **dendrogram**.

### Agglomerative

Bottom-up:

```text
Each point starts as its own cluster
        ↓
Merge closest clusters
        ↓
Repeat
```

### Divisive

Top-down:

```text
Start with one cluster
        ↓
Split it
        ↓
Split again
```

Agglomerative clustering is widely used in practical workflows.

### Linkage

Common linkage methods include:

- Single
- Complete
- Average
- Ward

The linkage choice influences the resulting cluster structure.

### Scikit-Learn

```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(
    n_clusters=4,
    linkage="ward"
)

labels = model.fit_predict(X)
```

---

# 8. DBSCAN

**DBSCAN = Density-Based Spatial Clustering of Applications with Noise**

DBSCAN is a **density-based clustering algorithm**.

It identifies dense regions separated by lower-density regions. Sparse observations can be labeled as **noise**, and the method can discover many irregular cluster shapes.

### Important parameters

#### `eps`
Neighborhood radius.

```python
eps=0.5
```

#### `min_samples`
Minimum number of samples required in a neighborhood for a dense region.

```python
min_samples=5
```

Example:

```python
from sklearn.cluster import DBSCAN

model = DBSCAN(
    eps=0.5,
    min_samples=5
)

labels = model.fit_predict(X)
```

Scikit-Learn uses:

```text
-1
```

for noise points.

---

# 9. MEAN SHIFT CLUSTERING

Mean Shift is a **mode-seeking** clustering approach.

It moves observations toward regions of higher estimated density. Dense regions form modes around which observations can be grouped.

Unlike K-Means, Mean Shift does not directly require the number of clusters to be specified in advance. Its behavior depends on the neighborhood scale or bandwidth.

Scikit-Learn:

```python
from sklearn.cluster import MeanShift

model = MeanShift()
labels = model.fit_predict(X)
```

---

# 10. GAUSSIAN MIXTURE MODEL

A **Gaussian Mixture Model (GMM)** is a probability-based or model-based clustering method.

It represents data as a mixture of probability distributions.

A major feature is **soft clustering**.

Example membership probabilities:

```text
Cluster 0 → 0.10
Cluster 1 → 0.75
Cluster 2 → 0.15
```

Scikit-Learn:

```python
from sklearn.mixture import GaussianMixture

model = GaussianMixture(
    n_components=3,
    random_state=42
)

model.fit(X)

labels = model.predict(X)
probabilities = model.predict_proba(X)
```

---

# 11. HARD CLUSTERING VS SOFT CLUSTERING

### Hard Clustering

Each observation receives one cluster assignment.

K-Means is a common example.

### Soft Clustering

An observation can have membership probabilities for multiple clusters.

GMM provides probabilistic membership.

---

# 12. K-MEANS vs HIERARCHICAL vs DBSCAN

| Feature | K-Means | Hierarchical | DBSCAN |
|---|---|---|---|
| Main idea | Centroids | Nested groups | Density |
| Specify K directly | Yes | Not necessarily | No fixed K |
| Noise handling | Not explicit | Not designed for it | Yes |
| Irregular shapes | Limited | Depends on linkage | Good for many irregular shapes |
| Main parameters | `n_clusters` | linkage / cluster selection | `eps`, `min_samples` |

No single clustering algorithm is universally correct. The choice depends on cluster shape, density, noise, dataset size, and the goal of the analysis.

---

# 13. SILHOUETTE SCORE

Silhouette Score is an internal clustering evaluation metric.

It considers:

- How close a sample is to its own cluster.
- How separated it is from neighboring clusters.

Its range is:

```text
-1 to +1
```

Higher values generally indicate better separation and cohesion, but the score should be interpreted together with the data and application.

Scikit-Learn:

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
```

---

# 14. FEATURE SCALING

Scaling can be important for distance-based algorithms.

Example:

```text
Age       → 18–70
Income    → 10,000–1,000,000
```

The larger-scale feature can dominate distance calculations.

A common approach:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

# 15. PCA AND CLUSTERING

PCA is **not a clustering algorithm**.

PCA is a dimensionality-reduction technique. It can help:

- Reduce dimensions
- Remove some redundancy
- Visualize high-dimensional data
- Inspect cluster structure

Example:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

Clustering can then be applied to an appropriate representation of the data.

---

# 16. PRACTICAL CLUSTERING WORKFLOW

```text
Raw Dataset
     ↓
Understand Features
     ↓
Clean Data
     ↓
Handle Missing Values
     ↓
Select Features
     ↓
Scale / Normalize When Appropriate
     ↓
Choose Clustering Method
     ↓
Train Algorithm
     ↓
Analyze Clusters
     ↓
Evaluate
     ↓
Visualize
     ↓
Interpret Results
```

---

# 17. IMPORTANT SCikit-Learn CLASSES

| Class | Purpose |
|---|---|
| `KMeans` | Centroid-based clustering |
| `AgglomerativeClustering` | Hierarchical clustering |
| `DBSCAN` | Density-based clustering |
| `MeanShift` | Mode-seeking clustering |
| `GaussianMixture` | Probability-based clustering |
| `silhouette_score` | Internal evaluation |
| `StandardScaler` | Feature scaling |
| `PCA` | Dimensionality reduction |

---

# 18. APPLICATIONS

Clustering can be used for:

1. Customer segmentation
2. Market analysis
3. Image analysis
4. Document organization
5. Recommendation systems
6. Anomaly detection
7. Social network analysis
8. Scientific data analysis
9. Geographic data analysis
10. Exploratory data analysis

---

# 19. KEY TERMS

| Term | Meaning |
|---|---|
| Clustering | Grouping similar observations |
| Cluster | A discovered group |
| Centroid | Center of a centroid-based cluster |
| K-Means | Centroid-based clustering |
| Hierarchical Clustering | Nested cluster structure |
| Dendrogram | Tree representation of hierarchy |
| DBSCAN | Density-based clustering with noise handling |
| `eps` | DBSCAN neighborhood radius |
| `min_samples` | DBSCAN density threshold |
| Mean Shift | Mode-seeking clustering |
| GMM | Gaussian Mixture Model |
| Hard Clustering | One cluster per sample |
| Soft Clustering | Probabilistic membership |
| Inertia | K-Means squared-distance measure |
| Silhouette Score | Internal cluster separation/cohesion measure |
| PCA | Principal Component Analysis |

---

# 20. QUICK REVISION

1. Clustering is an **unsupervised learning** technique.
2. It works without predefined target labels.
3. K-Means is centroid-based.
4. K-Means requires the number of clusters.
5. Hierarchical clustering builds nested groups.
6. A dendrogram represents hierarchical structure.
7. DBSCAN is density-based.
8. DBSCAN can identify noise.
9. DBSCAN can handle many irregular cluster shapes.
10. Mean Shift is a mode-seeking method.
11. GMM is a probability-based clustering approach.
12. K-Means commonly gives hard assignments.
13. GMM can provide soft membership probabilities.
14. Silhouette Score is an internal clustering metric.
15. Feature scaling can matter for distance-based methods.
16. PCA is dimensionality reduction, not clustering.
17. Different algorithms can produce different groupings.
18. There is no universally correct clustering algorithm.
19. Dataset structure should guide algorithm selection.
20. Evaluation and interpretation are important after clustering.

---

# CONCLUSION

Clustering is an important part of unsupervised machine learning.

Different algorithms define clusters differently:

```text
K-Means       → Centroids
Hierarchical  → Nested relationships
DBSCAN        → Density
Mean Shift    → Density modes
GMM           → Probability distributions
```

Understanding these differences helps us select an appropriate approach for different datasets and real-world problems.

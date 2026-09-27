# Hierarchical Clustering Explained

## 1. What is Hierarchical Clustering?

Hierarchical clustering is an **unsupervised machine learning** technique used to group similar data points into clusters by building a **hierarchy (tree) of clusters**. Unlike k-means, it does not require you to specify the number of clusters in advance — instead it produces a tree-like structure called a **dendrogram** that shows how clusters merge or split at every level of similarity, and you can "cut" the tree at any level to get the number of clusters you want.

There are two main strategies:

1. **Agglomerative (bottom-up)** — start with every point as its own cluster and repeatedly merge the closest pair of clusters until only one cluster remains.
2. **Divisive (top-down)** — start with all points in one cluster and repeatedly split it into smaller clusters until every point is its own cluster.

---

## 2. Agglomerative Clustering (Bottom-Up)

This is the more commonly used approach. The algorithm works as follows:

### Algorithm
1. Treat each of the `n` data points as an individual cluster (so you start with `n` clusters).
2. Compute the **distance matrix** — pairwise distances between all clusters.
3. Merge the two clusters that are **closest** to each other into a single cluster.
4. Update the distance matrix to reflect the distance between the new cluster and all other clusters.
5. Repeat steps 3–4 until only one cluster remains (or a stopping criterion is met).
6. Represent the merge history as a **dendrogram**.

### Why "bottom-up"?
Because the process starts from individual leaves (points) and works its way up to a single root (one cluster containing everything) — much like building a family tree upward from individuals to a common ancestor.

### Complexity
- Naively O(n³) time and O(n²) space, because at every step you scan the distance matrix to find the minimum. Optimized implementations (e.g., using priority queues) bring this down to roughly O(n² log n).
- This makes agglomerative clustering **expensive for very large datasets**.

---

## 3. Divisive Clustering (Top-Down)

Divisive clustering works in the opposite direction:

### Algorithm
1. Start with all `n` points in a single cluster.
2. Split the cluster into two sub-clusters using some criterion (e.g., a flat clustering algorithm like k-means with k=2, or by finding the point that is most dissimilar to the rest).
3. Recursively repeat the split on each resulting cluster.
4. Stop when each cluster contains a single point, or when a stopping condition is reached.

### Why is it less common?
- Choosing the "best" way to split a cluster at each step is a harder combinatorial problem than choosing the best pair to merge (there are exponentially many ways to split a set of points into two groups).
- It is generally more computationally expensive than agglomerative clustering, so it is used less often in practice, though it can produce better results when the natural structure of the data has a few large, well-separated groups.

---

## 4. Distance Matrix

The **distance matrix** is an `n × n` symmetric matrix where the entry `(i, j)` represents the distance (dissimilarity) between data points `i` and `j`. It is the foundational data structure hierarchical clustering operates on.

|        | A   | B   | C   | D   |
|--------|-----|-----|-----|-----|
| **A**  | 0   | d(A,B) | d(A,C) | d(A,D) |
| **B**  | d(A,B) | 0 | d(B,C) | d(B,D) |
| **C**  | d(A,C) | d(B,C) | 0 | d(C,D) |
| **D**  | d(A,D) | d(B,D) | d(C,D) | 0 |

### Common Distance Metrics (between individual points)
- **Euclidean distance**: `sqrt(Σ(xᵢ - yᵢ)²)` — most common, used for continuous numeric data.
- **Manhattan distance**: `Σ|xᵢ - yᵢ|` — sum of absolute differences, robust to outliers.
- **Cosine distance**: `1 - cos(θ)` — used for high-dimensional / text data where direction matters more than magnitude.
- **Hamming distance**: for categorical/binary data — counts differing positions.

### Linkage Criteria (how to measure distance *between clusters*, not just points)
Once clusters contain more than one point, we need a rule to define the distance between two clusters:

| Linkage | Definition | Characteristic |
|---|---|---|
| **Single linkage** | Minimum distance between any pair of points, one from each cluster | Can produce long, "chained" clusters; sensitive to noise |
| **Complete linkage** | Maximum distance between any pair of points, one from each cluster | Produces compact, tighter clusters |
| **Average linkage** | Average distance between all pairs of points across the two clusters | Balanced compromise between single and complete |
| **Centroid linkage** | Distance between the centroids (means) of the two clusters | Intuitive but can cause "inversions" in the dendrogram |
| **Ward's method** | Merges the pair of clusters that leads to the minimum increase in total within-cluster variance | Tends to produce evenly sized, compact clusters; very popular default |

The choice of linkage significantly changes the shape and quality of the resulting clusters, so it's often worth experimenting with more than one.

---

## 5. Dendrogram

A **dendrogram** is a tree diagram that records the sequence of merges (or splits) performed during hierarchical clustering.

### How to read it
- Each **leaf** at the bottom represents a single data point.
- Each **horizontal merge line (or "U" shape)** represents a point where two clusters were joined.
- The **height (y-axis)** at which two clusters merge represents the **distance/dissimilarity** between them — the higher the merge point, the less similar the two clusters were.
- **Cutting the dendrogram horizontally** at a chosen height gives you a specific number of clusters — everything below the cut line stays grouped, and the number of vertical lines the cut crosses equals the number of clusters.

### Choosing the number of clusters from a dendrogram
A common heuristic is to look for the **longest vertical distance** that isn't crossed by any horizontal merge line — cutting through the middle of that gap tends to give a natural, well-separated clustering.

---

## 6. Agglomerative vs. Divisive — Summary

| Aspect | Agglomerative | Divisive |
|---|---|---|
| Direction | Bottom-up (merge) | Top-down (split) |
| Starting point | n singleton clusters | 1 cluster with all points |
| Computational cost | O(n² log n) typical | Generally higher (splitting is combinatorially harder) |
| Popularity | Very common, widely implemented (scipy, sklearn) | Rare in practice |
| Best suited for | General-purpose clustering, smaller/medium datasets | Cases where a few large, well-defined groups are expected |

---

## 7. Advantages and Disadvantages

### Advantages
- No need to pre-specify the number of clusters.
- Produces an interpretable dendrogram showing relationships at every scale.
- Works with any valid distance metric — flexible for numeric, categorical, or mixed data.
- Deterministic — same input always produces the same result (unlike k-means, which depends on random initialization).

### Disadvantages
- Computationally expensive for large datasets (O(n²) or worse in time/space).
- Once a merge or split is made, it **cannot be undone** (greedy algorithm) — an early "wrong" merge propagates through the whole hierarchy.
- Sensitive to noise and outliers, especially with single linkage.
- Choice of distance metric and linkage method strongly affects results, and there's no single "correct" choice.

---

## 8. Real-World Applications
- Gene expression analysis / bioinformatics (grouping genes or samples with similar expression patterns)
- Document and text clustering
- Customer segmentation in marketing
- Image segmentation
- Social network analysis (community detection)
- Taxonomy and phylogenetic tree construction

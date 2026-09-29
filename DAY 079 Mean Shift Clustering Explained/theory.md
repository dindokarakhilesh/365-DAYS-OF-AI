# Mean Shift Clustering Explained

## 1. What is Mean Shift Clustering?

Mean Shift is an **unsupervised, centroid-based, non-parametric** clustering algorithm. Instead of asking you for the number of clusters (like k-means), it finds clusters by locating the **dense regions (modes)** of the data distribution. Every point "climbs the hill" of the estimated density until it reaches a peak, and points that climb to the same peak belong to the same cluster.

Key ideas:
- The number of clusters is **discovered automatically** from the data.
- The only important hyperparameter is the **bandwidth** (the "window size").
- Clusters can have arbitrary shapes in the sense that they are defined by density peaks, not by a fixed number of centroids (though each cluster is still represented by a mode).

---

## 2. Intuition: Sliding Windows Climbing Density Hills

Imagine the data as points scattered on a 2D plane:

1. Place a circular **window** (with radius = bandwidth) centered on a point.
2. Compute the **mean** of all data points that fall inside the window.
3. **Shift** the window's center to that mean.
4. Repeat until the center stops moving (convergence).

Because the mean of points inside a window is always pulled toward the denser side, the window slides uphill towards a local density maximum. When many windows converge to the same location, that location is a **cluster center (mode)**.

---

## 3. Kernel Density Estimation (KDE) — The Math Behind It

Mean Shift is closely tied to **Kernel Density Estimation**, a way of estimating the probability density function (PDF) of the data without assuming a distribution.

For `n` points `x₁, …, xₙ` in `d` dimensions, the KDE at a point `x` is:

```
f̂(x) = (1 / (n · hᵈ)) · Σᵢ K( (x − xᵢ) / h )
```

where:
- `K` is the **kernel function** (a smooth, symmetric bump),
- `h` is the **bandwidth**, which controls the width of each bump.

Each data point contributes a small bump; summing all bumps gives a smooth density surface. The peaks of that surface are the **modes** that Mean Shift searches for.

### Common Kernels
| Kernel | Formula (in terms of distance `r = ‖x − xᵢ‖`) | Notes |
|---|---|---|
| **Flat (uniform)** | `1` if `r ≤ h`, else `0` | Simple window; all points inside count equally. This is what scikit-learn's `MeanShift` uses. |
| **Gaussian** | `exp(−r² / (2h²))` | Smooth; nearby points weigh more than far ones. |

### Gradient of the density and the mean shift vector
Taking the gradient of the KDE shows that it points toward higher density. The **mean shift vector** is:

```
m(x) = ( Σᵢ K(xᵢ − x) · xᵢ ) / ( Σᵢ K(xᵢ − x) )  −  x
```

It is the difference between the **kernel-weighted mean** of the neighbors and the current position `x`. Updating `x ← x + m(x)` is a step of gradient ascent on the density, with an automatically adapted step size.

---

## 4. Bandwidth (Window Size) — The Most Important Parameter

The bandwidth `h` controls how large the neighborhood is when computing the mean.

| Bandwidth | Effect |
|---|---|
| **Too small** | Many tiny windows, each finds its own little peak → **too many clusters** (over-segmentation), noisy density estimate. |
| **Too large** | Windows cover multiple true clusters → peaks blur together → **too few clusters** (under-segmentation). |
| **Just right** | Peaks correspond to real structure in the data. |

### How to choose the bandwidth
- **`sklearn.cluster.estimate_bandwidth`**: estimates a bandwidth using the average distance to the k-th nearest neighbors, controlled by a `quantile` parameter (default 0.3). Lower quantile → smaller bandwidth → more clusters.
- **Domain knowledge**: choose a window size that matches the scale at which you consider points "similar".
- **Cross-checking with a KDE plot and the silhouette score**: try several values and inspect.
- Scale your features first (e.g., `StandardScaler`) because bandwidth is a single value applied to all dimensions.

---

## 5. The Algorithm Step by Step

1. **Initialize**: use every data point as a starting seed (or a subset via *bin seeding* for speed).
2. **For each seed**, repeat:
   1. Find all points within `bandwidth` of the current center.
   2. Compute their mean (the new center).
   3. Stop when the shift is smaller than a small tolerance (or a max number of iterations is reached).
3. **Merge modes**: seeds that converged to (nearly) the same location — within `bandwidth` of each other — are merged into a single cluster center. When two centers are close, the one with more points inside its window is kept.
4. **Assign labels**: each data point is assigned to its nearest final cluster center.

### Pseudocode
```
for each seed s:
    while True:
        neighbors = points within bandwidth of s
        new_s = mean(neighbors)
        if ||new_s - s|| < tol: break
        s = new_s
    record (s, len(neighbors))

remove near-duplicate centers (keep the one with the most neighbors)
label every point by its nearest surviving center
```

---

## 6. Scikit-learn Implementation Notes

```python
from sklearn.cluster import MeanShift, estimate_bandwidth

bw = estimate_bandwidth(X, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bw, bin_seeding=True)
labels = ms.fit_predict(X)
centers = ms.cluster_centers_
```

Important parameters:
- **`bandwidth`**: window radius. If `None`, it is estimated automatically with `estimate_bandwidth` (which can be slow on large data).
- **`bin_seeding`**: if `True`, seeds are placed on a coarse grid instead of at every point → much faster.
- **`min_bin_freq`**: minimum number of points a bin needs to be used as a seed (with `bin_seeding=True`).
- **`cluster_all`**: if `True` (default), every point is assigned to a cluster; if `False`, orphan points (not within any seed's window) get label `-1`, which acts like outlier detection.
- **`max_iter`**: max number of shift iterations per seed.
- **`n_jobs`**: parallelizes the seed processing.

Attributes after fitting: `cluster_centers_`, `labels_`, `n_iter_`.

---

## 7. Complexity

- Time: roughly **O(T · n²)** in the worst case, where `T` is the number of iterations, because each shift requires finding neighbors of every seed among `n` points. With low-dimensional data and `bin_seeding`, it is much faster in practice.
- Space: O(n).
- Not well suited for very large datasets or very high-dimensional data (density estimation degrades in high dimensions — the "curse of dimensionality").

---

## 8. Mean Shift vs. K-Means

| Aspect | Mean Shift | K-Means |
|---|---|---|
| Number of clusters | Found automatically | Must be specified (`k`) |
| Key parameter | Bandwidth | `k` |
| Cluster shape assumption | None explicit (density modes) | Roughly spherical, similar size |
| Outlier handling | Can leave orphans (`cluster_all=False`) | Every point forced into a cluster |
| Scalability | Slow for large `n` | Fast and scalable |
| Determinism | Deterministic given the same seeds | Depends on random initialization |

---

## 9. Advantages and Disadvantages

### Advantages
- No need to choose the number of clusters.
- Finds clusters of varying shapes and sizes based on density.
- Robust to outliers and only one main hyperparameter.
- Simple, intuitive, and mathematically grounded (mode seeking on a KDE).

### Disadvantages
- **Computationally expensive** — poor scaling with dataset size.
- **Bandwidth selection is critical** and the result can change drastically with it.
- A single global bandwidth performs poorly when clusters have very different densities.
- Struggles in high-dimensional spaces.

---

## 10. Real-World Applications
- **Image segmentation** and color quantization (clustering pixels in color/space)
- **Object tracking** in computer vision (e.g., the CamShift tracker)
- **Mode detection** and density-based feature analysis
- **Geospatial hotspot detection** (finding dense regions of GPS points)
- **Customer segmentation** when the number of segments is unknown

# Customer Segmentation with KMeans Clustering

## 1. Introduction

Customer Segmentation is the process of dividing customers into different groups based on their similarities, behavior, characteristics, or purchasing patterns.

In this project, we use **KMeans Clustering**, an unsupervised machine learning algorithm, to identify different customer segments.

The project also demonstrates how the clustering model can be used in a **Streamlit web application** for interactive customer segmentation.

---

## 2. What is Customer Segmentation?

Customer segmentation means grouping customers who have similar characteristics.

For example, a business may have customers with:

* Different income levels
* Different spending habits
* Different purchase frequencies
* Different ages
* Different purchase values

Instead of treating every customer in the same way, segmentation allows a business to understand these groups separately.

### General Workflow

```text
Customer Data
      ↓
Data Cleaning
      ↓
Feature Selection
      ↓
Feature Scaling
      ↓
KMeans Clustering
      ↓
Cluster Analysis
      ↓
Customer Segments
```

---

## 3. What is Clustering?

Clustering is an **unsupervised machine learning technique**.

In supervised learning, we normally have:

```text
Input → Known Target
```

But in clustering:

```text
Input → No Target
```

The algorithm automatically tries to discover patterns and groups inside the data.

Some popular clustering algorithms are:

* KMeans
* Hierarchical Clustering
* DBSCAN

In this project, we use **KMeans**.

---

# 4. What is KMeans Clustering?

KMeans is an unsupervised machine learning algorithm that divides data into a specified number of clusters.

The letter **K** represents the number of clusters we want to create.

For example:

```text
K = 3
```

means that the algorithm will try to divide the dataset into 3 clusters.

Each cluster has a **centroid**, which represents the center of that cluster.

---

## 5. How KMeans Works

KMeans follows an iterative process.

### Step 1 — Select K

First, we decide how many clusters we want.

Example:

```python
K = 5
```

---

### Step 2 — Initialize Centroids

KMeans initially selects cluster centers called **centroids**.

---

### Step 3 — Assign Data Points

Each data point is assigned to the nearest centroid.

The distance is commonly calculated using Euclidean distance.

---

### Step 4 — Recalculate Centroids

After assigning the points, KMeans calculates a new center for every cluster.

---

### Step 5 — Repeat

The assignment and centroid calculation process continues until the clusters become stable or the algorithm reaches its iteration limit.

---

## 6. KMeans Objective

KMeans tries to minimize the distance between data points and their assigned cluster centers.

Its objective can be represented as:

```text
Minimize:

Σ distance²(point, centroid)
```

This is commonly represented by **inertia** in Scikit-Learn.

---

# 7. Important KMeans Parameters

In Scikit-Learn, KMeans can be created using:

```python
from sklearn.cluster import KMeans

model = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
```

### `n_clusters`

Defines the number of clusters.

```python
n_clusters=5
```

means 5 clusters will be created.

---

### `random_state`

Controls the randomness used during initialization.

```python
random_state=42
```

Using a fixed value makes the result reproducible.

---

### `n_init`

Specifies how many times KMeans is run with different centroid initializations.

The best result based on the objective is selected.

---

# 8. Choosing the Number of Clusters

One of the main challenges in KMeans is deciding the value of **K**.

A commonly used technique is the:

## Elbow Method

The Elbow Method calculates the KMeans **inertia** for different values of K.

Example:

```python
inertias = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertias.append(model.inertia_)
```

Then we plot K against inertia.

```text
Inertia
  |
  |\
  | \
  |  \
  |   \__
  |      \____
  |
  +----------------
       K
```

The point where the improvement begins to slow down is often called the **elbow**.

That point can be used as a candidate value for K.

### Important

The Elbow Method is a heuristic.

There is not always one mathematically guaranteed "correct" value of K.

Other information such as cluster interpretability, business requirements, and validation metrics can also be considered.

---

# 9. What is Inertia?

**Inertia** measures how close the data points are to their assigned cluster centers.

In simple terms:

```text
Lower inertia
      ↓
Points are closer to their centroids
```

However, inertia generally decreases as the number of clusters increases.

Therefore, we should not simply choose the largest K because it has lower inertia.

This is why the Elbow Method is useful.

---

# 10. Feature Selection

Before applying KMeans, we need to select appropriate features.

For customer segmentation, possible features include:

* Age
* Annual Income
* Spending Score
* Number of Purchases
* Recency
* Frequency
* Monetary Value

Example:

```python
features = [
    "Annual_Income",
    "Spending_Score",
    "Purchases"
]

X = df[features]
```

The selected features should represent meaningful customer characteristics.

---

# 11. Feature Scaling

Feature scaling is very important for KMeans because KMeans is distance-based.

Suppose we have:

```text
Annual Income → 15,000 to 120,000

Spending Score → 1 to 100
```

Annual Income has a much larger numerical range.

Without scaling, the income feature can have a much stronger effect on the distance calculation.

Therefore, we commonly scale the features before clustering.

---

# 12. StandardScaler

Scikit-Learn provides `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Standardization approximately transforms each feature using:

```text
z = (x - mean) / standard deviation
```

After scaling, features are placed on a more comparable scale.

---

# 13. Training the KMeans Model

After preprocessing and scaling, we can train KMeans.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)
```

---

# 14. Getting Cluster Labels

After training, every customer receives a cluster label.

```python
df["Cluster"] = kmeans.labels_
```

Or:

```python
df["Cluster"] = kmeans.fit_predict(X_scaled)
```

The result may look like:

```text
Customer_ID    Cluster
1              2
2              0
3              4
4              1
5              2
```

---

# 15. Understanding Cluster Labels

Cluster numbers are only identifiers.

For example:

```text
Cluster 0
Cluster 1
Cluster 2
Cluster 3
Cluster 4
```

There is no inherent meaning that Cluster 4 is better than Cluster 1.

We need to examine the characteristics of each cluster.

---

# 16. Cluster Profiling

Cluster profiling helps us understand what makes each group different.

Example:

```python
profile = df.groupby("Cluster")[features].mean()

print(profile)
```

This can produce something similar to:

```text
Cluster    Income    Spending    Purchases
0          ...
1          ...
2          ...
3          ...
4          ...
```

We can then study the average characteristics of every cluster.

---

# 17. Naming Customer Segments

After examining the cluster profiles, meaningful names can be assigned.

For example, depending on the actual data, a cluster could represent:

* High-value customers
* Frequent customers
* Low-engagement customers
* Budget-oriented customers
* Occasional customers

These names should be based on the actual cluster statistics rather than the cluster number.

---

# 18. Visualizing Customer Segments

Visualization makes clusters easier to understand.

For example, we can visualize:

```python
plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Cluster"]
)
```

This creates a two-dimensional representation of the customer segments.

A visualization can help us identify whether different groups appear separated or overlapping.

---

# 19. Silhouette Score

The **Silhouette Score** is another metric that can be used to evaluate clustering structure.

In Scikit-Learn:

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(
    X_scaled,
    df["Cluster"]
)

print(score)
```

The silhouette score considers:

* How close a data point is to its own cluster.
* How far it is from other clusters.

Its value ranges approximately from:

```text
-1 to +1
```

A higher value generally indicates better separation under this metric.

However, the score should not be interpreted alone. Domain knowledge and cluster profiles are also important.

---

# 20. Customer Segmentation with Streamlit

**Streamlit** allows us to convert the Python machine learning workflow into an interactive web application.

The application can allow users to:

* Upload a customer CSV file.
* Select numerical features.
* Select the number of clusters.
* Train the KMeans model.
* View cluster profiles.
* View cluster distributions.
* Visualize customer segments.
* Download the segmented dataset.

---

# 21. Streamlit Workflow

The Streamlit application follows this workflow:

```text
Upload CSV
     ↓
Read Dataset
     ↓
Select Numerical Features
     ↓
Select Number of Clusters
     ↓
Handle Missing Values
     ↓
Scale Features
     ↓
Train KMeans
     ↓
Generate Cluster Labels
     ↓
Display Cluster Profile
     ↓
Visualize Clusters
     ↓
Download Results
```

---

# 22. Running the Streamlit Application

Install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn streamlit
```

Then run:

```bash
streamlit run app.py
```

Streamlit will start a local web application.

---

# 23. Advantages of KMeans for Customer Segmentation

KMeans has several advantages:

* Simple to understand.
* Easy to implement.
* Computationally efficient for many datasets.
* Works well with numerical features.
* Easy to visualize.
* Useful for discovering customer behavior patterns.
* Can be integrated into business dashboards.

---

# 24. Limitations of KMeans

KMeans also has limitations:

* The number of clusters must be selected.
* It is sensitive to feature scaling.
* It can be affected by outliers.
* Different initialization can produce different results.
* It generally works best when clusters have suitable geometric structure.
* It is not directly designed for categorical-only data.
* Cluster labels do not have inherent business meaning.

---

# 25. Real-World Applications

Customer segmentation can be used in:

### Marketing

Businesses can create targeted marketing campaigns for different customer groups.

### Personalized Offers

Different customer segments can receive different offers.

### Customer Retention

Businesses can identify groups with lower engagement and study their behavior.

### Product Recommendations

Customer behavior patterns can support recommendation systems.

### Loyalty Programs

Customer segments can help businesses design different loyalty strategies.

### Sales Analysis

Businesses can analyze purchasing patterns across customer groups.

---

# 26. Complete Project Architecture

```text
Customer Dataset
       │
       ▼
Data Loading
       │
       ▼
Data Inspection
       │
       ▼
Data Cleaning
       │
       ▼
Feature Selection
       │
       ▼
Feature Scaling
       │
       ▼
Elbow Method
       │
       ▼
Select K
       │
       ▼
KMeans Model
       │
       ▼
Cluster Labels
       │
       ▼
Cluster Profiling
       │
       ▼
Visualization
       │
       ▼
Streamlit Application
```

---

# 27. Important Python Libraries

## Pandas

Used for:

* Loading data
* Cleaning data
* Data manipulation
* Grouping and analysis

```python
import pandas as pd
```

---

## NumPy

Used for numerical operations.

```python
import numpy as np
```

---

## Matplotlib

Used for visualization.

```python
import matplotlib.pyplot as plt
```

---

## Scikit-Learn

Used for:

* KMeans
* StandardScaler
* Silhouette Score

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
```

---

## Streamlit

Used to create the interactive web application.

```python
import streamlit as st
```

---

# 28. Key Concepts to Remember

### Customer Segmentation

Dividing customers into meaningful groups.

### Clustering

An unsupervised learning technique used to discover groups.

### KMeans

An algorithm that divides data into K clusters.

### Centroid

The center of a cluster.

### Inertia

A measure of within-cluster squared distances.

### Elbow Method

A heuristic for selecting a suitable number of clusters.

### Feature Scaling

Transforming features to comparable scales.

### Cluster Profile

Summary of the characteristics of each cluster.

### Silhouette Score

A metric for assessing cluster separation and cohesion.

---

# 29. Project Outcome

After completing this project, we can:

1. Load customer data using Pandas.
2. Select useful numerical features.
3. Scale features using StandardScaler.
4. Use the Elbow Method to investigate possible values of K.
5. Train a KMeans clustering model.
6. Generate customer cluster labels.
7. Analyze cluster profiles.
8. Visualize customer segments.
9. Calculate a Silhouette Score.
10. Build an interactive Streamlit application.
11. Export the segmented customer dataset.

---

# 30. Conclusion

KMeans Clustering is a useful unsupervised machine learning technique for discovering groups of similar customers.

In this project, we combine:

```text
Python
+
Pandas
+
Scikit-Learn
+
KMeans
+
Feature Scaling
+
Elbow Method
+
Cluster Profiling
+
Visualization
+
Streamlit
```

to create an end-to-end **Customer Segmentation System**.

The machine learning model discovers the groups, while cluster profiling and domain knowledge help us understand what those groups represent.

This project demonstrates how an unsupervised machine learning algorithm can be taken from data preprocessing to an interactive application.

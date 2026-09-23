# 📊 Customer Segmentation with KMeans Clustering

A complete **Customer Segmentation Machine Learning project** using **Python, KMeans Clustering, Scikit-Learn, Matplotlib, and Streamlit**.

This project demonstrates how to divide customers into meaningful groups based on numerical customer characteristics and behavior.

---

## 🚀 Project Overview

Customer Segmentation is an important application of **Unsupervised Machine Learning**.

In this project, we use **KMeans Clustering** to automatically group customers based on selected numerical features.

The project contains:

* Data preprocessing
* Feature selection
* Feature scaling
* Elbow Method
* KMeans Clustering
* Cluster profiling
* Silhouette Score
* Data visualization
* Streamlit web application
* Segmented CSV export

---

# 📁 Project Structure

```text
DAY 68/
│
├── theory.md
├── code.ipynb
├── app.py
├── requirements.txt
├── README.md
│
└── data/
    └── README.md
```

### File Description

| File               | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `theory.md`        | Complete theoretical notes                           |
| `code.ipynb`       | Jupyter Notebook containing the complete ML workflow |
| `app.py`           | Interactive Streamlit application                    |
| `requirements.txt` | Required Python libraries                            |
| `README.md`        | Project documentation                                |
| `data/`            | Folder for customer datasets                         |

---

# 🧠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* Streamlit
* Jupyter Notebook

---

# 📦 Installation

## 1. Clone the Repository

First, clone the repository:

```bash
git clone https://github.com/dindokarakhilesh/365-DAYS-OF-AI.git
```

Then move into the project directory:

```bash
cd 365-DAYS-OF-AI
```

Navigate to the **DAY 68** folder:

```bash
cd "DAY 68"
```

> The exact folder name may differ depending on how the repository is organized.

---

# 🐍 2. Create a Virtual Environment

It is recommended to use a separate Python environment for the project.

### Using Conda

```bash
conda create -n customer-segmentation python=3.11
```

Activate it:

```bash
conda activate customer-segmentation
```

### Or using Python venv

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

---

# 📥 3. Install Required Libraries

Make sure you are inside the project folder containing `requirements.txt`.

Then run:

```bash
pip install -r requirements.txt
```

This installs all required dependencies.

---

# 📓 4. Run the Jupyter Notebook

The machine learning workflow can be explored through:

```text
code.ipynb
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
code.ipynb
```

Run the cells from top to bottom.

The notebook demonstrates:

```text
Dataset
   ↓
Data Inspection
   ↓
Feature Selection
   ↓
Feature Scaling
   ↓
Elbow Method
   ↓
KMeans
   ↓
Cluster Profiling
   ↓
Silhouette Score
   ↓
Visualization
   ↓
CSV Export
```

---

# 🌐 5. Run the Streamlit Application

The interactive application is located in:

```text
app.py
```

## ⚠️ Important

Do **not** run the application like a normal Python script:

```bash
python app.py
```

For Streamlit applications, use the Streamlit command:

```bash
streamlit run app.py
```

This is the correct way to start the application.

---

# ▶️ Complete Streamlit Run Steps

### Step 1 — Open Terminal

Open Command Prompt, PowerShell, Anaconda Prompt, or VS Code Terminal.

---

### Step 2 — Activate Your Environment

If you created a Conda environment:

```bash
conda activate customer-segmentation
```

Or activate your existing Python environment.

---

### Step 3 — Navigate to the Project Folder

Move to the folder containing `app.py`.

Example:

```bash
cd "F:\AI Engineer Roadmap\DAY 68"
```

The path will be different on every computer.

You can verify the files using:

```bash
dir
```

You should see something similar to:

```text
app.py
code.ipynb
requirements.txt
README.md
theory.md
```

---

### Step 4 — Install Dependencies

If you have not installed them yet:

```bash
pip install -r requirements.txt
```

---

### Step 5 — Start Streamlit

Run:

```bash
streamlit run app.py
```

After running the command, Streamlit will start a local web server.

Usually, the application will be available at:

```text
http://localhost:8501
```

You can open the displayed local URL in your browser.

---

# 📤 6. Upload a Customer Dataset

Once the Streamlit application opens:

1. Click **Upload Customer CSV**.
2. Select your `.csv` customer dataset.
3. The application will display a preview of the dataset.
4. Select the numerical features you want to use.
5. Select the number of clusters.
6. The KMeans model will be trained automatically.

---

# 📊 7. What the Streamlit App Shows

The application provides:

### Dataset Preview

Displays the first rows of the uploaded dataset.

### Dataset Information

Shows:

* Total customers
* Total columns

### Feature Selection

Allows you to select numerical columns for clustering.

### KMeans Configuration

Allows you to select the number of clusters `K`.

### Model Results

Displays:

* Number of customers
* Number of clusters
* Silhouette Score

### Cluster Profile

Displays the average feature values for every cluster.

### Customer Distribution

Shows how many customers belong to each cluster.

### Cluster Visualization

Displays the customer groups using the first two selected features.

### Segmented Dataset

Displays the complete dataset with the generated `Cluster` column.

### Download

You can download the final segmented dataset as:

```text
customer_segments.csv
```

---

# 🧪 Example Workflow

Suppose your dataset contains:

```text
Customer_ID
Age
Annual_Income
Spending_Score
Purchases
```

You could select:

```text
Annual_Income
Spending_Score
Purchases
```

Then select:

```text
K = 5
```

The application will generate:

```text
Customer → Cluster
```

For example:

```text
Customer 1 → Cluster 2
Customer 2 → Cluster 0
Customer 3 → Cluster 4
Customer 4 → Cluster 1
```

The cluster numbers are identifiers. Their actual meaning should be understood by examining the cluster profile.

---

# 📂 Dataset Requirements

The Streamlit application accepts:

```text
.csv
```

files.

The dataset should contain at least **two numerical features** for clustering.

Examples of useful features:

* Age
* Annual Income
* Spending Score
* Purchases
* Recency
* Frequency
* Monetary Value

The application automatically detects numerical columns.

---

# ⚠️ Common Problems

## Problem 1 — `streamlit is not recognized`

Install Streamlit:

```bash
pip install streamlit
```

Then try again:

```bash
streamlit run app.py
```

---

## Problem 2 — `ModuleNotFoundError`

Install all project dependencies:

```bash
pip install -r requirements.txt
```

---

## Problem 3 — Running `python app.py`

Do not use:

```bash
python app.py
```

Use:

```bash
streamlit run app.py
```

---

## Problem 4 — Wrong Folder

Make sure the terminal is inside the folder containing:

```text
app.py
```

Check using:

```bash
dir
```

If `app.py` is not displayed, navigate to the correct folder first.

---

# 🛑 Stop the Streamlit Application

To stop the running Streamlit server, return to the terminal and press:

```text
Ctrl + C
```

---

# 🔄 Project Workflow

```text
Customer Dataset
       ↓
Data Inspection
       ↓
Feature Selection
       ↓
Feature Scaling
       ↓
Elbow Method
       ↓
Select K
       ↓
KMeans Clustering
       ↓
Cluster Labels
       ↓
Cluster Profiling
       ↓
Visualization
       ↓
Streamlit Application
       ↓
Download Segmented Dataset
```

---

# 🎯 Learning Objectives

After completing this project, you will understand:

* What customer segmentation is
* What clustering means
* How KMeans works
* How to select clustering features
* Why feature scaling is important
* How the Elbow Method works
* How to train a KMeans model
* How to analyze cluster profiles
* How to calculate Silhouette Score
* How to visualize clusters
* How to build a Streamlit ML application
* How to export machine learning results

---

# 💡 Real-World Applications

Customer segmentation can be used for:

* Customer behavior analysis
* Marketing campaigns
* Personalized offers
* Customer retention analysis
* Product recommendations
* Loyalty programs
* Sales analysis
* Business intelligence

---

# 📌 Important Note

KMeans is an **unsupervised learning algorithm**.

The model creates clusters based on the selected features, but the cluster numbers themselves do not have predefined business meanings.

For example:

```text
Cluster 0
Cluster 1
Cluster 2
```

does not automatically mean:

```text
Bad Customer
Average Customer
Best Customer
```

The actual meaning should be determined by analyzing the characteristics of each cluster.

---

# 🚀 Future Improvements

Possible improvements for this project include:

* Automatic Elbow Method inside the Streamlit app
* Automatic K selection
* PCA-based visualization
* More clustering algorithms
* Customer segment naming
* Interactive Plotly visualizations
* Database integration
* Model persistence
* Advanced customer analytics
* Deployment to a cloud platform

---

# 👨‍💻 Project

**365 Days of AI — DAY 68**

Topic:

> **Customer Segmentation with KMeans Clustering: Real-World Example in Python & Streamlit**

Built as part of a continuous Artificial Intelligence and Machine Learning learning journey.

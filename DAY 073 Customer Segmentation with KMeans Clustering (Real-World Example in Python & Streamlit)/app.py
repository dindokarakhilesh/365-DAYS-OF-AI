import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------
# 1. Streamlit Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# 2. Application Title
# ---------------------------------------------------------

st.title("📊 Customer Segmentation with KMeans")

st.write(
    "Upload a customer CSV file and use KMeans Clustering "
    "to discover customer segments."
)


# ---------------------------------------------------------
# 3. Upload CSV File
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)


if uploaded_file is not None:

    # -----------------------------------------------------
    # 4. Load Dataset
    # -----------------------------------------------------

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )


    # -----------------------------------------------------
    # 5. Dataset Information
    # -----------------------------------------------------

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Customers",
            len(df)
        )

    with col2:
        st.metric(
            "Total Columns",
            len(df.columns)
        )


    # -----------------------------------------------------
    # 6. Find Numerical Columns
    # -----------------------------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()


    if len(numeric_columns) < 2:

        st.error(
            "The dataset must contain at least "
            "two numerical columns."
        )

        st.stop()


    # -----------------------------------------------------
    # 7. Feature Selection
    # -----------------------------------------------------

    st.subheader("Select Features")

    default_features = numeric_columns[
        :min(3, len(numeric_columns))
    ]

    selected_features = st.multiselect(
        "Choose numerical features for clustering:",
        numeric_columns,
        default=default_features
    )


    if len(selected_features) < 2:

        st.warning(
            "Please select at least two numerical features."
        )

        st.stop()


    # -----------------------------------------------------
    # 8. Select Number of Clusters
    # -----------------------------------------------------

    st.subheader("KMeans Configuration")

    max_k = min(10, len(df) - 1)

    if max_k < 2:

        st.error(
            "The dataset must contain at least "
            "three rows for clustering."
        )

        st.stop()


    k = st.slider(
        "Number of Clusters (K)",
        min_value=2,
        max_value=max_k,
        value=min(5, max_k)
    )


    # -----------------------------------------------------
    # 9. Prepare Data
    # -----------------------------------------------------

    working_data = df[
        selected_features
    ].copy()


    # -----------------------------------------------------
    # 10. Handle Missing Values
    # -----------------------------------------------------

    missing_values = working_data.isnull().sum().sum()

    if missing_values > 0:

        st.info(
            f"{missing_values} missing values found. "
            "Missing values will be replaced with "
            "the median of each feature."
        )

        working_data = working_data.fillna(
            working_data.median()
        )


    # -----------------------------------------------------
    # 11. Feature Scaling
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        working_data
    )


    # -----------------------------------------------------
    # 12. Train KMeans Model
    # -----------------------------------------------------

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = kmeans.fit_predict(
        X_scaled
    )


    # -----------------------------------------------------
    # 13. Calculate Silhouette Score
    # -----------------------------------------------------

    silhouette = silhouette_score(
        X_scaled,
        df["Cluster"]
    )


    # -----------------------------------------------------
    # 14. Display Metrics
    # -----------------------------------------------------

    st.subheader("Model Results")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Customers",
            len(df)
        )

    with col2:

        st.metric(
            "Clusters",
            k
        )

    with col3:

        st.metric(
            "Silhouette Score",
            f"{silhouette:.3f}"
        )


    # -----------------------------------------------------
    # 15. Cluster Profile
    # -----------------------------------------------------

    st.subheader("Cluster Profile")

    cluster_profile = (
        df.groupby("Cluster")[selected_features]
        .mean()
        .round(2)
    )

    cluster_profile["Customer_Count"] = (
        df["Cluster"]
        .value_counts()
        .sort_index()
    )

    st.dataframe(
        cluster_profile,
        use_container_width=True
    )


    # -----------------------------------------------------
    # 16. Cluster Distribution
    # -----------------------------------------------------

    st.subheader("Customer Distribution")

    cluster_counts = (
        df["Cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        cluster_counts
    )


    # -----------------------------------------------------
    # 17. Cluster Visualization
    # -----------------------------------------------------

    st.subheader("Customer Segments Visualization")

    if len(selected_features) >= 2:

        x_feature = selected_features[0]
        y_feature = selected_features[1]

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )


        for cluster in sorted(
            df["Cluster"].unique()
        ):

            cluster_data = df[
                df["Cluster"] == cluster
            ]

            ax.scatter(
                cluster_data[x_feature],
                cluster_data[y_feature],
                label=f"Cluster {cluster}"
            )


        ax.set_xlabel(
            x_feature
        )

        ax.set_ylabel(
            y_feature
        )

        ax.set_title(
            "Customer Segmentation using KMeans"
        )

        ax.legend()

        ax.grid()

        st.pyplot(fig)


    # -----------------------------------------------------
    # 18. Segmented Dataset
    # -----------------------------------------------------

    st.subheader("Segmented Customer Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )


    # -----------------------------------------------------
    # 19. Download Segmented Dataset
    # -----------------------------------------------------

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download Segmented CSV",
        data=csv_data,
        file_name="customer_segments.csv",
        mime="text/csv"
    )


else:

    # -----------------------------------------------------
    # 20. Instructions
    # -----------------------------------------------------

    st.info(
        "👆 Upload a customer CSV file to start "
        "the segmentation process."
    )


    st.markdown(
        """
        ### How to use this application

        1. Upload a CSV customer dataset.
        2. Select numerical features.
        3. Select the number of clusters.
        4. The application will train KMeans.
        5. Review the cluster profile.
        6. Analyze the visualization.
        7. Download the segmented dataset.

        ### Recommended Features

        Examples of useful customer features:

        - Annual Income
        - Spending Score
        - Number of Purchases
        - Recency
        - Frequency
        - Monetary Value
        """
    )
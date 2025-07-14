import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🌲 Wildfire Risk Prediction", layout="wide")

st.title("🔥 Wildfire Risk Prediction App")
st.markdown("Upload a dataset and predict wildfire risk zones and CO₂ emissions.")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Preprocessing
    df_clean = df.drop(columns=["country", "subnational1"], errors='ignore')
    df_clean.fillna(df_clean.mean(), inplace=True)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_clean)
    df_scaled = pd.DataFrame(scaled_features, columns=df_clean.columns)

    # Sidebar for PCA & KMeans
    st.sidebar.header("Settings")
    n_components = st.sidebar.slider("PCA Components", 2, min(len(df_scaled.columns), 10), 2)
    n_clusters = st.sidebar.slider("K-Means Clusters", 2, 10, 3)

    # PCA
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df_scaled)
    explained_var = pca.explained_variance_ratio_
    st.write(f"### PCA Explained Variance Ratio: {explained_var}")

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_pca)
    df_scaled["Cluster"] = clusters

    # Scatterplot
    fig, ax = plt.subplots()
    sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=clusters, palette="viridis", ax=ax)
    ax.set_title("Wildfire Risk Clusters")
    st.pyplot(fig)

    # Regression for CO2 emissions (if present)
    target_column = "gfw_gross_emissions_co2e_all_gases__Mg_yr-1"
    if target_column in df_scaled.columns:
        X = df_scaled.drop(columns=[target_column, "Cluster"], errors='ignore')
        y = df_scaled[target_column]

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        mse = np.mean((y - y_pred) ** 2)
        st.write(f"### Regression Model Mean Squared Error: {mse:.4f}")

        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=y, y=y_pred, ax=ax2)
        ax2.set_xlabel("Actual CO₂ Emissions")
        ax2.set_ylabel("Predicted CO₂ Emissions")
        ax2.set_title("CO₂ Emissions Prediction")
        st.pyplot(fig2)

    st.write("### Clustered Data Preview")
    st.dataframe(df_scaled)

    csv = df_scaled.to_csv(index=False).encode('utf-8')
    st.download_button("Download Clustered Data", csv, "wildfire_clusters.csv", "text/csv")

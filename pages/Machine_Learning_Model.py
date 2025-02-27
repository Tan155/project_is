import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.title("K-Nearest Neighbor (K-NN) AND K-MEANS CLUSTERING DEMO")

st.write("### 🍂🍂K-Nearest Neighbor (K-NN)🍂🍂")
csv_path = "phoneEdit.csv"
df = pd.read_csv(csv_path)

# Select product for analysis
product_options = df["Product"].unique()
selected_product = st.selectbox("Select Product", product_options)

# Filter information for only selected products and create a copy
product_df = df[df["Product"] == selected_product].copy()

if product_df.empty:
    st.write(f"No data available for {selected_product}.")
else:
    # Convert Country as number by Label Encoding
    label_encoder_country = LabelEncoder()
    product_df["Country_Label"] = label_encoder_country.fit_transform(
        product_df["Country"]
    )

    # select feature K-NN
    X = product_df[
        ["Price", "Country_Label"]
    ].values  # use Price and Country as features
    y = product_df["Country_Label"].values  # use Country_Label as target

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # select number k
    k_value = st.slider(
        "Select K (Number of Neighbors)", min_value=1, max_value=100, value=3
    )

    # create model K-NN
    knn = KNeighborsClassifier(
        n_neighbors=k_value, metric="manhattan", weights="distance"
    )
    knn.fit(X_scaled, y)

    # predict slider
    new_price = st.slider(
        "Select Phone Price", min_value=5000, max_value=40000, value=25000, step=500
    )
    country_input = st.selectbox("Select Country", product_df["Country"].unique())
    new_country_label = label_encoder_country.transform([country_input])[0]
    new_point = np.array([[new_price, new_country_label]])

    new_point_scaled = scaler.transform(new_point)

    # predict
    predicted_label = knn.predict(new_point_scaled)
    neighbors = knn.kneighbors(new_point_scaled, return_distance=False)

    # Draw Graph
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        x=product_df["Price"],
        y=product_df["Country_Label"],
        hue=product_df["Country"],
        alpha=0.6,
        edgecolor="k",
        ax=ax,
    )

    ax.scatter(
        new_point[:, 0],
        new_point[:, 1],
        color="red",
        s=100,
        label="New Point",
        edgecolors="black",
        linewidth=2,
    )

    for neighbor in neighbors[0]:
        ax.plot(
            [new_point[0, 0], product_df.iloc[neighbor]["Price"]],
            [new_point[0, 1], product_df.iloc[neighbor]["Country_Label"]],
            "r--",
            alpha=0.8,
            linewidth=2,
        )

    ax.set_xlabel(f"Price ({selected_product})")
    ax.set_ylabel("Country (Encoded)")
    ax.set_title(f"K-NN Classification (K = {k_value}) on {selected_product} Data")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    predicted_country = label_encoder_country.inverse_transform(predicted_label)[0]
    st.write(f"### 🔍 Predicted Country: **{predicted_country}**")


st.header("🪅🪅🪅🪅🪅🪅🪅🪅🪅🪅🪅🪅🪅")

st.write("### 🍁🍁K-MEANS CLUSTERING🍁🍁")

# Convert Country as number
df["Country_Label"] = LabelEncoder().fit_transform(df["Country"])

# select Feature
X_kmeans = df[["Price", "Country_Label"]].values

scaler = StandardScaler()
X_kmeans_scaled = scaler.fit_transform(X_kmeans)

st.write("##### Finding Optimal K using Elbow Method")
inertia = []

# test k 1 - 10
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_kmeans_scaled)
    inertia.append(kmeans.inertia_)

# Drawgrapg K using Elbow Method
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range, inertia, marker="o", linestyle="-")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method for Optimal K")
ax.grid(True)

st.pyplot(fig)

# select k by slider
selected_k = st.slider(
    "Select number of clusters (K)", min_value=2, max_value=10, value=3, step=1
)

# create model K-Means follow k
kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_kmeans_scaled)

st.write(f"### K-Means Clustering with K = {selected_k}")

# draw scatter
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    x=df["Price"],
    y=df["Country_Label"],
    hue=df["Cluster"],
    palette="tab10",
    alpha=0.7,
    edgecolor="k",
    ax=ax,
)

ax.set_xlabel("Price")
ax.set_ylabel("Country (Encoded)")
ax.set_title(f"K-Means Clustering (K = {selected_k})")
ax.legend(title="Cluster")
ax.grid(True)

st.pyplot(fig)

st.write("### 📌 Cluster Summary")

cluster_summary = df.groupby("Cluster").agg(
    Count=("Price", "count"), Avg_Price=("Price", "mean")
)

st.write(cluster_summary)


st.write("Theoretical pictures come from the subject Intelligent systems.")

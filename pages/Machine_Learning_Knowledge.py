import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

st.title("Machine Learning Model")

# Model
st.write(
    "This data set from Chat GPT Generate of Phone Brand User by No prepare data set and Data have 1000 data"
)

st.write("Feature: Customer_ID, Gender, Country, Product, Price, Product_Model")

csv_path = "phone.csv"  # Set path csv
df = pd.read_csv(csv_path)
st.write("### DATA SET ARE NOT PREPARE:")  # Data are not prepare
st.write(df.head(5))

# Outlier data and missing data
st.write("### This is data which have Outlier data and missing data")
st.write("Missing Data")
st.write(df.isnull().sum(), df.describe())

st.write("#### First we have to delete Missig data")
codeMissing = """df.dropna(inplace=True)"""

df.dropna(inplace=True)

st.code(codeMissing, language="python")

st.write(df.isnull().sum())

st.write("#### Second Fix Outlier Data thst is Price lower 5,000 and Upper 40000 ")

codeOutlier = """df = df[(df['Price'] >= 5000) & (df['Price'] <= 40000)]"""

df = df[(df["Price"] >= 5000) & (df["Price"] <= 40000)]

st.code(codeOutlier, language="python")

st.write(df.describe())

df.to_csv("phoneEdit.csv", index=False)

# df.to_csv("phone_edited.csv", index=False)


# K-NN
st.header("⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️")
st.header("K-Nearest Neighbor (K-NN) :sunglasses:")

st.write("#### Supervise learning algorithm K-NN")

st.write("##### predictions calculates the distance between the input data")

st.write(
    "##### K is the number of neighbors to consider and the main hyperparameter of the model."
)

st.image("knn1.png")


st.write(
    "##### Each machine learning model has their own parameters that need to be set before prior to the start of the model training call hyperparameters"
)

st.write("#### Hyperparameters Tuning")

st.image("knn2.png")

# Develop Model K-NN
st.write("### Develop Model K-Nearest Neighbor (K-NN)")

st.write("Select X AND Y for Geanerate Scatter")
code_select_knn = """iphone_df = df[df["Product"] == "iPhone"].copy()  # Select Data X
label_encoder_country = LabelEncoder()  # Convert Data Country
iphone_df["Country_Label"] = label_encoder_country.fit_transform(iphone_df["Country"])

X_knn = iphone_df[["Price", "Country_Label"]].values
y_knn = iphone_df["Country_Label"].values"""
st.code(code_select_knn, language="python")


st.write("Create Model, Select k Value and Point of k-nn")
code_select_k = """k_value = 3 #Number K

knn = KNeighborsClassifier(n_neighbors=k_value, metric="manhattan", weights="distance") #Create Model
knn.fit(X_knn_scaled, y_knn)

new_price = 15000
new_country_label = 3
new_point = np.array([[new_price, new_country_label]])  # Point x,y of k-nn

new_point_scaled = scaler.transform(new_point)"""
st.code(code_select_k, language="python")


st.write("Predic And Draw Graph")
code_knn_drawgraph = """predicted_label = knn.predict(new_point_scaled) #predict
neighbors = knn.kneighbors(new_point_scaled, return_distance=False)

fig, ax = plt.subplots(figsize=(8, 5)) #Draw Graph
sns.scatterplot(
    x=iphone_df["Price"],
    y=iphone_df["Country_Label"],
    hue=iphone_df["Country"],
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

for neighbor in neighbors[0]: #Draw Line go to nearest neighbor
    ax.plot(
        [new_point[0, 0], iphone_df.iloc[neighbor]["Price"]],
        [new_point[0, 1], iphone_df.iloc[neighbor]["Country_Label"]],
        "r--",
        alpha=0.8,
        linewidth=2,
    )

ax.set_xlabel("Price (iPhone Only)")
ax.set_ylabel("Country (Encoded)")
ax.set_title("K-NN Classification (K = 10) on iPhone Data")
ax.legend()
ax.grid(True)

st.pyplot(fig)"""
st.code(code_knn_drawgraph, language="python")

iphone_df = df[df["Product"] == "iPhone"].copy()  # Select Data X
label_encoder_country = LabelEncoder()  # Convert Data Country
iphone_df["Country_Label"] = label_encoder_country.fit_transform(iphone_df["Country"])

X_knn = iphone_df[["Price", "Country_Label"]].values
y_knn = iphone_df["Country_Label"].values

scaler = StandardScaler()
X_knn_scaled = scaler.fit_transform(X_knn)

k_value = 10  # Number K

knn = KNeighborsClassifier(
    n_neighbors=k_value, metric="manhattan", weights="distance"
)  # Create Model
knn.fit(X_knn_scaled, y_knn)

new_price = 15000
new_country_label = 3
new_point = np.array([[new_price, new_country_label]])  # Point x,y of k-nn

new_point_scaled = scaler.transform(new_point)

predicted_label = knn.predict(new_point_scaled)  # predict
neighbors = knn.kneighbors(new_point_scaled, return_distance=False)

fig, ax = plt.subplots(figsize=(8, 5))  # Draw Graph
sns.scatterplot(
    x=iphone_df["Price"],
    y=iphone_df["Country_Label"],
    hue=iphone_df["Country"],
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

for neighbor in neighbors[0]:  # Draw Line go to nearest neighbor
    ax.plot(
        [new_point[0, 0], iphone_df.iloc[neighbor]["Price"]],
        [new_point[0, 1], iphone_df.iloc[neighbor]["Country_Label"]],
        "r--",
        alpha=0.8,
        linewidth=2,
    )

ax.set_xlabel("Price (iPhone Only)")
ax.set_ylabel("Country (Encoded)")
ax.set_title("K-NN Classification (K = 10) on iPhone Data")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# คืนค่าผลลัพธ์ของการทำนาย
predicted_country = label_encoder_country.inverse_transform(predicted_label)[0]
st.write(f"### 🔍 Predicted Country: **{predicted_country}**")

st.header("✨✨✨✨✨✨✨✨✨✨✨✨✨✨")
st.title("K-Means Clustering Algorithm 🤠")

st.write("#### Unsupervise learning K-Means Clustering")

st.write(
    "###### Finding similarities between data according to the characteristics found in the data and grouping similar data objects into clusters"
)

st.image("k-means1.png")

st.write(
    "##### k-means clustering can groups people of similar sizes, targeted marketing and so on"
)

st.write("#### How K-Means Clustering?")

st.write(
    "###### Initial Step: Select number of Cluster (K) and randomly select center point of each cluster (Centroid)."
)
st.image("k-means2.png")

st.write(
    "###### Step 1: Calculate distance from each data to each Centroid. Then Assign each data to the closest cluster."
)
st.image("k-means3.png")

st.write("###### Step 2: Recalculate the centroid of each cluster.")

st.write("###### Repeat Step 1 and 2 until the centroids don’t change.")
st.image("k-means4.png")
st.image("k-means5.png")

st.write(
    "##### Select K with Elbow method Calculate the Within-Cluster-Sum of Squared Errors (WCSS) for different values of k"
)

st.image("k-means6.png")

# Development k-means clustering
st.write("## Develop Model K-Means Clustering")

st.write("Select feature for k-means")
code_kmeans_data = """label_encoder = LabelEncoder()  # convert number
df["Country_Label"] = label_encoder.fit_transform(df["Country"])
X_kmeans = df[["Price", "Country_Label"]].values  # Select feature"""
st.code(code_kmeans_data, language="python")

st.write("Test k for create Elbow Method")
code_train_k = """inertia = []  # Finding Optimal K using Elbow Method

K_range = range(1, 11)  # test k 1 - 10
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_kmeans_scaled)
    inertia.append(kmeans.inertia_)"""

st.code(code_train_k, language="python")

st.write("Drawgraph k-means")
code_kmeans_drawgraph = """fig, ax = plt.subplots(figsize=(8, 5))  # Draw Graph Elbow Method
ax.plot(K_range, inertia, marker="o", linestyle="-")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method for Optimal K")
ax.grid(True)

st.pyplot(fig)


optimal_k = 3

kmeans = KMeans(
    n_clusters=optimal_k, random_state=42, n_init=10
)  # create model K-Means by Optimal K
df["Cluster"] = kmeans.fit_predict(X_kmeans_scaled)

st.write(f"### K-Means Clustering with K = {optimal_k}")


fig, ax = plt.subplots(figsize=(8, 5))  # Draw Scatter
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
ax.set_title(f"K-Means Clustering (K = {optimal_k})")
ax.legend(title="Cluster")
ax.grid(True)

st.pyplot(fig)"""

st.code(code_kmeans_drawgraph, language="python")


label_encoder = LabelEncoder()  # convert number
df["Country_Label"] = label_encoder.fit_transform(df["Country"])

X_kmeans = df[["Price", "Country_Label"]].values  # Select feature

scaler = StandardScaler()
X_kmeans_scaled = scaler.fit_transform(X_kmeans)

inertia = []  # Finding Optimal K using Elbow Method

K_range = range(1, 11)  # test k 1 - 10
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_kmeans_scaled)
    inertia.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))  # Draw Graph Elbow Method
ax.plot(K_range, inertia, marker="o", linestyle="-")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method for Optimal K")
ax.grid(True)

st.pyplot(fig)


optimal_k = 3

kmeans = KMeans(
    n_clusters=optimal_k, random_state=42, n_init=10
)  # create model K-Means by Optimal K
df["Cluster"] = kmeans.fit_predict(X_kmeans_scaled)

st.write(f"### K-Means Clustering with K = {optimal_k}")


fig, ax = plt.subplots(figsize=(8, 5))  # Draw Scatter
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
ax.set_title(f"K-Means Clustering (K = {optimal_k})")
ax.legend(title="Cluster")
ax.grid(True)

st.pyplot(fig)

st.write("### 📌 Cluster Summary")
cluster_summary = df.groupby("Cluster").agg(
    Count=("Price", "count"), Avg_Price=("Price", "mean")
)

st.write(cluster_summary)

st.write("Theoretical pictures come from the subject Intelligent systems.")

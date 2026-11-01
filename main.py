"""
Lab 9: K-Means and Hierarchical Clustering on Wholesale Customers
Dataset: data/Wholesale customers data.csv (download from UCI)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Wholesale customers data.csv"
if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Download it from UCI (see README.md).")
df = pd.read_csv(DATA_PATH)

print("Dataset overview:")
print(df.head())
print("Shape:", df.shape)
print("Missing values:\n",df.isnull().sum())
print("Duplicate rows:",df.duplicated().sum())

features = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
X = df[features]

# Exploratory data analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=df, x="Channel", ax=axes[0])
sns.countplot(data=df, x="Region", ax=axes[1])
fig.suptitle("Channel and Region Distribution")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
for ax, col in zip(axes.flat, features):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
for ax, col in zip(axes.flat, features):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(f"Box plot of {col}")
plt.tight_layout()
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("IQR outlier counts:")
for col in features:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    count = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"{col}: {count}")

# Standardize the six annual-spending variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means: compare k = 1 through 10 by within-cluster sum of squares
wcss = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    model.fit(X_scaled)
    wcss.append(model.inertia_)
plt.plot(range(1, 11), wcss, marker="o", linestyle="--")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS (inertia)")
plt.title("K-Means Elbow Method")
plt.xticks(range(1, 11))
plt.tight_layout()
plt.show()

silhouette_table = []
for k in range(2, 11):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    silhouette_table.append((k, silhouette_score(X_scaled, labels)))
silhouette_df = pd.DataFrame(silhouette_table, columns=["k", "silhouette_score"])
print("Silhouette scores:\n",silhouette_df.to_string(index=False))
K = int(silhouette_df.loc[silhouette_df.silhouette_score.idxmax(), "k"])
print("Selected k:",K)

kmeans_model = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans_labels = kmeans_model.fit_predict(X_scaled)
df_kmeans = df.copy()
df_kmeans["Kmeans_Cluster"] = kmeans_labels
print("K-Means sizes:\n",df_kmeans.Kmeans_Cluster.value_counts().sort_index())

# Ward linkage agglomerative hierarchical clustering
hierarchical_model = AgglomerativeClustering(n_clusters=K, metric="euclidean", linkage="ward")
hierarchical_labels = hierarchical_model.fit_predict(X_scaled)
df_hierarchical = df.copy()
df_hierarchical["Hierarchical_Cluster"] = hierarchical_labels
print("Hierarchical sizes:\n",df_hierarchical.Hierarchical_Cluster.value_counts().sort_index())

Z = linkage(X_scaled, method="ward", metric="euclidean")
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode="level", p=6, leaf_rotation=90, leaf_font_size=7)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer/cluster")
plt.ylabel("Ward Linkage Distance")
plt.tight_layout()
plt.show()

# Two-dimensional PCA view
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
plot_df = pd.DataFrame({"PC1":X_pca[:,0], "PC2":X_pca[:,1], "Kmeans_Cluster":kmeans_labels})
plt.figure(figsize=(10, 6))
sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Kmeans_Cluster", palette="Set1", s=65)
plt.title("K-Means Clusters in PCA Space")
plt.tight_layout()
plt.show()

# Cluster sizes and spending patterns
print("Cluster size comparison:")
print(pd.DataFrame({"K-Means":pd.Series(kmeans_labels).value_counts().sort_index(),
                    "Hierarchical":pd.Series(hierarchical_labels).value_counts().sort_index()}))
profile = df[features].assign(Cluster=kmeans_labels).groupby("Cluster")[features].mean()
print("K-Means customer spending profiles:\n",profile.round(1))
sns.heatmap(profile, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Mean Annual Spending by K-Means Cluster")
plt.tight_layout()
plt.show()

channel_mix = pd.crosstab(df["Channel"], kmeans_labels, normalize="columns")*100
print("Channel mix by cluster (%):\n",channel_mix.round(2))

# Internal metrics: additional comparison beyond the notebook outputs
for name, labels in [("K-Means", kmeans_labels), ("Hierarchical", hierarchical_labels)]:
    print(f"{name}: silhouette={silhouette_score(X_scaled,labels):.4f}, "
          f"Davies-Bouldin={davies_bouldin_score(X_scaled,labels):.4f}, "
          f"Calinski-Harabasz={calinski_harabasz_score(X_scaled,labels):.2f}")

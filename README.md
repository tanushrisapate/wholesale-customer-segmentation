# Wholesale Customer Segmentation with K-Means and Hierarchical Clustering

This Machine Learning Lab 9 project groups wholesale customers based on **annual spending across six product categories** using unsupervised learning.

## Work performed

- Dataset inspection, missing-value and duplicate checks
- EDA: count plots, histograms, box plots, correlation and IQR outlier counts
- Feature standardization with `StandardScaler`
- K-Means clustering, elbow plot and silhouette score selection
- Agglomerative Hierarchical Clustering (Ward linkage) and dendrograms
- PCA visualization, cluster-size comparison and spending profiles

## Models

- **K-Means** using k-means++ initialization
- **Agglomerative Clustering** using Ward linkage and Euclidean distance

The uploaded notebook selects **3 clusters** using the silhouette score. Its recorded cluster sizes are **45 / 393 / 2** for K-Means and **6 / 281 / 153** for hierarchical clustering. Cluster labels are arbitrary, so label 0 from one method should not be treated as equivalent to label 0 from the other.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

To use Jupyter Notebook instead:

```bash
jupyter notebook notebook/Lab_9_KMeans_Hierarchical_Clustering.ipynb
```

## Dataset

Download **Wholesale customers data.csv** from the [UCI Wholesale Customers dataset](https://archive.ics.uci.edu/dataset/292/wholesale+customers). Place it in `data/Wholesale customers data.csv`.

This dataset has **440 records** and **8 columns**. The clustering input uses the six spending columns only. `Channel` and `Region` are used for exploratory analysis, not for training the clusters.

Dataset citation: Cardoso, M. (2013), *Wholesale customers*, UCI Machine Learning Repository, DOI: 10.24432/C5030X. License: CC BY 4.0.

## Technologies

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · SciPy · Jupyter

## Author

**Tanushri Sapate**  
B.Tech CSE (AI & ML) · Section B-B4 · Roll No. 55

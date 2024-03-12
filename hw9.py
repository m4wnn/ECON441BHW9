# %% [markdown]
# # 0.) Import and Clean data

# %%
import toolz as tz
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import QuantileTransformer 
from sklearn.preprocessing import StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

# %%
df = pd.read_csv("data/Country-data.csv", sep = ",")
country = df['country']
# %%
# %%
X = tz.pipe(
    df,
    lambda x: x.drop(columns=['country']),
    lambda x: StandardScaler().fit_transform(x),
    lambda x: pd.DataFrame(x, columns=df.columns[1:])
)

# %% [markdown]
# # 1.) Fit a kmeans Model with any Number of Clusters
# %%
kmeans = KMeans(n_clusters=5, n_init='auto').fit(X)

# %% [markdown]
# # 2.) Pick two features to visualize across
# %%
x1_index = 0
x2_index = len(X.columns) - 1

# %%
scatter = plt.scatter(
    X.iloc[:, x1_index],
    X.iloc[:, x2_index], 
    c=kmeans.labels_, 
    cmap='viridis',
    label='Clusters'
)

centers = plt.scatter(
    kmeans.cluster_centers_[:, x1_index], 
    kmeans.cluster_centers_[:, x2_index], 
    marker='s', 
    color='black', 
    s=100, 
    label='Centers'
)

plt.xlabel(X.columns[x1_index])
plt.ylabel(X.columns[x2_index])
plt.title('Scatter Plot of Customers')

# Generate legend
plt.legend()

plt.grid()
plt.show()

# %% [markdown]
# # 3.) Check a range of k-clusters and visualize to find the elbow. Test 30 different random starting places for the centroid means
# 
elbow_ = lambda k, init: KMeans(n_clusters=k, n_init=init).fit(X).inertia_
# %%
elbow_results = pd.DataFrame({
    'k': range(1, 31),
    'inertia': [elbow_(k, 30) for k in range(1, 31)]
})
# %%
plt.plot(
    elbow_results['k'],
    elbow_results['inertia'],
    marker='o'
)
plt.grid()
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Plot')

# %% [markdown]
# # 4.) Use the above work and economic critical thinking to choose a number of clusters. Explain why you chose the number of clusters and fit a model accordingly.

# %% [markdown]
# # 6.) Do the same for a silhoutte plot

# %%
from sklearn.metrics import silhouette_score

# %%
silhouette_score_ = lambda k, init: silhouette_score(
    X, KMeans(n_clusters=k, n_init=init).fit_predict(X)
)
# %%
silhouette_results = pd.DataFrame({
    'k': range(2, 31),
    'silhouette': [silhouette_score_(k, 30) for k in range(2, 31)]
})
# %%
plt.plot(
    silhouette_results['k'],
    silhouette_results['silhouette'],
    marker='o'
)
plt.grid()
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Plot')

# %% [markdown]
# # 7.) Create a list of the countries that are in each cluster. Write interesting things you notice.

# %%
def get_countries_in_cluster(kmeans, countries, x):
    k_labels = kmeans.labels_
    results = X.copy()
    results['country'] = countries
    results['cluster'] = k_labels
    return results

X_k = get_countries_in_cluster(kmeans, country, X)
# %%


# %%


# %%


# %%


# %%


# %%
#### Write an observation

# %% [markdown]
# #8.) Create a table of Descriptive Statistics. Rows being the Cluster number and columns being all the features. Values being the mean of the centroid. Use the nonscaled X values for interprotation

# %%


# %%


# %%


# %% [markdown]
# # 9.) Write an observation about the descriptive statistics.

# %%




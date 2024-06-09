import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the data
file_path = 'babyNamesUSYOB-mostpopular.csv'
data = pd.read_csv(file_path)

# Prepare the data for clustering
X = data[['YearOfBirth', 'Number']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-means clustering
range_n_clusters = range(2, 11)
silhouette_scores = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.title('Silhouette Score for Various Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Optimal number of clusters
optimal_n_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]

# Perform K-means clustering with the optimal number of clusters
kmeans_optimal = KMeans(n_clusters=optimal_n_clusters, random_state=42)
cluster_labels_optimal = kmeans_optimal.fit_predict(X_scaled)

# Add cluster labels to the original data
data['Cluster'] = cluster_labels_optimal

# Plot the clusters
plt.figure(figsize=(12, 8))
for cluster in range(optimal_n_clusters):
    plt.scatter(X_scaled[cluster_labels_optimal == cluster, 0], X_scaled[cluster_labels_optimal == cluster, 1], label=f'Cluster {cluster}')

plt.title('Clusters of Baby Names by Year of Birth and Number')
plt.xlabel('Standardized Year of Birth')
plt.ylabel('Standardized Number of Babies')
plt.legend()
plt.grid(True)
plt.show()

# Calculate and display the silhouette score for the optimal clustering
silhouette_avg_optimal = silhouette_score(X_scaled, cluster_labels_optimal)
print(f'Silhouette Score for the optimal clustering: {silhouette_avg_optimal}')

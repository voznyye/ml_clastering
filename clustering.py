import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
file_path = 'babyNamesUSYOB-full.csv'
data = pd.read_csv(file_path)

# Assume we want to cluster based on numerical features
# Keep only numerical columns
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Cluster using k-means++
kmeans = KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=300, random_state=42)
kmeans.fit(scaled_data)

# Save the results
data['Cluster'] = kmeans.labels_
data.to_csv('clustered_data.csv', index=False)

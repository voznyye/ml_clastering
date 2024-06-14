import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

matplotlib.use('agg')

# Load data with clusters
file_path = 'babyNamesUSYOB-mostpopular.csv'
data = pd.read_csv(file_path)

# Keep only numerical columns
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Elbow method
sse = []
k_range = range(1, 30)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

# Visualization of the elbow method
plt.figure(figsize=(10, 8))
plt.plot(k_range, sse, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared errors (SSE)')
plt.title('Elbow Method For Optimal k')
plt.savefig('visual/elbow_method.png')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use('agg')

# Load the clustered data
file_path = 'clustered_data.csv'
data = pd.read_csv(file_path)

# Define the columns to be used
features = ['YearOfBirth', 'Number']

# Standardize the numerical columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[features])

# Apply PCA for visualization in 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Add PCA results to the original data
data['PCA1'] = pca_result[:, 0]
data['PCA2'] = pca_result[:, 1]

# Set up the figure
plt.figure(figsize=(14, 10))

# Create a scatter plot
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, palette='viridis', s=100, alpha=0.7, edgecolor='k')

# Enhance the plot with titles and labels
plt.title('K-means Clustering Results', fontsize=16)
plt.xlabel('Year of birth', fontsize=14)
plt.ylabel('Number', fontsize=14)
plt.legend(title='Cluster', fontsize=12, title_fontsize=14)
plt.grid(True)

# Center the plot around the origin
plt.xlim(data['PCA1'].min() - 1, data['PCA1'].max() + 1)
plt.ylim(data['PCA2'].min() - 1, data['PCA2'].max() + 1)

# Add a line at the origin for better visualization
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)

plt.tight_layout()

# Save the plot
plt.savefig('visual/cluster_visualization.png')

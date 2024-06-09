# visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib

matplotlib.use('agg')

# Загрузка данных с кластерами
data = pd.read_csv('clustered_data.csv')

# Оставляем только числовые столбцы для PCA
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# PCA для визуализации в 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(numerical_data)

data['PCA1'] = pca_result[:, 0]
data['PCA2'] = pca_result[:, 1]

# Визуализация
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, palette='viridis')
plt.title('K-means Clustering Results')
plt.savefig('cluster_visualization.png')
plt.show()

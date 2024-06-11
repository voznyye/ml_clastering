# silhouette_method.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use('agg')

# Загрузка данных с кластерами
file_path = 'clustered_data.csv'
data = pd.read_csv(file_path)

# Оставляем только числовые столбцы
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# Стандартизация данных
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Метод силуэта
silhouette_scores = []
k_range = range(4, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    score = silhouette_score(scaled_data, kmeans.labels_)
    silhouette_scores.append(score)
    print('done')

# Визуализация метода силуэта
plt.figure(figsize=(10, 8))
plt.plot(k_range, silhouette_scores, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method For Optimal k')
plt.savefig('visual/silhouette_method.png')
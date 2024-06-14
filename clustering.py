# kmeans_clustering.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Загрузка данных
file_path = 'babyNamesUSYOB-mostpopular.csv'
data = pd.read_csv(file_path)

# Предположим, что нам нужно кластеризовать данные на основе числовых признаков
# Оставляем только числовые столбцы
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# Стандартизация данных
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Кластеризация методом k-means++
kmeans = KMeans(n_clusters=8, init='k-means++', random_state=42)
kmeans.fit(scaled_data)

# Сохранение результатов
data['Cluster'] = kmeans.labels_
data.to_csv('clustered_data.csv', index=False)

# elbow_method.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Загрузка данных
file_path = 'babyNamesUSYOB-mostpopular.csv'
data = pd.read_csv(file_path)

# Оставляем только числовые столбцы
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# Стандартизация данных
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Метод локтя
sse = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

# Визуализация метода локтя
plt.figure(figsize=(10, 8))
plt.plot(k_range, sse, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared errors (SSE)')
plt.title('Elbow Method For Optimal k')
plt.savefig('elbow_method.png')
plt.show()

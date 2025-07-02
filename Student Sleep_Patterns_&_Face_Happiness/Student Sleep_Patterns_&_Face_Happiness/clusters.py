import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from sklearn.neighbors import NearestNeighbors  # [[9]]

# قراءة البيانات
data = pd.read_csv('E:/level_four_IT/data_mining/cleaned_student_data.csv')
print(data.head())
print(data.info())

# اختيار الأعمدة المناسبة
X = data[['Sleep_Duration', 'Study_Hours']]

# تطبيع البيانات (يجب أن يكون أول خطوة قبل التحليل) [[6]][[9]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =====================================================
# K-Means Clustering
# =====================================================
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)  # تحديد n_init بشكل صريح
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for K-Means')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

# تطبيق K-Means مع العدد الأمثل من المجموعات
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# إضافة المجموعات إلى البيانات الأصلية
data['Cluster'] = clusters

# رسم النتائج مع تحسين العرض [[8]]
plt.figure(figsize=(12, 8))
for i in range(optimal_k):
    plt.scatter(X_scaled[clusters == i, 0], 
                X_scaled[clusters == i, 1], 
                label=f'Cluster {i+1}', 
                s=50,
                alpha=0.7)

plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], 
            s=200, 
            c='yellow', 
            label='Centroids',
            marker='X',
            edgecolor='black')
plt.title('Customer Segmentation using K-Means')
plt.xlabel('Sleep Duration (Scaled)')
plt.ylabel('Study Hours (Scaled)')
plt.legend()
plt.grid(True)
plt.show()

# تحليل المجال لكل مجموعة [[8]]
print("تحليل المجموعات:")
for cluster in range(optimal_k):
    subset = data[data['Cluster'] == cluster]
    print(f"\nالمجموعة {cluster+1}:")
    print(f"  عدد الطلاب: {len(subset)}")
    print(f"  متوسط النوم: {subset['Sleep_Duration'].mean():.2f} ساعات")
    print(f"  متوسط الدراسة: {subset['Study_Hours'].mean():.2f} ساعات")
    print(f"  مدى الأعمار: {subset['Age'].min()} - {subset['Age'].max()} سنة")

# =====================================================
# DBSCAN Clustering مع تحسينات [[5]][[9]]
# =====================================================
# تحديد الـ eps المناسب باستخدام K-distance graph
nbrs = NearestNeighbors(n_neighbors=5).fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)
plt.figure(figsize=(10, 6))
plt.plot(np.sort(distances[:, 4]))
plt.title('K-distance Graph for DBSCAN Parameter Selection')
plt.xlabel('Data Points Sorted by Distance')
plt.ylabel('4th Nearest Neighbor Distance')
plt.show()

# تطبيق DBSCAN مع معلمات محسنة
dbscan = DBSCAN(eps=0.4, min_samples=5)
clusters_dbscan = dbscan.fit_predict(X_scaled)

# إضافة المجموعات مع معالجة النقاط الضجيج [[5]]
data['Cluster_DBSCAN'] = clusters_dbscan

# رسم النتائج مع تمييز النقاط الضجيج [[5]]
plt.figure(figsize=(12, 8))
unique_clusters = np.unique(clusters_dbscan)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

for idx, cluster in enumerate(unique_clusters):
    mask = clusters_dbscan == cluster
    label = f'Cluster {cluster}' if cluster != -1 else 'Noise'
    plt.scatter(X_scaled[mask, 0], 
                X_scaled[mask, 1],
                color=colors[idx] if cluster != -1 else 'gray',
                label=label,
                s=50,
                alpha=0.6 if cluster != -1 else 0.3,
                marker='o' if cluster != -1 else 'x')

plt.title('Customer Segmentation using DBSCAN')
plt.xlabel('Sleep Duration (Scaled)')
plt.ylabel('Study Hours (Scaled)')
plt.legend()
plt.grid(True)
plt.show()

# تحليل النتائج مع استثناء النقاط الضجيج [[5]]
valid_clusters = clusters_dbscan != -1
print(f"\nعدد النقاط الضجيج: {sum(clusters_dbscan == -1)}")
print(f"عدد المجموعات الفعلية: {len(np.unique(clusters_dbscan[valid_clusters]))}")

cluster_summary_dbscan = data[valid_clusters].groupby('Cluster_DBSCAN')[[
    'Sleep_Duration', 'Study_Hours']].mean()
print("\nملخص المجموعات لـ DBSCAN:")
print(cluster_summary_dbscan)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

weight_height = np.array([[55,155],
     [60,145],
     [72,176],
     [105,175],
     [65,156],
     [95,170],
     [50,152],
     [70,151],
     [65,165],
     [80,180],
     [98, 182]])

# Elbow Method to find K value
distorsions = []
for k in range(1, 5):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(weight_height)
    distorsions.append(kmeans.inertia_) # adding the sum of squared errors

# plotting the chart
fig = plt.figure(figsize=(20, 5))
plt.plot(range(1, 5), distorsions)
plt.grid(True)
plt.title('Elbow curve')

plt.scatter(weight_height[:,0],weight_height[:,1], label='True Position')

kmeans = KMeans(n_clusters=2, max_iter=100)
kmeans.fit(weight_height)

# Finding the centroids
print(kmeans.cluster_centers_)

# Getting the cluster mapping
print(kmeans.labels_)

#plotting the cluster
plt.scatter(weight_height[:,0],weight_height[:,1], c=kmeans.labels_, cmap='rainbow')

# Ploting the clusters with the centroid
plt.scatter(weight_height[:,0], weight_height[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')


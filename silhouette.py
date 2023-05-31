from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

Tk().withdraw() 
filename = askopenfilename()
print(filename)
data = pd.read_csv(filename)

x = np.array(data.loc[:, 'Filtered AnkleLeftX - SpineBaseX'])
v = np.array(data.loc[:, 'VelocityX'])
t = np.array(data.loc[:, 'Time'])
X_std = np.zeros(shape=(len(x), 2))
for i in range(len(x)):
	X_std[i] = [x[i], v[i]]

 
for i, k in enumerate([4]):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    
    # Run the Kmeans algorithm
    km = KMeans(n_clusters=k)
    labels = km.fit_predict(X_std)
    centroids = km.cluster_centers_

    # Get silhouette samples
    silhouette_vals = silhouette_samples(X_std, labels)

    # Silhouette plot
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # Get the average silhouette score and plot it
    avg_score = np.mean(silhouette_vals)
    print("Silhouette coefficient for k = ", k, ": ", avg_score)

    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
    ax1.set_yticks([])
    ax1.set_xlim([-0.1, 1])
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster labels')
    ax1.set_title('Silhouette plot for the various clusters', y=1.02);
    
    # Scatter plot of data colored with labels
    ax2.scatter(X_std[:, 0], X_std[:, 1], c=labels)
    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)
    ax2.set_xlim([-600, 1300])
    ax2.grid()
    ax2.set_xlabel('Velocity', fontsize=12)
    ax2.set_ylabel('$X_{RelativeAnkle}$', fontsize=12)
    ax2.set_title('Visualization of clustered data', y=1.02)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.suptitle(f'Silhouette analysis using k = {k}', fontsize=16, fontweight='semibold', y=1.05)
    plt.show()
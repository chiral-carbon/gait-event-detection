import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from math import sqrt
from sklearn.cluster import KMeans
from scipy.signal import argrelextrema, butter, filtfilt

data = pd.read_csv('kinectdata45deg.csv')            
X = np.array(data.loc[:, 'AnkleLeftX'])
Z = np.array(data.loc[:, 'AnkleLeftZ'])

def residual(fc_opt, sig):
    C = 0.802
    fs = 30.0
    wn = (fc_opt/C)/(fs/2)
    b, a = butter(2, wn , btype = 'low', output = 'ba')
    sigf = filtfilt(b, a, sig)
    return sigf

Xf = residual(2.68, X)
Zf = residual(2.71, Z)
maxpos = np.array(argrelextrema(Xf, np.greater))[0]           
minpos = np.array(argrelextrema(Xf, np.less))[0]               	    

data = np.zeros(shape=(len(Xf),2))
for i in range(len(Xf)):
    data[i] = np.array([Xf[i], Zf[i]])

K = input("No. of clusters: ")
I = input("No. of iterations: ")

kmeans = KMeans(n_clusters=K)  
kmeans.fit(data) 
print(kmeans.cluster_centers_)  
print(kmeans.labels_)  



#K Means clustering
J = np.zeros(shape=(I))             #stores cost for each iteration 
C = np.zeros(shape=(I, len(data)))  #stores cluster labels for each iteration
Mu = np.zeros(shape=(I, K, 2))      #stores cluster centers for each iteration

#Running I iterations for cost function(distortion) computing
for j in range(I):
    c = np.zeros(shape=(len(data)))
    mu_k = np.zeros(shape=(K, 2)) 
    mu_c = np.zeros(shape=(len(data), 2)) 
    fj = np.zeros(shape=(len(X))) 
    #Random initialization of cluster centers
    for k in range(K):
        mu_k[k] = data[np.random.randint(len(data), size=(1))]
    #Cluster assignment step
    for i in range(len(data)):
        dist = []
        for k in range(K):
	    dist.append(round(sqrt((data[i][0]-mu_k[k][0])**2 + (data[i][1]-mu_k[k][1])**2),2))  #Euclidean distance b/w centroid and ith point
        for k in range(K):
	    if dist[k] == min(dist):
	        c[i] = k
		mu_c[i] = mu_k[k]
    	fj[i] = min(dist)
    C[j] = c 
    #Move cluster centroid step
    for k in range(K):
        mu_k[k] = np.mean(data[c==k], axis=0)

    Mu[j] = mu_k
    #Computing the cost function for jth iteration
    J[j] = np.mean(fj)


#Determining cluster centers and labels based on iteration comprising minimum distortion
for j in range(I):
    if J[j] == min(J):
	min_j = j	
	min_mu = Mu[j]
	min_c = C[j]  

print(min_mu)
print(min_c)

#Plotting the unclustered and clustered points
ax = np.array([plt.subplot(221), plt.subplot(223), plt.subplot(122)])

ax[0].scatter(data[:,0],data[:,1], label='True Position') 
ax[0].scatter(Xf[maxpos], Zf[maxpos], label = 'HS')
ax[0].scatter(Xf[minpos], Zf[minpos], label = 'TO') 
ax[0].grid(); ax[0].legend()
ax[0].set_title('Before clustering')

ax[1].scatter(data[:,0], data[:,1], c=min_c, cmap='rainbow')  
ax[1].scatter(min_mu[:,0], min_mu[:,1], color='black')
ax[1].scatter(Xf[maxpos], Zf[maxpos], label = 'HS')
ax[1].scatter(Xf[minpos], Zf[minpos], label = 'TO')
ax[1].grid(); ax[1].legend() 
ax[1].set_title('After clustering (K='+str(K)+')')

ax[2].scatter(min_j, min(J))
ax[2].plot(range(1,I+1,1), J)
ax[2].set_xlabel('Iteration')
ax[2].set_ylabel('Distortion')
ax[2].set_title('Cost function v/s Iterations')
ax[2].grid()

plt.suptitle('K Means Clustering for ' + str(I) + ' iterations on left ankle x-z coordinate data', fontsize=16)
plt.tight_layout()
plt.show()

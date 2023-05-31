from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.signal import argrelextrema, butter, filtfilt
from Tkinter import Tk
from tkFileDialog import askopenfilename
from residual import residual
from gaitTSP import StSw
from EventCoor import detect
from math import sqrt
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


def kmeansSK(K, data):
    #sklearn inbuilt functions for K Means Clustering
    kmeans = KMeans(n_clusters=K)  
    kmeans.fit(data) 
    print("\n From sklearn.cluster: \n")
    print(kmeans.cluster_centers_)  
    print(kmeans.labels_)  

    #counting the no. of points and percentage of points in each cluster
    count = np.zeros(shape=(K))
    cent = np.zeros(shape=(K))
    for i, val in enumerate(kmeans.labels_):
        count[val] = count[val]+1

    cent = count/float(l) * 100
    for i in range(K):
        print("n(cluster %d): %d \t percent: %.3f"%(i, count[i], cent[i]))
    
    return kmeans.cluster_centers_, kmeans.labels_ #returns cluster centroids and cluster labels of each data point

def kmeansCluster(K, I, data):
    #K Means clustering
    J = np.zeros(shape=(I))             #stores cost for each iteration 
    C = np.zeros(shape=(I, len(data)))  #stores cluster labels for each iteration
    Mu = np.zeros(shape=(I, K, dim))    #stores cluster centers for each iteration

    #Running I iterations for cost function(distortion) computing
    for j in range(I):
        c = np.zeros(shape=(len(data)))
        mu_k = np.zeros(shape=(K, dim)) 
        mu_c = np.zeros(shape=(len(data), dim)) 
        fj = np.zeros(shape=(len(X))) 
        #Random initialization of cluster centers
        for k in range(K):
             mu_k[k] = data[np.random.randint(len(data), size=(1))]
        #Cluster assignment step
        for i in range(len(data)):
            dist = []
            for k in range(K):
	        #dist.append(round(np.abs(data[i][0]-mu_k[k][0])+np.abs(data[i][1]-mu_k[k][1]), 2))  #City block/Manhattan distance
	            dist.append(round(sqrt((data[i][0]-mu_k[k][0])**2+(data[i][1]-mu_k[k][1])**2),2))  #Euclidean distance 
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
    print("\n\n My implementation:\n")
    print(min_mu)
    print(min_c)

    #counting the no. of points and percentage of points in each cluster
    count = np.zeros(shape=(K))
    cent = np.zeros(shape=(K))
    for i, val in enumerate(min_c):
        count[int(val)] = count[int(val)]+1

    cent = count/float(l) * 100
    for i in range(K):
        print("n(cluster %d): %d\tpercent: %.3f"%(i, count[i], cent[i]))

    return min_mu, min_c, min_j, J, count, cent#returns cluster centroids and cluster labels of each data point

def plot(data, min_mu, min_j, K, I, dim, Xf, Vf, hs_pos, to_pos, J):
   
    #Plotting the unclustered and clustered points

    ax = np.array([plt.subplot(221), plt.subplot(223), plt.subplot(122)])

    ax[0].scatter(data[:,0],data[:,1], label='True Position') 
    ax[0].scatter(Xf[hs_pos[0]], Vf[hs_pos[0]], label = 'HS')
    ax[0].scatter(Xf[to_pos[1]], Vf[to_pos[1]], label = 'TO') 
    ax[0].set_xlabel('X coordinate')
    ax[0].set_ylabel('Linear Velocity')
    ax[0].grid(); ax[0].legend()
    ax[0].set_title('Before clustering (HS and TO from coordinate based algo)')

    ax[1].scatter(data[:,0], data[:,1], c=min_c, cmap='rainbow')  
    ax[1].scatter(min_mu[:,0], min_mu[:,1], color='black', label='Cluster centroids')
    ax[1].set_xlabel('X coordinate')
    ax[1].set_ylabel('Linear Velocity')
    ax[1].grid(); ax[1].legend() 
    ax[1].set_title('After clustering (K='+str(K)+')')
    
    #Cost function vs Iterations plot
    ax[2].plot(range(1,I+1,1), J)
    ax[2].scatter([min_j, 0], [min(J), min(J)], color='r', label = 'Min. cost')
    ax[2].plot([min_j, 0], [min(J), min(J)], color='r')
    ax[2].set_xlabel('Iteration')
    ax[2].set_ylabel('Distortion')
    ax[2].set_title('Cost function v/s Iterations')
    ax[2].grid()
    ax[2].legend()

    plt.suptitle('K-Means clustering for %d iterations on %d features'%(I, dim), fontsize=16)
    plt.show()


def confusion(y_true, y_pred, yes, pred_yes):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = float(tp) / yes 
    recall = float(tp) / pred_yes
    fscore = (2*precision*recall)/(precision + recall)

    print("True Positive: %d\nTrue Negative: %d\nFalse Positive: %d\nFalse Negative: %d"%(tp,tn,fp,fn))
    print("Accuracy: %.2f"%((tn+tp)/float(len(data))*100))
    print("Misclassification Rate: %.2f"%((fp+fn)/float(len(data))*100)) 
    print("Recall: %.2f"%(recall))
    print("Precision: %.2f"%(precision))
    print("F-score: %.2f"%(fscore))


if __name__ == "__main__":
    Tk().withdraw() 
    filename = askopenfilename()
    print(filename)

    f = pd.read_csv(filename)
    X = np.array(f.loc[:, 'AnkleLeftX'])
    t = np.array(f.loc[:, 'Time'])
    frame = np.array(f.loc[:, 'Frames'])
    V = [0]
    for i in range(len(X)-1):
        V.append((X[i+1]-X[i])/(t[i+1]-t[i])) #Velocity data
   

    fs = input("Sampling frequency: ")
    Xf = residual(X, fs, show=False)
    Vf = residual(V, fs, show=False)

    
    hs_pos, to_pos, str_time, cycles = detect(Xf, t) #coordinate based algo - HS and T0    
    stance, swing, st, sw = StSw(hs_pos, to_pos, t)
  
    dim = input("No. of features: ") #2 features for stance and swing phase clustering(K=2); 1.  x coordinate data, 2. linear velocity data
    K = input("No. of clusters: ")
    I = input("No. of iterations: ")
    l = len(Xf)
    data = np.zeros(shape=(l,dim))
    for i in range(l):
        data[i] = np.array([Xf[i], Vf[i]])

    min_mu, min_c, min_j, J, count, cent = kmeansCluster(K, I, data)
    """
    #For the first gait cycle
    l = maxpos[1]-maxpos[0] + 1
    for i in range(l):
        data[i] = np.array([Xf[maxpos[0]+i], Vf[maxpos[0]+i]])
    """
      
    	    
    yes = 0 #stores no. of actual YES, here YES - stance phase
    y_true = np.zeros(shape=(len(Xf)))
    #Calculating stance phase from 
    for i in range(len(stance)):
        lt = stance[i][1] - stance[i][0] + 1
        for j in range(lt):
	    y_true[stance[i][0]+j] = 1
	    yes = yes + 1
    for i in range(len(swing)):
        lt = swing[i][1] - swing[i][0] + 1
        for j in range(lt):
	    y_true[swing[i][0]+j] = 2

    if stance[0][0] > swing[0][0]:
        y_true[y_true == 0] = 1
    else:
        y_true[y_true == 0] = 2


    pred_yes = 0 #stores predicted YES - yes is stance phase
    y_pred = np.zeros(shape=(len(Xf)))
    for i in range(l):
        if count[0]>count[1]:
            if min_c[i] == 0:
	        y_pred[i] = 1
	        pred_yes = pred_yes + 1
	    else:
	        y_pred[i] = 2
        else:
	    if min_c[i] == 1:
	        y_pred[i] = 1
	        pred_yes = pred_yes + 1
	    else:
	        y_pred[i] = 2

    confusion(y_true, y_pred, yes, pred_yes)

    c = input("Create new csv with labeled data for stance and swing phase? \nType True for yes and False for no: ")
    if c==True:
        myData = []
        myData.append(["Stance Phase = 1, Swing Phase = 2"])
        myData.append(["Frame","Time","Filtered AnkleLeftX ", "Filtered AnkleVelocityX", "Phase from Coordinate algo", "Phase from Kmeans"])
        for i in range(len(data)):
            myData.append([frame[i], t[i], Xf[i], Vf[i], y_true[i], y_pred[i]])
        myFile = open(filename+'_Stance and Swing phases from KMeans and Coordinate algo.csv','w')
        with myFile:
            writer = csv.writer(myFile) 
            writer.writerows(myData) 

    c = input("Show graphical representation? \nType True for yes and False for no: ")
    if c==True:
	plot(data, min_mu, min_j, K, I, dim, Xf, Vf, hs_pos, to_pos, J)

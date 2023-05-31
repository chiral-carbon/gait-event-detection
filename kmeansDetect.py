from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.signal import argrelextrema, butter, filtfilt
from Tkinter import Tk
from tkFileDialog import askopenfilename
from residual import residual
from gaitTSP import StSw
from EventVel import detect
from math import sqrt
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


def kmeansSK(K, data):
    #sklearn inbuilt functions for K Means Clustering
    kmeans = KMeans(n_clusters=K)  
    kmeans.fit(data) 
    #print("\n From sklearn.cluster: \n")
    #print(kmeans.cluster_centers_)  
    #print(kmeans.labels_)  
   
    #counting the no. of points and percentage of points in each cluster
    count = np.zeros(shape=(K))
    cent = np.zeros(shape=(K))
    #for i, val in enumerate(kmeans.labels_):
     #   count[val] = count[val]+1

    cent = count/float(len(data)) * 100
    #for i in range(K):
     #   print("n(cluster %d): %d \t percent: %.3f"%(i, count[i], cent[i]))
    
    return kmeans.cluster_centers_, kmeans.labels_, count, cent #returns cluster centroids and cluster labels of each data point

def kmeansCluster(K, I, data):
    #K Means clustering
    J = np.zeros(shape=(I))             #stores cost for each iteration 
    C = np.zeros(shape=(I, len(data)))  #stores cluster labels for each iteration
    Mu = np.zeros(shape=(I, K, dim))    #stores cluster centers for each iteration

    #Running I iterations for computing cost function(distortion) 
    for j in range(I):
        c = np.zeros(shape=(len(data)))
        mu_k = np.zeros(shape=(K, dim)) 
        mu_c = np.zeros(shape=(len(data), dim)) 
        fj = np.zeros(shape=(len(data))) 
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
	    min_j = j	#stores minimum cost
	    min_mu = Mu[j] #stores cluster centroids from min. cost iteration
	    min_c = C[j]  #stores cluster labels from min. cost iteration
    #print("\n\n K means clusters:\n")
    #print(min_mu)
    #print(min_c)

    #counting the no. of points and percentage of points in each cluster
    count = np.zeros(shape=(K))
    cent = np.zeros(shape=(K))
    for i, val in enumerate(min_c):
        count[int(val)] = count[int(val)]+1

    cent = count/float(len(data)) * 100
    #for i in range(K):
    #    print("n(cluster %d): %d\tpercent: %.3f"%(i, count[i], cent[i]))

    return min_mu, min_c, min_j, J, count, cent#returns cluster centroids and cluster labels of each data point

def plot(data, mu, cp, j, K, I, dim, Xf, Vf, hs, to, J):
   
    #Plotting the unclustered and clustered points

    ax = np.array([plt.subplot(221), plt.subplot(223), plt.subplot(122)])

    ax[0].scatter(data[:,0],data[:,1], label='True Position') 
    ax[0].scatter(Xf[hs], Vf[hs], label = 'HS')
    ax[0].scatter(Xf[to], Vf[to], label = 'TO') 
    ax[0].set_xlabel('X coordinate')
    ax[0].set_ylabel('Linear Velocity')
    ax[0].grid(); ax[0].legend()
    ax[0].set_title('Before clustering (HS and TO from coordinate based algo)')

    ax[1].scatter(data[:,0], data[:,1], c=cp, cmap='rainbow')  
    ax[1].scatter(mu[:,0], mu[:,1], color='black', label='Cluster centroids')
    ax[1].set_xlabel('X coordinate')
    ax[1].set_ylabel('Linear Velocity')
    ax[1].grid(); ax[1].legend() 
    ax[1].set_title('After clustering (K=%d)'%K)
    
    #Cost function vs Iterations plot
    ax[2].plot(range(1,I+1,1), J)
    ax[2].scatter([j, 0], [min(J), min(J)], color='r', label = 'Min. cost')
    ax[2].plot([j, 0], [min(J), min(J)], color='r')
    ax[2].set_xlabel('Iteration')
    ax[2].set_ylabel('Distortion')
    ax[2].set_title('Cost function v/s Iterations')
    ax[2].grid()
    ax[2].legend()

    plt.suptitle('K-Means clustering for %d iterations on %d features'%(I, dim), fontsize=16)
    plt.show()


def confusion(y_true, y_pred, yes, pred_yes):
    #Confusion matrix geneartion 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() # true neg, false, pos, false neg, true pos returned
    precision = float(tp) / (tp+fp)
    recall = float(tp) / (tp+fn)
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
    X = np.array(f.loc[:, 'accX'])
    Y = np.array(f.loc[:, 'accY'])
    Z = np.array(f.loc[:, 'accZ'])
    t = np.array(f.loc[:, 'Time'])
    frame = np.array(f.loc[:, 'Frame'])
    #Vf = [0]
    Tk().withdraw() 
    groundfile = askopenfilename()
    print(groundfile)
    f2 = pd.read_csv(groundfile)
    lf_hs = np.array(f2.loc[:, 'HS'])
    lf_to = np.array(f2.loc[:, 'TO'])
    fs = input("Sampling frequency: ")
    Xf = residual(X, fs, show=False)
    Yf = residual(Y, fs, show=False)
    Zf = residual(Z, fs, show=False)

    #for i in range(len(Xf)-1):
     #   Vf.append((Xf[i+1]-Xf[i])/(t[i+1]-t[i])) #Velocity data - 3 point estimation

    #hs_pos, to_pos, str_time, cycles = detect(Vf, t) #velocity based algo - HS and T0    
    stance, swing = StSw(lf_hs, lf_to, t) #stance and swing phase frame limits for all cycles from velocity based algorithm
    #ignore st, sw obtained from StSw()
  
    dim = 2 #2 features for stance and swing phase clustering(K=2); 1. x coordinate data, 2. linear velocity data
    K = input("No. of clusters: ")
    I = input("No. of iterations: ")
    l = len(Xf)
    data = np.zeros(shape=(l,dim))
    for i in range(l):
        data[i] = np.array([Xf[i], Yf[i]])


    min_mu, min_c, min_j, J, count, cent = kmeansCluster(K, I, data) 
    #min_mu, min_c, count, cent = kmeansSK(K, data)
    print("\n\n K means clusters:\n")
    print(min_mu)
    print(min_c)
    
    #In y_true and y_pred, the data points in stance and swing are labeled. 1 is my representation for stance phase, 2 denotes swing phase
    yes = 0 #stores no. of actual YES, here YES - stance phase
    y_true = np.zeros(shape=(len(Xf)))
    #Calculating stance phase points 
    for i in range(len(stance)):
        lt = stance[i][1] - stance[i][0] + 1
        for j in range(lt):
	        y_true[stance[i][0]+j] = 1
	        yes = yes + 1
    #Calculating swing phase points
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
        if count[0]>count[1]: #Checks which cluster (unlabeled clusters, labels are 0 and 1) has more data points. 
  	        #Since stance has more data points than swing, 0 will denote stance if count[0] is more than count[1]. We then represent stance and 
	        #swing by 1 and 2 respectively temporarily.
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

    # Renaming: Stance - 1; Swing - 0 from erstwhile Stance - 1; Swing - 2
    y_true[y_true == 2] = 0
    y_pred[y_pred == 2] = 0

    
    #confusion matrix returns values that help in comparing stance and swing phase values from both the algorithms  
    confusion(y_true, y_pred, yes, pred_yes) 
    
    event = []
    #Determining HS and TO from clustered data
    phase = y_pred
    hs_pos = []#stores HS frames
    for i in range(len(phase)):
	    if i!=0 and phase[i]==1 and phase[i-1]==0:
	        hs_pos.append(i+1)
	    
    to_pos = [] #stores TO frames
    for i in range(len(phase)):
	    if i!=0 and phase[i]==0 and phase[i-1]==1:
	        to_pos.append(i+1)
	   
    event = []	
    for i in range(1, len(Xf)):
        hs = []; to = []
        for j in to_pos:
	        if i == j:
	            event.append("TOE OFF")
	            to.append(True)
	        else:
	            to.append(False)
        for j in hs_pos:	
            if i == j:
                event.append("HEEL STRIKE")
                hs.append(True)
            else:
                hs.append(False)
        if not np.any(to) and not np.any(hs):
	        event.append("-")
    event.append("-")

    cyc = [] #stores data cycle-wise
    for i in range(len(hs)-1):
        cyc.append(np.zeros(shape=(hs_pos[i+1]-hs_pos[i],dim)))
        for j, val in enumerate(range(hs_pos[i], hs_pos[i+1])):  #one gait cycle - from HS to frame before next consecutive HS
            cyc[i][j] = np.array([Xf[j], Yf[i]]) #cyc[i] is (i+1)th gait cycle and contains data in this gait cycle. data:(x_coordinate, x_vel)

    nst=0; nsw = 0 
    for i, c in enumerate(cyc):
        min_mu, min_c, min_j, J, count, cent = kmeansCluster(K, I, c)
        #min_mu, min_c, count, cent = kmeansSK(K, c)
        if i==0:
            mu = min_mu; cp = min_c#; j = min_j
            if count[0]>count[1]: 
                nst = nst + cent[0]; nsw = nsw + cent[1]
            else:
                nst = nst + cent[1]; nsw = nsw + cent[0]

    avg_st = float(nst) / len(cyc)
    avg_sw = float(nsw) / len(cyc)
    print("Average stance and swing percentages from the gait cycles:")
    print("Stance: " + str(avg_st))
    print("Swing: " + str(avg_sw))
    print("No. of gait cycles: %d "%(len(cyc)))
    print("No. of gait cycles from GRF: %d "%(len(lf_hs)))
    print("HS Frames: " + str(hs_pos))
    print("TO Frames: " + str(to_pos))


    c = input("Create new csv with labeled data for stance and swing phase with gait event? \nType True for yes and False for no: ")
    if c==True:
        myData = []
        myData.append(["Stance Phase = 1, Swing Phase = 0"])
        myData.append(["Frame","Time","AccX", "AccY", "Gait event from GRF", "Phase from Kmeans", "Gait Event from k-means"])
        for i in range(len(data)):
            myData.append([frame[i], t[i], Xf[i], Yf[i], y_true[i], y_pred[i], event[i]])
        myFile = open(filename+'_stance and swing from KMeans and GRF.csv','w')
        with myFile:
            writer = csv.writer(myFile) 
            writer.writerows(myData) 

    c = input("Show graphical k-means representation of the 1st gait cycle? \nType True for yes and False for no: ")
    if c==True:
	    plot(cyc[0], mu, cp, j, K, I, dim, Xf, Yf, hs[0], to[0], J)

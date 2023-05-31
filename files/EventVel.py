import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from Tkinter import Tk
from tkFileDialog import askopenfilename
from residual import residual
import csv

def detect(yf, time):
    hs_pos = []
    for i in range(len(yf)-1):
        if yf[i+1] < yf[i] and yf[i+1] < 0 and yf[i] > 0: 
	    hs_pos.append(i)  #HS at point of transition from positive to negative
    to_pos = []
    for i in range(len(yf)-1):
        if yf[i+1] > yf[i] and yf[i+1] > 0 and yf[i] < 0:
	    to_pos.append(i)  #TO at point of transition from negative to positive

 
    cycles = []
    for i in range(0, len(hs_pos)-1) :
        cycles.append([])	
        for j in range(hs_pos[i], hs_pos[i+1]) :
            cycles[i].append(yf[j])  #gait cycles from HS to next consecutive HS

    st = time[hs_pos]
    strt = []
    for i in range(len(st)-1):
	strt.append(np.abs(st[i+1]-st[i]))
    str_time = np.mean(strt)


    #returns filtered data, hs_pos, to_pos, stride time, gait cycles

    return hs_pos, to_pos, str_time, cycles

def plot(hs_pos, to_pos, yf, y):
    """
    #To plot the individual cycles over each other 
    c = 1
    for cycle in cycles : 
        plt.plot(cycle, label = 'Cycle' + str(c))
        c = c + 1
    """
    plt.plot(y, color = 'orange', label='unfiltered velocity')
    plt.plot(yf, color = 'g', label = 'Filtered velocity signal')
    plt.scatter(hs_pos, yf[hs_pos], color = 'k', label = 'HS')
    plt.scatter(to_pos, yf[to_pos], color = 'b', label = 'TO')
    plt.title('Left Ankle X direction velocity variation')
    plt.xlabel('Frames')
    plt.ylabel('Signal')
    plt.grid()
    plt.legend() 
    plt.show()


if __name__ == "__main__":
    #Browsing for csv file
    Tk().withdraw() 
    filename = askopenfilename()
    print(filename)
    data = pd.read_csv(filename)

    #Storing the left ankle X coordinate data and spine base data. To use a different feature, uncomment lines 65-66 and comment out lines67-70
    #name = input("Feature name from file: ")
    #y = np.array(data.loc[:, name])
    d = np.array(data.loc[:, 'AnkleLeftX'])
    sacrum = np.array(data.loc[:, 'SpineBaseX'])
    time = np.array(data.loc[:, 'Time'])
    frame = np.array(data.loc[:, 'Frames'])
    y = []
    for i in range(len(d)-1) :
        y.append((d[i+1]-d[i])/(time[i+1]-time[i])) #x direction velocity from x coordinate ankle data relative to spine base data

    fs = input("Enter sampling frequency: ")
    yf = residual(y, fs, show=False)
    hs_pos, to_pos, str_time, cycles = detect(yf, time)

    print("Heel strike frames: "+ str(hs_pos))
    print("Toe off frames: "+ str(to_pos)) 

    meanf = np.mean(yf)       

    c = input("Create new csv with filtered and unfiltered X direction velocity data? \nType True for yes and False for no: ")
    if c==True:
	myData = []
	myData.append(["Frame","Time","VelocityX", "Filtered VelocityX"])
	for i in range(len(y)):
    	    myData.append([frame[i], time[i], y[i], yf[i]])
	myFile = open(filename+'filt_velocity.csv','w')
	with myFile:
    	    writer = csv.writer(myFile) 
    	    writer.writerows(myData) 
    	    
    c = input("Plot graph for Velocity vs frames? \nType True for yes and False for no: ")
    if c==True:
	plot(hs_pos, to_pos, yf, y)


    


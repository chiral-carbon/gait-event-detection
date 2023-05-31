import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from Tkinter import Tk
from tkFileDialog import askopenfilename
from residual import residual
from gaitTSP import StSw
import csv

def detect(yf, time):
    hs_pos = []
    for i in range(len(yf)-1):
        if yf[i+1] < yf[i] and yf[i+1] < 0 and yf[i] > 0: 
	    hs_pos.append(i+1)  #HS at point of transition from positive to negative
    to_pos = []
    for i in range(len(yf)-1):
        if yf[i+1] > yf[i] and yf[i+1] > 0 and yf[i] < 0:
	    to_pos.append(i+1)  #TO at point of transition from negative to positive

 
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

def plot(hs_pos, to_pos, y):
    """
    #To plot the individual cycles over each other 
    c = 1
    for cycle in cycles : 
        plt.plot(cycle, label = 'Cycle' + str(c))
        c = c + 1
    """
    plt.plot(y, color = 'g', label = 'Velocity signal')
    plt.scatter(hs_pos, y[hs_pos], color = 'k', label = 'HS')
    plt.scatter(to_pos, y[to_pos], color = 'b', label = 'TO')
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

    #Storing the left ankle X coordinate data and spine base data. To use a different feature, uncomment the following two lines
    #name = input("Feature name from file: ")
    #y = np.array(data.loc[:, name])
    d = np.array(data.loc[:, 'Filtered AnkleLeftX - SpineBaseX'])
    time = np.array(data.loc[:, 'Time'])
    frame = np.array(data.loc[:, 'Frames'])
    y = []
    for i in range(len(d)-1) :
        y.append((d[i+1]-d[i])/(time[i+1]-time[i])) #x direction velocity from x coordinate ankle data relative to spine base data

    c = input("Create new csv with velocity and x coordinate data? \nType True for yes and False for no: ")
    if c==True:
	myData = []
	myData.append(["Frames","Time","Filtered AnkleLeftX - SpineBaseX", "VelocityX"])
	for i in range(len(y)):
    	    myData.append([frame[i], time[i], d[i], y[i]])
	myFile = open(filename+'_velocity.csv','w')
	with myFile:
    	    writer = csv.writer(myFile) 
    	    writer.writerows(myData) 

    hs_pos, to_pos, str_time, cycles = detect(y, time)

    print("Heel strike frames: "+ str(hs_pos))
    print("Toe off frames: "+ str(to_pos)) 
      
    event = []
    for i in range(1, len(y)):
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

    c = input("Create new csv with gait events? \nType True for yes and False for no: ")
    if c==True:
	myData = []
	myData.append(["Frames","Time","VelocityX", "Gait Event"])
	for i in range(len(y)):
    	    myData.append([frame[i], time[i], y[i], event[i]])
	myFile = open(filename+'_velEvent.csv','w')
	with myFile:
    	    writer = csv.writer(myFile) 
    	    writer.writerows(myData) 
    	    
    c = input("Plot graph for Velocity vs frames? \nType True for yes and False for no: ")
    if c==True:
	plot(hs_pos, to_pos, y)


    


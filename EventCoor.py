from scipy.signal import argrelextrema, butter, filtfilt
from Tkinter import Tk
from tkFileDialog import askopenfilename
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from residual import residual

def detect(yf, time):
    hs_pos = np.array(argrelextrema(yf, np.greater))[0]    #frames of HS at maxima
    to_pos = np.array(argrelextrema(yf, np.less))[0]        #Frames of TO at minima

    cycles = []
    for i in range(0, len(hs_pos)-1) :
        cycles.append([])	
        for j in range(hs_pos[i], hs_pos[i+1]) :
            cycles[i].append(yf[j])    #cycles from HS to next consecutive HS 

    st = time[hs_pos]
    strt = []
    for i in range(len(st)-1):
	strt.append(np.abs(st[i+1]-st[i]))
    str_time = np.mean(strt)

    #returns hs_pos, to_pos, gait cycles

    return hs_pos, to_pos, str_time, cycles

def plot(hs_pos, to_pos, yf, frame):
    """
    #To plot the individual cycles over each other, first change function definition to plot(hs_pos, to_pos, yf, y, cycles)
    c = 1
    for cycle in cycles : 
        plt.plot(cycle, label = 'Cycle' + str(c))
        c = c + 1
    """
    mean = np.mean(yf)
    plt.plot(yf, color = 'g', label = 'Filtered x coordinate signal')
    plt.scatter(hs_pos, yf[hs_pos], color = 'k', label = 'HS')
    plt.scatter(to_pos, yf[to_pos], color = 'b', label = 'TO')
    plt.plot([0, frame[-1]],[mean, mean], label = 'Mean of signal')
    plt.title('Left Ankle X Coordinate Variation wrt to Spine Base')
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
    yf = np.array(data.loc[:, 'Filtered AnkleLeftX - SpineBaseX'])
    time = np.array(data.loc[:, 'Time'])
    frame = np.array(data.loc[:, 'Frames'])

    hs_pos, to_pos, str_time, cycles = detect(yf, time)

               	        
    print("Heel strike frames: "+ str(hs_pos))
    print("Toe off frames: "+ str(to_pos)) 
    
    event = []; hs = []; to = []
	
    for i in range(1, len(yf)):
        hs = []; to = []
        for j in range(len(to_pos)):
	    if i == to_pos[j]:
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
    
    #Csv for filtered and unfiltered data
    c = input("Create new csv with gait events? \nType True for yes and False for no: ")
    if c==True:
	myData = []
	myData.append(["Frames","Time","Filtered Signal", "Gait Event"])
	for i in range(len(yf)):
    	    myData.append([frame[i], time[i], yf[i], event[i]])
	myFile = open(filename+'_coorEvent.csv','w')
	with myFile:
    	    writer = csv.writer(myFile) 
    	    writer.writerows(myData) 
    
    c = input("Plot graph for X coordinate vs frames? \nType True for yes and False for no: ")
    if c==True:
	plot(hs_pos, to_pos, yf, frame)




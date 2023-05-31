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
    first = [] #the points not in any cycle, before the first detected HS or TO
    for i in range (0, hs_pos[0]) :
        first.append(yf[i])	                   
    
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

def plot(hs_pos, to_pos, yf, y, frame):
    """
    #To plot the individual cycles over each other, first change function definition to plot(hs_pos, to_pos, yf, y, cycles)
    c = 1
    for cycle in cycles : 
        plt.plot(cycle, label = 'Cycle' + str(c))
        c = c + 1
    """
    mean = np.mean(yf)
    plt.plot(y, color = 'orange', label='Unfiltered velocity')
    plt.plot(yf, color = 'g', label = 'Filtered velocity signal')
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

    #Storing the left ankle X coordinate data and spine base data. To use a different feature, uncomment lines 50-51 and comment out lines52-55
    #name = input("Feature name from file: ")
    #y = np.array(data.loc[:, name])          
    ankle = np.array(data.loc[:, 'AnkleLeftX'])
    sacrum = np.array(data.loc[:, 'SpineBaseX'])
    time = np.array(data.loc[:, 'Time'])
    frame = np.array(data.loc[:, 'Frames'])
    y=ankle-sacrum

    fs = input("Enter sampling frequency: ")
    yf = residual(y, fs, show=False)
    hs_pos, to_pos, str_time, cycles = detect(yf, time)

               	        
    print("Heel strike frames: "+ str(hs_pos))
    print("Toe off frames: "+ str(to_pos)) 
    """
    #Csv for filtered and unfiltered data
    c = input("Create new csv with filtered and unfiltered X coordinate data? \nType True for yes and False for no: ")
    if c==True:
	myData = []
	myData.append(["Frame","Time","AnkleLeftX-SpineBaseX", "Filtered data"])
	for i in range(len(y)):
    	    myData.append([frame[i], time[i], y[i], yf[i]])
	myFile = open(filename+'_filt.csv','w')
	with myFile:
    	    writer = csv.writer(myFile) 
    	    writer.writerows(myData) 
    """
    c = input("Plot graph for X coordinate vs frames? \nType True for yes and False for no: ")
    if c==True:
	plot(hs_pos, to_pos, yf, y, frame)




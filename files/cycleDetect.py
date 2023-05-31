"""

%	Detecting different gait cycles using knee angle variation observations.
%	Detection is done by extracting gait cycles from local minima to local minima.

"""

import pandas as pd
import matplotlib.pyplot as plt
from numpy import array, less
from scipy.signal import argrelextrema

data = pd.read_csv('q11.csv')                             # Reading the csv file and storing knee angles in a list 
q11 = array(data.loc[:, 'q11'])

def cycleDetect(q11):
    minpos = array(argrelextrema(q11, less))[0]           # Stores frame no. of all occurences of minima
    mintab = q11[minpos]                                  # Stores values of all occurences of minima
    
    loc_min = []
    cycles = []

    for i, val in enumerate(minpos) :
        if mintab[i] < -0.6 :
	    loc_min.append(val)                           # Stores only the extreme minima values, which denote beginning of a new cycle
    
    cycles.append([])
    for i in range (0, loc_min[0]) :
	cycles[0].append(q11[i])	                  # Stores the first gait cycle 
    
    for i in range(0, len(loc_min)-1) :
	cycles.append([])	
	for j in range(loc_min[i], loc_min[i+1]) :
            cycles[i+1].append(q11[j])                    # Stores remaining gait cycles	    

    return loc_min, mintab[mintab < -0.6], cycles


if __name__=="__main__":
    frame, angle, cycles = cycleDetect(q11)
    
    # Plotting the graph
    c=1
    for cycle in cycles :
	plt.plot(cycle, label = 'Cycle '+ str(c))
	c = c+1
    """
    %plt.plot(q11, label = 'Left Knee Angle')
    %plt.scatter(frame, angle, color='red', label = 'Points of Minima')
   
    pt = 
    plt.plot(cycles[1], label = 'Cycle 1')
    plt.scatter(pt, cycles[1], color = 'red', label = 'Phases in one cycle') 
    """
    plt.title('Left Knee Angle Variation')
    plt.xlabel('Frames')
    plt.ylabel('Angle')
    plt.grid()
    plt.legend() 
    plt.show()

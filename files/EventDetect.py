import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema, butter, filtfilt
from sklearn.metrics import confusion_matrix

data = pd.read_csv('kinectdata45deg.csv')            
ankle = np.array(data.loc[:, 'AnkleLeftX'])
sacrum = np.array(data.loc[:, 'SpineBaseX'])
ankleY = np.array(data.loc[:, 'AnkleLeftY'])
footY = np.array(data.loc[:, 'FootLeftY'])
time = np.array(data.loc[:, 'Time'])
frame = np.array(data.loc[:, 'Frames'])
d = ankle-sacrum
y = []
for i in range(len(d)-1) :
    y.append((d[i+1]-d[i])/(time[i+1]-time[i]))

fc_optv = 2.68
C = 0.802
fs = 30.0
wn = (fc_optv/C)/(fs/2)
b, a = butter(2, wn, btype = 'low', output = 'ba')
yf = filtfilt(b, a, y)
meanf = np.mean(yf)
fv = []
cycv = []
pos = []; to_pos = []
for i in range(len(yf)-1):
    if yf[i+1] < yf[i] and yf[i+1] < 0 and yf[i] > 0:
	pos.append(i)
for i in range(len(yf)-1):
    if yf[i+1] > yf[i] and yf[i+1] > 0 and yf[i] < 0:
        to_pos.append(i)
for i in range (0, pos[0]) :
    fv.append(yf[i])	                   
    
for i in range(0, len(pos)-1) :
    cycv.append([])	
    for j in range(pos[i], pos[i+1]) :
        cycv[i].append(yf[j])  


yy = ankle - sacrum
fc_optx = 2.68
fc_opty = 2.82
fc_optFy = 2.83
C = 0.802
fs = 30.0
wn = (fc_optx/C)/(fs/2)
b, a = butter(4, wn, btype = 'low', output = 'ba')
yyf = filtfilt(b, a, yy)
mean = np.mean(yf)
maxpos = np.array(argrelextrema(yyf, np.greater))[0]  
minpos = np.array(argrelextrema(yyf, np.less))[0]                    
maxtab = yyf[maxpos]
mintab = yyf[minpos]

wn = (fc_opty/C)/(fs/2)
b, a = butter(4, wn, btype = 'low', output = 'ba')
ankleYf = filtfilt(b, a, ankleY)
wn = (fc_optFy/C)/(fs/2)
bf, af = butter(4, wn, btype = 'low', output = 'ba')
footYf = filtfilt(bf, af, footY)
cycles = []
first = []
for i in range (0, maxpos[0]) :
    first.append(yyf[i])	                   
    
for i in range(0, len(maxpos)-1) :
    cycles.append([])	
    for j in range(maxpos[i], maxpos[i+1]) :
        cycles[i].append(yyf[j])                    	    
"""
c = 1
for cycle in cycles : 
    plt.plot(cycle, label = 'Cycle' + str(c))
    c = c + 1
"""

print(maxpos)
print(pos)
print(minpos)
print(to_pos)
#plt.plot(y, color = 'g', label='unfiltered velocity')
plt.plot(yf, color = 'orange', label = 'Filtered velocity signal')
plt.scatter(pos, yf[pos], color = 'g', label = 'HS from velocity algorithm')
#plt.scatter(to_pos, yf[to_pos], color = 'g', label = 'TO from velocity algorithm')
#plt.plot(cycv[0], label = 'Velocity algo gait cycle')

#plt.plot(yy, color = 'b', label='unfiltered x coordinate')
plt.plot(yyf, color = 'red', label = 'Filtered x-coordinate signal ')
plt.scatter(maxpos, yyf[maxpos], color = 'k', label = 'HS from x-coordinate algorithm')
#plt.scatter(minpos, yyf[minpos], color = 'k', label = 'TO from x-coordinate algorithm')
plt.title('Comparison of the two algorithms - HS event')
plt.xlabel('Frames')
plt.ylabel('Signal')
plt.grid()
plt.legend() 
plt.show()


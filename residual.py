from scipy.signal import butter, filtfilt
from scipy.interpolate import UnivariateSpline, CubicSpline
from Tkinter import Tk
from tkFileDialog import askopenfilename
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

def plot(y, fs, freq, res, fclim, fc_opt, B, A):
    #Plotting the graph of residual amplitude vs cutoff frequency

    plt.plot(freq, res, label = 'Residual cutoff frequency curve')
    ylin = np.poly1d([B, A])(freq)
    plt.plot(freq, ylin, 'r--', linewidth=2)
    plt.plot(freq[fclim[0]], res[fclim[0]], 'r>', freq[fclim[1]], res[fclim[1]], 'r<', ms=9) #plots extrapolated straight line
    plt.plot([0, freq[-1]], [A, A], 'r-', linewidth=2)
    plt.plot([fc_opt, fc_opt], [0, A], 'r-', linewidth=2) #plots the line from point of intersection of residual amp v freq curve to x-axis
    plt.plot(fc_opt, 0, 'ro', markersize=7, clip_on=False, zorder=9, label='$Fc_{opt}$ = %.2f Hz' % fc_opt) #fc_opt on frequency axis(x-axis)
    plt.xlabel('Cutoff Frequency [Hz]')
    plt.ylabel('Root Mean Squared Residual')
    plt.title('Residual Analysis')
    plt.grid()
    plt.legend()
    plt.show()
    
def residual(y, fs, show):
    fclim = []  #stores limits of the linear part of the curve; i.e, the two extreme points
    res = []    #stores residual value b/w filtered and unfiltered signals
    freq = np.arange(0.1, 10.1, 0.1)  #range of cutoff frequencies

    # Correct the cutoff frequency for the number of passes in the filter
    C = 0.802  # for dual pass; C = (2**(1/npasses)-1)**0.25

    for fc in freq:
        wn = (fc/C)/(fs/2)
	#4th order butterworth filter; changed order to 2 as filfilt() returns twice the order of input signal
        b, a = butter(2, wn, btype = 'low', output = 'ba')
        yf = filtfilt(b, a, y) #filtering for each cutoff frequency.
        res = np.hstack((res, np.sqrt(np.mean((yf - y)**2)))) # residual between filtered and unfiltered signals


    # find the optimal cutoff frequency by fitting an exponential curve
    # y = A*exp(B*x)+C to the residual data and consider that the tail part
    # of the exponential (which should be the noisy part of the residuals)
    # decay starts after 3 lifetimes (exp(-3), 95% drop)
    if not len(fclim) or np.any(fclim < 0) or np.any(fclim > fs/2):
        fc1 = 0
        fc2 = int(0.95*(len(freq)-1))
	# log of exponential turns the problem to first order polynomial fit
        # make the data always greater than zero before taking the logarithm
        reslog = np.log(np.abs(res[fc1:fc2 + 1] - res[fc2]) + 1000 * np.finfo(np.float).eps)
        Blog, Alog = np.polyfit(freq[fc1:fc2 + 1], reslog, 1)
        fcini = np.nonzero(freq >= -3 / Blog)  # 3 lifetimes
        fclim = [fcini[0][0], fc2] if np.size(fcini) else []
    else:
        fclim = [np.nonzero(freq >= fclim[0])[0][0], np.nonzero(freq >= fclim[1])[0][0]]

    # find fc_opt with linear fit y=A+Bx of the noisy part of the residuals
    if len(fclim) and fclim[0] < fclim[1]:
        B, A = np.polyfit(freq[fclim[0]:fclim[1]], res[fclim[0]:fclim[1]], 1)
        # optimal cutoff frequency is the frequency where y[fc_opt] = A
        roots = UnivariateSpline(freq, res - A, s=0).roots()
        fc_opt = roots[0] if len(roots) else None
    else:
        fc_opt = None
    
    #Filtering with optimum cutoff frequency
    wn = (fc_opt/C)/(fs/2)
    bf, af = butter(2, wn, btype = 'low', output = 'ba')
    filtered = filtfilt(bf, af, y)
    
    print("Optimum Cutoff Frequency: %.2f"%fc_opt)

    if show==True:
        plot(y, f, freq, res, fclim, fc_opt, B, A)
    
    return filtered

if __name__ == "__main__":
    #Browsing for csv file
    Tk().withdraw() 
    filename = askopenfilename()
    print(filename)
    data = pd.read_csv(filename)

    #Storing the left ankle X coordinate data and spine base data. To use a different feature, uncomment lines 87-88 and comment out lines89-92

    #name = input("Feature name from file: ")
    #y = np.array(data.loc[:, name])
    ankle = np.array(data.loc[:, 'AnkleLeftX'])
    spine = np.array(data.loc[:, 'SpineBaseX']) 
    time = np.array(data.loc[:, 'Time'])
    frame = np.array(data.loc[:, 'Frames'])
    y=ankle
    print(y)
    for i in range(len(y)):
        if np.isnan(y[i]) == True:	
            cs = CubicSpline(time[0:i-1], y[0:i-1]) #spline interpolation from first value till previous value
	    y[i] = cs(time)[i] #Storing interpolated value in missing/blank/NaN array entry 
    
    #Obtaining fc_opt from residual analysis function
    f = input("Enter sampling frequency: ")
    show = input("Plot the graph of residual v frequency? \nType True for yes and False for no: ")
    filtered = residual(y, f, show)

  
    c = input("Create new csv with filtered and unfiltered values? \nType True for yes and False for no: ")
    if c==True:
        #Creating csv with  filtered values after Cubic Spline interpolation of unfiltered values  
        myData = []
        myData.append(["Frames","Time","AnkleLeftX - SpineBaseX", "Filtered AnkleLeftX - SpineBaseX"])
        for i in range(len(y)):
            myData.append([frame[i], time[i], y[i], filtered[i]])
        myFile = open(filename+'_filt.csv','w')
        with myFile:
            writer = csv.writer(myFile) 
            writer.writerows(myData)


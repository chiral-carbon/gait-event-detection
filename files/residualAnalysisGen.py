###Program to find the avg cutoff frequency of all the columns in the dataset and to store the newly filtered (with the avg fc_opt) dataset in ####a new csv.


from scipy.signal import butter, filtfilt
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from Tkinter import Tk
from tkFileDialog import askopenfilename
Tk().withdraw() 
filename = askopenfilename()
print(filename)

data = pd.read_csv(filename)
head = data.columns.values   
u, v = np.shape(data)

columns = []; opt = []
for j in range(2, v):
    columns.append([])
    columns[j-2].append(data.loc[:, head[j]])

fs = input("Enter the sampling frequency: ")
C = 0.802  # for dual pass; C = (2**(1/npasses)-1)**0.25
freq = np.arange(0.1, 10.1, 0.1)

def plot(freq, res, B, A, fclim, fc_opt, i):
#   Plotting the graph 
    plt.plot(freq, res, label = 'Residual cutoff frequency curve for ' + str(head[i+2]))
    ylin = np.poly1d([B, A])(freq)
    plt.plot(freq, ylin, 'r--', linewidth=2)
    plt.plot(freq[fclim[0]], res[fclim[0]], 'r>',
         freq[fclim[1]], res[fclim[1]], 'r<', ms=9)
    plt.plot([0, freq[-1]], [A, A], 'r-', linewidth=2)
    plt.plot([fc_opt, fc_opt], [0, A], 'r-', linewidth=2)
    plt.plot(fc_opt, 0, 'ro', markersize=7, clip_on=False, zorder=9, label='$Fc_{opt}$ = %.2f Hz' % fc_opt)
    plt.xlabel('Cutoff Frequency [Hz]')
    plt.ylabel('Root Mean Squared Residual')
    plt.title('Residual Analysis')
    plt.grid()
    plt.legend()
    plt.show()

for i, y in enumerate(columns):
    fclim = []
    res = []
    print(y)
    for i in range(len(y)):
        if np.isnan(y[i]) == True:	
            cs = CubicSpline(time[0:i-1], y[0:i-1]) #spline interpolation from first value till previous value
	    y[i] = cs(time)[i] #Storing interpolated value in missing/blank/NaN array entry 

    for fc in freq:
        wn = (fc/C)/(fs/2)
        b, a = butter(2, wn, btype = 'low', output = 'ba')
        yf = filtfilt(b, a, y)
        res.append(np.sqrt(np.mean((y-yf)**2)))

    if not len(fclim) or np.any(fclim < 0) or np.any(fclim > f/2):
        fc1 = 0
        fc2 = int(0.95*(len(freq)-1))
        reslog = np.log(np.abs(res[fc1:fc2 + 1] - res[fc2]) + 1000 * np.finfo(np.float).eps)
        Blog, Alog = np.polyfit(freq[fc1:fc2 + 1], reslog, 1)
        fcini = np.nonzero(freq >= -3 / Blog)  # 3 lifetimes
        fclim = [fcini[0][0], fc2] if np.size(fcini) else []
    else:
        fclim = [np.nonzero(freq >= fclim[0])[0][0], np.nonzero(freq >= fclim[1])[0][0]]


    if len(fclim) and fclim[0] < fclim[1]:
        B, A = np.polyfit(freq[fclim[0]:fclim[1]], res[fclim[0]:fclim[1]], 1)
        roots = UnivariateSpline(freq, res - A, s=0).roots()
        fc_opt = roots[0] if len(roots) else None
	
    else:
        fc_opt = None
    
    opt.append(fc_opt)
    #x, y = [[freq[fclim[0]], freq[fclim[1]], fc_opt], [res[fclim[0]], res[fclim[1]], 0]]
    #m = (y[1] - y[0])/(x[1]-x[0]) ; c = y[0] - m*x[0]
    #x.append(0); y.append(c)
    
    #Filtering with optimum cutoff frequency
    wn = (fc_opt/C)/(fs/2)
    bf, af = butter(2, wn, btype = 'low', output = 'ba')
    filtered = filtfilt(bf, af, y)

    #for i in range(head):
	

print("Optimal frequency obtained by residual analysis: " + str(np.mean(opt)))


	

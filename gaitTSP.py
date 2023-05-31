from scipy.signal import butter, filtfilt
from scipy.interpolate import UnivariateSpline
from Tkinter import Tk
from tkFileDialog import askopenfilename
from residual import residual
from EventCoor import detect
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def StSw(hs_li, to_li, time):
    #Function to determine stance and swing phase frame limits
    #hs = np.array(hs_li); to = np.array(to_li)

    hs = hs_li
    to = to_li
    
    swing = []; stance = []; swt = []; stt = []
    #Determines whether first detected TO has frame number before first detected HS 
    #Proceeds as 
    if hs[0] > to[0]:
	    for i in range(len(to)):
	        swing.append([to[i], hs[i]-1])
            #swt.append(time[hs[i]-1] - time[to[i]])
	    for i in range(len(hs)-1):
	        stance.append([hs[i], to[i+1]-1])
           # stt.append(time[to[i+1]-1] - time[hs[i]])
    else:
	    for i in range(len(hs)-1):
	        swing.append([to[i], hs[i+1]-1])
           # stt.append(time[to[i]] - time[hs[i+1]-1])
	    for i in range(len(to)):
	        stance.append([hs[i], to[i]])
            #swt.append(time[hs[i]] - time[to[i]-1])
    
    #returns stance phases' frame limits, swing phases' frame limits, stance time, swing time
    return stance, swing
#round(np.mean(stt),3), round(np.mean(swt),3)

def plot(signal, hs, to, ax):
    for i in range(2):
        ax[i].scatter(hs[i], signal[i][hs[i]], color = 'k', label = 'HS')
        ax[i].scatter(to[i], signal[i][to[i]], color = 'r', label = 'TO')
        ax[i].plot(signal[i], color = 'orange', label = 'Opt. filtered')
        ax[i].set_ylabel('X direction velocity')
        ax[i].grid()
        ax[i].legend() 
    ax[1].set_xlabel('Frames')
    ax[0].set_title('Left Ankle')
    ax[1].set_title('Right Ankle')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #Browsing for csv
    Tk().withdraw() 
    filename = askopenfilename()
    print(filename)
   
    data = pd.read_csv(filename) 
  
    left = np.array(data.loc[:, 'AnkleLeftX'])  #left leg data; change name for different body feature
    right = np.array(data.loc[:, 'AnkleRightX'])#right leg data; change name for different body feature
    time = np.array(data.loc[:, 'Time'])
    
    fs = input("Sampling frequency: ")
    leftfilt = residual(left, fs, show=False) #residual analysis
    rightfilt = residual(right, fs, show=False) #residual analysis

    """
    signal = left ankle and right ankle filtered data 
    hs = heel strike for left and right 
    to = toe off for " " " "
    cycles = gait cycles for left and right from velocity based treadmill algo
    stance = stance phase frame limits for left and right
    swing = swing phase frame limits for left and right
    str_time = stride time - index 0 has left ankle data, 1 contains right ankle
    
    All the variables have two indices - 0 for left leg, 1 for right leg
    
    Similarly, swingtime, stanceTime, singleSupp have 0 index for left leg, 1 index data for right leg
    """
    hs = []; to = []; str_time = []; cycles = []
    hs.append([]); to.append([]); str_time.append([]); cycles.append([])
    hs[0], to[0], str_time[0], cycles[0] = detect(leftfilt, time)
    hs.append([]); to.append([]); str_time.append([]); cycles.append([])
    hs[1], to[1], str_time[1], cycles[1] = detect(rightfilt, time)
    print(hs, to,str_time, cycles)
    print("\n\n------TEMPORAL PARAMETERS------\n")
    strideTime = np.mean(str_time)
    print("Stride time: " + str(round(strideTime, 3)) + "sec")
    stepTime = np.mean(np.abs(time[hs[0]] - time[hs[1]]))
    print("Step time: " + str(round(stepTime, 3)) + "sec")
    
    stance = []; swing = []; stanceTime = []; swingTime = []
    stance.append([]); swing.append([]); stanceTime.append([]); swingTime.append([])
    stance[0], swing[0], stanceTime[0], swingTime[0] = StSw(hs[0], to[0], time)
    stance.append([]); swing.append([]); stanceTime.append([]); swingTime.append([])
    stance[1], swing[1], stanceTime[1], swingTime[1] = StSw(hs[1], to[1], time)
    print("Stance time for left leg: " + str(stanceTime[0]) + " sec")
    print("Swing time for left leg: " + str(swingTime[0]) + " sec")
    print("Stance time for right leg: " + str(stanceTime[1]) + " sec")
    print("Swing time for right leg: " + str(swingTime[1]) + " sec")

    #Single support time for left leg = swing time for right leg and vice-versa
    #Stance phase = first double support + single support + second double support
    #  => double support time = (stance time(for left)- single support time(for left))/2
    singleSupp = []
    singleSupp.append([])
    singleSupp[0] = swing[1]
    singleSupp.append([])
    singleSupp[1] = swing[0]
    print("Single support time for left leg: " + str(swingTime[1]) + " sec")
    print("Single support time for right leg: " + str(swingTime[0]) + " sec")

    print("Double support time: " + str(round(np.mean((stanceTime[0]-swingTime[1])/2),3)) + " sec")

    #Stride length = left step length + right step length
    print("\n\n------SPATIAL PARAMETERS------\n")
    stepLeft = np.abs(leftfilt[hs[0]] - rightfilt[hs[0]])
    print("Step length (left foot): " + str(round(np.mean(stepLeft), 3)) + "m")
    stepRight = np.abs(rightfilt[hs[1]] - leftfilt[hs[1]])
    print("Step length (right foot): " + str(round(np.mean(stepRight), 3)) + "m")
    strideLength = stepLeft + stepRight 
    print("Stride length: " + str(round(np.mean(strideLength),3)) + "m")
    

    #Step width = right step - left step (Z axis assumed to be lateral)
    left_width = np.array(data.loc[:, 'AnkleLeftZ'])
    right_width = np.array(data.loc[:, 'AnkleRightZ'])
    stepWidth = np.mean(np.abs(left_width[hs[0]] - right_width[hs[1]]))
    print("Step width: " + str(round(stepWidth,3)) + "m")
    
    signal = [leftfilt,rightfilt]
    ax = np.array([plt.subplot(211), plt.subplot(212)])
    plot(signal, hs, to, ax)


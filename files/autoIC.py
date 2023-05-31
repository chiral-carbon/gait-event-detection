import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
from scipy.signal import butter, filtfilt
from Tkinter import Tk
from tkFileDialog import askopenfilename

Tk().withdraw() 
filename = askopenfilename()
print(filename)
data = pd.read_csv(filename)

leftX = np.array(data.loc[:, 'AnkleLeftX'])
leftZ = np.array(data.loc[:, 'AnkleLeftX'])
left = np.vstack((leftX, leftZ)).T

rightX = np.array(data.loc[:, 'AnkleRightX'])
rightZ = np.array(data.loc[:, 'AnkleRightZ'])
right = np.vstack((rightX, rightZ)).T

print(np.shape(left))
print(np.shape(right))

mid = (left + right) / 2 
print(mid)
midpt = np.mean(mid)
print(mid)
print(midpt)
print(left)
print(right)

mlin = []
for m in mid:
    mlin.append(np.abs(midpt - m))


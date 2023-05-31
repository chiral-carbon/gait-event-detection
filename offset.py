from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
import matplotlib as 

Tk.withdraw()
file = askopenfilename()
print(file)
data = pd.read_csv(file)

v = np.array(data.loc[:, 'Phase from Velocity algo'])
k = np.array(data.loc[:, 'Phase from Kmeans'])

offset = np.zeros(11)

for i in range(len(data)):
	if v[i] == k[i]:
		offset[0] += 1
	if v[i]!= k[i]:
		
		


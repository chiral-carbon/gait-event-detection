import matplotlib.pyplot as plt 
import numpy as np
import csv

frame = []
cyc2 = []
cyc3 = []
with open('gc-sp.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
       frame.append(float(row[0]))
       cyc2.append(float(row[1]))
       cyc3.append(float(row[2]))

plt.plot(frame,cyc2,label='2nd cycle')
plt.plot(frame,cyc3,label='3rd cycle')
plt.grid()
plt.xlabel('Frames')
plt.ylabel('Angle')
plt.title('Left Hip Angle variation over 2 gait cycles')
plt.legend()
plt.show()
    
data = pd.read_csv('gc-spt.csv')
c = np.array(data.loc[0:249,'Hsignal'])
print(np.corrcoef(c, rowvar = True)) # 1.0

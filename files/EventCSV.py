import csv
import numpy as np
from EventDetect import ankleYf, footYf, minpos, maxpos, frame

yc = []
for i in range(minpos[0], maxpos[0]):
    yc.append(ankleYf[i]) 
"""
print("Frame"+str(maxpos[0])+": "+str(cycy[0][0])+"\t<--First HS")
for i in range(len(cycy[0])-1):
    if cycy[0][i+1] == ankleYf[minpos[1]]:
	print ("Frame"+str(minpos[1])+": "+str(cycy[0][i+1])+"\t<--First TO")
    else:
	print("Frame"+str(maxpos[0]+i+1)+": "+str(cycy[0][i+1]))
print("Frame"+str(maxpos[1])+": "+str(cycy[1][0])+"\t<--Next HS")
"""
myData = []
c = []
myData.append(["Frame", "Filtered AnkleLeftY(kinectdata45deg.csv)", "Event", "Filtered FootLeftY(kinectdata45deg.csv)"])
hs = []; to = []
	
for i in range(1, len(ankleYf)):
    hs = []; to = []
    for j in range(len(minpos)):
	if i == minpos[j]:
	    c.append("TOE OFF")
	    to.append(True)
	else:
	    to.append(False)
    for j in range(len(maxpos)):	
	if i == maxpos[j]:
	    c.append("HEEL STRIKE")
	    hs.append(True)
        else:
	    hs.append(False)
    if not np.any(to) and not np.any(hs):
	c.append("-")
c.append("-")
for i in range(len(frame)):
    myData.append([frame[i], ankleYf[i], c[i], footYf[i]])
myFile = open('Left Ankle TO and HS.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(myData)

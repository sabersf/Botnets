import os
import numpy as np
from pylab import *
import matplotlib.pyplot  as pyplot

#Reading the Events ID name from the EID.txt
EID_set = []
file1 = open('EID.txt')
for line in file1:
        EID = line.strip()
        EID_set.append(EID)
file1.close()

#Let's try it on the "Cutwail" bot first, then generalize it
def power_law(name):
	file1 = open('C:\\Users\\USER\\Documents\\Bitsight\\BitSightData\\botnets\\' + name +".txt")
	No_attack = [[0] * 401 for _ in range(5917)]
	#53045
	day = 0
	for line in file1:
		EID = line.split()
		for event in EID:
			No_attack[EID_set.index(event)][day] += 1
		day += 1
	file1.close()
	active_days = np.zeros(5917)
	x = np.zeros(401)
	y = np.zeros(401)
	for i in range(0,5917):
		for j in range(0,day):
			if No_attack[i][j] > 0:
				active_days[i] += 1
		x[active_days[i]] += 1
	y = x / 5917
	y = -1 * log(y)
	for i in range(0,401):
		x[i] = log(i)
	plt.plot(x[1:],y[1:],'ro')
	plt.title("Botnet: " + name)
	plt.ylabel('Neg log probability of infection')
	plt.xlabel('Log length of infection')
	savefig(name +".pdf", format = 'pdf')

	

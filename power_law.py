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
	file1 = open( name +".txt")
	No_attack = [[0] * 401 for _ in range(5917)]
	day = 0
	for line in file1:
		EID = line.split()
		for event in EID:
			No_attack[EID_set.index(event)][day] += 1
		day += 1
	file1.close()
	

	
power_law("Cutwail")

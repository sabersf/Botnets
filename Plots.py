
# coding: utf-8

# In[1]:

from __future__ import print_function
import matplotlib.pyplot as plt

# coding: utf-8

# In[33]:

__author__ = 'Saber Shokat Fadaee'

#from gensim import corpora, models, similarities
#from gensim.models.doc2vec import TaggedDocument, LabeledSentence, Doc2Vec
#import gensim
from sklearn import manifold, datasets
import numpy as np
from itertools import chain
import multiprocessing
import csv
import matplotlib as ml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from matplotlib.backends.backend_pdf import PdfPages
import random

import numpy as np
import lda
from sklearn.datasets import make_checkerboard
from sklearn.datasets import samples_generator as sg
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.cluster.bicluster import SpectralCoclustering

from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals.six import iteritems
from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import v_measure_score
from sklearn.utils.extmath import *
from sklearn.metrics import consensus_score

import operator

storage = {}
i = 1.0
EID_set = set()
botnet_set = set()
event_set = set()
drop_ent = []

file1 = open('drop_ent.txt')
for line in file1:
    drop_ent.append(line.strip())
file1.close()

file1 = open('EID.txt')
for line in file1:
        EID = line.strip()
        EID_set.add(EID)
file1.close()

file1= open("botnets.txt")
for line in file1:
    botnet = line.strip()
    botnet_set.add(botnet)
file1.close()

count = np.loadtxt("count.txt")

botnet_family = []
file1= open("bot_relations.txt")
for line in file1:
    botnet_family.append(line.strip().split())
file1.close()
#Plus one for the unidentified classes
num_classes = len(botnet_family) + 1


EID_set = sorted(EID_set)
botnet_set = sorted(botnet_set)
event_set = sorted(event_set)


# In[34]:

#Set colors to each category
def sec_to_col(argument):
    switcher = {
		'Aerospace/Defense': 'aqua',
		'Business Services': 'blueviolet',
		'Consumer Goods': 'brown',
		'Education': 'coral',
		'Energy/Resources': 'crimson',
		'Engineering': 'darkgreen',
		'Finance': 'gold',
		'Food Production': 'green',
		'Government/Politics': 'lime',
		'Healthcare/Wellness': 'magenta',
		'Insurance': 'mintcream',
		'Legal': 'olive',
		'Manufacturing': 'orchid',
		'Media/Entertainment': 'peru',
		'Nonprofit/NGO': 'purple',
		'Real Estate': 'red',
		'Retail': 'skyblue',
		'Technology': 'silver',
		'Telecommunications': 'tomato',
		'Tourism/Hospitality': 'peachpuff',
		'Transportation': 'rosybrown',
		'Unknown': 'dimgray',
		'Utilities': 'royalblue',
    }
    return switcher.get(argument, "yellow")



#Set color to the different sizes
	
def size_to_col(argument):
    switcher = {
		'0-100': 'red',
		'100-1000': 'blue',
		'1000-10000': 'brown',
		'10000-50000': 'green',
		'50000+': 'gold',
		'Unknown': 'lime',
    }
    return switcher.get(argument, "yellow")

# Assigns the topics to the documents in corpus

col = []
col_size = []

sector = {}
count_range = {}

#Adding extra information
with open('extra.csv', 'rb' ) as theFile:
    reader = csv.DictReader( theFile )
    for line in reader:
		ind = int(line['']) 
		eid = line['entity_id_hash']
		sec = line['industry_sector']
		cnt = line['employee_count_range']
		sector[eid] = sec
		count_range[eid] = cnt

#Set numbers to each category
def sec_to_num(argument):
    switcher = {
		'Aerospace/Defense': 0,
		'Business Services': 1,
		'Consumer Goods': 2,
		'Education': 3,
		'Energy/Resources': 4,
		'Engineering': 5,
		'Finance': 6,
		'Food Production': 7,
		'Government/Politics': 8,
		'Healthcare/Wellness': 9,
		'Insurance': 10,
		'Legal': 11,
		'Manufacturing': 12,
		'Media/Entertainment': 13,
		'Nonprofit/NGO': 14,
		'Real Estate': 15,
		'Retail': 16,
		'Technology': 17,
		'Telecommunications': 18,
		'Tourism/Hospitality': 19,
		'Transportation': 20,
		'Unknown': 21,
		'Utilities': 22,
    }
    return switcher.get(argument, 23)
#Set numbers to each size
def size_to_num(argument):
    switcher = {
		'0-100': 50,
		'100-1000': 500,
		'1000-10000': 5000,
		'10000-50000': 50000,
		'50000+': 100000,
		'Unknown': 1,
    }
    return switcher.get(argument, 6)

#Set numbers to each size
def size_to_category(argument):
    switcher = {
		'0-100': 1,
		'100-1000': 2,
		'1000-10000': 3,
		'10000-50000': 4,
		'50000+': 5,
		'Unknown': 0,
    }
    return switcher.get(argument, 6)

#Set category to each number
def num_to_sec(argument):
    switcher = {
		0:'Aerospace/Defense',
		1:'Business Services',
		2:'Consumer Goods',
		3:'Education',
		4:'Energy/Resources',
		5:'Engineering',
		6:'Finance',
		7:'Food Production',
		8:'Government/Politics',
		9:'Healthcare/Wellness',
		10:'Insurance',
		11:'Legal',
		12:'Manufacturing',
		13:'Media/Entertainment',
		14:'Nonprofit/NGO',
		15:'Real Estate',
		16:'Retail',
		17:'Technology',
		18:'Telecommunications',
		19:'Tourism/Hospitality',
		20:'Transportation',
		21:'Unknown',
		22:'Utilities',
    }
    return switcher.get(argument,23)
#Set numbers to each size
def num_to_size(argument):
    switcher = {
		1:'0-100',
		2:'100-1000',
		3:'1000-10000',
		4:'10000-50000',
		5:'50000+',
		0:'Unknown',
    }
    return switcher.get(argument, 6)


# In[36]:

def in_list(item,L):
    for i in L:
        if item in i:
            return L.index(i)
    return num_classes - 1
def bot_to_vector(bot):
    output = [0] * num_classes
    output[in_list(bot, botnet_family)] = 1
    return output
def included_entry(entry_name):
	#remove the unwanted sectors
    if sector[entry_name] == 'Education':
        return False
    if sector[entry_name] == 'Technology':
        return False
    if sector[entry_name] == 'Tourism/Hospitality':
        return False
    if sector[entry_name] == 'Telecommunications':
        return False
    if sector[entry_name] == 'Unknown':
        return False    
    return True
def sectors_count(botnet_group):
    sectors_count = [0]*23
    res = dict()
    for i in range(len(EID_set_new)):
        if count_new[botnet_group,i] > 0:
            sectors_count[sec_to_num(sector[EID_set_new[i]])] += count_new[botnet_group,i]
    for i in range(23):
        res[num_to_sec(i)] = sectors_count[i]
    return res

def sectors_count_botnet(bot):
    sectors_count = [0]*23
    for i in range(len(EID_set_new)):
        if count_new1[bot,i] > 0:
            sectors_count[sec_to_num(sector[EID_set_new[i]])] += count_new1[bot,i]
    return sectors_count






print(sum(included_entry(entity) for entity in EID_set))

#Normalizing the count matrix by dividing the attacks on each entity by its size
#for i in range(207):
#    for j in range(5916):
#        count[i,j] = (count[i,j] + 0.0) / (0.0 + size_to_num(count_range[EID_set[j]]))


index  = 2899
#index = 4475
count_new1 = np.zeros((207,index))
#Build a new count matrix excluding the unwanted sectors
index = 0
EID_set_new = []
for i in range(len(EID_set)):
    if included_entry(EID_set[i]) and (EID_set[i] not in drop_ent):
        count_new1[:,index] = count[:,i]
        EID_set_new.append(EID_set[i])
        index += 1
print("Total number of entities: %d"%index)
count_new = np.zeros((num_classes,index))

#for i in range(len(botnet_set)):
#    count_new[in_list(botnet_set[i], botnet_family) ,:] += count_new1[i,:]

for i in range(len(botnet_set)):
    count_new[in_list(botnet_set[i], botnet_family) ,:] += count_new1[i,:]

print(count_new.shape)    


sum_count_new = 0
for i in range(num_classes):
    sum_count_new += sum(count_new[i,:])
print("Total number of attacks by botnets on the new entities : %d"%sum_count_new)


# In[ ]:

#Draw plots of bot-family sector
def draw_plots():
    for i in range(num_classes-1):
        x = range(23)
        y = sectors_count(i).values()
        labels = sectors_count(i).keys()
        plt.figure(figsize=(16,18))
        plt.plot(x, y, 'r-')
        plt.title(("Group: %d. Contains botnets like: %s %s")%(i+1,botnet_family[i][0],botnet_family[i][1]))
        plt.xticks(x, labels, rotation='vertical')
        plt.savefig("Group_%d.png"%(i+1))
        plt.close()


# In[2]:

sector_count = [0] * 23
labels = []
for i in range(5916):
    sector_count[sec_to_num(sector[EID_set[i]])] += 1
for i in range(23):
    labels.append(num_to_sec(i))
sec_labels = labels
for i in range(23):
    print(sec_labels[i], sector_count[i])


# In[101]:

sector_per_person = [0] * 23
total_sum = 0
total_num = 0
for i in range(5916):
    temp_sum = 0
    for j in range(207):
        if count[j][i] > 0:
            temp_sum += 1
    sector_per_person[sec_to_num(sector[EID_set[i]])] += temp_sum
    total_sum += temp_sum
    total_num += 1.0
for i in range(23):
    sector_per_person[i]= (1.0*sector_per_person[i]) / (sector_count[i] * 1.0)
    print(sec_labels[i], sector_per_person[i])
ave = (total_sum*1.0)/(total_num*1.0)
print("Total sum: %d, total nume: %d, average: %.3f "%(total_sum,total_num, ave ))


fig, ax = plt.subplots()
ax.bar(range(23), sector_per_person, align="center", color = 'cyan', width = 1)
fig.autofmt_xdate()
ax.set_title('Average number of attacks per entity in each sector')
ax.set_ylabel('Number of attacks')
ax.axhline(y = ave, linestyle='--')
fig.figsize = (25,35)
# Set the ticks to be at the edges of the bins.
ax.set_xticks([i+0.2 for i in range(23)])
ax.tick_params(labelsize=8) 
ax.set_xticklabels(sec_labels,rotation='vertical') 
fig.savefig('sector_per_person.pdf', dpi=fig.dpi)


# In[100]:

observed_botnets = 0
hist_num_sec = [0] * 141
index = 0
for i in range(206):
    if 'unk' not in botnet_set[i].lower():
        temp_sec_count = [0] * 23
        for j in range(5916):
            if count[i][j]:
                temp_sec_count[sec_to_num(sector[EID_set[j]])] = 1
        hist_num_sec[index] = sum(temp_sec_count)
        observed_botnets += 1
        index += 1

fig, ax = plt.subplots()
ax.hist(hist_num_sec, bins=24, color='darkblue', width = 0.8)
fig.autofmt_xdate()
ax.set_title('Histogram of the number of different sectors each \n botnet attacks among the observed bot families')
ax.set_ylabel('Number of unique bot families that attack that many sectors')
fig.figsize = (24,40)
x_pos = [i+0.35 for i in range(1,24)]
for i in range(2,23):
    x_pos[i] -= 0.05 * (i+1)
for i in range(11,23):
    x_pos[i] += 0.5 
x_pos[11] -=0.25
ax.set_xticks(x_pos)
ax.tick_params(labelsize=8) 

hist_labels = [i for i in range(1,24)]
ax.set_xticklabels(hist_labels, ha = 'center') 
ax.set_xlim(0,24)
ax.set_ylim(0,32)
fig.savefig('hist_sec.pdf', dpi=fig.dpi)


# In[ ]:


fig, ax = plt.subplots()
ax.bar(range(23), sector_count, align="center", color = 'green', width = 1)
fig.autofmt_xdate()
ax.set_title('Proportions of different organization sectors in our data set')
ax.set_ylabel('Number of unique entities in that sector')
fig.figsize = (25,35)
# Set the ticks to be at the edges of the bins.
ax.set_xticks([i+0.2 for i in range(23)])
ax.tick_params(labelsize=8) 
ax.set_xticklabels(sec_labels,rotation='vertical') 

fig.savefig('sector.pdf', dpi=fig.dpi)


# In[ ]:

size_count = [0] * 6
labels = []
for i in range(5916):
    size_count[size_to_category(count_range[EID_set[i]])] += 1
for i in range(6):
    labels.append(num_to_size(i))
size_labels = labels
print(labels, size_count)


# In[ ]:

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.bar(range(6), size_count, align="center", color = 'blue', width = 1)
fig.autofmt_xdate()
ax.set_title('Proportions of different organization sizes in our data set')
ax.set_ylabel('Number of unique entities in with that number of employees')
fig.figsize = (25,35)
# Set the ticks to be at the edges of the bins.
ax.set_xticks([i+0.2 for i in range(6)])
ax.tick_params(labelsize=8) 
ax.set_xticklabels(size_labels,rotation='vertical') 

fig.savefig('size.pdf', dpi=fig.dpi)


# In[ ]:




# In[ ]:

import csv
data = list(csv.reader(open("sector_size.csv")))
del data[0]
for i in range(len(data)):
    del data[i][0]
data = np.array(data).astype(np.int32)


# In[ ]:




# In[ ]:

column_labels = size_labels
row_labels = sec_labels

fig, ax = plt.subplots()
heatmap = ax.pcolor(data, cmap='bwr')
fig.autofmt_xdate()

fig.figsize = (45,35)
# put the major ticks at the middle of each cell
ax.set_xticks([i+0.2 for i in range(6)])
ax.set_yticks([i+0.25 for i in range(23)])

ax.set_title('Heatmap of sector and size relationships among different organizations')

# want a more natural, table-like display
#ax.invert_yaxis()
#ax.xaxis.tick_top()

ax.set_xticklabels(size_labels, ha = 'center')
ax.set_yticklabels(sec_labels, ha = 'right')

fig.savefig('sec_size.pdf', dpi=fig.dpi)
plt.show()


# In[ ]:




# In[ ]:

N = 3
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

r_vals = [5.6,5.6,20]
rects1 = ax.bar(ind, r_vals, width, color='r')
dt_vals = [18.52,22.22,33.33]
rects2 = ax.bar(ind+width, dt_vals, width, color='g')
dp_vals = [25.92,25.92,41.67]
rects3 = ax.bar(ind+width*2, dp_vals, width, color='b')

ax.set_title('Prediction accuracy of different methods')
ax.set_xlabel('Predication target')
ax.set_ylabel('Percentage')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('Sector', 'Sector*', 'Size') )
ax.legend( (rects1[0], rects2[0], rects3[0]), ('Random', 'Decision Tree', 'Deep Neural Network'), loc = 2 )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
fig.savefig('predictions.pdf', dpi=fig.dpi)

plt.show()


# In[ ]:




# In[27]:

hist_num_sec


# In[ ]:




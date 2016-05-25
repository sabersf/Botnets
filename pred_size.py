from __future__ import print_function

__author__ = 'Saber Shokat Fadaee'

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
		0:'0-100',
		1:'100-1000',
		2:'1000-10000',
		3:'10000-50000',
		4:'50000+',
		5:'Unknown',
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


# In[37]:

count_new.shape


# In[67]:

#Create test, train data sets for deep net input/output
from sklearn.cross_validation import train_test_split

input1 = []
output = []

#Predicting the sector
if False:
    #18 is the number of sec that has any elements
    #30 is the min that all sectors can get
    count_sec = [0]* 23
    while sum(count_sec) < 18*30:
        i = random.randint(0, len(EID_set_new) - 1)
        if count_sec[sec_to_num(sector[EID_set_new[i]])] >= 30:
            continue
        else:
            input1.append(count_new[:,i])
            sec_to_vec = [0]*23
            sec_to_vec[sec_to_num(sector[EID_set_new[i]])] = 1
            count_sec[sec_to_num(sector[EID_set_new[i]])] += 1
            output.append(sec_to_vec)

#Predicting the size
count_size = [0] * 6
while sum(count_size) < 5 * 168:
    i = random.randint(0, len(EID_set_new) - 1)
    if size_to_category(count_range[EID_set_new[i]]) == 0:
        continue # as we are not interested in the unknown size
    if count_size[size_to_category(count_range[EID_set_new[i]])] >= 168:
        continue
    else:
        input1.append(count_new[:,i])
        size_to_vec = [0,0,0,0,0,0]
        size_to_vec[size_to_category(count_range[EID_set_new[i]])] = 1
        count_size[size_to_category(count_range[EID_set_new[i]])] += 1
        output.append(size_to_vec)
        
print(len(input1), len(input1[0]))
print(len(output), len(output[0]))



# In[ ]:




# In[71]:

print(count_size)
#Randomizing the input
inp = np.array(input1)
out = np.array(output)
X_train, X_test, Y_train, Y_test = train_test_split(inp, out, test_size=0.1, random_state=42)
print(len(X_train), len(X_test), len(Y_train), len(Y_test) )


# In[72]:

np.random.seed(1337)  # for reproducibility
import theano
#import theano.sandbox.cuda
#theano.sandbox.cuda.use("gpu0")

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


#training the sequential model
model = Sequential()
model.add(Dense(32, input_shape=(20,)))
model.add(Activation('tanh'))
model.add(Dropout(0.05))


model.add(Dense(64))
model.add(Activation('tanh'))
model.add(Dropout(0.05))


#model.add(Dense(32))
#model.add(Activation('tanh'))
#model.add(Dropout(0.05))

#model.add(Dense(16))
#model.add(Activation('tanh'))
#model.add(Dropout(0.05))

#model.add(Dense(8))
#model.add(Activation('tanh'))
#model.add(Dropout(0.05))

model.add(Dense(6))
model.add(Activation('softmax'))

model.summary()
batch_size = 8
nb_classes = 23
nb_epoch = 500

try:
    #target = open("NN_out.txt", 'w')
    model.compile(loss='categorical_crossentropy', optimizer=SGD(), class_mode="categorical")

    history = model.fit(X_train, Y_train,  batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, validation_data=(X_test, Y_test))
except Exception,e: 
    print(str(e))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score)

p = model.predict(X_test)
yy = np.argmax(p, axis=1)
yyy = np.argmax(Y_test, axis=1)

a = np.equal(yy, yyy)
test_acc = ( 100.0 * (0.0 + sum(a)) / (len(a) + 0.0 ))

p = model.predict(X_train)
yy = np.argmax(p, axis=1)
yyy = np.argmax(Y_train, axis=1)

a = np.equal(yy, yyy)
train_acc = ( 100.0 * (0.0 + sum(a)) / (len(a) + 0.0 ))
print("NB_EPOCH : " , str(nb_epoch) , " Score: " , str(score) , " test accuracy: " , str(test_acc) , " Train accuracy: "  , str(train_acc) + "\n")
#target.close()


# In[73]:

print(yy)


# In[74]:

print(yyy)


# In[ ]:

X_train[0]


ans = 0.0
p = model.predict(X_test)
yy = np.argmax(p, axis=1)
yyy = np.argmax(Y_test, axis=1)

for i in range(len(Y_test)):
    pred = p[i].argsort()[-2:][::-1]
    if np.argmax(Y_test[i]) in pred:
        ans += 1.0
print("Top 2 prediction precision: %.2f"%(100.0*ans / (len(Y_test) + 0.0)))


# In[75]:

ans = 0.0
p = model.predict(X_test)
yy = np.argmax(p, axis=1)
yyy = np.argmax(Y_test, axis=1)

for i in range(len(Y_test)):
    pred = p[i].argsort()[-3:][::-1]
    if np.argmax(Y_test[i]) in pred:
        ans += 1.0
print("Top 3 prediction precision: %.2f"%(100.0*ans / (len(Y_test) + 0.0)))


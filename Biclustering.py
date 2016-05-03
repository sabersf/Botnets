__author__ = 'Saber Shokat Fadaee'

#import libraries
from gensim import corpora, models, similarities
from gensim.models import ldamodel
from gensim.models import HdpModel
import tsne
import numpy as np
from numpy import *
import os
import logging
from bokeh.server.serverbb import prune
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from itertools import chain
import csv
import matplotlib as ml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

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



#Read input files

storage = {}
i = 1.0
EID_set = set()
botnet_set = set()
event_set = set()

file1 = open('EID.txt')
for line in file1:
        EID = line.strip()
        EID_set.add(EID)
file1.close()

file1 = open("events_date.txt")
for line in file1:
        event_date = line.strip()
        event_set.add(event_date)
file1.close()

file1= open("botnets.txt")
for line in file1:
    botnet = line.strip()
    botnet_set.add(botnet)
file1.close()

EID_set = sorted(EID_set)
botnet_set = sorted(botnet_set)
event_set = sorted(event_set)


#Read the matrix and adjust the errors if
#  there are any. (i.e: NANs and infs)

count = np.loadtxt("count.txt")
where_are_NaNs = isnan(count)
count[where_are_NaNs] = 0

where_are_infs = isinf(count)
count[where_are_infs] = 0


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
		ind = int(line['index']) 
		eid = line['entity_id_hash']
		sec = line['industry_sector']
		cnt = line['employee_count_range']
		sector[eid] = sec
		count_range[eid] = cnt


#Assiging the colors		
for v in new_model.vocab:
	col.append(sec_to_col(sector[v]))

for v in new_model.vocab:
	col_size.append(size_to_col(count_range[v]))

# Check for each botnet how many times it attacked unique entities
# and the intensity of the attacks
attack_num = [0 for i in range(207)]
for i in range(207):
    c = np.where(count[i] > 0)
    attack_num[i] = int(count[i,c].shape[1])
intensity = [0 for i in range(207)]
for i in range(207):
    intensity[i] = sum(count[i])

#Making the bot-bot and ent-ent matrix by
# multiplying bot-ent matrix on its transpose matrix
count_t = count.transpose()
bot_mat = np.dot(count,count.T)
ent_mat = np.dot(count.T, count)

#Heatmap of bot-bot matrix
plt.pcolor(bot_mat)
plt.title('Entities heatmap')
plt.axis([0, 5917, 0, 5917])
plt.colorbar()
plt.show()

#Plotting the attack histogram
plt.hist(attack_num)
plt.title("Number of attacks histogram")
#plt.xlabel("Botnet #")
#plt.ylabel("The number of entities it attacks")
plt.show()

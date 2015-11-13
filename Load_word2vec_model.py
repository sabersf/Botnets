__author__ = 'Saber Shokat Fadaee'

from gensim import corpora, models, similarities
from gensim.models import ldamodel
from gensim.models import HdpModel
import tsne
import numpy as np
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
	
#Reading from the gensim dictionary
dictionary = corpora.Dictionary.load('Repwords.dict')
corpus = corpora.MmCorpus('Rep.mm')

#Creating a TF/IDF model
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

#Trains the LSI model
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=207)
corpus_lsi = lsi[corpus_tfidf]
lsi.print_topics(1)

#Trains the HDP model
hdp = HdpModel(corpus, id2word=dictionary)

# Trains the LDA models.    
lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=3, chunksize=10000, update_every=1, passes=1)

# Prints the topics in LDA
for top in lda.print_topics():
    print top
print



#Initializing an empty array to store all the word2vec vectors
arr = np.empty((100L,), float)

#Adding each vector to that array
for v in new_model.vocab:
	arr = np.vstack((arr, new_model[v]))
	
#Dimension reduction using the TSNE algorithm	
Y = tsne.tsne(arr, no_dims = 2)
Y_3 = tsne.tsne(arr, no_dims = 3)


#	arr = np.append(arr, new_model[v], axis=0)
#	arr = np.append(arr,new_model[v])
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
	
#2D	plot
plt.scatter(Y[:,0], Y[:,1], c = col_size)
plt.show()		
	
#3D plot
col = c.tolist()

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

fig = plt.figure()
ax = fig.gca(projection='3d')

cntr = 0 
for v in new_model.vocab:
	ax.scatter(Y_3[cntr,0], Y_3[cntr,1],Y_3[cntr,2], c = col_size[cntr], marker='o')
	cntr = cntr + 1

ax.scatter(Y_3[:,0], Y_3[:,1],Y_3[:,2], c = ['r' , 'b'], marker='^')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

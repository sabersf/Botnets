
# coding: utf-8

# In[6]:

import numpy as np
import tflearn
import tensorflow as tf
import csv
import random
from sklearn.cross_validation import train_test_split
from tflearn.helpers import regularizer
import math
import tflearn.variables as va


# In[2]:

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


# In[3]:

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


# In[91]:

count_sec = [0]* 23
input1 = []
output = []
#18 is the number of sec that has any elements
#30 is the min that all sectors can get
while sum(count_sec) < 23*30:
    i = random.randint(0, len(EID_set) - 1)
    if count_sec[sec_to_num(sector[EID_set[i]])] >= 30:
        continue
    else:
        input1.append(count[:,i])
        sec_to_vec = [0]*23
        sec_to_vec[sec_to_num(sector[EID_set[i]])] = 1
        count_sec[sec_to_num(sector[EID_set[i]])] += 1
        output.append(sec_to_vec)
print(len(input1), len(input1[0]))
print(len(output), len(output[0]))
inp = np.array(input1)
out = np.array(output)
X_train, X_test, Y_train, Y_test = train_test_split(inp, out, test_size=0.05, random_state=42)


# In[92]:

# Define a dnn using Tensorflow
with tf.Graph().as_default() as session:

    # Model variables
    X = tf.placeholder("float", [None, len(X_train[0])])
    Y = tf.placeholder("float", [None, len(Y_train[0])])

    # Multilayer perceptron
    def dnn(x):
        with tf.variable_scope('Layer1'):
            # Creating variable using TFLearn
            W1 = va.variable(name='W', shape=[len(X_train[0]), 256],
                             initializer='uniform_scaling',
                             regularizer='L2')
            b1 = va.variable(name='b', shape=[256])
            x = tf.nn.tanh(tf.add(tf.matmul(x, W1), b1))

        with tf.variable_scope('Layer2'):
            W2 = va.variable(name='W', shape=[256, 64],
                             initializer='uniform_scaling',
                             regularizer='L2')
            b2 = va.variable(name='b', shape=[64])
            x = tf.nn.tanh(tf.add(tf.matmul(x, W2), b2))

        with tf.variable_scope('Layer3'):
            W3 = va.variable(name='W', shape=[64, 64],
                             initializer='uniform_scaling',regularizer='L2')
            b3 = va.variable(name='b', shape=[64])
            x = tf.add(tf.matmul(x, W3), b3)

        with tf.variable_scope('Layer4'):
            W4 = va.variable(name='W', shape=[64, len(Y_train[0])],
                             initializer='uniform_scaling',regularizer='L2')
            b4 = va.variable(name='b', shape=[len(Y_train[0])])
            x = tf.add(tf.matmul(x, W4), b4)
            
        return x,W4,b4

    net, W4,b4 = dnn(X)
    const = .01
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, Y) + const*tf.nn.l2_loss(W4) +const*tf.nn.l2_loss(b4) )
    #optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1)), tf.float32), name='acc')


    trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer, metric=accuracy, batch_size=256)

    trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=3, tensorboard_dir='/tmp/tflearn_logs/')

    trainer.fit({X: X_train, Y: Y_train}, val_feed_dicts={X: X_test, Y: Y_test},
                n_epoch=100, show_metric=True, run_id='Variables_example')
    print("Accuracy on the test test: %.2f"% (100. * trainer.session.run(accuracy, feed_dict={X:X_test, Y:Y_test})))
    pred = trainer.session.run(tf.argmax(Y, 1),feed_dict={X:X_test, Y:Y_test})


# In[93]:

total = 0
correct = 0
count_pred = np.zeros((23))

for i in range(len(pred)):
    total += 1
    count_pred[pred[i]] += 1
    if np.argmax(pred[i]) == np.argmax(Y_test[i]):
        correct += 1
print(total, correct, np.var(pred))
for i in range(23):
    print("Sector: %s number of predictions: %.0f"%(num_to_sec(i), count_pred[i]))


# In[ ]:




# In[ ]:




# In[35]:

#Implementation with TFLEARN
tf.reset_default_graph()
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.8)
tnorm = tflearn.initializations.uniform(minval=-1.0, maxval=1.0)

# Building DNN
nn = tflearn.input_data(shape=[None, len(X_train[0])])
Input = nn
nn = tflearn.fully_connected(nn, 256, activation='elu', weights_init=tnorm, name = "layer_1")
nn = tflearn.dropout(nn, 0.5)
nn = tflearn.fully_connected(nn, 512, activation='elu', weights_init=tnorm, name = "layer_2")
nn = tflearn.dropout(nn, 0.5)
nn = tflearn.fully_connected(nn, 512, activation='elu', weights_init=tnorm, name = "layer_3")
nn = tflearn.dropout(nn, 0.5)
nn = tflearn.fully_connected(nn, 256, activation='elu', weights_init=tnorm, name = "layer_4")
nn = tflearn.dropout(nn, 0.5)
nn = tflearn.fully_connected(nn, 64, activation='elu', weights_init=tnorm, name = "layer_5")
nn = tflearn.dropout(nn, 0.5)
Hidden_state = nn
nn = tflearn.fully_connected(nn, len(Y_train[0]), activation='elu', weights_init=tnorm, name = "layer_6")
Output = nn    
#custom_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#    out_layer, tf_train_labels) +
#    0.01*tf.nn.l2_loss(hidden_weights) +
#    0.01*tf.nn.l2_loss(hidden_biases) +
#    0.01*tf.nn.l2_loss(out_weights) +
#    0.01*tf.nn.l2_loss(out_biases))


# Regression, with mean square error
net = tflearn.regression(nn, optimizer='SGD' , learning_rate=0.001, loss ='categorical_crossentropy', metric=None)

# Training the auto encoder
model = tflearn.DNN(net, tensorboard_verbose=3)
model.fit( X_train,  Y_train, n_epoch=20, validation_set=0.1, run_id="bitsight_nn", batch_size=128)


# In[36]:

pred = model.predict(X_test)


# In[65]:

total = 0
correct = 0
count_pred = np.zeros((23))

for i in range(len(pred)):
    total += 1
    count_pred[np.argmax(pred[i])] += 1
    if np.argmax(pred[i]) == np.argmax(Y_test[i]):
        correct += 1
print(total, correct)
for i in range(23):
    print("Sector: %s number of predictions: %.0f"%(num_to_sec(i), count_pred[i]))


# In[ ]:




# In[61]:

256, 128 , 64, const = 0.01 , result = Accuracy on the test test: 7.25, var = 45.52
256, 128 , 64, const = 0.10 , result = Accuracy on the test test: 13.04
256, 128 , 64, const = 1.00 , result = Accuracy on the test test: 
256, 128 , 64, const = 100.0 , result = Accuracy on the test test: 7.25, var = 45.52



# In[ ]:




# In[ ]:




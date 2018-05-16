#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 11:47:49 2018

@author: apple
"""

import numpy as np
import pandas as pd

data = pd.read_csv('creditcard.csv')
shuffled_data = data.sample(frac = 1)

one_hot_data = pd.get_dummies(shuffled_data, columns = ['Class'])

normalized_data = (one_hot_data - one_hot_data.min()) / (one_hot_data.max() - one_hot_data.min())

X = normalized_data.drop(['Class_0','Class_1'], axis = 1)
y = normalized_data[['Class_0','Class_1']]

X_ar, y_ar = np.asarray(X.values, dtype = 'float32'), np.asarray(y.values, dtype = 'float32')
train_size = int(0.8 * len(X_ar))

(X_train_raw, y_train_raw) = (X_ar[:train_size], y_ar[:train_size])
(X_test_raw, y_test_raw) = (X_ar[train_size:], y_ar[train_size:])

count_legit, count_fraud = np.unique(data['Class'], return_counts = True)[1]

fraud_ratio = float(count_fraud) / (count_legit + count_fraud)   #0.00172748563062

#print 'Percent of fraudulent transactions: ', fraud_ratio

weighting = 1 / fraud_ratio
y_train_raw[:, 1] = y_train_raw[:, 1] * weighting

import tensorflow as tf

#number of columns
input_dimensions = X_ar.shape[1]

output_dimensions = y_ar.shape[1]

num_layer_1 = 100

num_layer_2 = 150

X_train_node = tf.placeholder(tf.float32, [None, input_dimensions], name = 'X_train')

y_train_node = tf.placeholder(tf.float32,[None, output_dimensions], name = 'y_train')

X_test_node = tf.constant(X_test_raw, name = 'X_test')

y_test_node = tf.constant(y_test_raw, name = 'y_test')

##need weights of the layers and bias of the layer
##values will be changing as we want to optimize the loss
# a matrix of zeros
weight_1_node = tf.Variable(tf.zeros([input_dimensions, num_layer_1]), name ='weight_1')

biases_1_node = tf.Variable(tf.zeros([num_layer_1]), name = 'biases_1')


weight_2_node = tf.Variable(tf.zeros([num_layer_1, num_layer_2]), name ='weight_2')

biases_2_node = tf.Variable(tf.zeros([num_layer_2]), name = 'biases_2')


weight_3_node = tf.Variable(tf.zeros([num_layer_2, output_dimensions]), name ='weight_3')

biases_3_node = tf.Variable(tf.zeros([output_dimensions]), name = 'biases_3')


#print input_dimensions  #30
#print output_dimensions  #2
#print y_train_raw[:,1]
###define the neural network
def network (input_tensor):
    layer1 = tf.nn.sigmoid(tf.matmul(input_tensor, weight_1_node) + biases_1_node)
  #prevent the model from coming lazy, use dropout
    layer2 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(layer1, weight_2_node) + biases_2_node), 0.85)
    layer3 = tf.nn.softmax(tf.matmul(layer2, weight_3_node) + biases_3_node)
    return layer3
    

y_train_pred = network(X_train_node)
y_test_pred = network(X_test_node)

cross_entropy = tf.losses.softmax_cross_entropy(y_train_node, y_train_pred)

optimizer = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

def calculate_accuracy(actual, predicted):
    actual = np.argmax(actual, 1)   #get indices 
    predicted = np.argmax(predicted, 1)  #get indices
    return 100 * np.sum(np.equal(predicted, actual))/predicted.shape[0]

num_epochs = 100  #the higher, more time to train, the better the model(higher accuracy)

import time

with tf.Session() as session:   ## this 'session' variable to be of tf.Session()
    tf.global_variables_initializer().run()
    for epoch in range(num_epochs):
        
        start_time = time.time()
        
        _, cross_entropy_score = session.run([optimizer, cross_entropy],
                                             feed_dict = {X_train_node: X_train_raw,
                                                          y_train_node: y_train_raw})
        if epoch % 10 == 0:
            
            timer = time.time() - start_time
            ##{0: .4f} means printing 4 decimal places
            print ('Epoch: {}'.format(epoch),'Current loss: {0:.4f}'.format(cross_entropy_score), 
                   'Elapsed time: {0: .2f} seconds'.format(timer))
            
            
            final_y_test = y_test_node.eval()
            final_y_test_pred = y_test_pred.eval() 
            final_accuracy = calculate_accuracy(final_y_test, final_y_test_pred)
            print "Current accuracy: {0: .2f}%".format(final_accuracy)
            
    final_y_test = y_test_node.eval()
    final_y_test_pred = y_test_pred.eval() 
    final_accuracy = calculate_accuracy(final_y_test, final_y_test_pred)
    print "Current accuracy: {0: .2f}%".format(final_accuracy)   
            

final_fraud_y_test = final_y_test[final_y_test[:,1] == 1]
final_fraud_y_test_pred = final_y_test_pred[final_y_test[:,1] == 1]
final_fraud_accuracy = calculate_accuracy(final_fraud_y_test, final_fraud_y_test_pred)
print 'Final fraud specific accuracy: {0:.2f}'.format(final_fraud_accuracy)  # 82.00%

   





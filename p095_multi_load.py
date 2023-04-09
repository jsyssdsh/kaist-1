#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


tf.set_random_seed(777)

xy=np.loadtxt('p095_train.txt', np.float32)
# print(xy)

x_data=xy[:,:-1]
# print(x_data)

y_data=xy[:,-1:]
# print(y_data)


W = tf.Variable(tf.random_uniform([3,1], -1., 1.))
 
hypothesis = tf.matmul(x_data, W)  
cost = tf.reduce_mean(tf.square(hypothesis-y_data))
 
a = tf.Variable(0.1) #0.05로하면 해결됨 ***
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
 
init = tf.global_variables_initializer()
 
sess = tf.Session()
sess.run(init)
 
for step in range(2001):
    sess.run(train)
    if step%20 == 0:
        print(step, sess.run(cost), sess.run(W))
         




              
              
              

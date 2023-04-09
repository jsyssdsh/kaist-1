#-*- coding: utf-8 -*-

import tensorflow as tf
# from numpy.lib.function_base import gradient
tf.set_random_seed(777)

x_data = [1,2,3,5]
y_data = [1,2,3,5]

W = tf.Variable(tf.random_normal([1], -1., 1.))

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

hypothesis = X * W  

cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate=0.2
gradient = tf.reduce_mean((W*X-Y)*X)
descent= W-learning_rate*gradient #New W
update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost,feed_dict={X: x_data, Y: y_data}), sess.run(W))
        
        

              
              
              

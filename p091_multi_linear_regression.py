#-*- coding: utf-8 -*-

import tensorflow as tf

tf.set_random_seed(777)

x1_data = [1,0,3,0,5] #발산되는경우 있음 *****
x2_data = [0,2,0,4,0] #발산되는경우 있음 *****
y_data  = [1,2,3,4,5]

w1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
w2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

x1=tf.placeholder(tf.float32)
x2=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)


hypothesis = w1*x1_data+w2*x2_data + b  

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1) #0.05로하면 해결됨 ***
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

feed={x1:x1_data, x2:x2_data, y:y_data}

for step in range(2001):
    sess.run(train, feed)
    if step%20 == 0:
        print(step, sess.run(cost), sess.run(w1),sess.run(w2), sess.run(b))
        
# print(sess.run(hypothesis, {x1:7, x2:0}))
print(sess.run(hypothesis, {x1:0, x2:6}))    



              
              
              

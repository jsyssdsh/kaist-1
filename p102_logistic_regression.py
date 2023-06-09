import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

xy=np.loadtxt("p102_train.txt", dtype=np.float32) 
#print(xy)
# .csv�̸� delimiter=',' �߰�

x_data=xy[:,:-1]
#print(x_data) slicing
y_data=xy[:,-1:]
#print(y_data) slicing
 
X=tf.placeholder(tf.float32,[None,3])
Y=tf.placeholder(tf.float32,[None,1])  
  
W=tf.Variable(tf.random_uniform([3,1],-1.0, 1.0))
   
# h=tf.matmul(X,W)
# hypothesis=tf.div(1.,1.+tf.exp(-h))
hypothesis=tf.sigmoid(tf.matmul(X,W))
 
cost=-tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
   
a=tf.Variable(0.1)
optimizer=tf.train.GradientDescentOptimizer(a)
train=optimizer.minimize(cost)
   
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
   
for step in range(2001):
    sess.run(train,feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step, sess.run(cost,feed_dict={X:x_data, Y:y_data}), sess.run(W))

p1=sess.run(hypothesis,feed_dict={X:[[1,2,2]]})         
print(p1>0.5,p1)
p2=sess.run(hypothesis,feed_dict={X:[[1,5,5]]})
print(p2>0.5,p2)
p3=sess.run(hypothesis,feed_dict={X:[[1,4,3],[1,3,5]]})
print(p3>0.5,p3)

import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

xy=np.loadtxt("p110_train.txt", dtype=np.float32)
#print(xy)

# #x0 x1 x2 y[A   B   C]
# 1   2   1   0   0   1
# 1   3   2   0   0   1
# 1   3   4   0   0   1
# 1   5   5   0   1   0
# 1   7   5   0   1   0
# 1   2   5   0   1   0
# 1   6   6   1   0   0
# 1   7   7   1   0   0

x_data=xy[:,:3]
#print(x_data)
y_data=xy[:,3:]
#print(y_data)
 
X=tf.placeholder(tf.float32,[None,3])
Y=tf.placeholder(tf.float32,[None,3])
 
W=tf.Variable(tf.random_uniform([3,3],-1.0, 1.0))
  
h=tf.matmul(X,W)
hypothesis=tf.nn.softmax(h)
  
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
  
a=tf.Variable(0.01)
optimizer=tf.train.GradientDescentOptimizer(a)
train=optimizer.minimize(cost)

predicted=tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy=tf.reduce_mean(tf.cast(predicted, dtype=tf.float32))

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
  
for step in range(10001):
    sess.run(train,feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step, sess.run(cost,feed_dict={X:x_data, Y:y_data}), sess.run(W))
        
print()      
# a=sess.run(hypothesis,{X:[[1,11,7]]}) 
# print(a, sess.run(tf.argmax(a, 1)))
# b=sess.run(hypothesis,feed_dict={X:[[1,3,4]]}) 
# print(b, sess.run(tf.argmax(b, 1)))
# c=sess.run(hypothesis,feed_dict={X:[[1,1,0]]}) 
# print(c, sess.run(tf.argmax(c, 1)))
# all=sess.run(hypothesis,feed_dict={X:[[1,11,7],[1,3,4],[1,1,0]]}) 
# print(all, sess.run(tf.argmax(all, 1)))

print(sess.run(accuracy,feed_dict={X:x_data, Y:y_data}))



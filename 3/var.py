# -*- coding: utf-8 -*- 
import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 3])
#print(X)
# x : placeholder

x_data = [[1, 2, 3], [4, 5, 6]]
# init with normal distribution
#W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))

# can change like
W = tf.Variable([[0.1, 0.1],[0.2, 0.2], [0.3, 0.3]])

expr = tf.matmul(X,W) + b


sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("=== x_data ===")
print(x_data)
print("=== W ===")
print(sess.run(W))
print("=== b ===")
print(sess.run(b))
print("=== expr ===")
print(sess.run(expr, feed_dict={X:x_data})) #what?? 


sess.close()

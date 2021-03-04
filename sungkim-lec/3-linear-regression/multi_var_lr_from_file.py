# -*- coding: utf-8 -*- 
import tensorflow as tf
import numpy as np

# multi Variable linear regression 구현
# 지저분하니 매트릭스 형태로 바꾸자
# TODO 다차원 형태에 익숙해지는게 포인트. 좀더 연습해보자

xy = np.loadtxt('data.csv', delimiter=',', dtype=np.float32)
# slicing 참고
x_data = xy[:, 0:-1] # row:all, col:0~2 (-1:except last one)
y_data = xy[:, [-1]] # 그냥 -1만 하면 한 row에 다담김

# placeholders for a tensor that will be always fed.
# NOTE shape 순서는 역순으로 읽는게 보기 편한듯?
X = tf.placeholder(tf.float32, shape=[None, 3]) # None: 몇개 들어와도 상관x
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# XXX after reduce, cost's dimension??
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 2e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global Variables in the graph.
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], \
            feed_dict={X: x_data, Y:y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction: \n", hy_val)

# Ask my score
print("Your score will be ", sess.run(
    hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ", sess.run(hypothesis,
                                        feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
'''
# 각각의 shape 를 확인해보자
print(sess.run(tf.shape(x_data)))
print(sess.run(tf.shape(W)))
print(sess.run(tf.shape(hypothesis), feed_dict={X:x_data}))
print(sess.run(tf.shape(cost)))
'''

# -*- coding: utf-8 -*- 
# NOTE plot 은 gui에서만 된다... putty는 못해요 ㅠㅠ
import tensorflow as tf
import matplotlib.pyplot as plt

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
#W = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
# reduce ?? : 차원 축소한다는 의미 ? map-reduce 의 reduce??
# 같은 말이겠구나?
# TODO reduce 에 axis를 넣어서 차원을 어떻게 감소시키는지 실험해보자
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Minimize: gradient descent using derivative:
learning_rate = 0.1
gradient = tf.reduce_mean((W*X-Y)*X) #기울기, 편미분으로 구함 
descent = W - learning_rate * gradient
update = W.assign(descent)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())
# Variables for plotting cost function
for step in range(21):
    sess.run(update, feed_dict={X:x_data, Y:y_data}) # update 를 실행시킴, X,Y
                                                     # 주면서...
    print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

# 기울기(w)가 1인걸 맞추는 문제.. 는 아니고
# 이제 cost를 최소로 만들어주는 W값을 구해야지!

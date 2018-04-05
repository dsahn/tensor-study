# -*- coding: utf-8 -*- 
# NOTE plot 은 gui에서만 된다... putty는 못해요 ㅠㅠ
import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
# reduce ?? : 차원 축소한다는 의미 ? map-reduce 의 reduce??
# 같은 말이겠구나?
# TODO reduce 에 axis를 넣어서 차원을 어떻게 감소시키는지 실험해보자
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())
# Variables for plotting cost function
W_val = []
cost_val = []
for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost,W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

# Show the cost function
plt.plot(W_val, cost_val)
plt.show()

# 기울기(w)가 1인걸 맞추는 문제.. 는 아니고
# 이제 cost를 최소로 만들어주는 W값을 구해야지!

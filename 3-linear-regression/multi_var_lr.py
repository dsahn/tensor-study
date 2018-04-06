# -*- coding: utf-8 -*- 
import tensorflow as tf

# multi Variable linear regression 구현
# 지저분하니 매트릭스 형태로 바꾸자
# TODO 다차원 형태에 익숙해지는게 포인트. 좀더 연습해보자

'''
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]
'''
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

# placeholders for a tensor that will be always fed.
# NOTE shape 순서는 역순으로 읽는게 보기 편한듯?
X = tf.placeholder(tf.float32, shape=[None, 3]) # None: 몇개 들어와도 상관x
Y = tf.placeholder(tf.float32, shape=[None, 1])
'''
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
''' 

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
'''
#Y = tf.placeholder(tf.float32)
w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')
'''

hypothesis = tf.matmul(X, W) + b
'''
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
'''
"""
## 중간 결과 찍어보기
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(hypothesis, feed_dict={X:[[1,2,3], [4,5,6]]}))
# result : n * 1 matrix
"""
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# XXX after reduce, cost's dimension??
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global Variables in the graph.
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], \
            feed_dict={X: x_data, Y:y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val)
        print("Prediction: ")
        print(hy_val)

print(sess.run(tf.shape(x_data)))
print(sess.run(tf.shape(W)))
print(sess.run(tf.shape(hypothesis), feed_dict={X:x_data}))
print(sess.run(tf.shape(cost)))
'''
'''

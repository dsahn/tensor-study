# -*- coding: utf-8 -*- 
import tensorflow as tf

# linear regression 실습 
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# uniform dist. 로 값을 랜덤하게 넣어주자
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# placeholoder
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# 상관관계가 선형관계라고 가정하고 w,b를 찾아내는 문제
# W: 가중치(weight), b: bias
hypo = W * X + b

# loss function
# 손실값 : 예측값과 얼마나 차이가 나는가? 를 나타내는값
# 손실값이 작을수록 모델을 잘 설명하는것
# 학습 : 손실값을 최소로 만들자!
# 차이의 제곱의 평균
# 이차식 형태로 나올듯?
cost = tf.reduce_mean(tf.square(hypo - Y))

# 경사 하강법(gredient descent)
# 최적화 함수 : w,b값을 변경해가면서 손실을 최소화하는 함수
# 기울기가 낮은쪽으로 계속 이동시키면서 최적의 해를 찾음
# learning_rate : 학습률. 얼마나 급격하게 이동시킬지. hyperparameter
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train_op = optimizer.minimize(cost) # cost를 최소화 시키자!

with tf.Session() as sess: #소멸자 부르는 기능 RAII같은거 
    sess.run(tf.global_variables_initializer())

    for step in range(500):
        _, cost_val = sess.run([train_op, cost], \
                feed_dict = {X: x_data, Y: y_data})
        print(step, cost_val, sess.run(W), sess.run(b))

    print("X:5, Y:", sess.run(hypo, feed_dict={X:5}))
    print("X:2.5, Y:", sess.run(hypo, feed_dict={X:2.5}))

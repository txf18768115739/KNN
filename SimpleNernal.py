import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.100], dtype=tf.float32)
b = tf.Variable([-.100], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
print(loss)
# optimizer,定义一个优化器，使用梯度下降优化方法
optimizer = tf.train.GradientDescentOptimizer(0.01)
#优化器最小化loss
train = optimizer.minimize(loss)

# training data
x_train = [1,2,3,4,5,6]
y_train = [0,-1,-2,-3,-4,-5]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong

#迭代一千次后，参数已经训练好
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

#送入训练好的参数，求出loss
# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


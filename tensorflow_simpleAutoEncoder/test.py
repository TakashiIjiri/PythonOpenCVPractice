import numpy as np
import tensorflow as tf

#詳解ディープラーニング p. 094 より

# modeling
# y = w^t x + b
w = tf.Variable(tf.zeros([2,1]))
b = tf.Variable(tf.zeros([1]  ))
x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.nn.sigmoid (tf.matmul(x,w) + b)

cross_entropy = -tf.reduce_sum(t*tf.log(y) + (1-t) * tf.log(1-y))
train_step    =  tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)



#dataset
X = np.array([[0,0], [1,0], [0,1], [1,1]])
Y = np.array([[0], [1], [1], [1]])


#training
init = tf.global_variables_initializer()
sess = tf.Session() #これに初期化/step/変数取得などの機能がある
sess.run(init)

for epock in range(200) :
    sess.run(train_step, feed_dict={x:X, t:Y})

#results
classified = correct_prediction.eval(session=sess, feed_dict={x:X, t:Y})
print(classified)

prob = y.eval(session=sess, feed_dict={x:X})
print(prob)

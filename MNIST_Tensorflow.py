import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 28,28,1])
Y = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool)
'''
b = tf.Variable(tf.random_normal([10]))
keep_prob = tf.placeholder(tf.float32)
'''


# layer 1
L1 = tf.layers.conv2d(X, 32, [3, 3], activation=tf.nn.relu, padding='SAME')
L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2], padding='SAME')
L1 = tf.layers.dropout(L1, 0.7, is_training)
'''
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.1))
L1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding="SAME", use_cudnn_on_gpu=False))
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 
L1 = tf.nn.dropout(L1, keep_prob)
'''

# layer 2
L2 = tf.layers.conv2d(L1, 64, [3, 3])
L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
L2 = tf.layers.dropout(L2, 0.7, is_training)
'''
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.1))
L2 = tf.nn.relu(tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME', use_cudnn_on_gpu=False))
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob)
'''

#layer 3
L3 = tf.contrib.layers.flatten(L2)
L3 = tf.layers.dense(L3, 256, activation=tf.nn.relu)
L3 = tf.layers.dropout(L2, 0.5, is_training)

# fully connected
'''
FL = tf.reshape(L2, [-1, 7 * 7 * 64])
FW = tf.get_variable("FW", shape=[7 * 7 * 64, 10], initializer = tf.contrib.layers.xavier_initializer())
hypothesis = tf.matmul(FL, FW) + b
'''

FL = tf.layers.dense(L3, 10, activation=None)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=FL, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

# train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

training_epochs = 15
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

print('Learning started')
for epoch in range(training_epochs):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        feed_dict = {X: batch_xs, Y: batch_ys, is_training: True}
        _, cost_val = sess.run([optimizer, cost], feed_dict=feed_dict)
        total_cost += cost_val
    print('Epoch:','%04d'%(epoch + 1),'cost =','{:.5f}'.format(total_cost / total_batch))
print('Learning Finished!')
 
# Test
correct_prediction = tf.equal(tf.argmax(FL, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:',sess.run(accuracy,feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1), Y:mnist.test.labels, is_training: False}))
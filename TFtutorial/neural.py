import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

theta1 = tf.Variable(tf.truncated_normal([784,100]),tf.float32)
b1= tf.Variable(tf.truncated_normal([100]),tf.float32)

theta2 = tf.Variable(tf.truncated_normal([100,10]),tf.float32)
b2 = tf.Variable(tf.zeros([10]),tf.float32)

x = tf.placeholder(tf.float32, shape=[None,784])
y = tf.placeholder(tf.float32, shape = [None,10])

learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

plt.imshow(np.reshape(mnist.test.images[0],(28,28)),cmap=cm.binary)
plt.show()

l1 = tf.nn.relu(tf.matmul(x,theta1) +b1)
l2 = tf.matmul(l1, theta2)+b2
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y, logits =l2))
trainop = tf.train.AdamOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()
print("내려감")
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(mnist.train.num_examples / batch_size)
        print("epoch 진입함")
        # Fit the line.
        for step in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data

            sess.run(trainop, feed_dict={x: batch_xs, y: batch_ys})

            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch + 1), "cost=" ,avg_cost)
    print ("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(l2, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None,10])

w1 = tf.Variable(tf.truncated_normal([784,100]),tf.float32)
b1 = tf.Variable(tf.zeros([100]),tf.float32)

w2 = tf.Variable(tf.truncated_normal([100,10]), tf.float32)
b2 = tf.Variable(tf.zeros([10]),tf.float32)

layer1 = tf.nn.relu(tf.matmul(x ,w1)+b1)
layer2 = tf.matmul(layer1, w2)+b2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y , logits=layer2))

trainop = tf.train.AdamOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    batch_size = 150
    runbatches = int(mnist.train.num_examples / batch_size)
    for epoch in range(25):
        avg_cost = 0.0
        for runs in range(runbatches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, cost_in_batch = sess.run([trainop, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += cost_in_batch
        print('epoch ', epoch ,'th cost: ', avg_cost)
    correct_prediction = tf.equal(tf.argmax(layer2, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

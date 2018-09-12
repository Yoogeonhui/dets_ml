import tensorflow as tf

# x: [1,2,3,4,1,2,3,4] y: [4,3,2,1,2,1,0,-1] z;[1,1,1,1,0,0,0,0]

a=tf.Variable(0.0, tf.float32)
b= tf.Variable(0.0, tf.float32)
c=tf.Variable(0.0,tf.float32)

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
z= tf.placeholder(tf.float32)

hypothesis = tf.sigmoid(a*x+b*y+c)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=z, logits=hypothesis))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init= tf.global_variables_initializer()
sess= tf.Session()
sess.run(init)
for i in range(15000):
    sess.run(train, feed_dict={x: [1,2,3,4,1,2,3,4], y: [4,3,2,1,2,1,0,-1], z:[1,1,1,1,0,0,0,0]})
print("a ", sess.run(a)," b ",sess.run(b),"c ",sess.run(c))
import tensorflow as tf

# x: [0,1,2,3] y: [-1,-2,-3,-4]

a=tf.Variable(0.0, tf.float32)
b= tf.Variable(0.0, tf.float32)
c=tf.Variable(0.0,tf.float32)

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

hypothesis = a*x*x+b*x+c

loss = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init= tf.global_variables_initializer()
sess= tf.Session()
sess.run(init)
for i in range(1500):
    sess.run(train, feed_dict={x:[0,1,2,3], y:[-1,-2,-3,-4]})
print("a ", sess.run(a)," b ",sess.run(b), "c" ,sess.run(c))
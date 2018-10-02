import tensorflow as tf
tf.reset_default_graph()
a = tf.Variable(tf.constant(2))
init_op = tf.global_variables_initializer()
modify_op = a.assign(tf.constant(5))

sess = tf.Session()
sess.run(init_op)
print(sess.run(a))
sess.run(modify_op)
print(sess.run(a))
sess.run(init_op)
print(sess.run(a))
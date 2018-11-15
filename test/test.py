import tensorflow as tf

x = tf.constant([[1., 2., 3.], [4., 5., 6.]])

with tf.Session() as sess:
    x = sess.run(x)

    mean1 = sess.run(tf.reduce_mean(x))
    mean2 = sess.run(tf.reduce_mean(x, 0))
    mean3 = sess.run(tf.reduce_mean(x, 1))

    print(x)
    print()
    print(mean1)
    print()
    print(mean2)
    print()
    print(mean3)
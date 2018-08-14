import tensorflow as tf

# with tf.Session() as sess:
#     img = tf.gfile.FastGFile('D:/task/bicubic/HR/01.jpg', 'r').read()
#     img = tf.read_file('HR/01.jpg')
#     img = tf.image.decode_jpeg(img, channels=3)
#     h = tf.shape(img)[0]
#     print(sess.run(tf.shape(img)))

a = tf.constant(1)
b = tf.Variable(tf.truncated_normal((2, 2), dtype=tf.float32, stddev=0.1))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(b))
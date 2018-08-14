import os
from PIL import Image

input_dir = 'LR/'
save_dir = 'HR/'


# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     input_names = os.listdir(input_dir)
#     for i in range(len(input_names)):
#         img = tf.read_file(input_dir+input_names[i])
#         img = tf.image.decode_jpeg(img, channels=3)
#
#         h = tf.shape(img)[0]
#         w = tf.shape(img)[1]
#         img = tf.reshape(img, [1, h, w, 3])
#         img = tf.image.resize_bicubic(img, size=[h//2, w//2])
#
#         img = tf.reshape(img, [h//2, w//2, 3])
#         img = tf.image.convert_image_dtype(img, dtype=tf.uint8)
#
#         img = tf.cast(img, tf.float32)
#         img = img/255.0
#
#         img = tf.image.encode_jpeg(img)
#
#
#
#         file = tf.gfile.GFile(save_dir+input_names[i], 'w')
#         #sess.run(img)
#         file.write(img.eval())
#        # sess.run(file)
#         file.close()

nameList = os.listdir(input_dir)

for i in range(len(nameList)):
    im = Image.open(input_dir+nameList[i])
    w, h = im.size
    im_new = im.resize((w*2, h*2), Image.BICUBIC)
    im_new.save(save_dir+nameList[i])

# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import numpy as np


def get_data_list(inpDir):
    pathList = []
    labelList = []
    nameList = []
    for im_path in os.listdir(inpDir):
        nameList.append(im_path)
        im_name = im_path.split('.')[0]
        labelList.append(im_name.split('_')[1])
        pathList.append(os.path.join(inpDir, im_path))

    return pathList, labelList, nameList


def get_batch(inpDir, batch_size):
    # get images paths for all items
    inpList, labelList, _ = get_data_list(inpDir)

    # a simple judgement for consistency
    num_images = len(inpList)
    if num_images <= 0:
        raise ValueError("Not Found Images!")

    # cnvert them to tf.string
    inpList = tf.cast(inpList, tf.string)
    labelList = tf.cast(labelList, tf.string)

    # build a input queue for file names
    inpQueue = tf.train.slice_input_producer([inpList, labelList],
                                             shuffle=False,
                                             name='file_name_queue')

    labels = inpQueue[1]
    # construct ops for reading a single example
    with tf.name_scope('read_single_example'):
        inpImage = tf.read_file(inpQueue[0])

    # decode images
    with tf.name_scope('decode_single_example'):
        inpImage = tf.image.decode_image(inpImage, channels=3)

    # DO SOME AUGMENTATION: resize images with same shape

    with tf.name_scope('crop_or_pad_resize'):
        inpImage = tf.random_crop(inpImage, [256, 256, 3])

    inpBatch, labelBatch = tf.train.batch([inpImage, labels],
                                           batch_size=batch_size,
                                           num_threads=1,  # 注意这个参数
                                           capacity=8 * batch_size,
                                           dynamic_pad=False,
                                           name='batch_queue')
    # cast to tf.float32
    with tf.name_scope('cast_to_float32'):
        inpBatch = tf.cast(inpBatch, tf.float32)

    #Normalization
    with tf.name_scope('normalization'):
        inpBatch = inpBatch / 255.0

    return inpBatch, labelBatch


# %% TEST

import matplotlib.pyplot as plt

inpDIR = 'D:\\task\\TFRecords\\pictures'
batch_size = 2
inpBatch, labels = get_batch(inpDIR, batch_size=batch_size)
pathList, labelList, nameList = get_data_list(inpDIR)

with tf.Session() as sess:
    batch_index = 0
    tf.local_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop() and batch_index < 2:
            inp_batch, inp_label = sess.run([inpBatch, labels])
            for i in range(batch_size):
                imName = nameList[batch_index * batch_size + i]
                mean = sess.run(tf.reduce_mean(inp_batch[i, :, :, :]))
                var = np.var(inp_batch[i, :, :, :])
                plt.imshow(inp_batch[i, :, :, :])
                plt.title('input batch%d: image %d, label is %s' % (batch_index, i, inp_label[i]))
                plt.show()

                print('%s %s//%s' %(imName, mean, var))

            batch_index += 1
    except tf.errors.OutOfRangeError:
        print("Done!")
    finally:
        coord.request_stop()
        coord.join()


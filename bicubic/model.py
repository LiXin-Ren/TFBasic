import tensorflow as tf
import os
#import math

BATCH_SIZE = 2
LR_DIR = 'LR/'
HR_DIR = 'HR/'
TIMES = 2       #放大倍数

def get_file_name_list(file_dir):
    name_list = []
    for im in os.listdir(file_dir):
        name_list.append(file_dir + im)

    return name_list


def get_batch(LR_DIR, HR_DIR, batch_size):
    # get images paths for all items
    LRList = get_file_name_list(LR_DIR)
    HRList = get_file_name_list(HR_DIR)

    if len(LRList) <= 0 or len(HRList) <= 0:
        raise ValueError("Not Found Images!")

    # cnvert them to tf.string
    LRList = tf.cast(LRList, tf.string)
    HRList = tf.cast(HRList, tf.string)
    # build a input queue for file names
    inpQueue = tf.train.slice_input_producer([LRList, HRList],
                                             num_epochs=4,
                                             shuffle=False,
                                             capacity=32,
                                             shared_name=None,
                                             name='file_name_queue')

    # construct ops for reading a single example
    with tf.name_scope('read_single_example'):
        LRImage = tf.read_file(inpQueue[0])
        HRImage = tf.read_file(inpQueue[1])

    # decode images
    with tf.name_scope('decode_single_example'):
        LRImage = tf.image.decode_image(LRImage, channels=3)
        HRImage = tf.image.decode_image(HRImage, channels=3)

    with tf.name_scope('crop_or_pad_resize'):
        LRImage = tf.random_crop(LRImage, [256, 256, 3])
        HRImage = tf.random_crop(HRImage, [TIMES * 256, TIMES * 256, 3])

    LRBatch, HRBatch = tf.train.shuffle_batch([LRImage, HRImage],
                                      batch_size=batch_size,
                                      num_threads=4,
                                      capacity=8 * batch_size,
                                      min_after_dequeue=2 * batch_size,
                                      name='batch_queue')

    # cast to tf.float32
    with tf.name_scope('cast_to_float32'):
        LRBatch = tf.cast(LRBatch, tf.float32)
        HRBatch = tf.cast(HRBatch, tf.float32)

    # Normalization
    with tf.name_scope('normalization'):
        LRBatch = LRBatch / 255.0
        HRBatch = HRBatch / 255.0

    return LRBatch, HRBatch


def weight_variable(shape):
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape, name='b')
    return tf.Variable(initial)


def inference(LR_batch, TIMES):
    with tf.name_scope('conv'):
        conv_W = weight_variable([5, 5, 3, (TIMES**2)*3])
        conv_b = bias_variable([(TIMES**2)*3])
        HR_batch = tf.nn.conv2d(LR_batch, conv_W, strides=[1, 1, 1, 1], padding='SAME') + conv_b
    HR_batch = tf.depth_to_space(HR_batch, TIMES)
    return HR_batch


def losses(HR, pred_HR):
    with tf.name_scope('loss'):
        loss = tf.losses.mean_squared_error(HR, pred_HR)
    return loss


LR_batch, HR_batch = get_batch(LR_DIR, HR_DIR, BATCH_SIZE)
pre_HR = inference(LR_batch, TIMES)

loss = losses(HR_batch, pre_HR)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop():
            _, cost = sess.run([train_op, loss])
            print(cost)

    except tf.errors.OutOfRangeError:
        print("Done!")
    finally:
        coord.request_stop()
        coord.join()





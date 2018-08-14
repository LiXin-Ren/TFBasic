# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 19:44:49 2018

@author: zxlation
"""

# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def deepnn(inputs, keep_prob):
    """deepnn builds the graph for a deep net for classifying digits.

    Args:
        x: an input tensor with the dimensions (N_examples, 784), where 784 is the
        number of pixels in a standard MNIST image.

    Returns:
        A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
        equal to the logits of classifying the digit into one of 10 classes (the
        digits 0-9). keep_prob is a scalar placeholder for the probability of
        dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    inp_image = tf.reshape(inputs, [-1, 28, 28, 1])  # Note the format of input x: [batch, h, w, c]
    tf.summary.image('input_image', inp_image, max_outputs = 6)
    
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([5, 5, 1, 32])  # Note the format of filter: [h, w, in_c, out_c]
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(inp_image, W_conv1) + b_conv1)
    
    # 在TensorBoard中统计h_conv1的直方图
    tf.summary.histogram('conv1_histo',  h_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 32, 64])  # [h, w, in_c, out_c]
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    
    # 在TensorBoard中统计h_conv2的直方图
    tf.summary.histogram('h_conv2',  h_conv2)
    conv2_image = tf.expand_dims(h_conv2[:, :, :, 0], axis = 3)
    tf.summary.image('conv2_act', conv2_image, max_outputs = 6)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of down-sampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # 在TensorBoard中统计h_fc1的直方图
    tf.summary.histogram('h_fc1',  h_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  # Note here, no application of tf.nn.relu()
    
    # 在TensorBoard中统计h_fc1的直方图
    tf.summary.histogram('logits',  logits)
    
    return logits, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # 预备：导入数据
    with tf.name_scope("data"):
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # 第1步：定义placeholder，inputs表示我们的输入特征, labels表示对应的真实标签
    with tf.name_scope("io"):
        inputs = tf.placeholder(tf.float32, [None, 784])
        labels = tf.placeholder(tf.float32, [None, 10])
        keep_prob = tf.placeholder(tf.float32)
    
    # 第2步：根据模型构建计算图
    with tf.name_scope("model"):
        logits, keep_prob = deepnn(inputs, keep_prob)
    
    # 第3步：定义损失函数(交叉熵)
    # E = -sum_{x} p(x)log[q(x)]
    with tf.name_scope("cross-entropy"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
        cross_entropy = tf.reduce_mean(cross_entropy)
    
    # 第4步：定义优化器
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4)
        train_op = optimizer.minimize(cross_entropy)
    
    # 合并所有summary操作
    summary_op = tf.summary.merge_all()
    
    # 构建测试计算图: 这个计算图实际上是评估部分，训练阶段也可以使用
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
    ##                       所有准备工作做好之后，下面就可以开始训练了。                            ##
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
    # 以上下文管理器的方式定义会话session
    with tf.Session() as sess:
        # 定义摘要记录器(summary writer)，这一条语句会在TensorBoard中显示我们的计算图
        writer = tf.summary.FileWriter("logs/cnn/", sess.graph)
    
        sess.run(tf.global_variables_initializer())
        for i in range(FLAGS.max_steps):
            # 获得数据
            batch = mnist.train.next_batch(FLAGS.batch_size)
            
            # 生成feed_dict
            feed_dict = {inputs: batch[0], labels: batch[1], keep_prob: 0.5}
            
            # 训练一个step
            _, train_ac, summ_str = sess.run([train_op, accuracy, summary_op], feed_dict = feed_dict)
            
            if ((i + 1) % FLAGS.log_freq == 0) or ((i + 1) == FLAGS.max_steps):
                print('step%d: training accuracy %.4f' % (i + 1, train_ac))
            
            if ((i + 1) % FLAGS.summ_freq == 0) or ((i + 1) == FLAGS.max_steps):
                writer.add_summary(summ_str, i + 1)
                
        
        # excute evaluation
        feed_dict = {inputs: mnist.test.images, labels: mnist.test.labels, keep_prob: 1.0}
        test_ac = sess.run(accuracy, feed_dict = feed_dict)
        print('test accuracy %.4f' % test_ac)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = 'datasets/mnist/input_data',
                        help = 'Directory for storing input data')
    parser.add_argument('--max_steps', type = int, default = 5000,
                        help = 'The maximum number of steps to train.')
    parser.add_argument('--log_freq', type = int, default = 50,
                        help = 'print logs during how many training steps.')
    parser.add_argument('--summ_freq', type = int, default = 200,
                        help = 'writing summaries during how many training steps.')
    parser.add_argument('--batch_size', type = int, default = 50,
                        help = 'batch size.')
    
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main = main, argv = [sys.argv[0]] + unparsed)
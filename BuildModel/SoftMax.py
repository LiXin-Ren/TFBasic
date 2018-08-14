# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 13:35:03 2018

@author: zxlation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

# 用于提取MNIS数据
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def model(inputs):
    # 定义权重和偏置项
    with tf.variable_scope("conv"):
        W = tf.get_variable("weights", shape = [784, 10], dtype = tf.float32)
        b = tf.get_variable("biases", shape = [10], dtype = tf.float32)
    
    # 模型对应的操作：卷积 + 偏置项
    logits = tf.matmul(inputs, W) + b
    
    # 在TensorBoard中统计logits的直方图
    tf.summary.histogram('logits',  logits)
    
    return logits


def main(_):
    # 预备：导入数据
    with tf.name_scope("data"):
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot = True)

    # 第1步：定义placeholder，inputs表示我们的输入特征, labels表示对应的真实标签
    with tf.name_scope("io"):
        inputs = tf.placeholder(tf.float32, [None, 784])
        labels = tf.placeholder(tf.float32, [None, 10])
    
    # 第2步：根据模型构建计算图
    with tf.name_scope("model"):
        logits = model(inputs)

    # 第3步：定义损失函数(交叉熵)
    # E = -sum_{x} p(x)log[q(x)]
    with tf.name_scope("cross-entropy"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = labels,
                                                                logits = logits)
        cross_entropy = tf.reduce_mean(cross_entropy)
    
    # 第4步：定义优化器
    with tf.name_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5)
        train_op  = optimizer.minimize(cross_entropy)
    
    # 合并所有summary操作
    summary_op = tf.summary.merge_all()
    
    
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
    ##                       所有准备工作做好之后，下面就可以开始训练了。                            ##
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
    # 定义TF会话
    sess = tf.InteractiveSession()
    
    # 初始(所有)化全局变量
    tf.global_variables_initializer().run()
    
    # 定义摘要记录器(summary writer)，这一条语句会在TensorBoard中显示我们的计算图
    writer = tf.summary.FileWriter("logs/softmax/", sess.graph)
    
    ### --> 开始训练
    for i in range(FLAGS.max_steps):
        # 从数据集中取出一个batch的数据
        batch_inputs, batch_labels = mnist.train.next_batch(100)
        
        # 建立feed_dict
        feed_dict = {inputs: batch_inputs, labels: batch_labels}
        
        # 采用Feeding机制训练一个step
        _, ce, summ_str = sess.run([train_op, cross_entropy, summary_op], feed_dict = feed_dict)
        
        # 定期打印日志
        if (i + 1) % FLAGS.log_freq == 0 or ((i + 1) == FLAGS.max_steps):
            # 写入日志
            writer.add_summary(summ_str, i + 1)
            
            # 打印交叉熵
            print("step%d: cross entropy = %.4f" % (i + 1, ce))
    ###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 训练结束

    # 测试训练好的模型：注意这里相当于又构建一个计算图，专门用于测试。
    # --> 构建测试计算图
    correct_pred = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # --> 测试计算图构建好了之后，就可以用sess.run进行测试了
    feed_dict = {inputs: mnist.test.images, labels: mnist.test.labels}
    ac = sess.run(accuracy, feed_dict = feed_dict)
    print("test accuracy = %.4f" % ac)
    
    # 记住这种单独定义sess的情况需要手动关闭sess，同时关闭摘要记录器。
    writer.close()
    sess.close()


"""---------------------------------------------------------------
  为了区分主执行文件还是被调用的文件，Python引入了一个变量__name__. 
  当文件是被调用时，__name__的值为模块名，当文件被执行时，__name__
  为'__main__'。这个特性，我们可以在每个模块中写上测试代码，这些测试
  代码仅当模块被Python直接执行时才会运行，代码和测试完美结合在一起.
------------------------------------------------------------------"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = 'datasets/mnist/input_data',
                        help = 'Directory for storing input data.')
    parser.add_argument('--max_steps', type = int, default = 1000,
                        help = 'The maximum number of steps to train.')
    parser.add_argument('--log_freq', type = int, default = 50,
                        help = 'print logs during how many training steps.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main = main, argv = [sys.argv[0]] + unparsed)
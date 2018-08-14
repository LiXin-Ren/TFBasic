# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:13:09 2018

@author: zxlation
"""

import tensorflow as tf
import os


def get_file_name_list(file_dir):
    name_list = []
    for im in os.listdir(file_dir):
        name_list.append(file_dir + im)
    
    return name_list


def get_batch(inpDir, batch_size):
    # get images paths for all items
    inpList = get_file_name_list(inpDir)
    
    # a simple judgement for consistency
    num_images = len(inpList)
    if num_images <= 0:
        raise ValueError("Not Found Images!")
    
    # cnvert them to tf.string
    inpList = tf.cast(inpList, tf.string)
    
    # build a input queue for file names
    inpQueue = tf.train.slice_input_producer([inpList],
                                             num_epochs = None,
                                             shuffle = False,
                                             capacity = 32,
                                             shared_name = None,
                                             #name = '操作名'
                                             name = 'file_name_queue')
    
    # construct ops for reading a single example
    with tf.name_scope('read_single_example'):
        inpImage = tf.read_file(inpQueue[0])
    
    # decode images
    with tf.name_scope('decode_single_example'):
        inpImage = tf.image.decode_image(inpImage, channels = 3)
    
    # DO SOME AUGMENTATION: resize images with same shape
    # 注意这里的形状必须是确定的，而且要一致可以形成batch。那么要是图像大小
    # 不一样而又不能随机裁剪时，该怎么办呢？
    with tf.name_scope('crop_or_pad_resize'):
        inpImage = tf.random_crop(inpImage, [256, 256, 3])
    
    
    # 生成batch，观察图像出现的顺序和文件中图像原来的顺序有什么变化。如果想
    # batch中的图像顺序与图像本身的顺序保持一致，思考应该怎么操作。
    #
    # 提示：在保证tf.train.slice_input_producer函数中的shuffle参数为
    # False的情况下，改用下面被注释掉的函数，并将 num_threads 设为1试试，
    # 想想为什么。
    
    #shuffle_batch()会随机打乱顺序，batch()不会打乱顺序
    inpBatch = tf.train.shuffle_batch([inpImage],
                                      batch_size = batch_size,
                                      num_threads = 4,
                                      capacity = 8*batch_size,
                                      min_after_dequeue = 2*batch_size,
                                      name = 'batch_queue')
    
    #batch()不会打乱顺序
#    inpBatch = tf.train.batch([inpImage],
#                              batch_size = batch_size,
#                              num_threads = 1,  # 注意这个参数
#                              capacity = 8*batch_size,
#                              dynamic_pad = False,
#                              name = 'batch_queue')
    
    # cast to tf.float32
    with tf.name_scope('cast_to_float32'):
        inpBatch = tf.cast(inpBatch, tf.float32)
    
    # Normalization
    with tf.name_scope('normalization'):
        inpBatch = inpBatch / 255.0
    
    return inpBatch

#%% TEST
    
import matplotlib.pyplot as plt

inpDIR = 'dataset/DIV2K/DIV2K_train_LR_bicubic/X4/'
batch_size = 4
inpBatch = get_batch(inpDIR, batch_size = batch_size)

with tf.Session() as sess:
    batch_index = 0
    tf.local_variables_initializer().run()
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    
    try:
        #coord分类，threads回归        
        while not coord.should_stop() and batch_index < 1:
            inp_batch = sess.run(inpBatch)
            for i in range(batch_size):
                plt.imshow(inp_batch[i, :, :, :])
                plt.title('input batch%d: image %d' % (batch_index, i))
                plt.show()
            
            batch_index += 1
    except tf.errors.OutOfRangeError:
        print("Done!")
    finally:
        coord.request_stop()
        coord.join()
    
    
###------------------------------------任务--------------------------------------###
#
# 1. 仿照上面的过程，将数据集的标签（label）也输入到TF中，同时保证输入图像和标签一一对应。
#
# 注：对于分类问题，标签可能是一系列数字；对于回归问题，标签可能是输入图像对应的另一幅图像。根据
#    自己要做的方向和具体问题，实现特征和标签成对输入到模型。
#
# 2. 通常在训练时，需要对训练数据进行shuffle处理，但在测试或验证时并不需要，而且为了让输入内容
#    与文件名一一对应，还应该保证输入顺序保持不变。根据上面程序的提示内容，编写一个小程序，实现
#    下面的功能：假设目标文件夹下有100幅图像：001.png, 002.png, ..., 100.png，将这100幅图
#    像依次读入到TF，计算它们像素值的平均值和均方差，然后打印出来。要求打印时按照这样的格式进行：
#         文件名: 平均值/均方差
#    平均值和均方差都只保留2位小数。图像数量自己确定，不能＜5幅图像。
#
#


    
    
    
    
    
    
    

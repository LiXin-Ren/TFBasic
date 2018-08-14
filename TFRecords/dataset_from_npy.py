# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 19:21:06 2018

@author: zxlation
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import time

degration = 'FFT'
scale = 4
mode  = 'train'

input_dir = 'dataset/IXI/IXI_%s_LR_%s/X%d/' % (mode, degration, scale)
label_dir = 'dataset/IXI/IXI_%s_HR/' % (mode)
tfrecords_dir = 'dataset/tfrecords/IXI_%s_%s_X%d.tfrecords' % (mode, degration, scale)
    
labH = 256
labW = 256
inpH = labH//scale
inpW = labW//scale
nChannels = 1
batch_size = 16
num_images = 500
MAX_VALUE = 32767.0


def plot_batch(inpBatch, labBatch):
    bSize = inpBatch.shape[0]
    assert bSize == labBatch.shape[0], "the size of input and label batch does not match."
    
    for j in range(bSize):
        plt.imshow(inpBatch[j, :, :, 0], cmap = 'gray')
        plt.title("Input Image%d" % j), plt.show()
        plt.imshow(labBatch[j, :, :, 0], cmap = 'gray')
        plt.title("Label Image%d" % j), plt.show()
    
    return bSize


def _read_py_func(inp_name, lab_name):
    """Python function to load npy files.
    
    Args:
        inp_name: bytes that indicate a str name of an input image.
        lab_name: bytes that indicate a str name of a label image.
    Returns:
        a paris of whole slices randomly selected from the inp/lab images
    """
    # Note: input names are in form bytes, which cannot be used as file paths.
    # It must be decoded as string.
    inp_name = inp_name.decode()
    lab_name = lab_name.decode()
    
    inp_path = os.path.join(input_dir, inp_name)
    lab_path = os.path.join(label_dir, lab_name)
    inp_im = np.array(np.load(inp_path)/MAX_VALUE, np.float32)
    lab_im = np.array(np.load(lab_path)/MAX_VALUE, np.float32)
    
    # select a slice randomly.
    randS = np.random.randint(low = 0, high = inp_im.shape[0])
    input_slice = inp_im[:, :, randS]
    label_slice = lab_im[:, :, randS]
    
    # expand a dimension for depth.
    input_slice = input_slice[:, :, np.newaxis]
    label_slice = label_slice[:, :, np.newaxis]
    
    # we can do some other preprocessing, e.g. patching and augmentaton here if necessary.
    # -- patching...
    # -- augmentation...
    # -- filtering...
    
    return input_slice, label_slice
    

def _parse_py_func(inp_name, lab_name):
    """Use tf.py_func() to invoke a user-defined python function.
       Arguments to tf.py_func() are those from dataset definition, and the
       results are those returned by '_read_py_func', but in the form of 
       tensorflow tensors.
    """
    input_slice, label_slice = tf.py_func(_read_py_func,
                                          [inp_name, lab_name],
                                          [tf.float32, tf.float32])
    
    return input_slice, label_slice


def _resize_tf_func(input_slice, label_slice):
    """Make the shape of inp/lab slices clear.
    
    Args:
        input_slice: a tensor that represents a slice of NIFTI data from inputs
        label_slice: same with input slice but from labels
    Returns:
        tensors with definite shapes.
    """
    input_slice.set_shape([inpH, inpW, nChannels])
    label_slice.set_shape([labH, labW, nChannels])
    
    return input_slice, label_slice


def main(unused_argv):
    # get file names for inputs and labels
    input_list = os.listdir(input_dir)
    label_list = os.listdir(label_dir)
    
    # create a traning dataset from name lists
    train_dataset = tf.contrib.data.Dataset.from_tensor_slices((input_list, label_list))
    
    # map1: load npy and select a slice from the volume data
    train_dataset = train_dataset.map(_parse_py_func,   # -- dataset.map()
                                      num_threads = 8,
                                      output_buffer_size = 8*batch_size)
    train_dataset = train_dataset.map(_resize_tf_func,  # -- dataset.map()
                                      num_threads = 2,
                                      output_buffer_size = 8*batch_size)
    
    train_dataset = train_dataset.repeat(10)
    train_dataset = train_dataset.shuffle(buffer_size = 16*batch_size)
    train_dataset = train_dataset.batch(batch_size)
    #iterator = train_dataset.make_one_shot_iterator()
    
    
    iterator = train_dataset.make_initializable_iterator()
    
    input_batch, label_batch = iterator.get_next()
    
    num_batch = 64
    with tf.Session() as sess:
        i = 0
        sess.run([tf.global_variables_initializer(), iterator.initializer])
        while i < num_batch:
            try:
                start_time = time.time()
                lr_img, hr_img = sess.run([input_batch, label_batch])
                duration = time.time() - start_time
                print("batch%d: %.4fs" % (i + 1, duration))
            except tf.errors.OutOfRangeError:
                print("End of the Dataset")
                break
            i += 1
    

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
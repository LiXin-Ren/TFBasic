# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 19:21:06 2018

@author: zxlation
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import time

degration = 'FFT'
scale = 4
mode  = 'train'

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
    """Plot a batch of data with matplotlib.pyplot
    
    Args:
        inpBatch: a batch of input images.
        labBatch: a corresponding batch of label data.
    Returns:
        an integer indicating batch size
    """
    bSize = inpBatch.shape[0]
    assert bSize == labBatch.shape[0], "the size of input and label batch does not match."
    
    for j in range(bSize):
        plt.imshow(inpBatch[j, :, :, 0], cmap = 'gray')
        plt.title("Input Image%d" % j), plt.show()
        plt.imshow(labBatch[j, :, :, 0], cmap = 'gray')
        plt.title("Label Image%d" % j), plt.show()
    
    return bSize


def augmentation(input_patch, label_patch):
    """ Data Augmentation with TensorFlow ops.
    
    Args:
        input_patch: input tensor representing an input patch or image
        label_patch: label tensor representing an target patch or image
    Returns:
        rotated input_patch and label_patch randomly
    """
    def no_trans():
        return input_patch, label_patch

    def vflip():
        inpPatch = input_patch[::-1, :, :]
        labPatch = label_patch[::-1, :, :]
        return inpPatch, labPatch
    
    def hflip():
        inpPatch = input_patch[:, ::-1, :]
        labPatch = label_patch[:, ::-1, :]
        return inpPatch, labPatch
    
    def hvflip():
        inpPatch = input_patch[::-1, ::-1, :]
        labPatch = label_patch[::-1, ::-1, :]
        return inpPatch, labPatch

    def trans():
        inpPatch = tf.image.transpose_image(input_patch[:, :, :])
        labPatch = tf.image.transpose_image(label_patch[:, :, :])
        return inpPatch, labPatch
    
    def tran_vflip():
        inpPatch = tf.image.transpose_image(input_patch)[::-1, :, :]
        labPatch = tf.image.transpose_image(label_patch)[::-1, :, :]
        return inpPatch, labPatch
    
    def tran_hflip():
        inpPatch = tf.image.transpose_image(input_patch)[:, ::-1, :]
        labPatch = tf.image.transpose_image(label_patch)[:, ::-1, :]
        return inpPatch, labPatch
        
    def tran_hvflip():
        inpPatch = tf.image.transpose_image(input_patch)[::-1, ::-1, :]
        labPatch = tf.image.transpose_image(label_patch)[::-1, ::-1, :]
        return inpPatch, labPatch
    
    rot = tf.random_uniform(shape = (), minval = 2, maxval = 9, dtype = tf.int32)    
    input_patch, label_patch = tf.case({tf.equal(rot, 2): vflip,
                                  tf.equal(rot, 3): hflip,
                                  tf.equal(rot, 4): hvflip,
                                  tf.equal(rot, 5): trans,
                                  tf.equal(rot, 6): tran_vflip,
                                  tf.equal(rot, 7): tran_hflip,
                                  tf.equal(rot, 8): tran_hvflip},
    default = no_trans, exclusive = True)
    
    return input_patch, label_patch


def _parse_tfrecords(example_proto):
    """ Parse a tfrecords file into batch tensors.
    
    Args:
        example_proto: serialized example
    Returns:
        a pair of tensors indicating input and label images respectively.
    """
    # parse the serialized examples:
    # must follow the format with that when converting tfrecords files!
    features = {"input_slice":tf.FixedLenFeature([], tf.string),
                "label_slice":tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features = features)
    
    # decode raw bytes/string accordingly
    input_raw = tf.decode_raw(parsed_features["input_slice"], tf.int16)
    label_raw = tf.decode_raw(parsed_features["label_slice"], tf.int16)
    
    # reshape to fixed shape
    input_image = tf.reshape(input_raw, [inpH, inpW, nChannels])
    label_image = tf.reshape(label_raw, [labH, labW, nChannels])
    
    # 
    input_image = tf.cast(input_image, tf.float32)
    label_image = tf.cast(label_image, tf.float32)
    
    input_image = input_image/MAX_VALUE
    label_image = label_image/MAX_VALUE
    
    input_image, label_image = augmentation(input_image, label_image)
    
    return input_image, label_image


def main(unused_argv):
    # create a traning dataset from name lists
    train_dataset = tf.contrib.data.TFRecordDataset([tfrecords_dir])
    
    # map1: load npy and select a slice from the volume data
    train_dataset = train_dataset.map(_parse_tfrecords,   # -- dataset.map()
                                      num_threads = 8,
                                      output_buffer_size = 8*batch_size)
    
    train_dataset = train_dataset.repeat(10)
    train_dataset = train_dataset.shuffle(buffer_size = 16*batch_size)
    train_dataset = train_dataset.batch(batch_size)
    #iterator = train_dataset.make_one_shot_iterator()
    iterator = train_dataset.make_initializable_iterator()
    
    input_batch, label_batch = iterator.get_next()
    
    num_batch = 64
    with tf.Session() as sess:
        #initialize global variables and iterator
        sess.run([tf.global_variables_initializer(), iterator.initializer])
        
        i = 0
        total_time = 0.0
        while i < num_batch:
            try:
                start_time = time.time()
                inpBatch, labBatch = sess.run([input_batch, label_batch])
                duration = time.time() - start_time
                total_time += duration
                print("batch%d: %.4fs" % (i + 1, duration))
                
                plot_batch(inpBatch, labBatch)
            except tf.errors.OutOfRangeError:
                print("End of the Dataset")
                break
            i += 1
        
        print("%.4fs/batch for batch size = %d." % (total_time/num_batch, batch_size))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
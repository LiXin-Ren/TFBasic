# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 12:24:42 2018

@author: zxlation
"""

import tensorflow as tf
import numpy as np
import sys
import os

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))


def convert_slice_to_tfrecords(input_dir, label_dir, save_dir, file_name):
    input_names = os.listdir(input_dir)
    label_names = os.listdir(label_dir)
    
    num_examples = len(input_names)
    if num_examples != len(label_names):
        raise ValueError("Number of LR images %s does not match number of HR images %s!" % 
                         (num_examples, len(label_names)))
    
    save_path = os.path.join(save_dir, file_name)
    writer = tf.python_io.TFRecordWriter(save_path)
    
    for i in range(num_examples):
        if (i + 1) % 10 == 0 or (i + 1) == num_examples:
            print("%d items were done!" % (i + 1))
            #sys.stdout.flush()
        
        try:
            # read from npys
            input_content = np.load(os.path.join(input_dir, input_names[i]))
            label_content = np.load(os.path.join(label_dir, label_names[i]))
            [inpH, inpW, inpSlices] = input_content.shape
            [labH, labW, labSlices] = label_content.shape
            
            if inpSlices != labSlices:
                print("The size for z-axis does not match (%d, %d)!" % (inpSlices, labSlices))
            
            input_raw = np.zeros([inpH, inpW, 1], np.int16)
            label_raw = np.zeros([labH, labW, 1], np.int16)
            for j in range(labSlices):
                input_raw[:, :, 0] = input_content[:, :, j]
                label_raw[:, :, 0] = label_content[:, :, j]
                
                input_bytes = tf.compat.as_bytes(input_raw.tostring())
                label_bytes = tf.compat.as_bytes(label_raw.tostring())
                feature = {"input_slice":_bytes_feature(input_bytes),
                           "label_slice":_bytes_feature(label_bytes)}
            
                example = tf.train.Example(features = tf.train.Features(feature = feature))
                writer.write(example.SerializeToString())
                
        except IOError as e:
            print("Could not read image %d!" % i)
            print("Error: %s" % e)
    
    writer.close()
    sys.stdout.flush()
    
    return num_examples
    
#####################################################################
degration = 'FFT'  # the type of degration
scale = 4          # the factor of downsampling
mode = 'train'
inp_image_path = "dataset/IXI/IXI_%s_LR_%s/X%d/" % (mode, degration, scale)
lab_image_path = "dataset/IXI/IXI_%s_HR/" % (mode)

save_dir = "dataset/tfrecords/"
file_name = 'IXI_%s_%s_X%d.tfrecords' % (mode, degration, scale)

num_examples = convert_slice_to_tfrecords(inp_image_path, lab_image_path, save_dir, file_name)
print("Done! %d examples are transformed!" % num_examples)

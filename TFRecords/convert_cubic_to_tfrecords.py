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


def convert_to_tfrecords(input_dir, label_dir, save_dir, file_name):
    """Convert a whole dataset including inputs and labels into a single tfrecords file.
    
    Args:
        input_dir: a string represents inputs directory
        label_dir: a string represents labels directory
        save_dir: the directory that to save result files
        file_name: tfrecords file name
            
    Returns:
        An integer indicating the number of examples.
    """
    input_names = os.listdir(input_dir)
    label_names = os.listdir(label_dir)
    
    num_examples = len(input_names)
    if num_examples != len(label_names):
        raise ValueError("Number of LR images %s does not match number of HR images %s!" % 
                         (num_examples, len(label_names)))
    
    save_path = os.path.join(save_dir, file_name)
    writer = tf.python_io.TFRecordWriter(save_path)
    for i in range(num_examples):
        if (i + 1) % 100 == 0 or (i + 1) == num_examples:
            print("%d items were done!" % (i + 1))
            sys.stdout.flush()
        
        try:
            input_content = np.load(os.path.join(input_dir, input_names[i]))
            label_content = np.load(os.path.join(label_dir, label_names[i]))
            input_content = np.transpose(input_content, [2, 0, 1])
            label_content = np.transpose(label_content, [2, 0, 1])
            input_content = input_content[:, :, :, np.newaxis]
            label_content = label_content[:, :, :, np.newaxis]
            
            [inpS, inpH, inpW, inpC] = input_content.shape
            [labS, labH, labW, labC] = label_content.shape
            assert inpS == labS, "The size for z-axis does not match (%d, %d)!" % (inpS, labS)
            
            input_raw = input_content.tostring()
            label_raw = label_content.tostring()
            
            example = tf.train.Example(features = tf.train.Features(feature = {
                    "input":_bytes_feature(input_raw),
                    "inpS": _int64_feature(inpS),
                    "inpH": _int64_feature(inpH),
                    "inpW": _int64_feature(inpW),
                    "inpC": _int64_feature(inpC),
                    "label":_bytes_feature(label_raw),
                    "labS": _int64_feature(labS),
                    "labH": _int64_feature(labH),
                    "labW": _int64_feature(labW),
                    "labC": _int64_feature(labC),}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print("Could not read image %d!" % i)
            print("Error: %s" % e)
            print("Skip it! \n")
    
    writer.close()
    
    return num_examples
    
#####################################################################
degration = 'FFT'  # the type of degration
scale = 4          # the factor of downsampling
mode = 'valid'
lr_image_path = "dataset/IXI/IXI_%s_LR_%s/X%d/" % (mode, degration, scale)
hr_image_path = "dataset/IXI/IXI_%s_HR/" % (mode)

save_dir = "dataset/tfrecords/"
file_name = "IXI_%s_%s_X%d.tfrecords" % (mode, degration, scale)

num_examples = convert_to_tfrecords(lr_image_path, hr_image_path, save_dir, file_name)
print("Done! %d examples are transformed!" % num_examples)

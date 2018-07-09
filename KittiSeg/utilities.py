import tensorflow as tf
#import cv2
import numpy as np

def PReLU(x, scope):
    # PReLU(x) = x if x > 0, alpha*x otherwise

    alpha = tf.get_variable(scope + "/alpha", shape=[1],
                initializer=tf.constant_initializer(0), dtype=tf.float32)

    output = tf.nn.relu(x) + alpha*(x - abs(x))*0.5

    return output

# function for 2D spatial dropout:
def spatial_dropout(x, drop_prob=0.2):
    # x is a tensor of shape [batch_size, height, width, channels]
  #  print('---------Using Spatial Dropout-------------')
    keep_prob = 1.0 - drop_prob
    input_shape = x.get_shape().as_list()

    batch_size = input_shape[0]
    channels = input_shape[3]

    # drop each channel with probability drop_prob:
    noise_shape = tf.constant(value=[batch_size, 1, 1, channels])
    x_drop = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape)

    output = x_drop

    return output

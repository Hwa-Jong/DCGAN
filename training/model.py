import tensorflow as tf
import numpy as np

from training.layers import *

def Generator(x, batch_size):
    w_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        # project and reshape
        x = Dense(name='project', x=x, units=4*4*1024, use_bias=False, kernel_init=w_init)
        x = tf.reshape(x, [-1, 4,4,1024])
        x = tf.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        #conv1
        x = conv2dT(name='Conv1', x=x, k_size=5, batch_size=batch_size, strides=2, channel_out=512, kernel_init=w_init)
        x = tf.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        #oncv2
        x = conv2dT(name='Conv2', x=x, k_size=5, batch_size=batch_size, strides=2, channel_out=256, kernel_init=w_init)
        x = tf.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        #oncv3
        x = conv2dT(name='Conv3', x=x, k_size=5, batch_size=batch_size, strides=2, channel_out=128, kernel_init=w_init)
        x = tf.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        #conv4
        x = conv2dT(name='Conv4', x=x, k_size=5, batch_size=batch_size, strides=2, channel_out=3, kernel_init=w_init)
        imgs = tf.nn.tanh(x)

    return imgs

def Discriminator(x):
    w_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        #conv1
        x = conv2d(name='Conv1', x=x, k_size=5, strides=2, channel_out=128, kernel_init=w_init)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = tf.nn.dropout(x, rate=0.3)

        #oncv2
        x = conv2d(name='Conv2', x=x, k_size=5, strides=2, channel_out=256, kernel_init=w_init)
        x = tf.layers.BatchNormalization()(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = tf.nn.dropout(x, rate=0.3)

        #oncv3
        x = conv2d(name='Conv3', x=x, k_size=5, strides=2, channel_out=512, kernel_init=w_init)
        x = tf.layers.BatchNormalization()(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = tf.nn.dropout(x, rate=0.3)

        #conv4
        x = conv2d(name='Conv4', x=x, k_size=5, strides=2, channel_out=1024, kernel_init=w_init)
        x = tf.layers.BatchNormalization()(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = tf.nn.dropout(x, rate=0.3)

        #output
        x = tf.layers.Flatten()(x)
        output = Dense(name='output', x=x, units=1, kernel_init=w_init)
    
    return output

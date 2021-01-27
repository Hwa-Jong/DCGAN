import tensorflow as tf


#-----------------------------------------------------------------------------------------------
def conv2d(name, x, k_size, strides, channel_out, use_bias = True, dtype=tf.float32, kernel_init = None):
    kernel = tf.get_variable(name, shape=(k_size, k_size, x.shape[-1], channel_out), dtype=dtype, initializer=kernel_init)
    x = tf.cast(x, dtype)
    x = tf.nn.conv2d(x, kernel, strides=[1, strides, strides, 1], padding='SAME', data_format='NHWC')
    if use_bias:
        bias = tf.get_variable(name+'_bias', shape=(channel_out), initializer=tf.initializers.zeros(), dtype=dtype)
        x = tf.nn.bias_add(x, bias)
    return x

def conv2dT(name, x, k_size, strides, batch_size, channel_out, use_bias = True, dtype=tf.float32, kernel_init = None):
    kernel = tf.get_variable(name, shape=(k_size, k_size, channel_out, x.shape[-1]), dtype=dtype, initializer=kernel_init)
    x = tf.nn.conv2d_transpose(value=x, filter=kernel, output_shape=(batch_size, x.shape[1]*strides, x.shape[2]*strides, channel_out), strides=[1, strides, strides, 1], padding='SAME', data_format='NHWC')
    if use_bias:
        bias = tf.get_variable(name+'_bias', shape=(channel_out), initializer=tf.initializers.zeros(), dtype=dtype)
        x = tf.nn.bias_add(x, bias)
    return x

def Dense(name, x, units, use_bias = True, dtype=tf.float32, kernel_init = None):
    w = tf.get_variable(name, shape=(x.shape[1], units), dtype=dtype, initializer=kernel_init)
    x = tf.cast(x, dtype)
    x = tf.matmul(x, w)
    if use_bias:
        bias = tf.get_variable(name+'_bias', shape=(units), initializer=tf.initializers.zeros(), dtype=dtype)
        x = tf.nn.bias_add(x, bias)
    return x

#-----------------------------------------------------------------------------------------------
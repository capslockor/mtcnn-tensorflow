import tensorflow as tf


class LayerFactory(object):
    AVAILABEL_PADDINGS = ('SAME', 'VALID')

    def __init__(self, network):
        self.__network = network

    @staticmethod
    def __validate_padding(padding):
        if padding not in LayerFactory.AVAILABEL_PADDINGS:
            raise Exception("Padding {} not valid")

    @staticmethod
    def __validate_grouping(channels_input, channels_output, group):
        if channels_input % group != 0:
            raise Exception("The number of channels in the input does not match the group")

        if channels_output % group != 0:
            raise Exception("The number of channels in the output does not match the group")

    @staticmethod
    def vectorize_input(input_layer):
        input_shape = input_layer.get_shape()
        if input_shape.ndims == 4:
            dim = 1
            for x in input_shape[1:].as_list():
                dim *= int(x)
            vectorized_input = tf.reshape(input_layer, [-1, dim])
        else:
            vectorized_input, dim = (input_layer, input_shape[-1].value)

        return vectorized_input, dim

    def __make_var(self, name, shape):
        return tf.get_variable(name=name, shape=shape, trainable=False)

    def new_feed(self, name, layer_shape):
        feed_data = tf.placeholder(tf.float32, layer_shape, name=name)
        self.__network.add_layer(name, layer_out=feed_data)

    def new_conv(self, name, num_input_channels, filter_size, num_output_channels, padding, use_relu=True, input_layername=None):
        layer_input = self.__network.get_layer(input_layername)
        shape = [filter_size, filter_size, num_input_channels, num_output_channels]
        with tf.variable_scope(name):
            weights = self.__make_var('weight', shape=shape)
            biases = self.__make_var('biases', shape=[num_output_channels])
            layer = tf.nn.conv2d(input=layer_input, filter=weights, strides=[1, 1, 1, 1], padding=padding)
            out = tf.nn.bias_add(layer, biases)
            if use_relu:
                with tf.name_scope('Prelu'):
                    shape = int(out.get_shape()[-1])
                    alpha = self.__make_var('alpha', shape=shape)
                    out = tf.nn.relu(out) + tf.multiply(alpha, -tf.nn.relu(-out))
        self.__network.add_layer(name, layer_out=out)

    def new_max_pooling(self, name, kernel_size=2, strides=2, padding="SAME", input_layername=None):
        with tf.name_scope(name):
            layer_input = self.__network.get_layer(input_layername)
            out = tf.nn.max_pool(value=layer_input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, strides, strides, 1], padding=padding)
        self.__network.add_layer(name, layer_out=out)

    def new_softmax(self, name, axis, input_layername=None):
        layer_input = self.__network.get_layer(input_layername)
        out = tf.nn.softmax(layer_input, dim=axis, name=name)
        self.__network.add_layer(name, layer_out=out)

    def new_full_connected(self, name, output_dim, relu=False, input_layername=None):
        with tf.variable_scope(name):
            layer_input = self.__network.get_layer(input_layername)
            vectorized_input, dimension = self.vectorize_input(layer_input)

            weights = self.__make_var('weight', shape=[dimension, output_dim])
            biases = self.__make_var('biases', shape=[output_dim])
            out = tf.nn.xw_plus_b(vectorized_input, weights, biases, name=name)
            if relu:
                out = tf.nn.relu(out)
        self.__network.add_layer(name, layer_out=out)


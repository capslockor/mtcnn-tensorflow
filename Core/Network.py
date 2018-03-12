import tensorflow as tf


class Network(object):
    def __init__(self, session=None):
        self._session = session
        self.__layers = {}
        self.__last_layer_name = None

        with tf.variable_scope(self.__class__.__name__.lower()):
            self._config()

    def _config(self):
        raise NotImplementedError("This method must be implemented by the network!")

    def add_layer(self, name, layer_out):
        self.__layers[name] = layer_out
        self.__last_layer_name = name

    def get_layer(self, name=None):
        if name is None:
            name = self.__last_layer_name

        return self.__layers[name]

    def set_weights(self, weights_values):
        network_name = self.__class__.__name__.lower()
        with tf.variable_scope(network_name):
            for layer_name in weights_values:
                with tf.variable_scope(layer_name, reuse=True):
                    for param_name, data in weights_values[layer_name].items():
                        try:
                            var = tf.get_variable(param_name)
                            self._session.run(var.assign(data))
                        except ValueError:
                            raise

    def feed(self, image):
        network_name = self.__class__.__name__.lower()
        with tf.variable_scope(network_name):
            return self._feed(image)

    def _feed(self, image):
        raise NotImplementedError("Method not implemented!")

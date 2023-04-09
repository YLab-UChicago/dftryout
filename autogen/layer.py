
# Parent class of all layer-related classes
#
# Usage:
#   1)  Checking if an object is a layer, for validation or categorization
#
# Note:
#   Current not implementing anything
class Layer:
    def __init__(self):
        return


# Class for storing convolutional layer information
class ConvLayer(Layer):
    def __init__(self, batch_size,
                 input_width, input_height, input_depth,
                 filter_width, filter_height, filter_depth,
                 padding, stride,
                 stationarity, loop_orders,blocking_scheme,
                 simd_length = None):
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.input_depth = input_depth
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_depth = filter_depth
        self.padding = padding
        self.stride = stride
        # Either "OS", "WS", or "IS"
        self.stationarity = stationarity
        # A tuple of :
        #   For OS:
        #   x    Permutations of f, w, h, d
        #
        #   For WS:
        #       Permutations of f, fh, fw, d
        #
        #   For IS:
        #       Permutations of TBD
        self.loop_orders = loop_orders

        # A list of block sizes in each dimension that is
        #  consistent with the length of the permutation
        self.blocking_scheme = blocking_scheme
        self.simd_length = simd_length


# Class for storing pooling layer information
class PoolLayer(Layer):
    def __init__(self, pool_type, batch_size,
                 input_width, input_height, input_depth,
                 pool_size):
        # Type of pooling:
        #    "MAX" for max-pooling
        #    "AVG" for average-pooling
        self.pool_type = pool_type

        # The number of batches
        self.batch_size = batch_size

        # Width of input feature map
        self.input_width = input_width
        # Height of input feature map
        self.input_height = input_height
        # Number of channels in the input feature map
        self.input_depth = input_depth

        # The width = height of the pooling window
        self.pool_size = pool_size


# Class for storing binary-transformation layer information
class BitransLayer(Layer):
    def __init__(self, batch_size,
                 input_width, input_height, input_depth):

        # The number of batches
        self.batch_size = batch_size

        # Width of input feature map
        self.input_width = input_width
        # Height of input feature map
        self.input_height = input_height
        # Depth of input feature map
        self.input_depth = input_depth


# Class for storing fully-connected layer information
class FCLayer(Layer):
    def __init__(self, batch_size, input_size, output_size):

        # The number of batches
        self.batch_size = batch_size

        # The size of input
        self.input_size = input_size

        # The size of output
        self.output_size = output_size

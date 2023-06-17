import tensorflow as tf
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

class QuantConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(QuantConv2D, self).__init__(**kwargs)
        self.conv2d = tf.keras.layers.Conv2D(filters, kernel_size)

    def call(self, inputs, training=None):
        output = self.conv2d(inputs)

        # Quantize the output to 8-bits during the forward pass
        if not training:
            output = tf.quantization.quantize_and_dequantize(output,
                                                             input_min=tf.reduce_min(output),
                                                             input_max=tf.reduce_max(output),
                                                             num_bits=8)
        return output

# To use the layer
layer = QuantConv2D(512, (3, 3))
# Build the layer
layer.build((None, 28, 28, 512))

# Generate some random data
data = tf.random.normal((1, 28, 28, 512))

# Time the inference
start_time = time.time()

# Use the layer
output = layer(data, training=False)

end_time = time.time()

print("Inference took {:.6f} seconds".format(end_time - start_time))

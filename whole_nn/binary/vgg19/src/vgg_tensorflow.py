import os
import time
import tensorflow as tf
import larq as lq
from tensorflow.keras import layers

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Define the binarization keyword arguments for the convolutional layers
binarization_kwargs = dict(input_quantizer="ste_sign",
                           kernel_quantizer="ste_sign",
                           kernel_constraint="weight_clip")

class BinarizedVGG19(tf.keras.Model):
    def __init__(self):
        super(BinarizedVGG19, self).__init__()

        self.conv1_1 = lq.layers.QuantConv2D(64, kernel_size=(3, 3), padding='same', input_shape=(224, 224, 3), **binarization_kwargs)
        self.conv1_2 = lq.layers.QuantConv2D(64, kernel_size=(3, 3), padding='same', **binarization_kwargs)
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv2_1 = lq.layers.QuantConv2D(128, kernel_size=(3, 3), padding='same', **binarization_kwargs)
        self.conv2_2 = lq.layers.QuantConv2D(128, kernel_size=(3, 3), padding='same', **binarization_kwargs)
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv3_1 = lq.layers.QuantConv2D(256, kernel_size=(3, 3), padding='same', **binarization_kwargs)
        self.conv3_2 = lq.layers.QuantConv2D(256, kernel_size=(3, 3), padding='same', **binarization_kwargs)
        self.conv3_3 = lq.layers.QuantConv2D(256, kernel_size=(3, 3), padding='same', **binarization_kwargs)
        self.conv3_4 = lq.layers.QuantConv2D(256, kernel_size=(3, 3), padding='same', **binarization_kwargs)
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv4_1 = lq.layers.QuantConv2D(512, kernel_size=(3, 3), padding='same', **binarization_kwargs)
        self.conv4_2 = lq.layers.QuantConv2D(512, kernel_size=(3, 3), padding='same', **binarization_kwargs)
        self.conv4_3 = lq.layers.QuantConv2D(512, kernel_size=(3, 3), padding='same', **binarization_kwargs)
        self.conv4_4 = lq.layers.QuantConv2D(512, kernel_size=(3, 3), padding='same', **binarization_kwargs)
        self.pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv5_1 = lq.layers.QuantConv2D(512, kernel_size=(3, 3), padding='same', **binarization_kwargs)
        self.conv5_2 = lq.layers.QuantConv2D(512, kernel_size=(3, 3), padding='same', **binarization_kwargs)
        self.conv5_3 = lq.layers.QuantConv2D(512, kernel_size=(3, 3), padding='same', **binarization_kwargs)
        self.conv5_4 = lq.layers.QuantConv2D(512, kernel_size=(3, 3), padding='same', **binarization_kwargs)
        self.pool5 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    def call(self, inputs):
        timings = []

        start_time = time.time()
        x = self.conv1_1(inputs)
        timings.append(('conv1_1', time.time() - start_time))

        start_time = time.time()
        x = self.conv1_2(x)
        timings.append(('conv1_2', time.time() - start_time))

        x = self.pool1(x)

        start_time = time.time()
        x = self.conv2_1(x)
        timings.append(('conv2_1', time.time() - start_time))

        start_time = time.time()
        x = self.conv2_2(x)
        timings.append(('conv2_2', time.time() - start_time))
        x = self.pool2(x)

        start_time = time.time()
        x = self.conv3_1(x)
        timings.append(('conv3_1', time.time() - start_time))

        start_time = time.time()
        x = self.conv3_2(x)
        timings.append(('conv3_2', time.time() - start_time))

        start_time = time.time()
        x = self.conv3_3(x)
        timings.append(('conv3_3', time.time() - start_time))

        start_time = time.time()
        x = self.conv3_4(x)
        timings.append(('conv3_4', time.time() - start_time))
        x = self.pool3(x)

        start_time = time.time()
        x = self.conv4_1(x)
        timings.append(('conv4_1', time.time() - start_time))

        start_time = time.time()
        x = self.conv4_2(x)
        timings.append(('conv4_2', time.time() - start_time))

        start_time = time.time()
        x = self.conv4_3(x)
        timings.append(('conv4_3', time.time() - start_time))

        start_time = time.time()
        x = self.conv4_4(x)
        timings.append(('conv4_4', time.time() - start_time))
        x = self.pool4(x)

        start_time = time.time()
        x = self.conv5_1(x)
        timings.append(('conv5_1', time.time() - start_time))

        start_time = time.time()
        x = self.conv5_2(x)
        timings.append(('conv5_2', time.time() - start_time))

        start_time = time.time()
        x = self.conv5_3(x)
        timings.append(('conv5_3', time.time() - start_time))

        start_time = time.time()
        x = self.conv5_4(x)
        timings.append(('conv5_4', time.time() - start_time))

        x = self.pool5(x)

        for layer_name, elapsed_time in timings:
            print(f"{layer_name} time: {elapsed_time:.4f} seconds")

        return x

model = BinarizedVGG19()
input_tensor = tf.random.normal([1, 224, 224, 3])
output_tensor = model(input_tensor)
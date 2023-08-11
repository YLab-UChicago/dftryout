import tensorflow as tf
import time
import sys

#Run with "python 18_ResNet.py <number of trials>"

# Configure TensorFlow to run on a single thread
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D, Dense, Add
from tensorflow.keras.models import Model

def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False):
    shortcut = x
    if conv_shortcut:
        shortcut = Conv2D(filters, 1, strides=stride)(x)
        shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x

def ResNet18(input_shape=(224, 224, 3), classes=1000):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    for _ in range(2): x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2, conv_shortcut=True)
    for _ in range(1): x = residual_block(x, 128)

    x = residual_block(x, 256, stride=2, conv_shortcut=True)
    for _ in range(1): x = residual_block(x, 256)

    x = residual_block(x, 512, stride=2, conv_shortcut=True)
    for _ in range(1): x = residual_block(x, 512)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(classes, activation='softmax')(x)

    return Model(inputs, outputs)

model = ResNet18()

# Define a representative dataset generator
def representative_dataset_gen():
    for _ in range(100):
        yield [tf.random.normal([1, 224, 224, 3])]

# Convert the model to TensorFlow Lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_types = [tf.int8]
quantized_tflite_model = converter.convert()

# Save the quantized model to a .tflite file
with open('quantized_resnet18.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='quantized_resnet18.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create dummy input data
input_shape = input_details[0]['shape']
input_data = tf.random.normal(input_shape, dtype=tf.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

# Measure and record execution time
with open('result.txt', 'w') as file:
    total_time = 0
    num_runs = int(sys.argv[1])
    for i in range(num_runs):
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        difference = (end_time - start_time) * 1000
        file.write(f"Trial {i + 1}: {difference:.4f} ms\n")
        total_time += difference
    total_time /= num_runs
    result_str = f"Average execution time for 8bit-quantized ResNet-18: {total_time:.4f} ms\n{num_runs} trials completed"
    print(result_str)
    file.write(result_str)

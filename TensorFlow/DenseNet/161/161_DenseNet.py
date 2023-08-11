import tensorflow as tf
import time
import sys
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D, Dense, Concatenate
from tensorflow.keras.models import Model

#Run with "python 161_DenseNet.py <number of trials>"

# Configure TensorFlow to run on a single thread
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Define the dense block
def dense_block(x, num_layers, growth_rate):
    for _ in range(num_layers):
        conv = Conv2D(growth_rate, (3, 3), padding='same', activation='relu')(x)
        x = Concatenate(axis=-1)([x, conv])
    return x

# Define the transition block
def transition_block(x, compression_factor):
    num_filters = int(tf.keras.backend.int_shape(x)[-1] * compression_factor)
    x = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    return x

def DenseNet161(input_shape=(224, 224, 3), classes=1000):
    inputs = Input(shape=input_shape)
    
    # Initial Convolution
    x = Conv2D(96, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Dense Blocks and Transition Blocks
    x = dense_block(x, 6, 48)
    x = transition_block(x, 0.5)
    x = dense_block(x, 12, 48)
    x = transition_block(x, 0.5)
    x = dense_block(x, 36, 48)
    x = transition_block(x, 0.5)
    x = dense_block(x, 24, 48)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Fully Connected Layer
    outputs = Dense(classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

model = DenseNet161()

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
with open('quantized_densenet161.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='quantized_densenet161.tflite')
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
    result_str = f"Average execution time for 8bit-quantized DenseNet-161: {total_time:.4f} ms\n{num_runs} trials completed"
    print(result_str)
    file.write(result_str)

import tensorflow as tf
import time
import sys

#Run with "python 11_VGG.py <number of trials>"

# Configure TensorFlow to run on a single thread
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def VGG11(input_shape=(224, 224, 3), classes=1000):
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # FC layers
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    outputs = Dense(classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

model = VGG11()

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
with open('quantized_vgg11.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='quantized_vgg11.tflite')
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
    result_str = f"Average execution time for 8bit-quantized VGG-11: {total_time:.4f} ms\n{num_runs} trials completed"
    print(result_str)
    file.write(result_str)

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import time
import sys

#Run with "python 50_ResNet.py <number of trials>"

# Configure TensorFlow to run on a single thread
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Load the ResNet-50 model
model = ResNet50(weights='imagenet', include_top=True)

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
with open('quantized_resnet50.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='quantized_resnet50.tflite')
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
    result_str = f"Average execution time for 8bit-quantized ResNet-50: {total_time:.4f} ms\n{num_runs} trials completed"
    print(result_str)
    file.write(result_str)

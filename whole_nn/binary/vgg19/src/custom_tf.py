import tensorflow as tf
import numpy as np
import time

# Create an instance of the ConvLayer
in_channels = 64
out_channels = 128
kernel_size = 3

# Create some random input data
batch_size = 1
input_height = 128
input_width = 128

input_data = np.random.rand(batch_size, input_height, input_width, in_channels).astype(np.float32)

# Define the ConvLayer
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(out_channels, kernel_size, input_shape=(input_height, input_width, in_channels))
])

# Define a generator for the representative dataset
def representative_dataset_gen():
    for _ in range(100):
        # Random input data that matches the input shape and type of the model
        yield [np.random.rand(1, 224, 224, 64).astype(np.float32)]


# Convert the model to a quantized model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # Explicitly set to INT8
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8

# Set the representative dataset
converter.representative_dataset = representative_dataset_gen

quantized_tflite_model = converter.convert()

# Time the execution of the original model
start_time = time.time()
model.predict(input_data)
end_time = time.time()
print(f"Execution time of original model: {end_time - start_time} seconds")

# Time the execution of the quantized model
interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model, num_threads=1)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

start_time = time.time()
interpreter.set_tensor(input_index, input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_index)
end_time = time.time()
print(f"Execution time of quantized model: {end_time - start_time} seconds")


import torch
import torch.nn as nn
import torch.quantization as quantization
import time

# Define a simple convolutional layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.conv(x)

# Create an instance of the ConvLayer
in_channels = 3
out_channels = 128
kernel_size = 3
conv_layer = ConvLayer(in_channels, out_channels, kernel_size)

# Create some random input data
batch_size = 1
input_channels = 3
input_height = 32
input_width = 32
input_data = torch.randn(batch_size, input_channels, input_height, input_width)

# Specify quantization configuration
qconfig = quantization.get_default_qconfig('fbgemm')
conv_layer.qconfig = qconfig

# Prepare the model for static quantization
conv_layer = quantization.prepare(conv_layer, inplace=True)

# Calibrate the model with representative data
conv_layer(input_data)

# Convert the model to quantized version
quantized_model = quantization.convert(conv_layer, inplace=True)

# Time the execution of the original model
start_time = time.time()
output = conv_layer(input_data)
end_time = time.time()
print(f"Execution time of original model: {end_time - start_time} seconds")

# Time the execution of the quantized model
start_time = time.time()
output = quantized_model(input_data)
end_time = time.time()
print(f"Execution time of quantized model: {end_time - start_time} seconds")

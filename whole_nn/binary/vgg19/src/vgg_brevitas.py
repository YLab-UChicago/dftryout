import os
os.environ["OMP_NUM_THREADS"] = "1"

import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantLinear
from brevitas.core.quant import BinaryQuant
import time
from brevitas.core.scaling import ConstScaling

class BinaryVGG(nn.Module):
    def __init__(self):
        super(BinaryVGG, self).__init__()
        self.features = nn.Sequential(
            QuantConv2d(3, 64, kernel_size=3, padding=1, 
                        weight_quant=BinaryQuant(scaling_impl=ConstScaling(1.0)),  
                        weight_bit_width=1),
            nn.ReLU(inplace=True),
            QuantConv2d(64, 64, kernel_size=3, padding=1, 
                        weight_quant=BinaryQuant(scaling_impl=ConstScaling(1.0)),  
                        weight_bit_width=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            QuantConv2d(64, 128, kernel_size=3, padding=1, 
                        weight_quant=BinaryQuant(scaling_impl=ConstScaling(1.0)),  
                        weight_bit_width=1),
            nn.ReLU(inplace=True),
            QuantConv2d(128, 128, kernel_size=3, padding=1, 
                        weight_quant=BinaryQuant(scaling_impl=ConstScaling(1.0)),  
                        weight_bit_width=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),


            QuantConv2d(128, 256, kernel_size=3, padding=1, 
                        weight_quant=BinaryQuant(scaling_impl=ConstScaling(1.0)),  
                        weight_bit_width=1),
            nn.ReLU(inplace=True),
            QuantConv2d(256, 256, kernel_size=3, padding=1, 
                        weight_quant=BinaryQuant(scaling_impl=ConstScaling(1.0)),  
                        weight_bit_width=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            QuantConv2d(256, 512, kernel_size=3, padding=1, 
                        weight_quant=BinaryQuant(scaling_impl=ConstScaling(1.0)),  
                        weight_bit_width=1),
            nn.ReLU(inplace=True),
            QuantConv2d(512, 512, kernel_size=3, padding=1, 
                        weight_quant=BinaryQuant(scaling_impl=ConstScaling(1.0)),  
                        weight_bit_width=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            QuantLinear(256 * 4 * 4, 4096, weight_quant=BinaryQuant(scaling_impl=ConstScaling(1.0))),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            QuantLinear(4096, 4096, weight_quant=BinaryQuant(scaling_impl=ConstScaling(1.0))),
            nn.ReLU(inplace=True),
            QuantLinear(4096, 10, weight_quant=BinaryQuant(scaling_impl=ConstScaling(1.0))),
        )

    def forward(self, x):
        for i, layer in enumerate(self.features):
            start = time.time()
            x = layer(x)
            end = time.time()
            if isinstance(layer, QuantConv2d):
                print(f"Time for conv layer {i}: {end-start} seconds")

        x = x.view(x.size(0), -1)

        for i, layer in enumerate(self.classifier):
            start = time.time()
            x = layer(x)
            end = time.time()
            if isinstance(layer, QuantLinear):
                print(f"Time for fc layer {i}: {end-start} seconds")

        return x

model = BinaryVGG()

# Initialize the model with random weights
for param in model.parameters():
    nn.init.normal_(param, mean=0, std=0.01)

# Create some random input data
input_data = torch.randn(1, 3, 32, 32)

import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
import torch.nn as nn




class BinaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input

def binarize(input):
    return BinaryActivation.apply(input)

class BinaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        binary_input = binarize(input)
        binary_weight = binarize(self.weight)
        output = nn.functional.linear(binary_input, binary_weight, self.bias)
        return output


class VGG_BNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG_BNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # layer 1 
            nn.BatchNorm2d(64),                         # layer 2
            nn.ReLU(inplace=True),                      # layer 3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),# layer 4
            nn.BatchNorm2d(64),                         # layer 5
            nn.ReLU(inplace=True),                      # layer 6
            nn.MaxPool2d(kernel_size=2, stride=2),      # layer 7

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # layer 8 
            nn.BatchNorm2d(128),                          # layer 9
            nn.ReLU(inplace=True),                        # layer 10
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # layer 11 !!
            nn.BatchNorm2d(128),                           # layer 12
            nn.ReLU(inplace=True),                         # layer 13
            nn.MaxPool2d(kernel_size=2, stride=2),         # layer 14

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # layer 15
            nn.BatchNorm2d(256),                            # layer 16
            nn.ReLU(inplace=True),                          # layer 17
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # layer 18 !!
            nn.BatchNorm2d(256),                            # layer 19
            nn.ReLU(inplace=True),                          # layer 20
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # layer 21
            nn.BatchNorm2d(256),                            # layer 22
            nn.ReLU(inplace=True),                          # layer 23
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # layer 24
            nn.BatchNorm2d(256),                            # layer 25
            nn.ReLU(inplace=True),                          # layer 26
            nn.MaxPool2d(kernel_size=2, stride=2),          # layer 27

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # layer 28
            nn.BatchNorm2d(512),                            # layer 29
            nn.ReLU(inplace=True),                          # layer 30
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # layer 31 !!
            nn.BatchNorm2d(512),                            # layer 32
            nn.ReLU(inplace=True),                          # layer 33
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # layer 34
            nn.BatchNorm2d(512),                            # layer 35
            nn.ReLU(inplace=True),                          # layer 36
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # layer 37
            nn.BatchNorm2d(512),                            # layer 38
            nn.ReLU(inplace=True),                          # layer 39
            nn.MaxPool2d(kernel_size=2, stride=2),          # layer 40

            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # layer 41
            nn.BatchNorm2d(512),                            # layer 42
            nn.ReLU(inplace=True),                          # layer 43
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # layer 44
            nn.BatchNorm2d(512),                            # layer 45
            nn.ReLU(inplace=True),                          # layer 46
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # layer 47
            nn.BatchNorm2d(512),                            # layer 48
            nn.ReLU(inplace=True),                          # layer 49
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # layer 50
            nn.BatchNorm2d(512),                            # layer 51
            nn.ReLU(inplace=True),                          # layer 52
            nn.MaxPool2d(kernel_size=2, stride=2),          # layer 53
        )

        self.classifier = nn.Sequential(
            BinaryLinear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            BinaryLinear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        layer_times = []
        for layer in self.features:
            start_time = time.time()
            x = layer(x)
            layer_times.append(time.time() - start_time)

        x = x.view(x.size(0), -1)

        for layer in self.classifier:
            start_time = time.time()
            x = layer(x)
            layer_times.append(time.time() - start_time)

        return x, layer_times



# Instantiate the model
model = VGG_BNN(num_classes=1000)

# Set the model to evaluation mode
model.eval()

# Create a random input tensor
input_tensor = torch.randn(1, 3, 224, 224)

# Perform inference and time each layer
output, layer_times = model(input_tensor)

# Print the time taken by each layer
for i, layer_time in enumerate(layer_times):
    print(f"Layer {i + 1}: {layer_time:.6f} seconds")
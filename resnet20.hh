#ifndef RESNET20_H
#define RESNET20_H

#include <iostream>

using namespace std;
class ConvolutionalLayer {
    public:
        ConvolutionalLayer(int input_channels, int output_channels, int kernel_size); 
        ~ConvolutionalLayer(); 
        int* forward(int* input, int input_width, int input_height);
    private:
        int input_channels;
        int output_channels;
        int kernel_size;
        int* weights;
        int* biases;
};


class ResNet20 {
    public:
        ResNet20();
        ~ResNet20();
        int* Forward(float* input, int input_width, int input_height, int input_channels);
    private:
        ConvolutionalLayer* conv1;
        ConvolutionalLayer* conv2;
        ConvolutionalLayer* conv3;
        ConvolutionalLayer* conv4;
        ConvolutionalLayer* conv5;
        ConvolutionalLayer* conv6;
        ConvolutionalLayer* conv7;
        ConvolutionalLayer* conv8;
    
    // ...
};

class FullyConnectedLayer {
    public:
        FullyConnectedLayer(int input_size, int output_size);
        ~FullyConnectedLayer();
        float* forward(float* input);    
    private:
        int input_size;
        int output_size;
        float* weights;
        float* biases;
};



#endif  // RESNET20_H

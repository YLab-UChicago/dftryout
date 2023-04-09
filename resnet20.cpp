#include <iostream>
#include "resnet20.hh"

using namespace std;


void binarize(float* inputMatrix, int input_size, float threshold, int* binarizedMatrix) {
    for (int i = 0; i < input_size; i++) {
        binarizedMatrix[i] = (int)((unsigned int)inputMatrix[i] >> 31);
    }
}


ResNet20::ResNet20() {
    cout << "initialize ResNet20" << endl;
         // Initialize layers
    conv1 = new ConvolutionalLayer(3, 64, 3);
    conv2 = new ConvolutionalLayer(64, 64, 3);
    conv3 = new ConvolutionalLayer(64, 128, 3);
    conv4 = new ConvolutionalLayer(128, 128, 3);
    conv5 = new ConvolutionalLayer(128, 256, 3);
    conv6 = new ConvolutionalLayer(256, 256, 3);
    conv7 = new ConvolutionalLayer(256, 512, 3);
    conv8 = new ConvolutionalLayer(512, 512, 3);
    // Initialize the weights and biases for the convolutional layers
    // and batch normalization layers
}

ResNet20::~ResNet20() {
    // Deallocate memory
    delete conv1;
    delete conv2;
    delete conv3;
    delete conv4;
    delete conv5;
    delete conv6;
    delete conv7;
    delete conv8;
}   

/* 
 * TODO: edit the input channel to be internally represented by ConvolutionalLayer.
 */
int* ResNet20::Forward(float* input, int width, int height, int input_channels) {
    // Perform the forward pass through the network, applying
    // convolutional layers, batch normalization, and ReLU activation
    // to the input and returning the output
    
    int input_size = width * height * input_channels;
    int* binarized_input = new int[input_size];
    binarize(input, input_size, 0, binarized_input);
    int* output = conv1->forward(binarized_input, width, height);
    output = conv2->forward(output, width - 2, height - 2);
    output = conv3->forward(output, width - 4, height - 4);
    output = conv4->forward(output, width - 6, height - 6);
    output = conv5->forward(output, width - 8, height - 8);
    output = conv6->forward(output, width - 10, height - 10);
    output = conv7->forward(output, width - 12, height - 12);
    output = conv8->forward(output, width - 14, height - 14);
    return output; 
}


FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size) {
    // Initialize weights and biases with random values
    weights = new float[input_size * output_size];
    biases = new float[output_size];
    for (int i = 0; i < input_size * output_size; i++) {
        weights[i] = rand();
    }
    for (int i = 0; i < output_size; i++) {
        biases[i] = rand();
    }
}   

FullyConnectedLayer::~FullyConnectedLayer() {
    // Deallocate memory
    delete[] weights;
    delete[] biases;
}   

float* FullyConnectedLayer::forward(float* input) {
    // Perform matrix multiplication and add biases
    float* output = new float[output_size];
    for (int i = 0; i < output_size; i++) {
        output[i] = biases[i];
        for (int j = 0; j < input_size; j++) {
            output[i] += input[j] * weights[i * input_size + j];
        }
    }
    return output;
}   

ConvolutionalLayer::ConvolutionalLayer(int input_channels, int output_channels, int kernel_size) {
    // Initialize weights and biases with random values
    weights = new int[input_channels * output_channels * kernel_size * kernel_size];
    biases = new int[output_channels];
    for (int i = 0; i < input_channels * output_channels * kernel_size * kernel_size; i++) {
        weights[i] = rand();
        weights[i] = (weights[i] > 0) ? 1 : 0; // binarize the weights
    }
    for (int i = 0; i < output_channels; i++) {
        biases[i] = rand();
        biases[i] = (biases[i] > 0 ) ? 1 : 0; //binarize the biases
    }
}   

ConvolutionalLayer::~ConvolutionalLayer() {
    // Deallocate memory
    delete[] weights;
    delete[] biases;
}   



/*
 * Performs the binary convolution
 * inputs: 
 *      input: a binarized input matrix 
 *      width: the width of the input matrix
 *      heigth: the height of the input matrix
 */
int* ConvolutionalLayer::forward(int* input, int width, int height) {
    int input_size = width * height * input_channels;
    int output_size = (width - kernel_size + 1) * (height - kernel_size + 1) * output_channels;
    int* binarized_matrix = new int[input_size];
    for (int i = 0; i < input_size; i ++) {
        binarized_matrix[i] = input[i];
    }
    // binarize(input, input_size, 1, threshold, binarized_matrix);

    int* output = new int[output_size];
    for (int i = 0; i < output_channels; i++) {
        for (int j = 0; j < height - kernel_size + 1; j++) {
            for (int k = 0; k < width - kernel_size + 1; k++) {
                int output_index = i * (height - kernel_size + 1) * 
                    (width - kernel_size + 1) + j * (width - kernel_size + 1) + k;
                output[output_index] = biases[i];
                for (int m = 0; m < input_channels; m++) {
                    for (int n = 0; n < kernel_size; n++) {
                        for (int p = 0; p < kernel_size; p++) {
                            int input_index = m * height * width + (j + n) * width + (k + p);
                            output[output_index] ^= 
                            binarized_matrix[input_index] &  
                            weights[i * input_channels * kernel_size * kernel_size + m * kernel_size * kernel_size + n * kernel_size + p];
                        }
                    }
                }
            }
        }
    }
    delete[] binarized_matrix;
    return output;
}


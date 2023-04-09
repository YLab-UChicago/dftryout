#include <iostream>
#include <cmath>

using namespace std;

void max_pooling_fc(const float *inputs, int input_width, int input_height, int input_depth, int pool_size, float *output)
{
    int output_width = input_width / pool_size;
    int output_height = input_height / pool_size;
    int output_depth = input_depth;
    
    for (int d = 0; d < output_depth; d++) {
        for (int i = 0; i < output_height; i++) {
            for (int j = 0; j < output_width; j++) {
                float max_val = -1000000.0;
                for (int k = 0; k < pool_size; k++) {
                    for (int l = 0; l < pool_size; l++) {
                        max_val = max(max_val, inputs[d * input_height * input_width + (i * pool_size + k) * input_width + j * pool_size + l]);
                    }
                }
                output[d * output_height * output_width + i * output_width + j] = max_val;
            }
        }
    }
}

void max_pooling_bn(const int *inputs, int input_width, int input_height, int input_depth, int pool_size, int *output)
{
    int output_width = input_width / pool_size;
    int output_height = input_height / pool_size;
    int output_depth = input_depth;
    
    for (int d = 0; d < output_depth; d++) {
        for (int i = 0; i < output_height; i++) {
            for (int j = 0; j < output_width; j++) {
                int curr = 0;
                for (int k = 0; k < pool_size; k++) {
                    for (int l = 0; l < pool_size; l++) {
                        curr = curr | inputs[d * input_height * input_width + (i * pool_size + k) * input_width + j * pool_size + l];
                    }
                }
                output[d * output_height * output_width + i * output_width + j] = curr;
            }
        }
    }
}

void max_pooling_bn(const int *inputs, int input_width, int input_height, int input_depth, int pool_size, int *output)
{
    int output_width = input_width / pool_size;
    int output_height = input_height / pool_size;
    int output_depth = input_depth;
    
    for (int d = 0; d < output_depth; d++) {
        for (int i = 0; i < output_height; i++) {
            for (int j = 0; j < output_width; j++) {
                int curr = 0;
                for (int k = 0; k < pool_size; k++) {
                    for (int l = 0; l < pool_size; l++) {
                        curr = curr | inputs[d * input_height * input_width + (i * pool_size + k) * input_width + j * pool_size + l];
                    }
                }
                output[d * output_height * output_width + i * output_width + j] = curr;
            }
        }
    }
}
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>
#include <iostream>
#include <cstdint>
using namespace std;
void binarize(float* inputMatrix, int input_size, int* binarizedMatrix) {
    for (int i = 0; i < input_size; i++) {
        binarizedMatrix[i] = (int)((unsigned int)inputMatrix[i] >> 31);
    }
}

int xnor_popcount(int a, int b) {
    return __builtin_popcount(~(a^b));
}

int main (int argc, char *argv[]) {
    FILE* pFile = fopen("out/durations/vgg1__15_03_2023_17_39_27.txt", "a");
    int batch_size;
    int height;
    int width;
    int depth;
    int filter_height;
    int filter_width;
    int num_filters;
    int padding;
    int strides;
    int h_block;
    int w_block;
    int f_block;
    int d_block;
    int curr;
    int8_t* inputs;
    int8_t* outputs;
    int8_t* filters;
    
    int output_depth;
    int pool_size;
    
    int input_size;
    float* input_bitrans;
    
    std::clock_t c_start;
    std::clock_t c_end;
    int layer_counter = 0;
    double time_elapsed_ms;
    
    batch_size = 1;
    height = 224;
    width = 224;
    depth = 32;
    num_filters = 32;
    filter_height = 3;
    filter_width = 3;
    padding = 0;
    strides = 1;
    h_block = 1;
    w_block = 1;
    d_block = 1;
    f_block = 1;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    inputs = (int8_t*)malloc(sizeof(int8_t)*batch_size*(height+2*padding)*(width+2*padding)*depth);
    outputs = (int8_t*)malloc(sizeof(int8_t)*batch_size*out_height*out_width*num_filters);
    filters = (int8_t*)malloc(sizeof(int8_t)*batch_size*filter_height*filter_width*num_filters*depth);
    
    c_start = std::clock();
    
    for (int b = 0; b < batch_size; b++) {
        for (int f = 0; f < num_filters; f += f_block) {
            for (int d = 0; d < depth; d += d_block) {
                for (int w = 0; w < out_width; w += w_block) {
                    for (int h = 0; h < out_height; h += h_block) {
                        for (int h_ = 0; h_ < h_block; h_++) {
                            for (int w_ = 0; w_ < w_block; w_++) {
                                for (int f_ = 0; f_ < f_block; f_++) {
                                    for (int d_ = 0; d_ < d_block; d_ ++) {
                                        float sum_block = 0;
                                        for (int i = 0; i < filter_height; i++) {
                                            for (int j = 0; j < filter_width; j++) {
                                                int input_h = (h+h_) * strides + i - padding;
                                                int input_w = (w+w_) * strides + j - padding;
                                                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                                                    sum_block += inputs[b * height * width * depth + input_h * width * depth + input_w * depth + (d+d_)]* filters[(f+f_) * filter_height * filter_width * depth + (d + d_)* filter_height * filter_width + i * filter_width + j];
                                                }
                                            }
                                        }
                                        outputs[b * out_height * out_width * num_filters + (h+h_) * out_width * num_filters + (w+w_) * num_filters + (f+f_)] = sum_block;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::fprintf(pFile, "%lf\n",time_elapsed_ms);
    
    std::free(inputs);
    std::free(outputs);
    std::free(filters);
}
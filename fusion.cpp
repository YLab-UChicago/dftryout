//our implementations go here
#include <iostream>
#include <cmath>
#include "utils.cpp"

using namespace std;


void conv_pooling_fp(const float* inputs, const float *filters, int batch_size, int height, int width, int depth, int num_filters, int filter_height, int filter_width, int padding, int strides, int pool_size, int height_block, int width_block,float *output)
{
    int out_height = (height - filter_height + 2 * padding) / strides + 1;
    int out_width = (width - filter_width + 2 * padding) / strides + 1;
    int out_depth = num_filters;
    int out_pool_height = out_height / pool_size;
    int out_pool_width = out_width / pool_size;
    


    for (int d = 0; d < out_depth; d ++) {
        for (int i = 0; i < out_pool_height; i += height_block) {
            for (int j = 0; j < out_pool_width; j += width_block) { 
                for (int i_ = 0; i_ < height_block; i_ ++) {
                    for (int j_ = 0; j_ < width_block; j_ ++) {
                        float max_val = -1000000.0;
                        for (int k = 0; k < pool_size; k ++) {
                            for (int l = 0; l < pool_size; l ++) {
                                int input_h = (i + i_) * pool_size + k;
                                int input_w = (j + j_) * pool_size + l;
                                if (input_h >= 0 && input_h < out_height && input_w >= 0 && input_w < out_width) {
                                    float sum = 0;
                                    for (int m = 0; m < filter_height; m ++) {
                                        for (int n = 0; n < filter_width; n ++) {
                                            int input_hh = input_h * strides + m - padding;
                                            int input_ww = input_w * strides + n - padding;
                                            if (input_hh >= 0 && input_hh < height && input_ww >= 0 && input_ww < width) {
                                                sum += inputs[input_hh * width * depth + input_ww * depth + d] * filters[d * filter_height * filter_width + m * filter_width + n];
                                            }
                                        }
                                    }
                                    if (sum > max_val) {
                                        max_val = sum;
                                    }
                                }
                            }
                            output[d * out_pool_height * out_pool_width + (i + i_) * out_pool_width + (j+j_)] = max_val;
                        } 
                    }
                    
                }
                
            }
        }

    } 
}

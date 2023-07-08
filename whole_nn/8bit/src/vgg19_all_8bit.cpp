#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>
#include <iostream>
#include <arm_neon.h>
#include <m5ops.h>
#include <algorithm>
using namespace std;


int conv_1_1(int64_t* inputs, int64_t* outputs, int64_t* filters) {
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
    int output_depth;
    
    height = 112;
    width = 112;
    depth = 3;
    num_filters = 64;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    

    for (int f = 0; f < num_filters; f++) {
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w ++) {
                for (int i = 0; i < filter_width; i++) {
                    for (int j = 0; j < filter_height; j ++) {
                        outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output);
                    }
                }
            }
        }
    }
}

int conv_1_2(int64_t* inputs, int64_t* outputs, int64_t* filters) {
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
    int output_depth;
    
    height = 112;
    width = 112;
    depth = 128;
    num_filters = 64;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    
    int64x2_t data1;
    int64x2_t data2;
    int64x2_t input_cache_0;
    int64x2_t input_cache_1;
    int64x2_t input_cache_2;
    int64x2_t input_cache_3;
    int64x2_t input_cache_4;
    int64x2_t input_cache_5;
    
    int64x2_t weight_cache_0;
    int64x2_t weight_cache_1;
    int64x2_t weight_cache_2;
    int64x2_t weight_cache_3;
    int64x2_t weight_cache_4;
    int64x2_t weight_cache_5;
    int64x2_t weight_cache_6;
    int64x2_t weight_cache_7;
    int64x2_t weight_cache_8;
    
    std::clock_t c_start;
    std::clock_t c_end;
    double time_elapsed_ms;
    c_start = std::clock();


    for (int f = 0; f < num_filters; f++) {
        input_cache_0 = vld1q_s64((const int64_t *) &inputs[(0 * width * depth /128 + 0) * 128 /64]);
        input_cache_1 = vld1q_s64((const int64_t *) &inputs[(1 * width * depth /128 + 1) * 128 /64]);
        input_cache_2 = vld1q_s64((const int64_t *) &inputs[(2 * width * depth /128 + 2) * 128 /64]);
        input_cache_3 = vld1q_s64((const int64_t *) &inputs[(3 * width * depth /128 + 3) * 128 /64]);
        input_cache_4 = vld1q_s64((const int64_t *) &inputs[(4 * width * depth /128 + 4) * 128 /64]);
        input_cache_5 = vld1q_s64((const int64_t *) &inputs[(5 * width * depth /128 + 5) * 128 /64]);
        weight_cache_0 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +0)*128/64]);
        weight_cache_1 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +1)*128/64]);
        weight_cache_2 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +2)*128/64]);
        weight_cache_3 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +3)*128/64]);
        weight_cache_4 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +4)*128/64]);
        weight_cache_5 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +5)*128/64]);
        weight_cache_6 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +6)*128/64]);
        weight_cache_7 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +7)*128/64]);
        weight_cache_8 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +8)*128/64]);
        int64x2_t output;
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w ++) {
                int i = 0;
                int j = 0;
                int input_h;
                int input_w;
                 
                i = 0;
                j = 0;
                data1 = veorq_s64(input_cache_0,weight_cache_0);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 0;
                j = 1;
                data1 = veorq_s64(input_cache_1,weight_cache_1);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                input_cache_0 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_0,weight_cache_2);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 0;
                data1 = veorq_s64(input_cache_2,weight_cache_3);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 1;
                data1 = veorq_s64(input_cache_3,weight_cache_4);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                input_cache_2 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_2,weight_cache_5);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 0;
                data1 = veorq_s64(input_cache_4,weight_cache_6);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 1;
                data1 = veorq_s64(input_cache_5,weight_cache_7);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                input_cache_4 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_4,weight_cache_8);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output);
                
                w ++;
                i = 0;
                j = 0;
                data1 = veorq_s64(input_cache_1,weight_cache_0);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 0;
                j = 1;
                data1 = veorq_s64(input_cache_0,weight_cache_1);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                input_cache_1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_1,weight_cache_2);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 0;
                data1 = veorq_s64(input_cache_3,weight_cache_3);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 1;
                data1 = veorq_s64(input_cache_2,weight_cache_4);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                input_cache_3 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_3,weight_cache_5);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 0;
                data1 = veorq_s64(input_cache_5,weight_cache_6);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 1;
                data1 = veorq_s64(input_cache_4,weight_cache_7);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                input_cache_5 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_5,weight_cache_8);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output);
                
            }
        }
    }
}

int maxpool_1(int64_t* inputs, int64_t* outputs) {
    return
}

int conv_2_1(int64_t* inputs, int64_t* outputs, int64_t* filters){
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
    int output_depth;
    
    height = 112;
    width = 112;
    depth = 64;
    num_filters = 128;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    
    int64x2_t data1;
    int64x2_t data2;
    int64x2_t input_cache_0;
    int64x2_t input_cache_1;
    int64x2_t input_cache_2;
    int64x2_t input_cache_3;
    int64x2_t input_cache_4;
    int64x2_t input_cache_5;
    
    int64x2_t weight_cache_0;
    int64x2_t weight_cache_1;
    int64x2_t weight_cache_2;
    int64x2_t weight_cache_3;
    int64x2_t weight_cache_4;
    int64x2_t weight_cache_5;
    int64x2_t weight_cache_6;
    int64x2_t weight_cache_7;
    int64x2_t weight_cache_8;
    
    std::clock_t c_start;
    std::clock_t c_end;
    double time_elapsed_ms;
    c_start = std::clock();


    for (int f = 0; f < num_filters; f++) {
        input_cache_0 = vld1q_s64((const int64_t *) &inputs[(0 * width * depth /128 + 0) * 128 /64]);
        input_cache_1 = vld1q_s64((const int64_t *) &inputs[(1 * width * depth /128 + 1) * 128 /64]);
        input_cache_2 = vld1q_s64((const int64_t *) &inputs[(2 * width * depth /128 + 2) * 128 /64]);
        input_cache_3 = vld1q_s64((const int64_t *) &inputs[(3 * width * depth /128 + 3) * 128 /64]);
        input_cache_4 = vld1q_s64((const int64_t *) &inputs[(4 * width * depth /128 + 4) * 128 /64]);
        input_cache_5 = vld1q_s64((const int64_t *) &inputs[(5 * width * depth /128 + 5) * 128 /64]);
        weight_cache_0 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +0)*128/64]);
        weight_cache_1 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +1)*128/64]);
        weight_cache_2 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +2)*128/64]);
        weight_cache_3 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +3)*128/64]);
        weight_cache_4 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +4)*128/64]);
        weight_cache_5 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +5)*128/64]);
        weight_cache_6 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +6)*128/64]);
        weight_cache_7 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +7)*128/64]);
        weight_cache_8 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +8)*128/64]);
        int64x2_t output;
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w ++) {
                int i = 0;
                int j = 0;
                int input_h;
                int input_w;
                 
                i = 0;
                j = 0;
                data1 = veorq_s64(input_cache_0,weight_cache_0);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 0;
                j = 1;
                data1 = veorq_s64(input_cache_1,weight_cache_1);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                input_cache_0 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_0,weight_cache_2);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 0;
                data1 = veorq_s64(input_cache_2,weight_cache_3);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 1;
                data1 = veorq_s64(input_cache_3,weight_cache_4);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                input_cache_2 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_2,weight_cache_5);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 0;
                data1 = veorq_s64(input_cache_4,weight_cache_6);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 1;
                data1 = veorq_s64(input_cache_5,weight_cache_7);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                input_cache_4 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_4,weight_cache_8);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output);
                
                w ++;
                i = 0;
                j = 0;
                data1 = veorq_s64(input_cache_1,weight_cache_0);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 0;
                j = 1;
                data1 = veorq_s64(input_cache_0,weight_cache_1);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                input_cache_1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_1,weight_cache_2);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 0;
                data1 = veorq_s64(input_cache_3,weight_cache_3);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 1;
                data1 = veorq_s64(input_cache_2,weight_cache_4);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                input_cache_3 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_3,weight_cache_5);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 0;
                data1 = veorq_s64(input_cache_5,weight_cache_6);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 1;
                data1 = veorq_s64(input_cache_4,weight_cache_7);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                input_cache_5 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_5,weight_cache_8);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output);
                
            }
        }
    }

}


int conv_2_2(int64_t* inputs, int64_t* outputs, int64_t* filters){
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
    int output_depth;
    
    height = 112;
    width = 112;
    depth = 128;
    num_filters = 128;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    
    int64x2_t data1;
    int64x2_t data2;
    int64x2_t input_cache_0;
    int64x2_t input_cache_1;
    int64x2_t input_cache_2;
    int64x2_t input_cache_3;
    int64x2_t input_cache_4;
    int64x2_t input_cache_5;
    
    int64x2_t weight_cache_0;
    int64x2_t weight_cache_1;
    int64x2_t weight_cache_2;
    int64x2_t weight_cache_3;
    int64x2_t weight_cache_4;
    int64x2_t weight_cache_5;
    int64x2_t weight_cache_6;
    int64x2_t weight_cache_7;
    int64x2_t weight_cache_8;
    
    std::clock_t c_start;
    std::clock_t c_end;
    double time_elapsed_ms;
    c_start = std::clock();


    // Convolution Layer 1
    // Filter = 3x3x256
    // Input Feature Map = 224x224x64
    // 

    // Convolution Layer 2
    // Filter = 3x3x256
    // Input Feature Map = 112x112x128
    // Stride = 1
    for (int f = 0; f < num_filters; f++) {
        input_cache_0 = vld1q_s64((const int64_t *) &inputs[(0 * width * depth /128 + 0) * 128 /64]);
        input_cache_1 = vld1q_s64((const int64_t *) &inputs[(1 * width * depth /128 + 1) * 128 /64]);
        input_cache_2 = vld1q_s64((const int64_t *) &inputs[(2 * width * depth /128 + 2) * 128 /64]);
        input_cache_3 = vld1q_s64((const int64_t *) &inputs[(3 * width * depth /128 + 3) * 128 /64]);
        input_cache_4 = vld1q_s64((const int64_t *) &inputs[(4 * width * depth /128 + 4) * 128 /64]);
        input_cache_5 = vld1q_s64((const int64_t *) &inputs[(5 * width * depth /128 + 5) * 128 /64]);
        weight_cache_0 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +0)*128/64]);
        weight_cache_1 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +1)*128/64]);
        weight_cache_2 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +2)*128/64]);
        weight_cache_3 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +3)*128/64]);
        weight_cache_4 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +4)*128/64]);
        weight_cache_5 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +5)*128/64]);
        weight_cache_6 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +6)*128/64]);
        weight_cache_7 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +7)*128/64]);
        weight_cache_8 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +8)*128/64]);
        int64x2_t output;
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w ++) {
                int i = 0;
                int j = 0;
                int input_h;
                int input_w;
                 
                i = 0;
                j = 0;
                data1 = veorq_s64(input_cache_0,weight_cache_0);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 0;
                j = 1;
                data1 = veorq_s64(input_cache_1,weight_cache_1);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                input_cache_0 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_0,weight_cache_2);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 0;
                data1 = veorq_s64(input_cache_2,weight_cache_3);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 1;
                data1 = veorq_s64(input_cache_3,weight_cache_4);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                input_cache_2 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_2,weight_cache_5);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 0;
                data1 = veorq_s64(input_cache_4,weight_cache_6);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 1;
                data1 = veorq_s64(input_cache_5,weight_cache_7);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                input_cache_4 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_4,weight_cache_8);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output);
                
                w ++;
                i = 0;
                j = 0;
                data1 = veorq_s64(input_cache_1,weight_cache_0);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 0;
                j = 1;
                data1 = veorq_s64(input_cache_0,weight_cache_1);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                input_cache_1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_1,weight_cache_2);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 0;
                data1 = veorq_s64(input_cache_3,weight_cache_3);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 1;
                data1 = veorq_s64(input_cache_2,weight_cache_4);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                input_cache_3 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_3,weight_cache_5);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 0;
                data1 = veorq_s64(input_cache_5,weight_cache_6);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 1;
                data1 = veorq_s64(input_cache_4,weight_cache_7);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                input_cache_5 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1 = veorq_s64(input_cache_5,weight_cache_8);
                output = vaddq_u8(output,vcntq_u8(data1));
                
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output);
                
            }
        }
    }

}

int maxpool_2(int64_t* inputs, int64_t* outputs) {
    for (int i = 0; i < 112; i += 2) {
        for (int j = 0; i < 112; i += 2) {
            int32x4_t max_val = vdupq_n_s32(0);
            for (int p = i; p < i + 2; p++) {
                for (int q = j; q < j + 2; q ++) {
                    int32x4_t vec = vld1q_s32(&inputs[])
                })
            }
        }
    }
    return
}

int conv_3_1(int64_t* inputs, int64_t* outputs, int64_t* filters) {
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
    int64_t* inputs;
    int64_t* outputs;
    int64_t* filters;
    int output_depth;
    
    height = 56;
    width = 56;
    depth = 256;
    num_filters = 256;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);
    
    int64x2x2_t data1;
    int64x2x2_t data2;
    int64x2x2_t input_cache_0;
    int64x2x2_t input_cache_1;
    int64x2x2_t input_cache_2;
    int64x2x2_t input_cache_3;
    
    int64x2x2_t weight_cache_0;
    int64x2x2_t weight_cache_1;
    int64x2x2_t weight_cache_2;
    int64x2x2_t weight_cache_3;
    int64x2x2_t weight_cache_4;
    int64x2x2_t weight_cache_5;
    int64x2x2_t weight_cache_6;
    int64x2x2_t weight_cache_7;
    int64x2x2_t weight_cache_8;
    
    for (int f = 0; f < num_filters; f++) {
        input_cache_0 = vld1q_s64_x2((const int64_t *) &inputs[(0 * width * depth /256 + 0) * 256 /64]);
        input_cache_1 = vld1q_s64_x2((const int64_t *) &inputs[(1 * width * depth /256 + 1) * 256 /64]);
        input_cache_2 = vld1q_s64_x2((const int64_t *) &inputs[(2 * width * depth /256 + 2) * 256 /64]);
        input_cache_3 = vld1q_s64_x2((const int64_t *) &inputs[(3 * width * depth /256 + 3) * 256 /64]);
        weight_cache_0 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +0)*256/64]);
        weight_cache_1 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +1)*256/64]);
        weight_cache_2 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +2)*256/64]);
        weight_cache_3 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +3)*256/64]);
        weight_cache_4 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +4)*256/64]);
        weight_cache_5 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +5)*256/64]);
        weight_cache_6 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +6)*256/64]);
        weight_cache_7 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +7)*256/64]);
        weight_cache_8 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +8)*256/64]);
        int64x2x2_t output;
        
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w ++) {
                int i = 0;
                int j = 0;
                int input_h;
                int input_w;
                 
                i = 0;
                j = 0;
                data1.val[0] = veorq_s64(input_cache_0.val[0],weight_cache_0.val[0]);
                data1.val[1] = veorq_s64(input_cache_0.val[1],weight_cache_0.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 0;
                j = 1;
                data1.val[0] = veorq_s64(input_cache_1.val[0],weight_cache_1.val[0]);
                data1.val[1] = veorq_s64(input_cache_1.val[1],weight_cache_1.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                input_cache_0 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(input_cache_0.val[0],weight_cache_2.val[0]);
                data1.val[1] = veorq_s64(input_cache_0.val[1],weight_cache_2.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 0;
                data1.val[0] = veorq_s64(input_cache_2.val[0],weight_cache_3.val[0]);
                data1.val[1] = veorq_s64(input_cache_2.val[1],weight_cache_3.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 1;
                data1.val[0] = veorq_s64(input_cache_3.val[0],weight_cache_4.val[0]);
                data1.val[1] = veorq_s64(input_cache_3.val[1],weight_cache_4.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                input_cache_2 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(input_cache_2.val[0],weight_cache_5.val[0]);
                data1.val[1] = veorq_s64(input_cache_2.val[1],weight_cache_5.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 0;
                input_h = h * strides +2;
                input_w = w * strides +0;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_6.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_6.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 1;
                input_h = h * strides +2;
                input_w = w * strides +1;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_7.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_7.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_8.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_8.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output.val[0]) + vaddvq_u8(output.val[1]);
                
            
                w ++;
                i = 0;
                j = 0;
                data1.val[0] = veorq_s64(input_cache_1.val[0],weight_cache_0.val[0]);
                data1.val[1] = veorq_s64(input_cache_1.val[1],weight_cache_0.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 0;
                j = 1;
                data1.val[0] = veorq_s64(input_cache_0.val[0],weight_cache_1.val[0]);
                data1.val[1] = veorq_s64(input_cache_0.val[1],weight_cache_1.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                input_cache_1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(input_cache_1.val[0],weight_cache_2.val[0]);
                data1.val[1] = veorq_s64(input_cache_1.val[1],weight_cache_2.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 0;
                data1.val[0] = veorq_s64(input_cache_3.val[0],weight_cache_3.val[0]);
                data1.val[1] = veorq_s64(input_cache_3.val[1],weight_cache_3.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 1;
                data1.val[0] = veorq_s64(input_cache_2.val[0],weight_cache_4.val[0]);
                data1.val[1] = veorq_s64(input_cache_2.val[1],weight_cache_4.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                input_cache_3 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(input_cache_3.val[0],weight_cache_5.val[0]);
                data1.val[1] = veorq_s64(input_cache_3.val[1],weight_cache_5.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 0;
                input_h = h * strides +2;
                input_w = w * strides +0;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_6.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_6.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 1;
                input_h = h * strides +2;
                input_w = w * strides +1;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_7.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_7.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_8.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_8.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output.val[0]) + vaddvq_u8(output.val[1]);
                
            }
        }
    }
}

int conv_3_2(int64_t* inputs, int64_t* outputs, int64_t* filters) {
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
    int64_t* inputs;
    int64_t* outputs;
    int64_t* filters;
    int output_depth;
    
    height = 56;
    width = 56;
    depth = 256;
    num_filters = 256;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);
    
    int64x2x2_t data1;
    int64x2x2_t data2;
    int64x2x2_t input_cache_0;
    int64x2x2_t input_cache_1;
    int64x2x2_t input_cache_2;
    int64x2x2_t input_cache_3;
    
    int64x2x2_t weight_cache_0;
    int64x2x2_t weight_cache_1;
    int64x2x2_t weight_cache_2;
    int64x2x2_t weight_cache_3;
    int64x2x2_t weight_cache_4;
    int64x2x2_t weight_cache_5;
    int64x2x2_t weight_cache_6;
    int64x2x2_t weight_cache_7;
    int64x2x2_t weight_cache_8;
    
    for (int f = 0; f < num_filters; f++) {
        input_cache_0 = vld1q_s64_x2((const int64_t *) &inputs[(0 * width * depth /256 + 0) * 256 /64]);
        input_cache_1 = vld1q_s64_x2((const int64_t *) &inputs[(1 * width * depth /256 + 1) * 256 /64]);
        input_cache_2 = vld1q_s64_x2((const int64_t *) &inputs[(2 * width * depth /256 + 2) * 256 /64]);
        input_cache_3 = vld1q_s64_x2((const int64_t *) &inputs[(3 * width * depth /256 + 3) * 256 /64]);
        weight_cache_0 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +0)*256/64]);
        weight_cache_1 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +1)*256/64]);
        weight_cache_2 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +2)*256/64]);
        weight_cache_3 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +3)*256/64]);
        weight_cache_4 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +4)*256/64]);
        weight_cache_5 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +5)*256/64]);
        weight_cache_6 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +6)*256/64]);
        weight_cache_7 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +7)*256/64]);
        weight_cache_8 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +8)*256/64]);
        int64x2x2_t output;
        
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w ++) {
                int i = 0;
                int j = 0;
                int input_h;
                int input_w;
                 
                i = 0;
                j = 0;
                data1.val[0] = veorq_s64(input_cache_0.val[0],weight_cache_0.val[0]);
                data1.val[1] = veorq_s64(input_cache_0.val[1],weight_cache_0.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 0;
                j = 1;
                data1.val[0] = veorq_s64(input_cache_1.val[0],weight_cache_1.val[0]);
                data1.val[1] = veorq_s64(input_cache_1.val[1],weight_cache_1.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                input_cache_0 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(input_cache_0.val[0],weight_cache_2.val[0]);
                data1.val[1] = veorq_s64(input_cache_0.val[1],weight_cache_2.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 0;
                data1.val[0] = veorq_s64(input_cache_2.val[0],weight_cache_3.val[0]);
                data1.val[1] = veorq_s64(input_cache_2.val[1],weight_cache_3.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 1;
                data1.val[0] = veorq_s64(input_cache_3.val[0],weight_cache_4.val[0]);
                data1.val[1] = veorq_s64(input_cache_3.val[1],weight_cache_4.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                input_cache_2 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(input_cache_2.val[0],weight_cache_5.val[0]);
                data1.val[1] = veorq_s64(input_cache_2.val[1],weight_cache_5.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 0;
                input_h = h * strides +2;
                input_w = w * strides +0;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_6.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_6.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 1;
                input_h = h * strides +2;
                input_w = w * strides +1;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_7.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_7.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_8.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_8.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output.val[0]) + vaddvq_u8(output.val[1]);
                
            
                w ++;
                i = 0;
                j = 0;
                data1.val[0] = veorq_s64(input_cache_1.val[0],weight_cache_0.val[0]);
                data1.val[1] = veorq_s64(input_cache_1.val[1],weight_cache_0.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 0;
                j = 1;
                data1.val[0] = veorq_s64(input_cache_0.val[0],weight_cache_1.val[0]);
                data1.val[1] = veorq_s64(input_cache_0.val[1],weight_cache_1.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                input_cache_1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(input_cache_1.val[0],weight_cache_2.val[0]);
                data1.val[1] = veorq_s64(input_cache_1.val[1],weight_cache_2.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 0;
                data1.val[0] = veorq_s64(input_cache_3.val[0],weight_cache_3.val[0]);
                data1.val[1] = veorq_s64(input_cache_3.val[1],weight_cache_3.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 1;
                data1.val[0] = veorq_s64(input_cache_2.val[0],weight_cache_4.val[0]);
                data1.val[1] = veorq_s64(input_cache_2.val[1],weight_cache_4.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                input_cache_3 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(input_cache_3.val[0],weight_cache_5.val[0]);
                data1.val[1] = veorq_s64(input_cache_3.val[1],weight_cache_5.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 0;
                input_h = h * strides +2;
                input_w = w * strides +0;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_6.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_6.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 1;
                input_h = h * strides +2;
                input_w = w * strides +1;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_7.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_7.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_8.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_8.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output.val[0]) + vaddvq_u8(output.val[1]);
                
            }
        }
    }
}

int conv_3_3(int64_t* inputs, int64_t* outputs, int64_t* filters) {
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
    int64_t* inputs;
    int64_t* outputs;
    int64_t* filters;
    int output_depth;
    
    height = 56;
    width = 56;
    depth = 256;
    num_filters = 512;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);
    
    int64x2x2_t data1;
    int64x2x2_t data2;
    int64x2x2_t input_cache_0;
    int64x2x2_t input_cache_1;
    int64x2x2_t input_cache_2;
    int64x2x2_t input_cache_3;
    
    int64x2x2_t weight_cache_0;
    int64x2x2_t weight_cache_1;
    int64x2x2_t weight_cache_2;
    int64x2x2_t weight_cache_3;
    int64x2x2_t weight_cache_4;
    int64x2x2_t weight_cache_5;
    int64x2x2_t weight_cache_6;
    int64x2x2_t weight_cache_7;
    int64x2x2_t weight_cache_8;
    
    for (int f = 0; f < num_filters; f++) {
        input_cache_0 = vld1q_s64_x2((const int64_t *) &inputs[(0 * width * depth /256 + 0) * 256 /64]);
        input_cache_1 = vld1q_s64_x2((const int64_t *) &inputs[(1 * width * depth /256 + 1) * 256 /64]);
        input_cache_2 = vld1q_s64_x2((const int64_t *) &inputs[(2 * width * depth /256 + 2) * 256 /64]);
        input_cache_3 = vld1q_s64_x2((const int64_t *) &inputs[(3 * width * depth /256 + 3) * 256 /64]);
        weight_cache_0 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +0)*256/64]);
        weight_cache_1 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +1)*256/64]);
        weight_cache_2 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +2)*256/64]);
        weight_cache_3 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +3)*256/64]);
        weight_cache_4 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +4)*256/64]);
        weight_cache_5 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +5)*256/64]);
        weight_cache_6 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +6)*256/64]);
        weight_cache_7 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +7)*256/64]);
        weight_cache_8 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +8)*256/64]);
        int64x2x2_t output;
        
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w ++) {
                int i = 0;
                int j = 0;
                int input_h;
                int input_w;
                 
                i = 0;
                j = 0;
                data1.val[0] = veorq_s64(input_cache_0.val[0],weight_cache_0.val[0]);
                data1.val[1] = veorq_s64(input_cache_0.val[1],weight_cache_0.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 0;
                j = 1;
                data1.val[0] = veorq_s64(input_cache_1.val[0],weight_cache_1.val[0]);
                data1.val[1] = veorq_s64(input_cache_1.val[1],weight_cache_1.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                input_cache_0 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(input_cache_0.val[0],weight_cache_2.val[0]);
                data1.val[1] = veorq_s64(input_cache_0.val[1],weight_cache_2.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 0;
                data1.val[0] = veorq_s64(input_cache_2.val[0],weight_cache_3.val[0]);
                data1.val[1] = veorq_s64(input_cache_2.val[1],weight_cache_3.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 1;
                data1.val[0] = veorq_s64(input_cache_3.val[0],weight_cache_4.val[0]);
                data1.val[1] = veorq_s64(input_cache_3.val[1],weight_cache_4.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                input_cache_2 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(input_cache_2.val[0],weight_cache_5.val[0]);
                data1.val[1] = veorq_s64(input_cache_2.val[1],weight_cache_5.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 0;
                input_h = h * strides +2;
                input_w = w * strides +0;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_6.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_6.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 1;
                input_h = h * strides +2;
                input_w = w * strides +1;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_7.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_7.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_8.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_8.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output.val[0]) + vaddvq_u8(output.val[1]);
                
            
                w ++;
                i = 0;
                j = 0;
                data1.val[0] = veorq_s64(input_cache_1.val[0],weight_cache_0.val[0]);
                data1.val[1] = veorq_s64(input_cache_1.val[1],weight_cache_0.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 0;
                j = 1;
                data1.val[0] = veorq_s64(input_cache_0.val[0],weight_cache_1.val[0]);
                data1.val[1] = veorq_s64(input_cache_0.val[1],weight_cache_1.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                input_cache_1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(input_cache_1.val[0],weight_cache_2.val[0]);
                data1.val[1] = veorq_s64(input_cache_1.val[1],weight_cache_2.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 0;
                data1.val[0] = veorq_s64(input_cache_3.val[0],weight_cache_3.val[0]);
                data1.val[1] = veorq_s64(input_cache_3.val[1],weight_cache_3.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 1;
                data1.val[0] = veorq_s64(input_cache_2.val[0],weight_cache_4.val[0]);
                data1.val[1] = veorq_s64(input_cache_2.val[1],weight_cache_4.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                input_cache_3 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(input_cache_3.val[0],weight_cache_5.val[0]);
                data1.val[1] = veorq_s64(input_cache_3.val[1],weight_cache_5.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 0;
                input_h = h * strides +2;
                input_w = w * strides +0;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_6.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_6.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 1;
                input_h = h * strides +2;
                input_w = w * strides +1;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_7.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_7.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_s64(data1.val[0],weight_cache_8.val[0]);
                data1.val[1] = veorq_s64(data1.val[1],weight_cache_8.val[1]);
                output.val[0] = vaddq_u8(output.val[0],vcntq_u8(data1.val[0]));
                output.val[1] = vaddq_u8(output.val[1],vcntq_u8(data1.val[1]));
                
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output.val[0]) + vaddvq_u8(output.val[1]);
                
            }
        }
    }
}

int maxpool_3(int64_t* inputs, int64_t* outputs) {
    return
}


int conv_4_1(int64_t* inputs, int64_t* outputs, int64_t* filters) {
    return
}

int conv_4_2(int64_t* inputs, int64_t* outputs, int64_t* filters) {
    return
}

int conv_4_3(int64_t* inputs, int64_t* outputs, int64_t* filters) {
    return
}

int maxpool_4(int64_t* inputs, int64_t* outputs) {
    return
}

int conv_5_1(int64_t* inputs, int64_t* outputs, int64_t* filters) {
    return
}

int conv_5_2(int64_t* inputs, int64_t* outputs, int64_t* filters) {
    return
}

int conv_5_3(int64_t* inputs, int64_t* outputs, int64_t* filters) {
    return
}

int maxpool_5(int64_t* inputs, int64_t* outputs) {
    return
}

int fc_4096(int64_t* inputs, int64_t* outputs, int64_t* weights) {
    return
}

int fc_1000(int64_t* inputs, int64_t* outputs, int64_t* weights) {
    return
}

int main (int argc, char *argv[]) {
    FILE *pFile = fopen("durations/vgg19_conv2_ext.txt", "a");
    int height;
    int width;
    int depth;
    int num_filters;
    int padding;
    int strides;
    int64_t* inputs;
    int64_t* outputs;
    int64_t* filters;
    int out_height;
    int out_width;
    

    std::clock_t c_start;
    std::clock_t c_end;
    double time_elapsed_ms;
    c_start = std::clock();


    // Convolution Layer 1_1
    // Filter = 3x3x64
    // Input Feature Map = 224x224x3
    height = 224;
    width = 224;
    depth = 64;
    num_filters = 64;
    padding = 2;
    strides = 1;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);

    conv_1_1(inputs, outputs, filters);

    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    std::fprintf(pFile, "Layer 1_1: %lf\n", time_elapsed_ms);
    std::free(inputs);
    std::free(filters);
    inputs = outputs;

    // Convolution Layer 1_2
    // Filter = 3x3x64
    // Input Feature Map = 224x224x64
    height = 224;
    width = 224;
    depth = 64;
    num_filters = 64;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);

    conv_1_2(inputs, outputs, filters);

    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    std::fprintf(pFile, "Layer 1_2: %lf\n", time_elapsed_ms);
    std::free(inputs);
    std::free(outputs);
    inputs = outputs;
    

    // Maximum Pooling Layer 1
    height = 224;
    width = 224;
    depth = 64;
    filter_height = 2;
    filter_width = 2;
    padding = 1;
    stride = 2;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);

    maxpool_1(inputs, outputs);

    std::free(inputs);
    inputs = outputs;


    // Convolution Layer 2_1
    // Filter = 3x3x256
    // Input Feature Map = 112x112x128
    // Stride = 1

    height = 112;
    width = 112;
    depth = 64;
    num_filters = 128;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);

    conv_2_2(inputs, outputs, filters);

    std::free(inputs);
    std::free(filters);
    inputs = outputs;

    // Convolution Layer 2_2
    // Filter = 3x3x128
    // Input Feature Map = 112x112x128
    // Stride = 1

    height = 112;
    width = 112;
    depth = 128;
    num_filters = 128;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);

    conv_2_2(inputs, outputs, filters);

    std::free(inputs);
    std::free(filters);
    inputs = outputs;

    // Maximum Pooling Layer 2
    height = 112;
    width = 112;
    depth = 128;
    filter_height = 2;
    filter_width = 2;
    padding = 1;
    stride = 2;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);

    maxpool_2(inputs, outputs);

    std::free(inputs);
    inputs = outputs;


    // Convolution Layer 3_1
    // Filter = 3x3x256
    // Input Feature Map = 56x56x128
    // Stride = 1

    height = 56;
    width = 56;
    depth = 128;
    num_filters = 256;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);

    conv_3_1(inputs, outputs, filters);

    std::free(inputs);
    std::free(filters);
    inputs = outputs;

    // Convolution Layer 3_2
    // Filter = 3x3x256
    // Input Feature Map = 56x56x256
    // Stride = 1

    height = 56;
    width = 56;
    depth = 256;
    num_filters = 256;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);

    conv_3_2(inputs, outputs, filters);

    std::free(inputs);
    std::free(filters);
    inputs = outputs;

    // Convolution Layer 3_3
    // Filter = 3x3x256
    // Input Feature Map = 56x56x256
    // Stride = 1

    height = 56;
    width = 56;
    depth = 256;
    num_filters = 256;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);

    conv_3_3(inputs, outputs, filters);

    std::free(inputs);
    std::free(filters);
    inputs = outputs;

    // Maximum Pooling Layer 3
    height = 56;
    width = 56;
    depth = 256;
    filter_height = 2;
    filter_width = 2;
    padding = 1;
    stride = 2;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);

    maxpool_3(inputs, outputs);

    std::free(inputs);
    inputs = outputs;

    // Convolution Layer 4_1
    // Filter = 3x3x512
    // Input Feature Map = 28x28x256
    // Stride = 1

    height = 28;
    width = 28;
    depth = 256;
    num_filters = 512;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);

    conv_4_1(inputs, outputs, filters);

    std::free(inputs);
    std::free(filters);
    inputs = outputs;

    // Convolution Layer 4_2
    // Filter = 3x3x512
    // Input Feature Map = 28x28x512
    // Stride = 1

    height = 28;
    width = 28;
    depth = 512;
    num_filters = 512;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);

    conv_4_2(inputs, outputs, filters);

    std::free(inputs);
    std::free(filters);
    inputs = outputs;

    // Convolution Layer 4_3
    // Filter = 3x3x512
    // Input Feature Map = 28x28x512
    // Stride = 1

    height = 28;
    width = 28;
    depth = 512;
    num_filters = 512;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);

    conv_4_3(inputs, outputs, filters);

    std::free(inputs);
    std::free(filters);
    inputs = outputs;

    // Maximum Pooling Layer 4
    height = 28;
    width = 28;
    depth = 256;
    filter_height = 2;
    filter_width = 2;
    padding = 1;
    stride = 2;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);

    maxpool_4(inputs, outputs);

    std::free(inputs);
    inputs = outputs;

    // Convolution Layer 5_1
    // Filter = 3x3x512
    // Input Feature Map = 28x28x512
    // Stride = 1

    height = 14;
    width = 14;
    depth = 512;
    num_filters = 512;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);

    conv_5_1(inputs, outputs, filters);

    std::free(inputs);
    std::free(filters);
    inputs = outputs;

    // Convolution Layer 5_2
    // Filter = 3x3x512
    // Input Feature Map = 14x14x512
    // Stride = 1

    height = 14;
    width = 14;
    depth = 512;
    num_filters = 512;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);

    conv_5_2(inputs, outputs, filters);

    std::free(inputs);
    std::free(filters);
    inputs = outputs;

    // Convolution Layer 5_3
    // Filter = 3x3x512
    // Input Feature Map = 14x14x512
    // Stride = 1

    height = 14;
    width = 14;
    depth = 512;
    num_filters = 512;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);

    conv_5_3(inputs, outputs, filters);

    std::free(inputs);
    std::free(filters);
    inputs = outputs;

    // Maximum Pooling Layer 5
    height = 14;
    width = 14;
    depth = 512;
    filter_height = 2;
    filter_width = 2;
    padding = 1;
    stride = 2;
    out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);

    maxpool_5(inputs, outputs);

    std::free(inputs);
    inputs = outputs;

    // Fully connected layer 1


}
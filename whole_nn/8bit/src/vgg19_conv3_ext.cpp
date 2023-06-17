#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>
#include <iostream>
#include <arm_neon.h>
#include <m5ops.h>
#include <algorithm>
using namespace std;



int main (int argc, char *argv[]) {
    FILE *pFile = fopen("durations/vgg19_conv3_ext.txt", "a");
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
    int vec_d;
    int64_t* inputs;
    short* outputs;
    int64_t* filters;
    int output_depth;
    std::clock_t c_start;
    std::clock_t c_end;
    double time_elapsed_ms;
    
    height = 56;
    width = 56;
    depth = 512;
    num_filters = 256;
    vec_d = 4;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64 * vec_d);
    outputs = (short *)malloc(sizeof(short) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64 * vec_d);
    
    int64x2x4_t data1;
    int64x2x4_t data2;
    
    int64x2x4_t weight_cache_0;
    int64x2x4_t weight_cache_1;
    int64x2x4_t weight_cache_2;
    int64x2x4_t weight_cache_3;
    int64x2x4_t weight_cache_4;
    
    c_start = std::clock();
    
    
    for (int f = 0; f < num_filters; f++) {
        for (int v = 0; v < vec_d; v++ ) {
        weight_cache_0 = vld1q_s64_x4((const int64_t*) &filters[(f * filter_height * filter_width * vec_d + v * filter_height + filter_width + 0)*depth/64]);
        weight_cache_1 = vld1q_s64_x4((const int64_t*) &filters[(f * filter_height * filter_width * vec_d + v * filter_height + filter_width + 1)*depth/64]);
        weight_cache_2 = vld1q_s64_x4((const int64_t*) &filters[(f * filter_height * filter_width * vec_d + v * filter_height + filter_width + 2)*depth/64]);
        weight_cache_3 = vld1q_s64_x4((const int64_t*) &filters[(f * filter_height * filter_width * vec_d + v * filter_height + filter_width + 3)*depth/64]);
        weight_cache_4 = vld1q_s64_x4((const int64_t*) &filters[(f * filter_height * filter_width * vec_d + v * filter_height + filter_width + 4)*depth/64]);
        int64x2x4_t output;

        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w ++) {
                int i = 0;
                int j = 0;
                int input_h;
                int input_w;
                 
                i = 0;
                j = 0;
                input_h = h * strides +0;
                input_w = w * strides +0;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data1.val[0] = vmulq_s8(data1.val[0],weight_cache_0.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],weight_cache_0.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],weight_cache_0.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],weight_cache_0.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                i = 0;
                j = 1;
                input_h = h * strides +0;
                input_w = w * strides +1;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data1.val[0] = vmulq_s8(data1.val[0],weight_cache_1.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],weight_cache_1.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],weight_cache_1.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],weight_cache_1.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data1.val[0] = vmulq_s8(data1.val[0],weight_cache_2.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],weight_cache_2.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],weight_cache_2.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],weight_cache_2.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                i = 1;
                j = 0;
                input_h = h * strides +1;
                input_w = w * strides +0;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data1.val[0] = vmulq_s8(data1.val[0],weight_cache_3.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],weight_cache_3.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],weight_cache_3.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],weight_cache_3.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                i = 1;
                j = 1;
                input_h = h * strides +1;
                input_w = w * strides +1;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data1.val[0] = vmulq_s8(data1.val[0],weight_cache_4.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],weight_cache_4.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],weight_cache_4.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],weight_cache_4.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data2 = vld1q_s64_x4((const int64_t *) &filters[(f * filter_height * filter_width * vec_d + v * filter_height + filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                i = 2;
                j = 0;
                input_h = h * strides +2;
                input_w = w * strides +0;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data2 = vld1q_s64_x4((const int64_t *) &filters[(f * filter_height * filter_width * vec_d + v * filter_height + filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                i = 2;
                j = 1;
                input_h = h * strides +2;
                input_w = w * strides +1;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data2 = vld1q_s64_x4((const int64_t *) &filters[(f * filter_height * filter_width * vec_d + v * filter_height + filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data2 = vld1q_s64_x4((const int64_t *) &filters[(f * filter_height * filter_width * vec_d + v * filter_height + filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                outputs[f / 64 * out_height * out_width + h * out_width * 64 + w * 64 + f % 64] = vaddvq_u8(output.val[0]) + vaddvq_u8(output.val[1]) + vaddvq_u8(output.val[2]) + vaddvq_u8(output.val[3]);
                
                w ++;
                i = 0;
                j = 0;
                input_h = h * strides +0;
                input_w = w * strides +0;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data1.val[0] = vmulq_s8(data1.val[0],weight_cache_0.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],weight_cache_0.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],weight_cache_0.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],weight_cache_0.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                i = 0;
                j = 1;
                input_h = h * strides +0;
                input_w = w * strides +1;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data1.val[0] = vmulq_s8(data1.val[0],weight_cache_1.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],weight_cache_1.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],weight_cache_1.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],weight_cache_1.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data1.val[0] = vmulq_s8(data1.val[0],weight_cache_2.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],weight_cache_2.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],weight_cache_2.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],weight_cache_2.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                i = 1;
                j = 0;
                input_h = h * strides +1;
                input_w = w * strides +0;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data1.val[0] = vmulq_s8(data1.val[0],weight_cache_3.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],weight_cache_3.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],weight_cache_3.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],weight_cache_3.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                i = 1;
                j = 1;
                input_h = h * strides +1;
                input_w = w * strides +1;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data1.val[0] = vmulq_s8(data1.val[0],weight_cache_4.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],weight_cache_4.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],weight_cache_4.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],weight_cache_4.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data2 = vld1q_s64_x4((const int64_t *) &filters[(f * filter_height * filter_width * vec_d + v * filter_height + filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                i = 2;
                j = 0;
                input_h = h * strides +2;
                input_w = w * strides +0;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data2 = vld1q_s64_x4((const int64_t *) &filters[(f * filter_height * filter_width * vec_d + v * filter_height + filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                i = 2;
                j = 1;
                input_h = h * strides +2;
                input_w = w * strides +1;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data2 = vld1q_s64_x4((const int64_t *) &filters[(f * filter_height * filter_width * vec_d + v * filter_height + filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(v * height * width + input_h * width+ input_w) * depth /64]);
                data2 = vld1q_s64_x4((const int64_t *) &filters[(f * filter_height * filter_width * vec_d + v * filter_height + filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                output.val[2] = vaddq_u8(output.val[2],data1.val[2]);
                output.val[3] = vaddq_u8(output.val[3],data1.val[3]);
                
                outputs[f / 64 * out_height * out_width + h * out_width * 64 + w * 64 + f % 64] = vaddvq_u8(output.val[0]) + vaddvq_u8(output.val[1]) + vaddvq_u8(output.val[2]) + vaddvq_u8(output.val[3]);
                
            }
        }
    }
    
    }
    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    std::fprintf(pFile, "%lf\n", time_elapsed_ms);
    std::free(inputs);
    std::free(outputs);
    std::free(filters);
}
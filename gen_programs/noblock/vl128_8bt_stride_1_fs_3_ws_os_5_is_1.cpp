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
    
    height = atoi(argv[1]);
    width = atoi(argv[2]);
    depth = atoi(argv[3]);
    num_filters = atoi(argv[4]);
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);
    int h;
    int w;
    int input_h;
    int input_w;
    
    int64x2_t data1;
    int64x2_t data2;
    int64x2_t input_cache_0;
    
    int64x2_t output_cache_0;
    int64x2_t output_cache_1;
    int64x2_t output_cache_2;
    int64x2_t output_cache_3;
    int64x2_t output_cache_4;
    
    m5_reset_stats(0, 0);
    
    for (int f = 0; f < num_filters; f++) {
        output_cache_0.val[0]=vdupq_n_u64(0);
        output_cache_1.val[0]=vdupq_n_u64(0);
        output_cache_2.val[0]=vdupq_n_u64(0);
        output_cache_3.val[0]=vdupq_n_u64(0);
        output_cache_4.val[0]=vdupq_n_u64(0);
        int64x2_t input_cache_0 = vld1q_s64((const int64_t *) &inputs[((0-padding) * width * depth /256 + (0-padding) * depth /256) * 128 /64]);
        int i;
        int j;
        for (i = 0; i < filter_height - 1; i ++) {
            for (j = 0; j < filter_width; j ++) {
                h = 0;
                w = 0;
                data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                
                
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_0.val[0],data2.val[0]);
                output_cache_0.val[0] = vaddq_u8(output_cache_0.val[0],data1.val[0]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                output_cache_1.val[0] = vaddq_u8(output_cache_1.val[0],data1.val[0]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                output_cache_2.val[0] = vaddq_u8(output_cache_2.val[0],data1.val[0]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                output_cache_3.val[0] = vaddq_u8(output_cache_3.val[0],data1.val[0]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                output_cache_4.val[0] = vaddq_u8(output_cache_4.val[0],data1.val[0]);
                for (h = 0; h < out_height; h++) {
                    for (w = 5; w < out_width; w++) {
                        input_h = h * strides + i - padding;
                        input_w = w * strides + j - padding;
                        data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * 128 /64]);
                        data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1.val[0]);
                    }
                }
            }
        }
        
        
        for (j = 0; j < filter_width - 1; j ++) {
            data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            
            
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_0.val[0], data2.val[0]);
            output_cache_0.val[0] = vaddq_u8(output_cache_0.val[0],data1.val[0]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1 = vld1q_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1.val[0] = vmulq_s8(data1.val[0], data2.val[0]);
            output_cache_1.val[0] = vaddq_u8(output_cache_1.val[0],data1.val[0]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1 = vld1q_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1.val[0] = vmulq_s8(data1.val[0], data2.val[0]);
            output_cache_2.val[0] = vaddq_u8(output_cache_2.val[0],data1.val[0]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1 = vld1q_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1.val[0] = vmulq_s8(data1.val[0], data2.val[0]);
            output_cache_3.val[0] = vaddq_u8(output_cache_3.val[0],data1.val[0]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1 = vld1q_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1.val[0] = vmulq_s8(data1.val[0], data2.val[0]);
            output_cache_4.val[0] = vaddq_u8(output_cache_4.val[0],data1.val[0]);
            
            for (h = 0; h < out_height; h++) {
                for (w = 5; w < out_width; w++) {
                    input_h = h * strides + i - padding;
                    input_w = w * strides + j - padding;
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * 128 /64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1.val[0]);
                }
            }
        }
        data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        
        
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_0.val[0],data2.val[0]);
        output_cache_0.val[0] = vaddq_u8(output_cache_0.val[0],data1.val[0]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_0.val[0]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1 = vld1q_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
        output_cache_1.val[0] = vaddq_u8(output_cache_1.val[0],data1.val[0]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_1.val[0]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1 = vld1q_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
        output_cache_2.val[0] = vaddq_u8(output_cache_2.val[0],data1.val[0]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_2.val[0]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1 = vld1q_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
        output_cache_3.val[0] = vaddq_u8(output_cache_3.val[0],data1.val[0]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_3.val[0]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1 = vld1q_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
        output_cache_4.val[0] = vaddq_u8(output_cache_4.val[0],data1.val[0]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_4.val[0]);
        
        
        for (h = 0; h < out_height; h++) {
            for (w = 5; w < out_width; w++) {
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * 128 /64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1.val[0]);
            }
        }
    }
}
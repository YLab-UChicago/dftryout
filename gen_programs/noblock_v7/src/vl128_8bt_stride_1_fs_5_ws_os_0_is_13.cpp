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
    depth = 128;
    num_filters = atoi(argv[3]);
    filter_height = 5;
    filter_width = 5;
    padding = 4;
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
    int64x2_t input_cache_1;
    int64x2_t input_cache_2;
    int64x2_t input_cache_3;
    int64x2_t input_cache_4;
    int64x2_t input_cache_5;
    int64x2_t input_cache_6;
    int64x2_t input_cache_7;
    int64x2_t input_cache_8;
    int64x2_t input_cache_9;
    int64x2_t input_cache_10;
    int64x2_t input_cache_11;
    int64x2_t input_cache_12;
    
    
    m5_reset_stats(0, 0);
    
    for (int f = 0; f < num_filters; f++) {
        int64x2_t input_cache_0 = vld1q_s64((const int64_t *) &inputs[(0 * width + 0) * 128 /64]);
        int64x2_t input_cache_1 = vld1q_s64((const int64_t *) &inputs[(0 * width + 1) * 128 /64]);
        int64x2_t input_cache_2 = vld1q_s64((const int64_t *) &inputs[(0 * width + 2) * 128 /64]);
        int64x2_t input_cache_3 = vld1q_s64((const int64_t *) &inputs[(0 * width + 3) * 128 /64]);
        int64x2_t input_cache_4 = vld1q_s64((const int64_t *) &inputs[(0 * width + 4) * 128 /64]);
        int64x2_t input_cache_5 = vld1q_s64((const int64_t *) &inputs[(1 * width + 0) * 128 /64]);
        int64x2_t input_cache_6 = vld1q_s64((const int64_t *) &inputs[(1 * width + 1) * 128 /64]);
        int64x2_t input_cache_7 = vld1q_s64((const int64_t *) &inputs[(1 * width + 2) * 128 /64]);
        int64x2_t input_cache_8 = vld1q_s64((const int64_t *) &inputs[(1 * width + 3) * 128 /64]);
        int64x2_t input_cache_9 = vld1q_s64((const int64_t *) &inputs[(1 * width + 4) * 128 /64]);
        int64x2_t input_cache_10 = vld1q_s64((const int64_t *) &inputs[(2 * width + 0) * 128 /64]);
        int64x2_t input_cache_11 = vld1q_s64((const int64_t *) &inputs[(2 * width + 1) * 128 /64]);
        int64x2_t input_cache_12 = vld1q_s64((const int64_t *) &inputs[(2 * width + 2) * 128 /64]);
        int i;
        int j;
        for (i = 0; i < filter_height - 1; i ++) {
            for (j = 0; j < filter_width; j ++) {
                h = 0;
                w = 0;
                data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                
                
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vmulq_s8(input_cache_0,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vmulq_s8(input_cache_1,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vmulq_s8(input_cache_2,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vmulq_s8(input_cache_3,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vmulq_s8(input_cache_4,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vmulq_s8(input_cache_5,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vmulq_s8(input_cache_6,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vmulq_s8(input_cache_7,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vmulq_s8(input_cache_8,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vmulq_s8(input_cache_9,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vmulq_s8(input_cache_10,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vmulq_s8(input_cache_11,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vmulq_s8(input_cache_12,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
                
                for (h = 0; h < out_height; h++) {
                    for (w = 13; w < out_width; w++) {
                        input_h = h * strides + i;
                        input_w = w * strides + j;
                        data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128+ input_w * depth /128) * 128 /64]);
                        data1 = vmulq_s8(data1,data2);
                        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
                    }
                }
            }
        }
        
        
        for (j = 0; j < filter_width - 1; j ++) {
            data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            
            
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vmulq_s8(input_cache_0, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vmulq_s8(input_cache_1, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vmulq_s8(input_cache_2, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vmulq_s8(input_cache_3, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vmulq_s8(input_cache_4, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vmulq_s8(input_cache_5, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vmulq_s8(input_cache_6, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vmulq_s8(input_cache_7, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vmulq_s8(input_cache_8, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vmulq_s8(input_cache_9, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vmulq_s8(input_cache_10, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vmulq_s8(input_cache_11, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vmulq_s8(input_cache_12, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
            
            
            for (h = 0; h < out_height; h++) {
                for (w = 13; w < out_width; w++) {
                    input_h = h * strides + i;
                    input_w = w * strides + j;
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * 128 /64]);
                    data1 = vmulq_s8(data1,data2);
                    outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
                }
            }
        }
        data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        
        
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vmulq_s8(input_cache_0,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vmulq_s8(input_cache_1,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vmulq_s8(input_cache_2,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vmulq_s8(input_cache_3,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vmulq_s8(input_cache_4,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vmulq_s8(input_cache_5,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vmulq_s8(input_cache_6,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vmulq_s8(input_cache_7,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vmulq_s8(input_cache_8,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vmulq_s8(input_cache_9,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vmulq_s8(input_cache_10,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vmulq_s8(input_cache_11,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vmulq_s8(input_cache_12,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
        
        
        for (h = 0; h < out_height; h++) {
            for (w = 13; w < out_width; w++) {
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128+ input_w * depth /128) * 128 /64]);
                data1 = vmulq_s8(data1,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1);
            }
        }
    }
m5_dump_reset_stats(0, 0);
free(inputs);
free(outputs);
free(filters);
}
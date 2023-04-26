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
    int64x2_t input_cache_13;
    int64x2_t input_cache_14;
    int64x2_t input_cache_15;
    int64x2_t input_cache_16;
    int64x2_t input_cache_17;
    int64x2_t input_cache_18;
    int64x2_t input_cache_19;
    int64x2_t input_cache_20;
    int64x2_t input_cache_21;
    int64x2_t input_cache_22;
    int64x2_t input_cache_23;
    int64x2_t input_cache_24;
    int64x2_t input_cache_25;
    int64x2_t input_cache_26;
    int64x2_t input_cache_27;
    int64x2_t input_cache_28;
    
    
    m5_reset_stats(0, 0);
    
    for (int f = 0; f < num_filters; f++) {
        int64x2_t input_cache_0 = vld1q_s64((const int64_t *) &inputs[((0) * width * depth /256 + (0) * depth /256) * 128 /64]);
        int64x2_t input_cache_1 = vld1q_s64((const int64_t *) &inputs[((0) * width * depth /256 + (1) * depth /256) * 128 /64]);
        int64x2_t input_cache_2 = vld1q_s64((const int64_t *) &inputs[((0) * width * depth /256 + (2) * depth /256) * 128 /64]);
        int64x2_t input_cache_3 = vld1q_s64((const int64_t *) &inputs[((1) * width * depth /256 + (0) * depth /256) * 128 /64]);
        int64x2_t input_cache_4 = vld1q_s64((const int64_t *) &inputs[((1) * width * depth /256 + (1) * depth /256) * 128 /64]);
        int64x2_t input_cache_5 = vld1q_s64((const int64_t *) &inputs[((1) * width * depth /256 + (2) * depth /256) * 128 /64]);
        int64x2_t input_cache_6 = vld1q_s64((const int64_t *) &inputs[((2) * width * depth /256 + (0) * depth /256) * 128 /64]);
        int64x2_t input_cache_7 = vld1q_s64((const int64_t *) &inputs[((2) * width * depth /256 + (1) * depth /256) * 128 /64]);
        int64x2_t input_cache_8 = vld1q_s64((const int64_t *) &inputs[((2) * width * depth /256 + (2) * depth /256) * 128 /64]);
        int64x2_t input_cache_9 = vld1q_s64((const int64_t *) &inputs[((3) * width * depth /256 + (0) * depth /256) * 128 /64]);
        int64x2_t input_cache_10 = vld1q_s64((const int64_t *) &inputs[((3) * width * depth /256 + (1) * depth /256) * 128 /64]);
        int64x2_t input_cache_11 = vld1q_s64((const int64_t *) &inputs[((3) * width * depth /256 + (2) * depth /256) * 128 /64]);
        int64x2_t input_cache_12 = vld1q_s64((const int64_t *) &inputs[((4) * width * depth /256 + (0) * depth /256) * 128 /64]);
        int64x2_t input_cache_13 = vld1q_s64((const int64_t *) &inputs[((4) * width * depth /256 + (1) * depth /256) * 128 /64]);
        int64x2_t input_cache_14 = vld1q_s64((const int64_t *) &inputs[((4) * width * depth /256 + (2) * depth /256) * 128 /64]);
        int64x2_t input_cache_15 = vld1q_s64((const int64_t *) &inputs[((5) * width * depth /256 + (0) * depth /256) * 128 /64]);
        int64x2_t input_cache_16 = vld1q_s64((const int64_t *) &inputs[((5) * width * depth /256 + (1) * depth /256) * 128 /64]);
        int64x2_t input_cache_17 = vld1q_s64((const int64_t *) &inputs[((5) * width * depth /256 + (2) * depth /256) * 128 /64]);
        int64x2_t input_cache_18 = vld1q_s64((const int64_t *) &inputs[((6) * width * depth /256 + (0) * depth /256) * 128 /64]);
        int64x2_t input_cache_19 = vld1q_s64((const int64_t *) &inputs[((6) * width * depth /256 + (1) * depth /256) * 128 /64]);
        int64x2_t input_cache_20 = vld1q_s64((const int64_t *) &inputs[((6) * width * depth /256 + (2) * depth /256) * 128 /64]);
        int64x2_t input_cache_21 = vld1q_s64((const int64_t *) &inputs[((7) * width * depth /256 + (0) * depth /256) * 128 /64]);
        int64x2_t input_cache_22 = vld1q_s64((const int64_t *) &inputs[((7) * width * depth /256 + (1) * depth /256) * 128 /64]);
        int64x2_t input_cache_23 = vld1q_s64((const int64_t *) &inputs[((7) * width * depth /256 + (2) * depth /256) * 128 /64]);
        int64x2_t input_cache_24 = vld1q_s64((const int64_t *) &inputs[((8) * width * depth /256 + (0) * depth /256) * 128 /64]);
        int64x2_t input_cache_25 = vld1q_s64((const int64_t *) &inputs[((8) * width * depth /256 + (1) * depth /256) * 128 /64]);
        int64x2_t input_cache_26 = vld1q_s64((const int64_t *) &inputs[((8) * width * depth /256 + (2) * depth /256) * 128 /64]);
        int64x2_t input_cache_27 = vld1q_s64((const int64_t *) &inputs[((9) * width * depth /256 + (0) * depth /256) * 128 /64]);
        int64x2_t input_cache_28 = vld1q_s64((const int64_t *) &inputs[((9) * width * depth /256 + (1) * depth /256) * 128 /64]);
        int i;
        int j;
        for (i = 0; i < filter_height - 1; i ++) {
            for (j = 0; j < filter_width; j ++) {
                h = 0;
                w = 0;
                data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                
                
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_0,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_0)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_1,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_1)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_2,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_2)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_3,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_3)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_4,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_4)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_5,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_5)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_6,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_6)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_7,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_7)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_8,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_8)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_9,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_9)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_10,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_10)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_11,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_11)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_12,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_12)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_13,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_13)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_14,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_14)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_15,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_15)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_16,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_16)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_17,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_17)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_18,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_18)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_19,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_19)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_20,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_20)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_21,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_21)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_22,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_22)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_23,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_23)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_24,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_24)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_25,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_25)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_26,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_26)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_27,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_27)));
                
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = veorq_s64(input_cache_28,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_28)));
                
                for (h = 0; h < out_height; h++) {
                    for (w = 29; w < out_width; w++) {
                        input_h = h * strides + i;
                        input_w = w * strides + j;
                        data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128+ input_w * depth /128) * 128 /64]);
                        data1 = veorq_s64(data1,data2);
                        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_28)));
                    }
                }
            }
        }
        
        
        for (j = 0; j < filter_width - 1; j ++) {
            data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            
            
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_0, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_0)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_1, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_1)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_2, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_2)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_3, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_3)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_4, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_4)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_5, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_5)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_6, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_6)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_7, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_7)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_8, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_8)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_9, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_9)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_10, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_10)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_11, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_11)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_12, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_12)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_13, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_13)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_14, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_14)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_15, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_15)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_16, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_16)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_17, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_17)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_18, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_18)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_19, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_19)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_20, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_20)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_21, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_21)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_22, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_22)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_23, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_23)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_24, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_24)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_25, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_25)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_26, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_26)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_27, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_27)));
            
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = veorq_s64(input_cache_28, data2);
            outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_28)));
            
            
            for (h = 0; h < out_height; h++) {
                for (w = 29; w < out_width; w++) {
                    input_h = h * strides + i;
                    input_w = w * strides + j;
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * 128 /64]);
                    data1 = veorq_s64(data1,data2);
                    outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_28)));
                }
            }
        }
        data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        
        
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_0,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_0)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_1,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_1)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_2,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_2)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_3,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_3)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_4,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_4)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_5,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_5)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_6,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_6)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_7,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_7)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_8,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_8)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_9,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_9)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_10,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_10)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_11,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_11)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_12,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_12)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_13,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_13)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_14,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_14)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_15,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_15)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_16,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_16)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_17,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_17)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_18,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_18)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_19,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_19)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_20,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_20)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_21,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_21)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_22,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_22)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_23,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_23)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_24,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_24)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_25,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_25)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_26,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_26)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_27,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_27)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = veorq_s64(input_cache_28,data2);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_28)));
        
        
        for (h = 0; h < out_height; h++) {
            for (w = 29; w < out_width; w++) {
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128+ input_w * depth /128) * 128 /64]);
                data1 = veorq_s64(data1,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(input_cache_28)));
            }
        }
    }
m5_dump_reset_stats(0, 0);
free(inputs);
free(outputs);
free(filters);
}
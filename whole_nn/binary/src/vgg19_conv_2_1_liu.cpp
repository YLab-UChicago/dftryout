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
    double time_elapsed_ms;
    
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
    inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);
    int h;
    int w;
    int input_h;
    int input_w;
    std::clock_t c_start;
    std::clock_t c_end;
    
    int64x1_t data1;
    int64x1_t data2;
    
    int64x1_t output_cache_0;
    int64x1_t output_cache_1;
    int64x1_t output_cache_2;
    int64x1_t output_cache_3;
    int64x1_t output_cache_4;
    int64x1_t output_cache_5;
    int64x1_t output_cache_6;
    int64x1_t output_cache_7;
    int64x1_t output_cache_8;
    int64x1_t output_cache_9;
    int64x1_t output_cache_10;
    int64x1_t output_cache_11;
    int64x1_t output_cache_12;
    int64x1_t output_cache_13;
    int64x1_t output_cache_14;
    int64x1_t output_cache_15;
    int64x1_t output_cache_16;
    int64x1_t output_cache_17;
    int64x1_t output_cache_18;
    int64x1_t output_cache_19;
    int64x1_t output_cache_20;
    int64x1_t output_cache_21;
    int64x1_t output_cache_22;
    int64x1_t output_cache_23;
    int64x1_t output_cache_24;
    int64x1_t output_cache_25;
    int64x1_t output_cache_26;
    int64x1_t output_cache_27;
    int64x1_t output_cache_28;
    
    c_start = std::clock();
    
    for (int f = 0; f < num_filters; f++) {
        output_cache_0=vdup_n_u64(0);
        output_cache_1=vdup_n_u64(0);
        output_cache_2=vdup_n_u64(0);
        output_cache_3=vdup_n_u64(0);
        output_cache_4=vdup_n_u64(0);
        output_cache_5=vdup_n_u64(0);
        output_cache_6=vdup_n_u64(0);
        output_cache_7=vdup_n_u64(0);
        output_cache_8=vdup_n_u64(0);
        output_cache_9=vdup_n_u64(0);
        output_cache_10=vdup_n_u64(0);
        output_cache_11=vdup_n_u64(0);
        output_cache_12=vdup_n_u64(0);
        output_cache_13=vdup_n_u64(0);
        output_cache_14=vdup_n_u64(0);
        output_cache_15=vdup_n_u64(0);
        output_cache_16=vdup_n_u64(0);
        output_cache_17=vdup_n_u64(0);
        output_cache_18=vdup_n_u64(0);
        output_cache_19=vdup_n_u64(0);
        output_cache_20=vdup_n_u64(0);
        output_cache_21=vdup_n_u64(0);
        output_cache_22=vdup_n_u64(0);
        output_cache_23=vdup_n_u64(0);
        output_cache_24=vdup_n_u64(0);
        output_cache_25=vdup_n_u64(0);
        output_cache_26=vdup_n_u64(0);
        output_cache_27=vdup_n_u64(0);
        output_cache_28=vdup_n_u64(0);
        int i;
        int j;
        for (i = 0; i < filter_height - 1; i ++) {
            for (j = 0; j < filter_width; j ++) {
                h = 0;
                w = 0;
                data2 = vld1_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                
                
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_0 = vadd_u8(output_cache_0,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_1 = vadd_u8(output_cache_1,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_2 = vadd_u8(output_cache_2,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_3 = vadd_u8(output_cache_3,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_4 = vadd_u8(output_cache_4,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_5 = vadd_u8(output_cache_5,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_6 = vadd_u8(output_cache_6,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_7 = vadd_u8(output_cache_7,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_8 = vadd_u8(output_cache_8,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_9 = vadd_u8(output_cache_9,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_10 = vadd_u8(output_cache_10,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_11 = vadd_u8(output_cache_11,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_12 = vadd_u8(output_cache_12,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_13 = vadd_u8(output_cache_13,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_14 = vadd_u8(output_cache_14,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_15 = vadd_u8(output_cache_15,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_16 = vadd_u8(output_cache_16,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_17 = vadd_u8(output_cache_17,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_18 = vadd_u8(output_cache_18,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_19 = vadd_u8(output_cache_19,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_20 = vadd_u8(output_cache_20,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_21 = vadd_u8(output_cache_21,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_22 = vadd_u8(output_cache_22,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_23 = vadd_u8(output_cache_23,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_24 = vadd_u8(output_cache_24,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_25 = vadd_u8(output_cache_25,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_26 = vadd_u8(output_cache_26,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_27 = vadd_u8(output_cache_27,data1);
                
                w++;
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
                data1 = veor_s64(data1,data2);
                output_cache_28 = vadd_u8(output_cache_28,data1);
                for (h = 0; h < out_height; h++) {
                    for (w = 29; w < out_width; w++) {
                        input_h = h * strides + i;
                        input_w = w * strides + j;
                        data1 = vld1_s64((const int64_t *) &inputs[(input_h * width * depth /128+ input_w * depth /128) * 128 /64]);
                        data1 = veor_s64(data1,data2);
                        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(data1)));
                    }
                }
            }
        }
        h = 0;
        w = 0;
        
        for (j = 0; j < filter_width - 1; j ++) {
            data2 = vld1_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            
            
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_0 = vadd_u8(output_cache_0,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_1 = vadd_u8(output_cache_1,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_2 = vadd_u8(output_cache_2,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_3 = vadd_u8(output_cache_3,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_4 = vadd_u8(output_cache_4,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_5 = vadd_u8(output_cache_5,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_6 = vadd_u8(output_cache_6,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_7 = vadd_u8(output_cache_7,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_8 = vadd_u8(output_cache_8,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_9 = vadd_u8(output_cache_9,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_10 = vadd_u8(output_cache_10,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_11 = vadd_u8(output_cache_11,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_12 = vadd_u8(output_cache_12,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_13 = vadd_u8(output_cache_13,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_14 = vadd_u8(output_cache_14,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_15 = vadd_u8(output_cache_15,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_16 = vadd_u8(output_cache_16,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_17 = vadd_u8(output_cache_17,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_18 = vadd_u8(output_cache_18,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_19 = vadd_u8(output_cache_19,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_20 = vadd_u8(output_cache_20,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_21 = vadd_u8(output_cache_21,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_22 = vadd_u8(output_cache_22,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_23 = vadd_u8(output_cache_23,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_24 = vadd_u8(output_cache_24,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_25 = vadd_u8(output_cache_25,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_26 = vadd_u8(output_cache_26,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_27 = vadd_u8(output_cache_27,data1);
            
            w++;
            input_h = h * strides + i;
            input_w = w * strides + j;
            data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
            data1 = veor_s64(data1, data2);
            output_cache_28 = vadd_u8(output_cache_28,data1);
            
            for (h = 0; h < out_height; h++) {
                for (w = 29; w < out_width; w++) {
                    input_h = h * strides + i;
                    input_w = w * strides + j;
                    data1 = vld1_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * 128 /64]);
                    data1 = veor_s64(data1,data2);
                    outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(data1)));
                }
            }
        }

        h = 0;
        w = 0;
        data2 = vld1_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        
        
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_0 = vadd_u8(output_cache_0,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_0)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_1 = vadd_u8(output_cache_1,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_1)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_2 = vadd_u8(output_cache_2,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_2)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_3 = vadd_u8(output_cache_3,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_3)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_4 = vadd_u8(output_cache_4,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_4)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_5 = vadd_u8(output_cache_5,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_5)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_6 = vadd_u8(output_cache_6,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_6)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_7 = vadd_u8(output_cache_7,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_7)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_8 = vadd_u8(output_cache_8,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_8)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_9 = vadd_u8(output_cache_9,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_9)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_10 = vadd_u8(output_cache_10,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_10)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_11 = vadd_u8(output_cache_11,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_11)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_12 = vadd_u8(output_cache_12,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_12)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_13 = vadd_u8(output_cache_13,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_13)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_14 = vadd_u8(output_cache_14,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_14)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_15 = vadd_u8(output_cache_15,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_15)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_16 = vadd_u8(output_cache_16,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_16)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_17 = vadd_u8(output_cache_17,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_17)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_18 = vadd_u8(output_cache_18,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_18)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_19 = vadd_u8(output_cache_19,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_19)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_20 = vadd_u8(output_cache_20,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_20)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_21 = vadd_u8(output_cache_21,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_21)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_22 = vadd_u8(output_cache_22,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_22)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_23 = vadd_u8(output_cache_23,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_23)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_24 = vadd_u8(output_cache_24,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_24)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_25 = vadd_u8(output_cache_25,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_25)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_26 = vadd_u8(output_cache_26,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_26)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_27 = vadd_u8(output_cache_27,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_27)));
        
        
        w++;
        input_h = h * strides + i;
        input_w = w * strides + j;
        data1 = vld1_s64((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*128/64]);
        data1 = veor_s64(data1,data2);
        output_cache_28 = vadd_u8(output_cache_28,data1);
        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(output_cache_28)));
        
        
        for (h = 0; h < out_height; h++) {
            for (w = 29; w < out_width; w++) {
                input_h = h * strides + i;
                input_w = w * strides + j;
                data1 = vld1_s64((const int64_t *) &inputs[(input_h * width * depth /128+ input_w * depth /128) * 128 /64]);
                data1 = veor_s64(data1,data2);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddv_u8(vcnt_u8(data1)));
            }
        }
    }
    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    printf("%lf\n", time_elapsed_ms);
free(inputs);
free(outputs);
free(filters);
}
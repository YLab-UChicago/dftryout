#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>
#include <iostream>
#include <arm_neon.h>
#include <algorithm>
#include <m5ops.h>

using namespace std;



int main(int argc, char *argv[])
{
    int height;
    int width;
    int depth;
    int filter_height;
    int filter_width;
    int num_filters;
    int padding;
    int strides;
    int curr;
    int64_t *inputs;
    int64_t *outputs;
    int64_t *filters;
    std::clock_t c_start;
    std::clock_t c_end;

    int output_depth;
    int pool_size;

    int input_size;
    int idx;

    int layer_counter = 0;
    double time_elapsed_ms;
    height = atoi(argv[1]);
    width = atoi(argv[2]);
    depth = 256;
    num_filters = atoi(argv[3]);
    filter_height = atoi(argv[4]);
    filter_width = atoi(argv[5]);
    padding = atoi(argv[4])-1;
    strides = atoi(argv[6]);
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);


    c_start = std::clock();
    for (int f = 0; f < num_filters; f ++)
    {
        for (int h = 0; h < height; h++) 
        {
            for (int w = 0; w < width; w ++) 
            {
                idx = h * width * depth / 64 + w * depth / 64;
                uint64x2x2_t data1 = vld1q_u64_x2((const uint64_t *)&inputs[idx]);
                for (int i = 0; i < filter_height; i ++)
                {
                    for (int j = 0; j < filter_width; j ++) 
                    {
                        if (!((w - j) % stride)) & (!(h - i) % stride) {
                            int output_h = floor((h - i) / strides);
                            int output_w = floor((w - j) / strides);
                            if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                                uint64x2x2_t data2 = vld1q_u64_x2((const uint64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                                uint64x2x2_t output;
                                output.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                                output.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                                outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(output.val[0]) + vaddvq_u8(output.val[1]);
                            }
                        }
                    }
                }
            }
        }
    }

c_end = std::clock();
time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
printf("%lf\n", time_elapsed_ms);

    std::free(inputs);
    std::free(outputs);
    std::free(filters);
}
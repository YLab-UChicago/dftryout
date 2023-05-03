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

    /* Please type the following in the command line
        ./<program_name> <input_height> <input_width> <input_depth (plase use 256)> <num_filters>
    */

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

    int output_depth;
    int pool_size;

    int input_size;
    int idx;

    int layer_counter = 0;
    double time_elapsed_ms;
    height = atoi(argv[1]);
    width = atoi(argv[2]);
    depth = 512;
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

    for (int f = 0; f < num_filters; f ++)
    {
        for (int h = 0; h < height; h++) 
        {
            for (int w = 0; w < width; w ++) 
            {
                idx = h * width * depth / 64 + w * depth / 64;
                uint64x2x4_t data1 = vld1q_u64_x4((const uint64_t *)&inputs[idx]);
                for (int i = 0; i < filter_height; i ++)
                {
                    for (int j = 0; j < filter_width; j ++) 
                    {
                        int output_h = (h  - i) / strides;
                        int output_w = (w - j) / strides;
                        if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                            uint64x2x4_t data2 = vld1q_u64_x4((const uint64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                            uint64x2x4_t output;
                            output.val[0] = veorq_u64(data1.val[0],data2.val[0]);
                            output.val[1] = veorq_u64(data1.val[1],data2.val[1]);
                            output.val[2] = veorq_u64(data1.val[2],data2.val[2]);
                            output.val[3] = veorq_u64(data1.val[3],data2.val[3]);
                            outputs[h * out_width * num_filters + w * num_filters + f] += 512 - 2 * (vaddvq_u8(vcntq_u8(output.val[0])) + vaddvq_u8(vcntq_u8(output.val[1]))+ vaddvq_u8(vcntq_u8(output.val[2]))+ vaddvq_u8(vcntq_u8(output.val[3])));
                        }
                    }
                }
            }
        }
    }


    std::free(inputs);
    std::free(outputs);
    std::free(filters);
}
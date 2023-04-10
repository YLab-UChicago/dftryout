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

    FILE *pFile = fopen("../durations/os_ws0_is0.txt", "a");
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
    int *outputs;
    int64_t *filters;

    int output_depth;
    int pool_size;

    int input_size;

    int layer_counter = 0;
    double time_elapsed_ms;
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
    outputs = (int *)malloc(sizeof(int) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);
    uint64x2x2_t data1;
    uint64x2x2_t data2;
    

    m5_reset_stats(0, 0);


    for (int h = 0; h < out_height; h++)
    {
        for (int w = 0; w < out_width; w++)
        {
            for (int f = 0; f < num_filters; f++)
                {
                int sum_block = 0;
                for (int i = 0; i < filter_height; i++)
                    {
                    for (int j = 0; j < filter_width; j++)
                        {
                        int input_h = h * strides + i - padding;
                        int input_w = w * strides + j - padding;
                        data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                        data2 = vld1q_u64_x2((const uint64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                        data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                        data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                        sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));
                    }
                }
                outputs[h * out_width * num_filters + w * num_filters + f] =sum_block;
            }
        }
    }

    m5_dump_reset_stats(0, 0);
    std::fprintf(pFile, "%lf\n", time_elapsed_ms);

    std::free(inputs);
    std::free(outputs);
    std::free(filters);
}
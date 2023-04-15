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

    FILE *pFile = fopen("../durations/os_ws9_is6.txt", "a");
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

    uint64x2x2_t weight_cache_1;
    uint64x2x2_t weight_cache_2;
    uint64x2x2_t weight_cache_3;
    uint64x2x2_t weight_cache_4;
    uint64x2x2_t weight_cache_5;
    uint64x2x2_t weight_cache_6;
    uint64x2x2_t weight_cache_7;
    uint64x2x2_t weight_cache_8;
    uint64x2x2_t weight_cache_9;

    uint64x2x2_t input_cache_1;
    uint64x2x2_t input_cache_2;
    uint64x2x2_t input_cache_3;
    uint64x2x2_t input_cache_4;
    uint64x2x2_t input_cache_5;
    uint64x2x2_t input_cache_6;


    m5_reset_stats(0, 0);

    for (int f = 0; f < num_filters; f++)
    {
        
        weight_cache_1 = vld1q_u64_x2((const uint64_t*) &filters[(f * filter_height * filter_width + 0)*4]);
        weight_cache_2 = vld1q_u64_x2((const uint64_t*) &filters[(f * filter_height * filter_width + 1)*4]);
        weight_cache_3 = vld1q_u64_x2((const uint64_t*) &filters[(f * filter_height * filter_width + 2)*4]);
        weight_cache_4 = vld1q_u64_x2((const uint64_t*) &filters[(f * filter_height * filter_width + 3)*4]);
        weight_cache_5 = vld1q_u64_x2((const uint64_t*) &filters[(f * filter_height * filter_width + 4)*4]);
        weight_cache_6 = vld1q_u64_x2((const uint64_t*) &filters[(f * filter_height * filter_width + 5)*4]);
        weight_cache_7 = vld1q_u64_x2((const uint64_t*) &filters[(f * filter_height * filter_width + 6)*4]);
        weight_cache_8 = vld1q_u64_x2((const uint64_t*) &filters[(f * filter_height * filter_width + 7)*4]);
        weight_cache_9 = vld1q_u64_x2((const uint64_t*) &filters[(f * filter_height * filter_width + 8)*4]);

        input_cache_1 = vld1q_u64_x2((const uint64_t *) &inputs[(0 * width * depth /256 + (0-padding) * depth /256) * depth /64]);
        input_cache_2 = vld1q_u64_x2((const uint64_t *) &inputs[(0 * width * depth /256 + (1-padding) * depth /256) * depth /64]);
        input_cache_3 = vld1q_u64_x2((const uint64_t *) &inputs[(0 * width * depth /256 + (2-padding) * depth /256) * depth /64]);
        input_cache_4 = vld1q_u64_x2((const uint64_t *) &inputs[(0 * width * depth /256 + (3-padding) * depth /256) * depth /64]);
        input_cache_5 = vld1q_u64_x2((const uint64_t *) &inputs[(0 * width * depth /256 + (4-padding) * depth /256) * depth /64]);
        input_cache_6 = vld1q_u64_x2((const uint64_t *) &inputs[(0 * width * depth /256 + (5-padding) * depth /256) * depth /64]);


        for (int h = 0; h < out_height; h++)
        {
            for (int w = 0; w < out_width; w++)
            {

                int sum_block = 0;
                int i;
                int j;

                int input_h;
                int input_w;
                
                input_h = h + 0 - padding;
                input_w = w + 0 - padding;
                
 
                data1.val[0] = veorq_u64(input_cache_1.val[0], weight_cache_1.val[0]);
                data1.val[1] = veorq_u64(input_cache_1.val[1], weight_cache_1.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                
                input_h = h + 0 - padding;
                input_w = w + 1 - padding;
                data1.val[0] = veorq_u64(input_cache_2.val[0], weight_cache_2.val[0]);
                data1.val[1] = veorq_u64(input_cache_2.val[1], weight_cache_2.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                input_h = h + 0 - padding;
                input_w = w + 2 - padding;
                input_cache_1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(input_cache_1.val[0], weight_cache_3.val[0]);
                data1.val[1] = veorq_u64(input_cache_1.val[1], weight_cache_3.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));


                input_h = h + 1 - padding;
                input_w = w + 0 - padding;
                data1.val[0] = veorq_u64(input_cache_3.val[0], weight_cache_4.val[0]);
                data1.val[1] = veorq_u64(input_cache_3.val[1], weight_cache_4.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                input_h = h  + 1 - padding;
                input_w = w  + 1 - padding;
                data1.val[0] = veorq_u64(input_cache_4.val[0], weight_cache_5.val[0]);
                data1.val[1] = veorq_u64(input_cache_4.val[1], weight_cache_5.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                input_h = h + 1 - padding;
                input_w = w + 2 - padding;
                input_cache_3 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(input_cache_3.val[0], weight_cache_6.val[0]);
                data1.val[1] = veorq_u64(input_cache_3.val[1], weight_cache_6.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                input_h = h  + 2 - padding;
                input_w = w  + 0 - padding;
                data1.val[0] = veorq_u64(input_cache_5.val[0], weight_cache_7.val[0]);
                data1.val[1] = veorq_u64(input_cache_5.val[1], weight_cache_7.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                input_h = h  + 2 - padding;
                input_w = w  + 1 - padding;
                data1.val[0] = veorq_u64(input_cache_6.val[0], weight_cache_8.val[0]);
                data1.val[1] = veorq_u64(input_cache_6.val[1], weight_cache_8.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                input_h = h + 2 - padding;
                input_w = w + 2 - padding;
                input_cache_5 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(input_cache_5.val[0], weight_cache_9.val[0]);
                data1.val[1] = veorq_u64(input_cache_5.val[1], weight_cache_9.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                outputs[h * out_width * num_filters + w * num_filters + f] = sum_block;

                w++;

                sum_block = 0;
                
                input_h = h + 0 - padding;
                input_w = w + 0 - padding;

                data1.val[0] = veorq_u64(input_cache_2.val[0], weight_cache_1.val[0]);
                data1.val[1] = veorq_u64(input_cache_2.val[1], weight_cache_1.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));


                input_h = h + 0 - padding;
                input_w = w + 1 - padding;
                data1.val[0] = veorq_u64(input_cache_1.val[0], weight_cache_2.val[0]);
                data1.val[1] = veorq_u64(input_cache_1.val[1], weight_cache_2.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                input_h = h + 0 - padding;
                input_w = w + 2 - padding;
                input_cache_2 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(input_cache_2.val[0], weight_cache_3.val[0]);
                data1.val[1] = veorq_u64(input_cache_2.val[1], weight_cache_3.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));


                input_h = h + 1 - padding;
                input_w = w + 0 - padding;
                data1.val[0] = veorq_u64(input_cache_4.val[0], weight_cache_4.val[0]);
                data1.val[1] = veorq_u64(input_cache_4.val[1], weight_cache_4.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                input_h = h  + 1 - padding;
                input_w = w  + 1 - padding;
                data1.val[0] = veorq_u64(input_cache_3.val[0], weight_cache_5.val[0]);
                data1.val[1] = veorq_u64(input_cache_3.val[1], weight_cache_5.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                input_h = h + 1 - padding;
                input_w = w + 2 - padding;
                input_cache_4 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(input_cache_4.val[0], weight_cache_6.val[0]);
                data1.val[1] = veorq_u64(input_cache_4.val[1], weight_cache_6.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                input_h = h  + 2 - padding;
                input_w = w  + 0 - padding;
                data1.val[0] = veorq_u64(input_cache_6.val[0], weight_cache_7.val[0]);
                data1.val[1] = veorq_u64(input_cache_6.val[1], weight_cache_7.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                input_h = h  + 2 - padding;
                input_w = w  + 1 - padding;
                data1.val[0] = veorq_u64(input_cache_5.val[0], weight_cache_8.val[0]);
                data1.val[1] = veorq_u64(input_cache_5.val[1], weight_cache_8.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                input_h = h + 2 - padding;
                input_w = w + 2 - padding;
                input_cache_6 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(input_cache_6.val[0], weight_cache_9.val[0]);
                data1.val[1] = veorq_u64(input_cache_6.val[1], weight_cache_9.val[1]);
                sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                outputs[h * out_width * num_filters + w * num_filters + f] = sum_block;
            }
        }
    }

    m5_dump_reset_stats(0, 0);
    std::fprintf(pFile, "%lf\n", time_elapsed_ms);

    std::free(inputs);
    std::free(outputs);
    std::free(filters);
}
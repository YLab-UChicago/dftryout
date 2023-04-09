#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>
#include <iostream>
#include <arm_neon.h>
#include <algorithm>

using namespace std;



int main(int argc, char *argv[])
{

    /* Please type the following in the command line
        ./<program_name> <input_height> <input_width> <input_depth (plase use 256)> <num_filters>
    */

    FILE *pFile = fopen("../durations/is_os6_s_ws9.txt", "a");
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
    int idx;

    std::clock_t c_start;
    std::clock_t c_end;
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
    uint64x2x2_t weight_cache_1;
    uint64x2x2_t weight_cache_2;
    uint64x2x2_t weight_cache_3;
    uint64x2x2_t weight_cache_4;
    uint64x2x2_t weight_cache_5;
    uint64x2x2_t weight_cache_6;
    uint64x2x2_t weight_cache_7;
    uint64x2x2_t weight_cache_8;
    uint64x2x2_t weight_cache_9;

    int output_cache_1;
    int output_cache_2;
    int output_cache_3;
    int output_cache_4;
    int output_cache_5;
    int output_cache_6;
    int i;
    int j;
    int output_h;
    int output_w;
    

    c_start = std::clock();

    for (int f = 0; f < num_filters; f ++)
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



        for (int h = 0; h < height; h++) 
        {
            for (int w = 0; w < width; w ++) 
            {
                idx = h * width * depth / 64 + w * depth / 64;
                data1 = vld1q_u64_x2((const uint64_t *)&inputs[idx]);
                
                i = 0; 
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data1.val[0] = veorq_u64(data1.val[0],weight_cache_1.val[0]);
                data1.val[1] = veorq_u64(data1.val[1],weight_cache_1.val[1]);
                output_cache_1 += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));
                outputs[h * out_width * num_filters + w * num_filters + f] += output_cache_1;

                i = 0; 
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data1.val[0] = veorq_u64(data1.val[0],weight_cache_2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1],weight_cache_2.val[1]);
                output_cache_1 = output_cache_2;
                output_cache_1 += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));
                
                i = 0; 
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data1.val[0] = veorq_u64(data1.val[0],weight_cache_3.val[0]);
                data1.val[1] = veorq_u64(data1.val[1],weight_cache_3.val[1]);
                output_cache_2 = 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                i = 1; 
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data1.val[0] = veorq_u64(data1.val[0],weight_cache_4.val[0]);
                data1.val[1] = veorq_u64(data1.val[1],weight_cache_4.val[1]);
                output_cache_3 += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));
                outputs[h * out_width * num_filters + w * num_filters + f] += output_cache_3;

                i = 1; 
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data1.val[0] = veorq_u64(data1.val[0],weight_cache_5.val[0]);
                data1.val[1] = veorq_u64(data1.val[1],weight_cache_5.val[1]);
                output_cache_3 = output_cache_4;
                output_cache_3 += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                i = 1; 
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_u64_x2((const uint64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_u64(data1.val[0],weight_cache_6.val[0]);
                data1.val[1] = veorq_u64(data1.val[1],weight_cache_6.val[1]);
                output_cache_4 = 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                i = 2; 
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_u64_x2((const uint64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_u64(data1.val[0],weight_cache_7.val[0]);
                data1.val[1] = veorq_u64(data1.val[1],weight_cache_7.val[1]);
                output_cache_5 += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));
                outputs[h * out_width * num_filters + w * num_filters + f] += output_cache_5;

                i = 2; 
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_u64_x2((const uint64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_u64(data1.val[0],weight_cache_8.val[0]);
                data1.val[1] = veorq_u64(data1.val[1],weight_cache_8.val[1]);
                output_cache_5 = output_cache_6;
                output_cache_5 += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

                i = 2; 
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_u64_x2((const uint64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_u64(data1.val[0],weight_cache_9.val[0]);
                data1.val[1] = veorq_u64(data1.val[1],weight_cache_9.val[1]);
                output_cache_6 += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));

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
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>
#include <iostream>
#include <smmintrin.h>
#include <immintrin.h>
#include <x86intrin.h>
using namespace std;

int main(int argc, char *argv[])
{
    /* Please type the following in the command line
        ./<program_name> <input_height> <input_width> <num_filters>
        Note that depth is defaulted and fixed to 256.
    */
    FILE *pFile = fopen("durations/simd_is.txt", "a");
    int height;
    int width;
    int depth;
    int filter_height;
    int filter_width;
    int num_filters;
    int padding;
    int strides;
    int curr;
    __m256i *inputs;
    short *outputs;
    __m256i *filters;
    __m256i filter;
    __m256i data1;
    __m256i data2;
    int output_depth;
    int pool_size;

    int input_size;
    float *input_bitrans;

    std::clock_t c_start;
    std::clock_t c_end;
    int layer_counter = 0;
    double time_elapsed_ms;

    height = atoi(argv[1]);
    width = atoi(argv[2]);
    depth = 1;
    num_filters = atoi(argv[3]);
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    inputs = (__m256i *)malloc(sizeof(__m256i) * (height + 2 * padding) * (width + 2 * padding) * depth);
    outputs = (short *)malloc(sizeof(short) * out_height * out_width * num_filters);
    filters = (__m256i *)malloc(sizeof(__m256i) * filter_height * filter_width * num_filters * depth);
    int idx;
    __m256i input;
    c_start = std::clock();

    for (int f = 0; f < num_filters; f++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                idx = h * width * depth + w * depth;
                input = inputs[idx];
                for (int i = 0; i < filter_height; i++)
                {
                    for (int j = 0; j < filter_width; j++)
                    {
                        int output_h = (h + padding - i) / strides;
                        int output_w = (w + padding - j) / strides;
                        filter = filters[f * filter_height * filter_width * depth + i * filter_width + j];
                        data1 = _mm256_xor_si256(input, filter);
                        data2 = _mm256_popcnt_epi64(data1);
                        outputs[output_h * out_width * num_filters + output_w * num_filters + f] += 256 - 2 * (data2[0] + data2[1] + data2[2] + data2[3]);
                    }
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
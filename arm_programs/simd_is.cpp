#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>
#include <iostream>
#include <smmintrin.h>
#include <immintrin.h>
#include <x86intrin.h>
using namespace std;
void binarize(float *inputMatrix, int input_size, int *binarizedMatrix)
{
    for (int i = 0; i < input_size; i++)
    {
        binarizedMatrix[i] = (int)((unsigned int)inputMatrix[i] >> 31);
    }
}

int xnor_popcount(int a, int b)
{
    return __builtin_popcount(~(a ^ b));
}

int main(int argc, char *argv[])
{
     /* Please type the following in the command line
        ./<program_name> <input_height> <input_width> <input_depth (plase use 256)> <num_filters>
    */

    FILE *pFile = fopen("durations/simd_os_ws9.txt", "a");
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
    short *outputs;
    int64_t *filters;

    int output_depth;
    int pool_size;

    int input_size;

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
    outputs = (short *)malloc(sizeof(short) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);
    uint64x2x2_t data1;
    uint64x2x2_t data2;

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
                        data_to_process = _mm256_xor_si256(input, filter);
                        data_to_process = _mm256_popcnt_epi64(data_to_process);
                        outputs[output_h * out_width * num_filters + output_w * num_filters + f] += 256 - 2 * (data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3]);
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
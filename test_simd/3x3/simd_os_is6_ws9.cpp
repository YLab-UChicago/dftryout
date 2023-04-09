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
    FILE *pFile = fopen("durations/simd_os_is6_ws9.txt", "a");
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
    __m256i *inputs;
    int *outputs;
    __m256i *filters;

    int output_depth;
    int pool_size;

    int input_size;
    float *input_bitrans;

    std::clock_t c_start;
    std::clock_t c_end;
    int layer_counter = 0;
    double time_elapsed_ms;

    height = 224;
    width = 224;
    depth = 1;
    num_filters = 32;
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    h_block = 1;
    w_block = 1;
    d_block = 1;
    f_block = 1;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    inputs = (__m256i *)malloc(sizeof(__m256i) * (height + 2 * padding) * (width + 2 * padding) * depth);
    outputs = (int *)malloc(sizeof(int) * out_height * out_width * num_filters);
    filters = (__m256i *)malloc(sizeof(__m256i) * filter_height * filter_width * num_filters * depth);

    c_start = std::clock();

    __m256i weight_cache_1;
    __m256i weight_cache_2;
    __m256i weight_cache_3;
    __m256i weight_cache_4;
    __m256i weight_cache_5;
    __m256i weight_cache_6;
    __m256i weight_cache_7;
    __m256i weight_cache_8;
    __m256i weight_cache_9;

    __m256i input_cache_1;
    __m256i input_cache_2;
    __m256i input_cache_3;
    __m256i input_cache_4;
    __m256i input_cache_5;
    __m256i input_cache_6;

    __m256i filter;
    __m256i data_to_process;

    input_cache_1 = _mm256_setr_epi64x(0, 0, 0, 0);
    input_cache_2 = _mm256_setr_epi64x(0, 0, 0, 0);
    input_cache_3 = _mm256_setr_epi64x(0, 0, 0, 0);
    input_cache_4 = _mm256_setr_epi64x(0, 0, 0, 0);
    input_cache_5 = _mm256_setr_epi64x(0, 0, 0, 0);
    input_cache_6 = _mm256_setr_epi64x(0, 0, 0, 0);

    for (int f = 0; f < num_filters; f++)
    {

        weight_cache_1 = filters[f * filter_height * filter_width + 0];
        weight_cache_2 = filters[f * filter_height * filter_width + 1];
        weight_cache_3 = filters[f * filter_height * filter_width + 2];
        weight_cache_4 = filters[f * filter_height * filter_width + 3];
        weight_cache_5 = filters[f * filter_height * filter_width + 4];
        weight_cache_6 = filters[f * filter_height * filter_width + 5];
        weight_cache_7 = filters[f * filter_height * filter_width + 6];
        weight_cache_8 = filters[f * filter_height * filter_width + 7];
        weight_cache_9 = filters[f * filter_height * filter_width + 8];

        for (int h = 0; h < out_height; h++)
        {
            for (int w = 0; w < out_width; w++)
            {

                int sum_block = 0;
                int i = 0;
                int j = 0;

                int input_h;
                int input_w;

                input_h = h * strides + 0 - padding;
                input_w = w * strides + 0 - padding;

                data_to_process = _mm256_xor_si256(input_cache_1, weight_cache_1);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                sum_block += 256 - 2 * (data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3]);
                input_cache_1 = input_cache_4;

                input_h = h * strides + 0 - padding;
                input_w = w * strides + 1 - padding;
                data_to_process = _mm256_xor_si256(input_cache_4, weight_cache_2);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                sum_block += 256 - 2 * (data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3]);

                input_h = h * strides + 0 - padding;
                input_w = w * strides + 2 - padding;
                data_to_process = inputs[input_h * width * depth + input_w * depth];
                input_cache_4 = data_to_process;
                data_to_process = _mm256_xor_si256(data_to_process, weight_cache_3);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                sum_block += 256 - 2 * (data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3]);

                input_h = h * strides + 1 - padding;
                input_w = w * strides + 0 - padding;
                data_to_process = _mm256_xor_si256(input_cache_2, weight_cache_4);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                sum_block += 256 - 2 * (data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3]);
                input_cache_2 = input_cache_5;

                input_h = h * strides + 1 - padding;
                input_w = w * strides + 1 - padding;
                data_to_process = _mm256_xor_si256(input_cache_5, weight_cache_5);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                sum_block += 256 - 2 * (data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3]);

                input_h = h * strides + 1 - padding;
                input_w = w * strides + 2 - padding;

                data_to_process = inputs[input_h * width * depth + input_w * depth];
                input_cache_5 = data_to_process;
                data_to_process = _mm256_xor_si256(data_to_process, weight_cache_6);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                sum_block += 256 - 2 * (data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3]);

                input_h = h * strides + 2 - padding;
                input_w = w * strides + 0 - padding;
                filter = filters[f * filter_height * filter_width + 6];
                data_to_process = _mm256_xor_si256(input_cache_3, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                sum_block += 256 - 2 * (data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3]);
                input_cache_3 = input_cache_6;

                input_h = h * strides + 2 - padding;
                input_w = w * strides + 1 - padding;
                filter = filters[f * filter_height * filter_width + 7];
                data_to_process = _mm256_xor_si256(input_cache_6, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                sum_block += 256 - 2 * (data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3]);

                input_h = h * strides + 2 - padding;
                input_w = w * strides + 2 - padding;
                data_to_process = inputs[input_h * width * depth + input_w * depth];
                input_cache_6 = data_to_process;
                filter = filters[f * filter_height * filter_width + 8];
                data_to_process = _mm256_xor_si256(data_to_process, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                sum_block += 256 - 2 * (data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3]);

                outputs[h * out_width * num_filters + w * num_filters + f] = sum_block;
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
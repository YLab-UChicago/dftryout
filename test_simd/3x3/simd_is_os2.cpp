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
    FILE *pFile = fopen("durations/simd_is_os2.txt", "a");
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
    __m256i filter;
    __m256i data_to_process;

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
    int idx;
    __m256i input;
    __m256i output_cache_1;
    __m256i output_cache_2;
    int output_h;
    int output_w;
    int i;
    int j;

    c_start = std::clock();

    for (int f = 0; f < num_filters; f++)
    {
        output_cache_1 = _mm256_setr_epi64x(0, 0, 0, 0);
        output_cache_2 = _mm256_setr_epi64x(0, 0, 0, 0);

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {

                idx = h * width * depth + w * depth;
                input = inputs[idx];

                i = 2;
                j = 2;

                output_h = (h + padding - j) / strides;
                output_w = (w + padding - i) / strides;

                filter = filters[f * filter_height * filter_width * depth + i * filter_width + j];
                data_to_process = _mm256_xor_si256(input, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                output_cache_1 += data_to_process;
                outputs[output_h * out_width * num_filters + output_w * num_filters + f] += 256 - 2 * (output_cache_1[0] + output_cache_1[1] + output_cache_1[2] + output_cache_1[3]);

                i = 2;
                j = 1;

                output_h = (h + padding - j) / strides;
                output_w = (w + padding - i) / strides;
                filter = filters[f * filter_height * filter_width * depth + i * filter_width + j];
                data_to_process = _mm256_xor_si256(input, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                output_cache_1 = data_to_process;

                i = 2;
                j = 0;

                output_h = (h + padding - j) / strides;
                output_w = (w + padding - i) / strides;
                filter = filters[f * filter_height * filter_width * depth + i * filter_width + j];
                data_to_process = _mm256_xor_si256(input, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                outputs[output_h * out_width * num_filters + output_w * num_filters + f] += 256 - 2 * (data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3]);

                i = 1;
                j = 2;

                output_h = (h + padding - j) / strides;
                output_w = (w + padding - i) / strides;
                filter = filters[f * filter_height * filter_width * depth + i * filter_width + j];
                data_to_process = _mm256_xor_si256(input, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                output_cache_2 += data_to_process;
                outputs[output_h * out_width * num_filters + output_w * num_filters + f] += 256 - 2 * (output_cache_2[0] + output_cache_2[1] + output_cache_2[2] + output_cache_2[3]);

                i = 1;
                j = 1;

                output_h = (h + padding - j) / strides;
                output_w = (w + padding - i) / strides;
                filter = filters[f * filter_height * filter_width * depth + i * filter_width + j];
                data_to_process = _mm256_xor_si256(input, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                output_cache_2 = data_to_process;

                i = 1;
                j = 0;

                output_h = (h + padding - j) / strides;
                output_w = (w + padding - i) / strides;
                filter = filters[f * filter_height * filter_width * depth + i * filter_width + j];
                data_to_process = _mm256_xor_si256(input, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                outputs[output_h * out_width * num_filters + output_w * num_filters + f] += 256 - 2 * (data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3]);

                i = 0;
                j = 2;

                output_h = (h + padding - j) / strides;
                output_w = (w + padding - i) / strides;
                filter = filters[f * filter_height * filter_width * depth + i * filter_width + j];
                data_to_process = _mm256_xor_si256(input, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                outputs[output_h * out_width * num_filters + output_w * num_filters + f] += 256 - 2 * (data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3]);

                i = 0;
                j = 1;

                output_h = (h + padding - j) / strides;
                output_w = (w + padding - i) / strides;
                filter = filters[f * filter_height * filter_width * depth + i * filter_width + j];
                data_to_process = _mm256_xor_si256(input, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);

                i = 0;
                j = 0;

                output_h = (h + padding - j) / strides;
                output_w = (w + padding - i) / strides;
                filter = filters[f * filter_height * filter_width * depth + i * filter_width + j];
                data_to_process = _mm256_xor_si256(input, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                outputs[output_h * out_width * num_filters + output_w * num_filters + f] += 256 - 2 * (data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3]);
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
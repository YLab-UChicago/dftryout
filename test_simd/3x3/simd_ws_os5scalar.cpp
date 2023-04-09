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
    FILE *pFile = fopen("durations/simd_ws_os5scalar.txt", "a");
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

    int output_cache_1;
    int output_cache_2;
    int output_cache_3;
    int output_cache_4;
    int output_cache_5;

    int h;
    int w;
    int input_h;
    int input_w;
    int i;
    int j;

    c_start = std::clock();

    for (int f = 0; f < num_filters; f++)
    {
        output_cache_1 = 0;
        output_cache_2 = 0;
        output_cache_3 = 0;
        output_cache_4 = 0;
        output_cache_5 = 0;
        for (i = 0; i < filter_height - 1; i++)
        {
            for (j = 0; j < filter_width; j++)
            {
                filter = filters[f * depth * filter_height * filter_width + i * filter_width + j];

                h = 0;
                w = 0;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data_to_process = inputs[input_h * width * depth + input_w * depth];
                data_to_process = _mm256_xor_si256(data_to_process, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                output_cache_1 += data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];

                h = 0;
                w = 1;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data_to_process = inputs[input_h * width * depth + input_w * depth];
                data_to_process = _mm256_xor_si256(data_to_process, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                output_cache_2 += data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];

                h = 0;
                w = 2;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data_to_process = inputs[input_h * width * depth + input_w * depth];
                data_to_process = _mm256_xor_si256(data_to_process, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                output_cache_3 += data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];

                h = 0;
                w = 3;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data_to_process = inputs[input_h * width * depth + input_w * depth];
                data_to_process = _mm256_xor_si256(data_to_process, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                output_cache_4 += data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];

                h = 0;
                w = 4;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data_to_process = inputs[input_h * width * depth + input_w * depth];
                data_to_process = _mm256_xor_si256(data_to_process, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                output_cache_5 += data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];

                for (h = 0; h < out_height; h++)
                {
                    for (w = 5; w < out_width; w++)
                    {
                        input_h = h * strides + i - padding;
                        input_w = w * strides + j - padding;
                        data_to_process = inputs[input_h * width * depth + input_w * depth];
                        data_to_process = _mm256_xor_si256(data_to_process, filter);
                        data_to_process = _mm256_popcnt_epi64(data_to_process);
                        outputs[h * out_width * num_filters + w * num_filters + f] = data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];
                    }
                }
            }
        }

        i = filter_height - 1;
        for (j = 0; j < filter_width - 1; j++)
        {
            filter = filters[f * depth * filter_height * filter_width + i * filter_width + j];
            h = 0;
            w = 0;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data_to_process = inputs[input_h * width * depth + input_w * depth];
            data_to_process = _mm256_xor_si256(data_to_process, filter);
            data_to_process = _mm256_popcnt_epi64(data_to_process);
            output_cache_1 += data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];

            h = 0;
            w = 1;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data_to_process = inputs[input_h * width * depth + input_w * depth];
            data_to_process = _mm256_xor_si256(data_to_process, filter);
            data_to_process = _mm256_popcnt_epi64(data_to_process);
            output_cache_2 += data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];

            h = 0;
            w = 2;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data_to_process = inputs[input_h * width * depth + input_w * depth];
            data_to_process = _mm256_xor_si256(data_to_process, filter);
            data_to_process = _mm256_popcnt_epi64(data_to_process);
            output_cache_3 += data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];

            h = 0;
            w = 3;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data_to_process = inputs[input_h * width * depth + input_w * depth];
            data_to_process = _mm256_xor_si256(data_to_process, filter);
            data_to_process = _mm256_popcnt_epi64(data_to_process);
            output_cache_4 += data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];

            h = 0;
            w = 4;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data_to_process = inputs[input_h * width * depth + input_w * depth];
            data_to_process = _mm256_xor_si256(data_to_process, filter);
            data_to_process = _mm256_popcnt_epi64(data_to_process);
            output_cache_5 += data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];

            for (h = 0; h < out_height; h++)
            {
                for (w = 5; w < out_width; w++)
                {
                    input_h = h * strides + i - padding;
                    input_w = w * strides + j - padding;
                    data_to_process = inputs[input_h * width * depth + input_w * depth];
                    data_to_process = _mm256_xor_si256(data_to_process, filter);
                    data_to_process = _mm256_popcnt_epi64(data_to_process);
                    outputs[h * out_width * num_filters + w * num_filters + f] = data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];
                }
            }
        }

        filter = filters[f * depth * filter_height * filter_width + i * filter_width + j];
        h = 0;
        w = 0;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data_to_process = inputs[input_h * width * depth + input_w * depth];
        data_to_process = _mm256_xor_si256(data_to_process, filter);
        data_to_process = _mm256_popcnt_epi64(data_to_process);
        output_cache_1 += data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];
        outputs[0 * out_width * num_filters + 0 * num_filters + f] = output_cache_1;

        h = 0;
        w = 1;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data_to_process = inputs[input_h * width * depth + input_w * depth];
        data_to_process = _mm256_xor_si256(data_to_process, filter);
        data_to_process = _mm256_popcnt_epi64(data_to_process);
        output_cache_2 += data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];
        outputs[0 * out_width * num_filters + 1 * num_filters + f] = output_cache_2;

        h = 0;
        w = 2;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data_to_process = inputs[input_h * width * depth + input_w * depth];
        data_to_process = _mm256_xor_si256(data_to_process, filter);
        data_to_process = _mm256_popcnt_epi64(data_to_process);
        output_cache_3 += data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];
        outputs[0 * out_width * num_filters + 2 * num_filters + f] = output_cache_3;

        h = 0;
        w = 3;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data_to_process = inputs[input_h * width * depth + input_w * depth];
        data_to_process = _mm256_xor_si256(data_to_process, filter);
        data_to_process = _mm256_popcnt_epi64(data_to_process);
        output_cache_4 += data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];
        outputs[0 * out_width * num_filters + 3 * num_filters + f] = output_cache_4;

        h = 0;
        w = 4;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data_to_process = inputs[input_h * width * depth + input_w * depth];
        data_to_process = _mm256_xor_si256(data_to_process, filter);
        data_to_process = _mm256_popcnt_epi64(data_to_process);
        output_cache_5 += data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];
        outputs[0 * out_width * num_filters + 4 * num_filters + f] = output_cache_5;

        for (h = 0; h < out_height; h++)
        {
            for (w = 5; w < out_width; w++)
            {
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data_to_process = inputs[input_h * width * depth + input_w * depth];
                data_to_process = _mm256_xor_si256(data_to_process, filter);
                data_to_process = _mm256_popcnt_epi64(data_to_process);
                outputs[h * out_width * num_filters + w * num_filters + f] = data_to_process[0] + data_to_process[1] + data_to_process[2] + data_to_process[3];
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
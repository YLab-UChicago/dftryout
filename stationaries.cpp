//our implementations go here
#include <iostream>
#include <cmath>
#include "utils.cpp"
#include <smmintrin.h>
#include <immintrin.h>
#include <x86intrin.h>
using namespace std;

void conv_os_fp(const float *inputs, const float *filters, int batch_size, int height, int width, int depth, int num_filters, int filter_height, int filter_width, int padding, int strides, float *output)
{
    int out_height = (height - filter_height + 2 * padding) / strides + 1;
    int out_width = (width - filter_width + 2 * padding) / strides + 1;

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w++) {
                for (int d = 0; d < depth; d ++) {
                    for (int f = 0; f < num_filters; f++) {
                        float sum = 0;
                        for (int i = 0; i < filter_height; i++) {
                            for (int j = 0; j < filter_width; j++) {
                                int input_h = h * strides + i - padding;
                                int input_w = w * strides + j - padding;
                                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                                    sum += inputs[b * height * width * depth + input_h * width * depth + input_w * depth + d] * filters[f * filter_height * filter_width * depth + d* filter_height * filter_width + i * filter_width + j];
                                }
                            }
                        }
                        output[b * out_height * out_width * num_filters + h * out_width * num_filters + w * num_filters + f] = sum;
                    }
                }
            }
        }
    }
}

void conv_os_128_bn(const __m128i *inputs, const __m128i *filters, int batch_size, int height, int width, int depth, int num_filters, int filter_height, int filter_width, int padding, int strides, float *output)
{
    int out_height = (height - filter_height + 2 * padding) / strides + 1;
    int out_width = (width - filter_width + 2 * padding) / strides + 1;

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w++) {
                for (int d = 0; d < depth; d++) {
                    for (int f = 0; f < num_filters; f++) {
                        int sum = 0;
                        for (int i = 0; i < filter_height; i++) {
                            for (int j = 0; j < filter_width; j++) {
                                int input_h = h * strides + i - padding;
                                int input_w = w * strides + j - padding;
                                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                                    __m128i tmp = _mm_popcnt_epi64(_mm_xor_si128(inputs[b * height * width * depth + input_h * width * depth + input_w * depth + d],filters[f * filter_height * filter_width * depth + d* filter_height * filter_width + i * filter_width + j]));
                                    sum += tmp[0]+tmp[1];
                                }
                            }
                        }
                        output[b * out_height * out_width * num_filters + h * out_width * num_filters + w * num_filters + f] = sum;
                    }
                }
            }
        }
    }
}

/* 
 *  Anchoring Stationarity: Output stationary
 *  Auxillary Stationarity: None
 */
void conv_os_none__256_bn(const __m256i *inputs, const __m256i *filters, int batch_size, int height, int width, int depth, int num_filters, int filter_height, int filter_width, int padding, int strides, float *output)
{
    int out_height = (height - filter_height + 2 * padding) / strides + 1;
    int out_width = (width - filter_width + 2 * padding) / strides + 1;

    for (int h = 0; h < out_height; h++) {
        for (int w = 0; w < out_width; w++) {
            for (int d = 0; d < depth; d++) {
                for (int f = 0; f < num_filters; f++) {
                    int sum = 0;
                    for (int i = 0; i < filter_height; i++) {
                        for (int j = 0; j < filter_width; j++) {
                            int input_h = h * strides + i - padding;
                            int input_w = w * strides + j - padding;
                            if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                                __m256i data_to_process = inputs[input_h * width * depth + input_w * depth + d];
                                __m256i filter = filters[f * filter_height * filter_width * depth + d* filter_height * filter_width + i * filter_width + j];
                                data_to_process = _mm256_xor_si256(data_to_process,filter);
                                data_to_process = _mm256_popcnt_epi64(data_to_process);
                                sum += 256 - 2*(data_to_process[0]+data_to_process[1]+data_to_process[2]+data_to_process[3]);
                            }
                        }
                    }
                    output[h * out_width * num_filters + w * num_filters + f] = sum;
                }
            }
        }
    }
}

/* 
 *  Anchoring Stationarity: Output stationary
 *  Auxillary Stationarity: Weight stationary
 *  For 16 Vector Registers of 128 bits
 *  
 *  Anchoring Stationarity takes 2 registers of size 256 bits (or 3, need to confirm)
 *  Let's try to cache 5 weights of 256 bits into vector registers
 */
void conv_os_ws_256_bn(const __m256i *inputs, const __m256i *filters, int height, int width, int depth, int num_filters, int filter_height, int filter_width, int padding, int strides, float *output)
{
    int out_height = (height - filter_height + 2 * padding) / strides + 1;
    int out_width = (width - filter_width + 2 * padding) / strides + 1;

    


        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w++) {
                for (int d = 0; d < depth; d++) {
                    for (int f = 0; f < num_filters; f++) {
                
                    int sum = 0;
                    for (int i = 0; i < filter_height; i++) {
                        for (int j = 0; j < filter_width; j++) {
                            int input_h = h * strides + i - padding;
                            int input_w = w * strides + j - padding;
                            if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                                __m256i data_to_process = inputs[input_h * width * depth + input_w * depth + d];
                                __m256i filter = filters[f * filter_height * filter_width * depth + d* filter_height * filter_width + i * filter_width + j];
                                data_to_process = _mm256_xor_si256(data_to_process,filter);
                                data_to_process = _mm256_popcnt_epi64(data_to_process);
                                sum += 256 - 2*(data_to_process[0]+data_to_process[1]+data_to_process[2]+data_to_process[3]);
                            }
                        }
                    }
                    output[h * out_width * num_filters + w * num_filters + f] = sum;
                }
            }
        }
    }
    
}

void conv_os_512_bn(const __m512i *inputs, const __m512i *filters, int batch_size, int height, int width, int depth, int num_filters, int filter_height, int filter_width, int padding, int strides, float *output)
{
    int out_height = (height - filter_height + 2 * padding) / strides + 1;
    int out_width = (width - filter_width + 2 * padding) / strides + 1;

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w++) {
                for (int d = 0; d < depth; d++) {
                    for (int f = 0; f < num_filters; f++) {
                        int sum = 0;
                        for (int i = 0; i < filter_height; i++) {
                            for (int j = 0; j < filter_width; j++) {
                                int input_h = h * strides + i - padding;
                                int input_w = w * strides + j - padding;
                                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                                    __m512i tmp = _mm512_popcnt_epi64(_mm512_xor_si512(inputs[b * height * width * depth + input_h * width * depth + input_w * depth + d],filters[f * filter_height * filter_width * depth + d* filter_height * filter_width + i * filter_width + j]));
                                    sum += tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
                                }
                            }
                        }
                        output[b * out_height * out_width * num_filters + h * out_width * num_filters + w * num_filters + f] = sum;
                    }
                }
            }
        }
    }
}


/* 
 * Anchoring Stationarity: Input stationary
 * Auxillary Stationarity: Weight stationary, 
 */
void conv_is_ws_256_bn(const __m256i *inputs, const __m256i *filters, int batch_size, int height, int width, int depth, int num_filters, int filter_height, int filter_width, int padding, int strides, float *output)
{
    int out_height = (height - filter_height + 2 * padding) / strides + 1;
    int out_width = (width - filter_width + 2 * padding) / strides + 1;
    
    for (int b = 0; b  < batch_size; b ++) {
        for (int i = 0; i < height;i ++) {
            for (int j = 0; j < width; j ++) {
                for (int k = 0; k < depth; k ++) {
                    int idx = b * height * width * depth + i * width * depth + j * depth + k;
                    const __m256i input = inputs[idx];
                    for (int x = 0; x < num_filters; x ++)  {
                        for (int y = 0; y < filter_height; y ++) {
                            for (int z = 0; z < filter_width; z ++) {
                                int filter_idx = b * num_filters * filter_height * filter_width * depth + x * filter_height * filter_width * depth + k * filter_height * filter_width + y * filter_width + z;
                                if ((i + padding - z) % strides != 0 || (j + padding - y) % strides != 0) {
                                    continue;
                                }
                                int output_h = (i + padding - z) / strides; 
                                int output_w = (j + padding - y) / strides;
                                if (output_h < 0 || output_w < 0 || output_h >= out_height || output_w >= out_width) {
                                    continue;
                                }
                                const __m256i filter = filters[filter_idx];
                                __m256i data_to_process = _mm256_xor_si256(input,filter);
                                data_to_process = _mm256_popcnt_epi64(data_to_process);
                                int output_idx = b * out_height * out_width * num_filters + output_h * out_width * num_filters + output_w * num_filters + x;
                                output[output_idx] += 256 - 2*(data_to_process[0]+data_to_process[1]+data_to_process[2]+data_to_process[3]);
                            }
                        }
                    }
                }
            }
        }

    }
}


void conv_os_bn(const int *inputs, const int *filters, int batch_size, int height, int width, int depth, int num_filters, int filter_height, int filter_width, int padding, int strides, int *output)
{
    int out_height = (height - filter_height + 2 * padding) / strides + 1;
    int out_width = (width - filter_width + 2 * padding) / strides + 1;

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w++) {
                for (int f = 0; f < num_filters; f++) {
                    float sum = 0;
                    for (int i = 0; i < filter_height; i++) {
                        for (int j = 0; j < filter_width; j++) {
                            int input_h = h * strides + i - padding;
                            int input_w = w * strides + j - padding;
                            if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                                sum += xnor_popcount(inputs[b * height * width * depth + input_h * width * depth + input_w * depth + f],filters[f * filter_height * filter_width + i * filter_width + j]);
                            }
                        }
                    }
                    output[b * out_height * out_width * num_filters + h * out_width * num_filters + w * num_filters + f] = sum;
                }
            }
        }
    }
}


void conv_ws_fp(const float *inputs, const float *filters, int batch_size, int height, int width, int depth, int num_filters, int filter_height, int filter_width, int padding, int strides, float *output)
{
    int out_height = (height - filter_height + 2 * padding) / strides + 1;
    int out_width = (width - filter_width + 2 * padding) / strides + 1;

    for (int b = 0; b < batch_size; b++) {
        for (int f = 0; f < num_filters; f++) {
            for (int i = 0; i < filter_height; i++) {
                for (int j = 0; j < filter_width; j++) {
                    for (int d = 0; d < depth; d++) {
                        float filter = filters[f * depth * filter_height * filter_width + d * filter_height * filter_width + i * filter_width + j];
                        for (int h = 0; h < out_height; h++) {
                            for (int w = 0; w < out_width; w++) {
                                int input_h = h * strides + i - padding;
                                int input_w = w * strides + j - padding;
                                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                                    output[b * out_height * out_width * num_filters + h * out_width * num_filters + w * num_filters + f] += inputs[b * height * width * depth + input_h * width * depth + input_w * depth + d] * filter;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void conv_ws_bn(const int *inputs, const int *filters, int batch_size, int height, int width, int depth, int num_filters, int filter_height, int filter_width, int padding, int strides, int *output)
{
    int out_height = (height - filter_height + 2 * padding) / strides + 1;
    int out_width = (width - filter_width + 2 * padding) / strides + 1;

    for (int b = 0; b < batch_size; b++) {
        for (int f = 0; f < num_filters; f++) {
            for (int i = 0; i < filter_height; i++) {
                for (int j = 0; j < filter_width; j++) {
                    for (int d = 0; d < depth; d++) {
                        int filter = filters[f * depth * filter_height * filter_width + d * filter_height * filter_width + i * filter_width + j];
                        for (int h = 0; h < out_height; h++) {
                            for (int w = 0; w < out_width; w++) {
                                int input_h = h * strides + i - padding;
                                int input_w = w * strides + j - padding;
                                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                                    output[b * out_height * out_width * num_filters + h * out_width * num_filters + w * num_filters + f] += xnor_popcount(inputs[b * height * width * depth + input_h * width * depth + input_w * depth + d],filter);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void conv_ws_fp_wrong(const float *inputs, const float *filters, int batch_size, int height, int width, int depth, int num_filters, int filter_height, int filter_width, int padding, int strides, float *output)
{
    int out_height = (height - filter_height + 2 * padding) / strides + 1;
    int out_width = (width - filter_width + 2 * padding) / strides + 1;

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w++) {
                for (int f = 0; f < num_filters; f++) {
                    for (int i = 0; i < filter_height; i++) {
                        for (int j = 0; j < filter_width; j++) {
                            for (int d = 0; d < depth; d++) {
                                int input_h = h * strides + i - padding;
                                int input_w = w * strides + j - padding;
                                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                                    float input = inputs[b * height * width * depth + input_h * width * depth + input_w * depth + d];
                                    float filter = filters[f * depth * filter_height * filter_width + d * filter_width * filter_height + i * filter_width + j];
                                    output[b * out_height * out_width * num_filters + h * out_width * num_filters + w * num_filters + f] += input * filter;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


void conv_ws_bn_wrong(const int *inputs, const int *filters, int batch_size, int height, int width, int depth, int num_filters, int filter_height, int filter_width, int padding, int strides, int *output)
{
    int out_height = (height - filter_height + 2 * padding) / strides + 1;
    int out_width = (width - filter_width + 2 * padding) / strides + 1;

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w++) {
                for (int f = 0; f < num_filters; f++) {
                    for (int i = 0; i < filter_height; i++) {
                        for (int j = 0; j < filter_width; j++) {
                            for (int d = 0; d < depth; d++) {
                                int input_h = h * strides + i - padding;
                                int input_w = w * strides + j - padding;
                                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                                    int input = inputs[b * height * width * depth + input_h * width * depth + input_w * depth + d];
                                    int filter = filters[f * depth * filter_height * filter_width + d * filter_width * filter_height + i * filter_width + j];
                                    output[b * out_height * out_width * num_filters + h * out_width * num_filters + w * num_filters + f] += xnor_popcount(input, filter);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


void fully_connected_fp(const float *inputs, const float *weights, const float *biases, int input_size, int output_size, float *output)
{
    for (int i = 0; i < output_size; i++) {
        float dot_product = 0.0;
        for (int j = 0; j < input_size; j++) {
            dot_product += inputs[j] * weights[j * output_size + i];
        }
        output[i] = dot_product + biases[i];
    }
}

void fully_connected_layer_bn(const int *inputs, const int *weights, const float *biases, int input_size, int output_size, int *output)
{
    for (int i = 0; i < output_size; i++) {
        float dot_product = 0.0;
        for (int j = 0; j < input_size; j++) {
            dot_product += xnor_popcount(inputs[j], weights[j * output_size + i]);
        }
        output[i] = (dot_product + biases[i] > 0.0) ? 1.0 : 0.0;
    }
}

void conv_os_bn_blk(const int *inputs, const int *filters, int batch_size, int height, int width, int depth, int num_filters, int filter_height, int filter_width, int padding, int strides, int *output, int h_block, int w_block, int f_block)
{
    int out_height = (height - filter_height + 2 * padding) / strides + 1;
    int out_width = (width - filter_width + 2 * padding) / strides + 1;

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < out_height; h += h_block) {
            for (int w = 0; w < out_width; w += w_block) {
                for (int f = 0; f < num_filters; f += f_block) {
                    for (int h_ = 0; h_ < h_block; h_++) {
                        for (int w_ = 0; w_ < w_block; w_++) {
                            for (int f_ = 0; f_ < f_block; f_++) {
                                float sum_block = 0;
                                for (int i = 0; i < filter_height; i++) {
                                    for (int j = 0; j < filter_width; j++) {
                                        int input_h = (h+h_) * strides + i - padding;
                                        int input_w = (w+w_) * strides + j - padding;
                                        if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                                            sum_block += xnor_popcount(inputs[b * height * width * depth + input_h * width * depth + input_w * depth + (f+f_)],filters[(f+f_) * filter_height * filter_width + i * filter_width + j]);
                                        }
                                    }
                                }
                                output[b * out_height * out_width * num_filters + (h+h_) * out_width * num_filters + (w+w_) * num_filters + (f+f_)] = sum_block;
                            }
                        }
                    }
                    
                }
            }
        }
    }
}


void conv_is_ws_256_bn()


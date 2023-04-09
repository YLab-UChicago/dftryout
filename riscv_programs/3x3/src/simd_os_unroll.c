#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
//#include <riscv_vector.h>
//#include <rvv-intrin.h>

int main (int argc, char *argv[]) {
    FILE* pFile = fopen("../durations/simd_os_is6.txt", "a");
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
    int* inputs;
    int* outputs;
    int* data_2_iptr;
    int* filters;
    
    int output_depth;
    int pool_size;
    
    int input_size;
    float* input_bitrans;
    
    clock_t c_start;
    clock_t c_end;
    int layer_counter = 0;
    double time_elapsed_ms;
    
    height = 224;
    width = 224;
    depth = 256;
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
    inputs = (int*)malloc(sizeof(int)*(height+2*padding)*(width+2*padding)*depth/32);
    outputs = (int*)malloc(sizeof(int)*out_height*out_width*num_filters);
    filters = (int*)malloc(sizeof(int)*filter_height*filter_width*num_filters*depth/32);
    
    c_start = clock();

    __epi_4xi64 input_cache_1;
    __epi_4xi64 input_cache_2;
    __epi_4xi64 input_cache_3;
    __epi_4xi64 input_cache_4;
    __epi_4xi64 input_cache_5;
    __epi_4xi64 input_cache_6;
    
    __epi_4xi64 filter;
    __epi_4xi64 data_1;
    __epi_4xi64 data_2;

    unsigned long int gvl = __builtin_epi_vsetvlmax(64, 4);

    for (int f = 0; f < num_filters; f ++) {

        input_cache_1 = __builtin_epi_vbroadcast_4xi64(0,gvl);
        input_cache_2 = __builtin_epi_vbroadcast_4xi64(0,gvl);
        input_cache_3 = __builtin_epi_vbroadcast_4xi64(0,gvl);
        input_cache_4 = __builtin_epi_vbroadcast_4xi64(0,gvl);
        input_cache_5 = __builtin_epi_vbroadcast_4xi64(0,gvl);
        input_cache_6 = __builtin_epi_vbroadcast_4xi64(0,gvl);

            for (int h = 0; h < out_height; h ++) {
                for (int w = 0; w < out_width; w ++) {

                    int sum_block = 0;
                    int i = 0;
                    int j = 0;

                    int input_h;
                    int input_w;

                    input_h = h * strides + 0 - padding;
                    input_w = w * strides + 0 - padding;
                    filter = __builtin_epi_vload_4xi64((const long *)&filters[8*f*filter_height*filter_width+0],gvl);
                    data_1 = __builtin_epi_vxor_4xi64(input_cache_1,filter,gvl);
                    sum_block += 256 - 2*( __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,0))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,1))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,2))+__builtin_popcount(__builtin_epi_vextract_4xi64(data_1,3)));
                    input_cache_1 = input_cache_4;

                    input_h = h * strides + 0 - padding;
                    input_w = w * strides + 1 - padding;
                    filter = __builtin_epi_vload_4xi64((const long *) &filters[8*f*filter_height*filter_width+1],gvl);
                    data_1 = __builtin_epi_vxor_4xi64(input_cache_4,filter,gvl);
                    sum_block += 256 - 2*( __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,0))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,1))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,2))+__builtin_popcount(__builtin_epi_vextract_4xi64(data_1,3)));

                    input_h = h * strides + 0 - padding;
                    input_w = w * strides + 2 - padding;
                    filter = __builtin_epi_vload_4xi64((const long *)&filters[8*f*filter_height*filter_width+2],gvl);
                    data_2 = __builtin_epi_vload_4xi64((const long *)&inputs[input_h * width * depth + input_w * depth],gvl);
                    input_cache_4 = data_2;
                    data_1 = __builtin_epi_vxor_4xi64(data_2,filter,gvl);
                    sum_block += 256 - 2*( __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,0))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,1))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,2))+__builtin_popcount(__builtin_epi_vextract_4xi64(data_1,3)));
                    

                    input_h = h * strides + 1 - padding;
                    input_w = w * strides + 0 - padding;
                    filter = __builtin_epi_vload_4xi64((const long *)&filters[8*f*filter_height*filter_width+3],gvl);
                    data_1 = __builtin_epi_vxor_4xi64(input_cache_2,filter,gvl);
                    sum_block += 256 - 2*( __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,0))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,1))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,2))+__builtin_popcount(__builtin_epi_vextract_4xi64(data_1,3)));

                    input_h = h * strides + 1 - padding;
                    input_w = w * strides + 1 - padding;
                    filter = __builtin_epi_vload_4xi64((const long *)&filters[8*f*filter_height*filter_width+4],gvl);
                    data_1 =__builtin_epi_vxor_4xi64(input_cache_5,filter,gvl);
                    sum_block += 256 - 2*( __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,0))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,1))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,2))+__builtin_popcount(__builtin_epi_vextract_4xi64(data_1,3)));


                    input_h = h * strides + 1 - padding;
                    input_w = w * strides + 2 - padding;
                    data_2 = __builtin_epi_vload_4xi64((const long *)&inputs[input_h * width * depth + input_w * depth],gvl);
                    input_cache_5 = data_2;
                    filter = __builtin_epi_vload_4xi64((const long *)&filters[8*f*filter_height*filter_width+5],gvl);
                    data_1 = __builtin_epi_vxor_4xi64(data_2,filter,gvl);
                    sum_block += 256 - 2*( __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,0))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,1))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,2))+__builtin_popcount(__builtin_epi_vextract_4xi64(data_1,3)));
                    

                    input_h = h * strides + 2 - padding;
                    input_w = w * strides + 0 - padding;
                    filter = __builtin_epi_vload_4xi64((const long *)&filters[8*f*filter_height*filter_width+6],gvl);
                    data_1 = __builtin_epi_vxor_4xi64(input_cache_3,filter,gvl);
                    sum_block += 256 - 2*( __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,0))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,1))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,2))+__builtin_popcount(__builtin_epi_vextract_4xi64(data_1,3)));


                    input_h = h * strides + 2 - padding;
                    input_w = w * strides + 1 - padding;
                    filter = __builtin_epi_vload_4xi64((const long *)&filters[8*f*filter_height*filter_width+7],gvl);
                    data_1 = __builtin_epi_vxor_4xi64(input_cache_6,filter,gvl);
                    sum_block += 256 - 2*( __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,0))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,1))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,2))+__builtin_popcount(__builtin_epi_vextract_4xi64(data_1,3)));

                    input_h = h * strides + 2 - padding;
                    input_w = w * strides + 2 - padding;
                    data_2 = __builtin_epi_vload_4xi64((const long *)&inputs[input_h * width * depth + input_w * depth],gvl);
                    input_cache_6 = data_2;
                    filter = __builtin_epi_vload_4xi64((const long *)&filters[8*f*filter_height*filter_width+8],gvl);
                    data_1 = __builtin_epi_vxor_4xi64(data_2,filter,gvl);
                    sum_block += 256 - 2*( __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,0))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,1))+ __builtin_popcount(__builtin_epi_vextract_4xi64(data_1,2))+__builtin_popcount(__builtin_epi_vextract_4xi64(data_1,3)));

                    outputs[h * out_width * num_filters + w * num_filters + f] = sum_block;
                }
            }
        }
        
    
    c_end = clock();
    time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    fprintf(pFile, "%lf\n",time_elapsed_ms);
    
    free(inputs);
    free(outputs);
    free(filters);
}

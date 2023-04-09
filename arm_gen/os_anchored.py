from csnake import CodeWriter

def gen_OS_anchored_program(cw: CodeWriter, precision, vec_len, aux_stationarity):
    num_weight_cache = aux_stationarity["WS"]
    num_input_cache = aux_stationarity["IS"]
    name = str(precision)+"_"+str(vec_len)+"_os_ws"+str(num_weight_cache)+"_is"+str(num_input_cache)
    
    cw.add_line("#include <stdio.h>")
    cw.add_line("#include <string.h>")
    cw.add_line("#include <math.h>")
    cw.add_line("#include <ctime>")
    cw.add_line("#include <immintrin.h>")
    cw.add_line("#include <iostream>")
    cw.add_line("#include <cstdint>")
    cw.add_line("using namespace std;")
    cw.add_line("")
    cw.add_line("")
    cw.add_line("")

    cw.add_line("int main (int argc, char *argv[]) {")
    cw.indent()
    cw.add_line("FILE* pFile = fopen(\"" +
                "out/durations/"+name+".txt\", \"a\");")
    cw.add_line("int height;")
    cw.add_line("int width;")
    cw.add_line("int depth;")
    cw.add_line("int filter_height;")
    cw.add_line("int filter_width;")
    cw.add_line("int num_filters;")
    cw.add_line("int padding;")
    cw.add_line("int strides;")
    cw.add_line("int h_block;")
    cw.add_line("int w_block;")
    cw.add_line("int f_block;")
    cw.add_line("int d_block;")
    cw.add_line("int curr;")
    cw.add_line("int64_t* inputs;")
    cw.add_line("short* outputs;")
    cw.add_line("int64_t* filters;")
    cw.add_line("int output_depth;")
    cw.add_line("std::clock_t c_start;")
    cw.add_line("std::clock_t c_end;")
    cw.add_line("double time_elapsed_ms;")
    cw.add_line("")

    cw.add_line("height = atoi(argv[1]);")
    cw.add_line("width = atoi(argv[2]);")
    cw.add_line("depth = atoi(argv[3]);")
    cw.add_line("num_filters = atoi(argv[4]);")
    cw.add_line("filter_height = "+ filter_dim +";")
    cw.add_line("filter_width = "+ filter_dim +";")
    cw.add_line("padding = atoi(argv[5]);")
    cw.add_line("strides = atoi(argv[6]);")
    cw.add_line("int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);")
    cw.add_line("int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);")
    cw.add_line("inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);")
    cw.add_line("outputs = (short *)malloc(sizeof(short) * out_height * out_width * num_filters);")
    cw.add_line("filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);")
    cw.add_line("")

    cw.add_line("uint64x2x2_t data1;")
    cw.add_line("uint64x2x2_t data2;")

    for i in num_input_cache:
        cw.add_line("uint64x2x2_t input_cache_"+str(i)+";")
    
    for i in num_weight_cache:
        cw.add_line("uint64x2x2_t weight_cache_"+str(i)+";")

    cw.add_line("std::clock();")

    cw.add_line("for (int f = 0; f < num_filters; f++) {")
    cw.indent()

    for i in num_input_cache:
        cw.add_line("input_cache_1 = vld1q_u64_x2((const uint64_t *) &inputs[((0-padding) * width * depth /256 + ("+str(i)+"-padding) * depth /256) * depth /64]);")

    for i in num_weight_cache:
        cw.add_line("weight_cache_1 = vld1q_u64_x2((const uint64_t*) &filters[(f * filter_height * filter_width +"+ str(i) +")]);")



def gen_OS_anchored_program_block(precision, vec_len, aux_stationarity, block_scheme):
    num_weight_cache = aux_stationarity["WS"]
    num_input_cache = aux_stationarity["IS"]
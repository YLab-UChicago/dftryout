from csnake import CodeWriter

def gen_IS_anchored_program(precision, vec_len, aux_stationarity):
    if vec_len == 128:
        vec_type = "int64x2_t"
        load_func = "vld1q_s64"
    elif vec_len == 256:
        vec_type = "int64x2x2_t"
        load_func = "vld1q_s64_x2"
    elif vec_len == 512:
        vec_type = "int64x2x4_t"
        load_func = "vld1q_s64_x4"
    else:
        raise ValueError("invalid vector length")

    if precision == 1:
        operation_func = "veorq_s64"
        getres_func_start = "vaddvq_u8(vcntq_u8"
        getres_func_end = "))"
    elif precision == 8:
        operation_func = "vmul_s8"
        getres_func_start = "vaddvq_u8"
        getres_func_end = ")"
    
    num_weight_cache = aux_stationarity["WS"]
    num_output_cache = aux_stationarity["OS"]

        num_vec_op = int(vec_len / 128)
    
    cw.add_line("#include <stdio.h>")
    cw.add_line("#include <string.h>")
    cw.add_line("#include <math.h>")
    cw.add_line("#include <ctime>")
    cw.add_line("#include <iostream>")
    cw.add_line("#include <arm_neon.h>")
    cw.add_line("#include <m5ops.h>")
    cw.add_line("#include <algorithm>")
    cw.add_line("using namespace std;")
    cw.add_line("")
    cw.add_line("")
    cw.add_line("")

    cw.add_line("int main (int argc, char *argv[]) {")
    cw.indent()
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
    cw.add_line("int64_t* outputs;")
    cw.add_line("int64_t* filters;")
    cw.add_line("int output_depth;")
    cw.add_line("")

    cw.add_line("height = atoi(argv[1]);")
    cw.add_line("width = atoi(argv[2]);")
    cw.add_line("depth = atoi(argv[3]);")
    cw.add_line("num_filters = atoi(argv[4]);")
    cw.add_line("filter_height = "+ str(fh) +";")
    cw.add_line("filter_width = "+ str(fw)+";")
    cw.add_line("padding = "+str(fh-1)+";")
    cw.add_line("strides = "+str(stride)+";")
    cw.add_line("int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);")
    cw.add_line("int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);")
    cw.add_line("inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);")
    cw.add_line("outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);")
    cw.add_line("filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);")
    cw.add_line("")

    cw.add_line(vec_type+" data1;")
    cw.add_line(vec_type+" data2;")

    for i in range(num_input_cache):
        cw.add_line(vec_type+" input_cache_"+str(i)+";")
    
    cw.add_line("")
    
    for i in range(num_weight_cache):
        cw.add_line(vec_type+" weight_cache_"+str(i)+";")
    cw.add_line("")

def gen_IS_anchored_program_block(precision, vec_len, aux_stationarity, block_scheme):
    num_weight_cache = aux_stationarity["WS"]
    num_output_cache = aux_stationarity["OS"]


cw = CodeWriter()
gen_OS_anchored_program(cw, 1, 256, 3,3, {"WS":9,"OS":6},1)
cw.write_to_file("gen_os_ws9_is6.cpp")
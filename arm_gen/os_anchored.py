from csnake import CodeWriter

def gen_OS_anchored_program(cw: CodeWriter, precision, vec_len, fh, fw, aux_stationarity,stride):
    num_weight_cache = aux_stationarity["WS"]
    num_input_cache = aux_stationarity["IS"]
    name = str(precision)+"_"+str(vec_len)+"_os_ws"+str(num_weight_cache)+"_is"+str(num_input_cache)
    
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
    cw.add_line("filter_height = "+ str(fh) +";")
    cw.add_line("filter_width = "+ str(fw)+";")
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

    for i in range(num_input_cache):
        cw.add_line("uint64x2x2_t input_cache_"+str(i)+";")
    
    cw.add_line("")
    
    for i in range(num_weight_cache):
        cw.add_line("uint64x2x2_t weight_cache_"+str(i)+";")
    cw.add_line("")
    cw.add_line("std::clock();")

    cw.add_line("")
    cw.add_line("for (int f = 0; f < num_filters; f++) {")
    cw.indent()

    for i in range(num_input_cache):
        cw.add_line("input_cache_"+str(i)+" = vld1q_u64_x2((const uint64_t *) &inputs[((0-padding) * width * depth /256 + ("+str(i)+"-padding) * depth /256) * depth /64]);")

    for i in range(num_weight_cache):
        cw.add_line("weight_cache_"+str(i)+" = vld1q_u64_x2((const uint64_t*) &filters[(f * filter_height * filter_width +"+ str(i) +")]);")

    cw.add_line("for (int h = 0; h < height; h++) {")
    cw.indent()
    cw.add_line("for (int w = 0; w < width; w ++) {")
    cw.indent()
    cw.add_line("int sum_block = 0;")
    cw.add_line("int i = 0;")
    cw.add_line("int j = 0;")
    cw.add_line("int input_h;")
    cw.add_line("int input_w;")
    cw.add_line(" ")

    input_cache_end = fw - stride;
    input_cache_indices = []
    curr_input_base = 0
    
    count = 0
    for i in range(fh):
        for j in range(input_cache_end):
            if count < num_input_cache:
                input_cache_indices.append(i*fw+j)
                count += 1
    
    input_var_name = "data1"
    weight_var_name = "data2"

    should_inc_w = False

    for a in range(fw):
        if (should_inc_w):
            cw.add_line("w ++;")
        else: 
            should_inc_w = True
        for i in range(fh):
            for j in range(fw):
                cw.add_line("input_h = h * strides +" + str(i) +" - padding;")
                cw.add_line("input_w = w * strides +" + str(j) +" - padding;")
                idx = (i * fw + j - 1)
                if idx in input_cache_indices:
                    input_var_name = "input_cache_"+str(input_cache_indices.index(idx))
                else: 
                    input_var_name = "data1"
                    cw.add_line("data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);")

                if idx < num_weight_cache and idx >= 0:
                    weight_var_name = "weight_cache_"+str(idx)
                else:
                    weight_var_name = "data2"
                    cw.add_line("data2 = vld1q_u64_x2((const uint64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);")

                
                cw.add_line("data1.val[0] = veorq_u64("+input_var_name+".val[0],"+weight_var_name+".val[0]);")
                cw.add_line("data1.val[1] = veorq_u64("+input_var_name+".val[1],"+weight_var_name+".val[1]);")
                cw.add_line("sum_block += 256 - 2 * (vaddvq_u8(vcntq_u8("+input_var_name+".val[0])) + vaddvq_u8(vcntq_u8("+input_var_name+".val[1])));")
                cw.add_line("")

        curr_input_base += 1 % fw

    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")

    cw.add_line("")
    cw.add_line("m5_dump_reset_stats(0, 0);")
    cw.add_line("std::free(inputs);")
    cw.add_line("std::free(outputs);")
    cw.add_line("std::free(filters);")

    cw.dedent()
    cw.add_line("}")




def gen_OS_anchored_program_block(precision, vec_len, aux_stationarity, block_scheme):
    num_weight_cache = aux_stationarity["WS"]
    num_input_cache = aux_stationarity["IS"]


#test

cw = CodeWriter()
gen_OS_anchored_program(cw, 1, 256, 3, 3, {"WS":4,"IS":3},1)
cw.write_to_file("test.cpp")


from csnake import CodeWriter
from nn_ext_dataflows.arm_gen.src.utils import generate_inout_sequence

def gen_OS_anchored_program(cw: CodeWriter, precision, vec_len, fh, fw, aux_stationarity,stride):

    # Depending on the specified vector length
    # We generate code with appropriate vector types
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

    # We have implemented both 1-bit (binary) and 8-bit precisions
    # For each precision, we define the operation for
    # performing multiplication (for binary we perform multiplication
    # through exclusive-or/xor).
    # We also define corresponding variables to aggregate results
    # after calculation is done through reduction sums
    # For binarized version, we also do popcount during this process
    if precision == 1:
        operation_func = "veorq_s64"
        getres_func_start = "vaddvq_u8(vcntq_u8"
        getres_func_end = "))"
    elif precision == 8:
        operation_func = "vmulq_s8"
        getres_func_start = "vaddvq_u8"
        getres_func_end = ")"

    # We get the number of vector variables allocated to 
    # each possible auxiliary stationarity from the parameter
    # in this case, weight or output
    num_weight_cache = aux_stationarity["WS"]
    num_input_cache = aux_stationarity["IS"]

    # As ARM ISA requires performing computations with strictly
    # 128 bits as the unit, we divide vector length by 128 bits
    # to determine the number of operations needed to perform
    # in order to cover the whole vector variables.
    num_vec_op = int(vec_len / 128)
    
    # Generate code for importing required C++ modules
    cw.add_line("#include <stdio.h>")
    cw.add_line("#include <string.h>")
    cw.add_line("#include <math.h>")
    cw.add_line("#include <ctime>")
    cw.add_line("#include <iostream>")
    cw.add_line("#include <arm_neon.h>")
    cw.add_line("#include <m5ops.h>")
    cw.add_line("#include <algorithm>")
    cw.add_line("using namespace std;")

    # Spacing
    cw.add_line("")
    cw.add_line("")
    cw.add_line("")

    # Start of the main function
    cw.add_line("int main (int argc, char *argv[]) {")
    cw.indent()

    # Generate code for Declaration of required variables
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
    cw.add_line("std::clock_t c_start;")
    cw.add_line("std::clock_t c_end;")
    cw.add_line("double time_elapsed_ms;")
    cw.add_line("int out_height;")
    cw.add_line("int out_width;")
    cw.add_line("")
    
    # Initializing of local variable values
    cw.add_line("height = atoi(argv[1]);")
    cw.add_line("width = atoi(argv[2]);")
    cw.add_line("depth = "+str(vec_len)+";")
    cw.add_line("num_filters = atoi(argv[3]);")
    cw.add_line("filter_height = "+ str(fh) +";")
    cw.add_line("filter_width = "+ str(fw)+";")
    cw.add_line("padding = "+str(fh-1)+";")
    cw.add_line("strides = "+str(stride)+";")
    cw.add_line("out_height = ceil((height - filter_height + 2 * padding) / strides + 1);")
    cw.add_line("out_width = ceil((width - filter_width + 2 * padding) / strides + 1);")
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

    cw.add_line("m5_reset_stats(0, 0);")

    cw.add_line("")
    cw.add_line("for (int f = 0; f < num_filters; f++) {")
    cw.indent()

    for i in range(num_input_cache):
        cw.add_line("input_cache_"+str(i)+" = "+load_func+"((const int64_t *) &inputs[(0 * width * depth /"+str(vec_len)+" + "+str(i)+") * "+str(vec_len)+" /64]);")

    for i in range(num_weight_cache):
        cw.add_line("weight_cache_"+str(i)+" = "+load_func+"((const int64_t*) &filters[(f * filter_height * filter_width +"+ str(i) +")*"+str(vec_len)+"/64]);")


    cw.add_line(vec_type+" output;")

    cw.add_line("for (int h = 0; h < out_height; h++) {")
    cw.indent()
    cw.add_line("for (int w = 0; w < out_width; w ++) {")
    cw.indent()
    cw.add_line("int i = 0;")
    cw.add_line("int j = 0;")
    cw.add_line("int input_h;")
    cw.add_line("int input_w;")
    cw.add_line(" ")

    input_cache_end = fw - stride;
    input_cache_indices = []
    num_icache_byrow = {}
    curr_input_base = 0
    
    count = 0
    for i in range(fh):
        num_icache_byrow[i] = 0
        for j in range(input_cache_end):
            if count < num_input_cache:
                input_cache_indices.append(i*fw+j)
                num_icache_byrow[i] = num_icache_byrow[i] + 1
                count += 1
    
    input_var_name = "data1"
    weight_var_name = "data2"

    icache_unroll_sequence = generate_inout_sequence(fw,fh,stride,num_icache_byrow)

    for a in range(fw-stride):
        if a > 0:
            cw.add_line("w ++;")

        for i in range(fh):
            for j in range(fw):
                
                idx = (i * fw + j)
                cw.add_line("i = "+str(i)+";")
                cw.add_line("j = "+str(j)+";")
                cw.add_line("input_h = h * strides +" + str(i)+";" )
                cw.add_line("input_w = w * strides +" + str(j)+";")
                cw.add_line("if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {")
                cw.indent()
                if idx in input_cache_indices:
                    input_var_name = "input_cache_"+str(icache_unroll_sequence[a][i][j])
                else: 
                    if num_icache_byrow[i] > 0:
                        if (idx - stride) in input_cache_indices:
                            input_var_name = "input_cache_"+str(icache_unroll_sequence[(a+1)%(fw-stride)][i][j-stride])
                            cw.add_line(input_var_name+" = "+ load_func+ "((const int64_t *) &inputs[(input_h * width * depth /"+str(vec_len)+" + input_w * depth /"+str(vec_len)+") * depth /64]);")
                    else:
                        input_var_name = "data1"
                        cw.add_line("data1 = "+ load_func+ "((const int64_t *) &inputs[(input_h * width * depth /"+str(vec_len)+" + input_w * depth /"+str(vec_len)+") * depth /64]);")
                    

                if idx < num_weight_cache and idx >= 0:
                    weight_var_name = "weight_cache_"+str(idx)
                else:
                    weight_var_name = "data2"
                    cw.add_line("data2 = "+load_func+"((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);")

                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        cw.add_line("data1.val["+str(n)+"] = "+operation_func+"("+input_var_name+".val["+str(n)+"],"+weight_var_name+".val["+str(n)+"]);")
                else:
                    cw.add_line("data1 = "+operation_func+"("+input_var_name+","+weight_var_name+");")

                if precision == 1:
                    if num_vec_op > 1:
                        for n in range(num_vec_op):
                            cw.add_line("output.val["+str(n)+"] = vaddq_u8(output.val["+str(n)+"],vcntq_u8(data1.val["+str(n)+"]));")
                    else:
                        cw.add_line("output = vaddq_u8(output,vcntq_u8(data1));")
                elif precision == 8:
                    if num_vec_op > 1:
                        for n in range(num_vec_op):
                            cw.add_line("output.val["+str(n)+"] = vaddq_u8(output.val["+str(n)+"],data1.val["+str(n)+"]);")
                    else:
                        cw.add_line("output = vaddq_u8(output,data1);")

                cw.add_line("")
                cw.dedent()
                cw.add_line("}")
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] = "
        if num_vec_op > 1:
            for n in range(num_vec_op):
                if n > 0:
                    res_string += " + "
                res_string += "vaddvq_u8(output.val["+str(n)+"])"
        else:
            res_string += "vaddvq_u8(output)"
        res_string+=";"
        cw.add_line(res_string)
        cw.add_line("")

        curr_input_base += stride

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

def gen_OS_anchored_program_real(cw: CodeWriter, precision, vec_len, fh, fw, aux_stationarity,stride):

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
        operation_func = "vmulq_s8"
        getres_func_start = "vaddvq_u8"
        getres_func_end = ")"

    num_weight_cache = aux_stationarity["WS"]
    num_input_cache = aux_stationarity["IS"]

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
    cw.add_line("std::clock_t c_start;")
    cw.add_line("std::clock_t c_end;")
    cw.add_line("double time_elapsed_ms;")
    cw.add_line("int curr;")
    cw.add_line("int64_t* inputs;")
    cw.add_line("int64_t* outputs;")
    cw.add_line("int64_t* filters;")
    cw.add_line("int output_depth;")
    cw.add_line("")

    cw.add_line("height = atoi(argv[1]);")
    cw.add_line("width = atoi(argv[2]);")
    cw.add_line("depth = "+str(vec_len)+";")
    cw.add_line("num_filters = atoi(argv[3]);")
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

    cw.add_line("c_start = std::clock();")
    cw.add_line("")
    cw.add_line("for (int f = 0; f < num_filters; f++) {")
    cw.indent()

    for i in range(num_input_cache):
        cw.add_line("input_cache_"+str(i)+" = "+load_func+"((const int64_t *) &inputs[("+str(i)+" * width * depth /"+str(vec_len)+" + "+str(i)+") * "+str(vec_len)+" /64]);")

    for i in range(num_weight_cache):
        cw.add_line("weight_cache_"+str(i)+" = "+load_func+"((const int64_t*) &filters[(f * filter_height * filter_width +"+ str(i) +")*"+str(vec_len)+"/64]);")


    cw.add_line(vec_type+" output;")

    cw.add_line("for (int h = 0; h < out_height; h++) {")
    cw.indent()
    cw.add_line("for (int w = 0; w < out_width; w ++) {")
    cw.indent()
    cw.add_line("int i = 0;")
    cw.add_line("int j = 0;")
    cw.add_line("int input_h;")
    cw.add_line("int input_w;")
    cw.add_line(" ")

    input_cache_end = fw - stride;
    input_cache_indices = []
    num_icache_byrow = {}
    curr_input_base = 0
    
    count = 0
    for i in range(fh):
        num_icache_byrow[i] = 0
        for j in range(input_cache_end):
            if count < num_input_cache:
                input_cache_indices.append(i*fw+j)
                num_icache_byrow[i] = num_icache_byrow[i] + 1
                count += 1
    
    input_var_name = "data1"
    weight_var_name = "data2"

    icache_unroll_sequence = generate_inout_sequence(fw,fh,stride,num_icache_byrow)

    for a in range(fw-stride):
        if a > 0:
            cw.add_line("w ++;")

        for i in range(fh):
            for j in range(fw):
                
                idx = (i * fw + j)
                cw.add_line("i = "+str(i)+";")
                cw.add_line("j = "+str(j)+";")
                cw.add_line("input_h = h * strides +" + str(i)+";" )
                cw.add_line("input_w = w * strides +" + str(j)+";")
                cw.add_line("if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {")
                cw.indent()
                if idx in input_cache_indices:
                    input_var_name = "input_cache_"+str(icache_unroll_sequence[a][i][j])
                else: 
                    if num_icache_byrow[i] > 0:
                        if (idx - stride) in input_cache_indices:
                            input_var_name = "input_cache_"+str(icache_unroll_sequence[(a+1)%(fw-stride)][i][j-stride])
                            cw.add_line(input_var_name+" = "+ load_func+ "((const int64_t *) &inputs[(input_h * width * depth /"+str(vec_len)+" + input_w * depth /"+str(vec_len)+") * depth /64]);")
                    else:
                        input_var_name = "data1"
                        cw.add_line("data1 = "+ load_func+ "((const int64_t *) &inputs[(input_h * width * depth /"+str(vec_len)+" + input_w * depth /"+str(vec_len)+") * depth /64]);")
                    

                if idx < num_weight_cache and idx >= 0:
                    weight_var_name = "weight_cache_"+str(idx)
                else:
                    weight_var_name = "data2"
                    cw.add_line("data2 = "+load_func+"((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);")

                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        cw.add_line("data1.val["+str(n)+"] = "+operation_func+"("+input_var_name+".val["+str(n)+"],"+weight_var_name+".val["+str(n)+"]);")
                else:
                    cw.add_line("data1 = "+operation_func+"("+input_var_name+","+weight_var_name+");")

                if precision == 1:
                    if num_vec_op > 1:
                        for n in range(num_vec_op):
                            cw.add_line("output.val["+str(n)+"] = vaddq_u8(output.val["+str(n)+"],vcntq_u8(data1.val["+str(n)+"]));")
                    else:
                        cw.add_line("output = vaddq_u8(output,vcntq_u8(data1));")
                elif precision == 8:
                    if num_vec_op > 1:
                        for n in range(num_vec_op):
                            cw.add_line("output.val["+str(n)+"] = vaddq_u8(output.val["+str(n)+"],data1.val["+str(n)+"]);")
                    else:
                        cw.add_line("output = vaddq_u8(output,data1);")

                cw.add_line("")
                cw.dedent()
                cw.add_line("}")
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] = "
        if num_vec_op > 1:
            for n in range(num_vec_op):
                if n > 0:
                    res_string += " + "
                res_string += "vaddvq_u8(output.val["+str(n)+"])"
        else:
            res_string += "vaddvq_u8(output)"
        res_string+=";"
        cw.add_line(res_string)
        cw.add_line("")

        curr_input_base += stride

    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.add_line("c_end = std::clock();")
    cw.add_line("time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;")
    cw.add_line("printf(\"%lf\\n\", time_elapsed_ms);")
    cw.add_line("")
    cw.add_line("std::free(inputs);")
    cw.add_line("std::free(outputs);")
    cw.add_line("std::free(filters);")

    cw.dedent()
    cw.add_line("}")

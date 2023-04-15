from csnake import CodeWriter
import math

def gen_WS_anchored_program(cw: CodeWriter, precision, vec_len, fh, fw, aux_stationarity,stride):

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

    num_input_cache = aux_stationarity["IS"]
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
    
    for i in range(num_output_cache):
        cw.add_line(vec_type+" output_cache_"+str(i)+";")
    cw.add_line("")
    cw.add_line("m5_reset_stats(0, 0);")
    cw.add_line("")
    cw.add_line("for (int f = 0; f < num_filters; f++) {")
    cw.indent()

    for i in range(num_output_cache):
        for n in range(num_vec_op):
            cw.add_line(vec_type+" output_cache_"+str(i)+".val["+str(n)+"]=vdupq_n_u64(0);")

    for i in range(num_input_cache):
        cw.add_line(vec_type+" input_cache_"+str(i)+" = " + load_func + "((const int64_t *) &inputs[(("+ str(i // fw)+ "-padding) * width * depth /256 + ("+ str(i % fw)+"-padding) * depth /256) * "+str(vec_len)+" /64]);")

    cw.add_line("int i;")
    cw.add_line("int j;")
    cw.add_line("for (i = 0; i < filter_height - 1; i ++) {")
    cw.indent()
    cw.add_line("for (j = 0; j < filter_width; j ++) {")
    cw.indent()
    cw.add_line("int h = 0;")
    cw.add_line("int w = 0;")
    cw.add_line("int input_h;")
    cw.add_line("int input_w;")
    cw.add_line("data2 = "+load_func+"((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*"+str(vec_len)+"/64]);")
    cw.add_line('')

    should_unroll_num = max(num_input_cache,num_output_cache)

    input_var_name = "data1"

    for i in range(should_unroll_num):
        cw.add_line("")
        if i > 0:
            cw.add_line("w++;")
        cw.add_line("input_h = h * strides + i - padding;")
        cw.add_line("input_w = w * strides + j - padding;")
        if i < num_input_cache:
            input_var_name = "input_cache_"+str(i)
        else:
            input_var_name = "data1"
            cw.add_line("data1 = "+load_func+"((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*"+str(vec_len)+"/64]);")
        for n in range(num_vec_op):
            cw.add_line("data1.val["+str(n)+"] = "+operation_func+"("+input_var_name+".val["+str(n)+"]"+"data2.val["+str(n)+"]);")
        if i < num_output_cache:
            for n in range(num_vec_op):
                cw.add_line("output_cache_"+str(i)+".val["+str(n)+"] = vaddq_u8(output_cache_"+str(i)+".val["+str(n)+"],data1.val["+str(n)+"]);")
        else:
            if precision == 1:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
                for n in range(num_vec_op):
                    res_string += getres_func_start+"("+input_var_name+".val["+str(n)+"]"+getres_func_end
                    if n < num_vec_op - 1:
                        res_string += "+"
                res_string += ");"
            elif precision == 8:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
                for n in range(num_vec_op):
                    res_string += getres_func_start+"("+input_var_name+".val["+str(n)+"]"+getres_func_end
                    if n < range(num_vec_op) - 1:
                        res_string += "+"

                res_string += ";"

            cw.add_line(res_string)
            cw.add_line("")
        
    cw.add_line("for (h = 0; h < out_height; h++) {")
    cw.indent()
    cw.add_line("for (w = "+ str(should_unroll_num)+"; w < out_width; w++) {")
    cw.indent()
    cw.add_line("input_h = h * strides + i - padding;")
    cw.add_line("input_w = w * strides + j - padding;")
    cw.add_line("data1 = "+load_func+"((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * "+str(vec_len)+" /64]);")

    for n in range(num_vec_op):
        cw.add_line("data1.val["+str(n)+"] = "+operation_func+"(data1.val["+str(n)+"],data2.val["+str(n)+"]);")

    if precision == 1:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
        for n in range(num_vec_op):
            res_string += getres_func_start+"("+input_var_name+".val["+str(n)+"]"+getres_func_end
            if n < num_vec_op - 1:
                res_string += "+"
        res_string += ");"
    elif precision == 8:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
        for n in range(num_vec_op):
            res_string += getres_func_start+"("+input_var_name+".val["+str(n)+"]"+ getres_func_end
            if n < range(num_vec_op) - 1:
                res_string += "+"

        res_string += ";"
    cw.add_line(res_string)
    cw.dedent()

    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.add_line("")
    cw.add_line("")
    cw.add_line("for (j = 0; j < filter_width - 1; j ++) {")
    cw.indent()
    cw.add_line("data2 = "+load_func+"((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*"+str(vec_len)+"/64]);")
    cw.add_line("")

    for i in range(should_unroll_num):
        cw.add_line("")
        if i > 0:
            cw.add_line("w++;")
        cw.add_line("input_h = h * strides + i - padding;")
        cw.add_line("input_w = w * strides + j - padding;")
        if i < num_input_cache:
            input_var_name = "input_cache_"+str(i)
        else:
            input_var_name = "data1"
            cw.add_line("data1 = "+load_func+"((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*"+str(vec_len)+"/64]);")
        for n in range(num_vec_op):
            cw.add_line("data1.val["+str(n)+"] = "+operation_func+"("+input_var_name+".val["+str(n)+"]"+"data2.val["+str(n)+"]);")
        if i < num_output_cache:
            for n in range(num_vec_op):
                cw.add_line("output_cache_"+str(i)+".val["+str(n)+"] = vaddq_u8(output_cache_"+str(i)+".val["+str(n)+"],data1.val["+str(n)+"]);")
        else:
            if precision == 1:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
                for n in range(num_vec_op):
                    res_string += getres_func_start+"("+input_var_name+".val["+str(n)+"]"+getres_func_end
                    if n < num_vec_op - 1:
                        res_string += "+"
                res_string += ");"
            elif precision == 8:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
                for n in range(num_vec_op):
                    res_string += getres_func_start+"("+input_var_name+".val["+str(n)+"]"+getres_func_end
                    if n < range(num_vec_op) - 1:
                        res_string += "+"

                res_string += ";"

            cw.add_line(res_string)
            cw.add_line("")

    cw.add_line("")
        
    cw.add_line("for (h = 0; h < out_height; h++) {")
    cw.indent()
    cw.add_line("for (w = "+ str(should_unroll_num)+"; w < out_width; w++) {")
    cw.indent()
    cw.add_line("input_h = h * strides + i - padding;")
    cw.add_line("input_w = w * strides + j - padding;")
    cw.add_line("data1 = "+load_func+"((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * "+str(vec_len)+" /64]);")

    for n in range(num_vec_op):
        cw.add_line("data1.val["+str(n)+"] = "+operation_func+"(data1.val["+str(n)+"],data2.val["+str(n)+"]);")

    if precision == 1:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
        for n in range(num_vec_op):
            res_string += getres_func_start+"("+input_var_name+".val["+str(n)+"]"+getres_func_end
            if n < num_vec_op - 1:
                res_string += "+"
        res_string += ");"
    elif precision == 8:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
        for n in range(num_vec_op):
            res_string += getres_func_start+"("+input_var_name+".val["+str(n)+"]"+ getres_func_end
            if n < range(num_vec_op) - 1:
                res_string += "+"

        res_string += ";"
    cw.add_line(res_string)
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.add_line("data2 = "+load_func+"((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*"+str(vec_len)+"/64]);")
    cw.add_line("")
    

    for i in range(should_unroll_num):
        cw.add_line("")
        if i > 0:
            cw.add_line("w++;")
        cw.add_line("input_h = h * strides + i - padding;")
        cw.add_line("input_w = w * strides + j - padding;")
        if i < num_input_cache:
            input_var_name = "input_cache_"+str(i)
        else:
            input_var_name = "data1"
            cw.add_line("data1 = "+load_func+"((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*"+str(vec_len)+"/64]);")
        for n in range(num_vec_op):
            cw.add_line("data1.val["+str(n)+"] = "+operation_func+"("+input_var_name+".val["+str(n)+"]"+"data2.val["+str(n)+"]);")
        if i < num_output_cache:
            for n in range(num_vec_op):
                cw.add_line("output_cache_"+str(i)+".val["+str(n)+"] = vaddq_u8(output_cache_"+str(i)+".val["+str(n)+"],data1.val["+str(n)+"]);")
            if precision == 1:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
                for n in range(num_vec_op):
                    res_string += getres_func_start+"(output_cache_"+str(i)+".val["+str(n)+"]"+getres_func_end
                    if n < num_vec_op - 1:
                        res_string += "+"
                res_string += ");"
            elif precision == 8:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
                for n in range(num_vec_op):
                    res_string += getres_func_start+"(output_cache_"+str(i)+".val["+str(n)+"]"+getres_func_end
                    if n < range(num_vec_op) - 1:
                        res_string += "+"

                res_string += ";"

            cw.add_line(res_string)
            cw.add_line("")
                
        else:
            if precision == 1:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
                for n in range(num_vec_op):
                    res_string += getres_func_start+"("+input_var_name+".val["+str(n)+"]"+getres_func_end
                    if n < num_vec_op - 1:
                        res_string += "+"
                res_string += ");"
            elif precision == 8:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
                for n in range(num_vec_op):
                    res_string += getres_func_start+"("+input_var_name+".val["+str(n)+"]"+getres_func_end
                    if n < range(num_vec_op) - 1:
                        res_string += "+"

                res_string += ";"

            cw.add_line(res_string)
            cw.add_line("")

    cw.add_line("")
        
    cw.add_line("for (h = 0; h < out_height; h++) {")
    cw.indent()
    cw.add_line("for (w = "+ str(should_unroll_num)+"; w < out_width; w++) {")
    cw.indent()
    cw.add_line("input_h = h * strides + i - padding;")
    cw.add_line("input_w = w * strides + j - padding;")
    cw.add_line("data1 = "+load_func+"((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * "+str(vec_len)+" /64]);")

    for n in range(num_vec_op):
        cw.add_line("data1.val["+str(n)+"] = "+operation_func+"(data1.val["+str(n)+"],data2.val["+str(n)+"]);")

    if precision == 1:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
        for n in range(num_vec_op):
            res_string += getres_func_start+"("+input_var_name+".val["+str(n)+"]"+getres_func_end
            if n < num_vec_op - 1:
                res_string += "+"
        res_string += ");"
    elif precision == 8:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
        for n in range(num_vec_op):
            res_string += getres_func_start+"("+input_var_name+".val["+str(n)+"]"+ getres_func_end
            if n < range(num_vec_op) - 1:
                res_string += "+"

        res_string += ";"
    cw.add_line(res_string)
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")



        







def gen_WS_anchored_program_block(precision, vec_len, aux_stationarity, block_scheme):
    num_input_cache = aux_stationarity["IS"]
    num_output_cache = aux_stationarity["OS"]


cw = CodeWriter()
gen_WS_anchored_program(cw, 1, 256, 3,3, {"IS":9,"OS":6},1)
cw.write_to_file("gen_os_ws9_is6.cpp")
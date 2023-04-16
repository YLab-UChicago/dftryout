from csnake import CodeWriter

def gen_IS_anchored_program(cw: CodeWriter, precision, vec_len, fh, fw, aux_stationarity,stride):
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
    cw.add_line("int idx;")
    cw.add_line("int64_t* inputs;")
    cw.add_line("int64_t* outputs;")
    cw.add_line("int64_t* filters;")
    cw.add_line("int i = 0;")
    cw.add_line("int j = 0;")
    cw.add_line("int input_h;")
    cw.add_line("int input_w;")
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

    
    
    for i in range(num_weight_cache):
        cw.add_line(vec_type+" weight_cache_"+str(i)+";")
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
            cw.add_line("output_cache_"+str(i)+".val["+str(n)+"]=vdupq_n_u64(0);")

    for i in range(num_weight_cache):
        cw.add_line("weight_cache_"+str(i)+" = "+load_func+"((const int64_t*) &filters[(f * filter_height * filter_width +"+ str(i) +")*"+str(vec_len)+"/64]);")

    cw.add_line("for (int h = 0; h < height; h++) {")
    cw.indent()
    cw.add_line("for (int w = 0; w < width; w ++) {")
    cw.indent()
    cw.add_line("idx = h * width * depth / 64 + w * depth / 64;")
    cw.add_line("input = "+load_func+"((const int64_t *)&inputs[idx]);")

    cw.add_line(" ")

    output_cache_end = fw - stride;
    output_cache_indices = []
    num_ocache_byrow = {}
    curr_output_base = 0
    
    count = 0
    for i in range(fh):
        num_ocache_byrow[i] = 0
        for j in range(output_cache_end):
            if count < num_output_cache:
                output_cache_indices.append(i*fw+j)
                num_ocache_byrow[i] = num_ocache_byrow[i] + 1
                count += 1

    for a in range(fw-stride):
        if a > 0:
            cw.add_line("w ++;")
        for i in range(fh):
            for j in range(fw):
                
                idx = (i * fw + j)

                cw.add_line("i = "+str(fh - 1 - i)+";")
                cw.add_line("j = "+str(fw - 1 - j)+";")
                set_new_cache = False
                add_to_cache = False
                write_output = False
                
                if idx in output_cache_indices:
                    add_to_cache = True
                    output_var_name = "output_cache_"+str(((curr_output_base+output_cache_indices.index(idx)) % (fw-stride))+i*(fw-stride)) 
                    if idx % fw < stride:
                        write_output = True
                else: 
                    if num_ocache_byrow[i] > 0 and (idx - num_ocache_byrow[i]) in output_cache_indices:
                        output_var_name = "output_cache_"+str(((curr_output_base+output_cache_indices.index(idx - num_ocache_byrow[i])) % (fw-stride))+i*(fw-stride))

                        set_new_cache = True
                        
                    else:
                        output_var_name = "data1"
                        cw.add_line("output_h = (h + padding - i) / strides;")
                        cw.add_line("output_w = (w + padding - j) / strides;")


    
                if idx < num_weight_cache and idx >= 0:
                    weight_var_name = "weight_cache_"+str(idx)
                else:
                    weight_var_name = "data2"
                    cw.add_line("data2 = "+load_func+"((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);")
               

                for n in range(num_vec_op):
                    cw.add_line("data1.val["+str(n)+"] = "+operation_func+"(input.val["+str(n)+"],"+weight_var_name+".val["+str(n)+"]);")

                if set_new_cache:
                    if precision == 1:
                        for n in range(num_vec_op):
                            cw.add_line(output_var_name + ".val["+str(n)+"] = data1.val["+str(n)+"]")
                    elif precision == 8:
                        for n in range(num_vec_op):
                            cw.add_line(output_var_name + ".val["+str(n)+"] = data1.val["+str(n)+"]") 

                elif add_to_cache:
                    for n in range(num_vec_op):
                        cw.add_line(output_var_name+".val["+str(n)+ "]= addq_u8("+output_var_name+".val["+str(n)+"],data1.val["+str(n)+"])")
                
                if write_output:
                    if precision == 1:
                        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
                        for n in range(num_vec_op):
                            res_string += getres_func_start+"("+output_var_name+".val["+str(n)+"]"+getres_func_end
                            if n < num_vec_op - 1:
                                res_string += "+"
                        res_string += ");"
                    elif precision == 8:
                        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
                        for n in range(num_vec_op):
                            res_string += getres_func_start+"("+output_var_name+".val["+str(n)+"]"+getres_func_end
                            if n < range(num_vec_op) - 1:
                                res_string += "+"

                        res_string += ";"

                    cw.add_line(res_string)
                    cw.add_line("")



                cw.add_line("")

        cw.add_line("")

        curr_output_base += 1



    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")

def gen_IS_anchored_program_block(precision, vec_len, aux_stationarity, block_scheme):
    num_weight_cache = aux_stationarity["WS"]
    num_output_cache = aux_stationarity["OS"]


cw = CodeWriter()
gen_IS_anchored_program(cw, 1, 256, 3,3, {"WS":5,"OS":6},1)
cw.write_to_file("gen_is_ws5_os6.cpp")
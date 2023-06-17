from csnake import CodeWriter
import math

def gen_WS_anchored_program(cw: CodeWriter, precision, vec_len, fh, fw, aux_stationarity,stride):

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
    # in this case, input or output
    num_input_cache = aux_stationarity["IS"]
    num_output_cache = aux_stationarity["OS"]

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

    # Memory allocation for feature maps
    cw.add_line("inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);")
    cw.add_line("outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);")
    cw.add_line("filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);")
    cw.add_line("int h;")
    cw.add_line("int w;")
    cw.add_line("int input_h;")
    cw.add_line("int input_w;")
    cw.add_line("")

    # Declaration of Vector Variables for active computation
    # (i.e. for anchoring stationarity and uncahced auxiliary stationarities)
    cw.add_line(vec_type+" data1;")
    cw.add_line(vec_type+" data2;")

    # Declaration of Vector Variables for auxiliary stationarities
    #   Vector Variables for auxiliary input data
    for i in range(num_input_cache):
        cw.add_line(vec_type+" input_cache_"+str(i)+";")
    cw.add_line("")
    #   Vector Variables for auxiliary output data
    for i in range(num_output_cache):
        cw.add_line(vec_type+" output_cache_"+str(i)+";")

    # Spacing
    cw.add_line("")
    # Start taking stats from this line
    cw.add_line("m5_reset_stats(0, 0);")

    # Spacing
    cw.add_line("")

    # Start of the computation loop
    cw.add_line("for (int f = 0; f < num_filters; f++) {")
    cw.indent()

    # Initializations of the Vector Variables for auxiliary output
    #   Outputs are initialized to 0's for accumulation
    for i in range(num_output_cache):
        if num_vec_op > 1:
            for n in range(num_vec_op):
                cw.add_line("output_cache_"+str(i)+".val["+str(n)+"]=vdupq_n_u64(0);")
        else:
            cw.add_line("output_cache_"+str(i)+"=vdupq_n_u64(0);")
    #   Inputs are initialized by loading from feature map
    for i in range(num_input_cache):
        cw.add_line(vec_type+" input_cache_"+str(i)+" = " + load_func + "((const int64_t *) &inputs[("+ str(i // fw)+ " * width + "+ str(i % fw)+") * "+str(vec_len)+" /64]);")


    #   Loops with filter width and filter height
    cw.add_line("for (int i = 0; i < filter_height - 1; i ++) {")
    cw.indent()
    cw.add_line("for (int j = 0; j < filter_width; j ++) {")
    cw.indent()
    cw.add_line("h = 0;")
    cw.add_line("w = 0;")
    cw.add_line("data2 = "+load_func+"((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*"+str(vec_len)+"/64]);")
    cw.add_line('')

    #   Calculate the magnitude of unrolling for utilizing
    #       Auxiliary Stationarity
    #   This is determined by the maximum of number of 
    #       caches allocated to auxiliary input stationarity 
    #       and the number of caches allocated to auxiliary
    #       output stationarity.
    should_unroll_num = max(num_input_cache,num_output_cache)

    #   By default, we use the vector variable "data1" for
    #       uncached input. This variable would be overwritten
    #       if we find that the input is actually cached.
    input_var_name = "data1"

    #   We perform unrolling by the previously determined magnitude. 
    for i in range(should_unroll_num):
        cw.add_line("")
        # We assume that the number of elements to cache is smaller than
        #   the width of both output and input feature maps
        if i > 0:
            cw.add_line("w++;")

        # Calculate the corresponding input height and input width
        cw.add_line("input_h = h * strides + i;")
        cw.add_line("input_w = w * strides + j;")

        # See if the input is already cached to determine the
        #   vector variable for the input in this iteration
        if i < num_input_cache:
            input_var_name = "input_cache_"+str(i)
        else:
            input_var_name = "data1"
            cw.add_line("data1 = "+load_func+"((const int64_t *) &inputs[ (input_h * width + input_w )* depth /64]);")

        # Performs multiplication
        if num_vec_op > 1:
            for n in range(num_vec_op):
                cw.add_line("data1.val["+str(n)+"] = "+operation_func+"("+input_var_name+".val["+str(n)+"]"+",data2.val["+str(n)+"]);")
        else:
            cw.add_line("data1 = "+operation_func+"("+input_var_name+",data2);")

        # If output is cached, accumulate to output
        # Else, write back to feature map
        # Performs Reduction Sum (and popcounts for binary precision)
        if i < num_output_cache:
            if num_vec_op > 1:
                for n in range(num_vec_op):
                    cw.add_line("output_cache_"+str(i)+".val["+str(n)+"] = vaddq_u8(output_cache_"+str(i)+".val["+str(n)+"],data1.val["+str(n)+"]);")
            else:
                cw.add_line("output_cache_"+str(i)+" = vaddq_u8(output_cache_"+str(i)+",data1);")
        else:
            if precision == 1:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        res_string += getres_func_start+"(data1.val["+str(n)+"]"+getres_func_end
                        if n < num_vec_op - 1:
                            res_string += "+"
                else:
                    res_string += getres_func_start+"("+"data1"+getres_func_end
                res_string += ");"
            elif precision == 8:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+getres_func_end
                        if n < num_vec_op - 1:
                            res_string += "+"
                else:
                    res_string += getres_func_start+"("+"data1"+getres_func_end

                res_string += ";"

            cw.add_line(res_string)
            cw.add_line("")
        
    cw.add_line("for (h = 0; h < out_height; h++) {")
    cw.indent()
    input_var_name = "data1"
    cw.add_line("for (w = "+ str(should_unroll_num)+"; w < out_width; w++) {")
    cw.indent()
    cw.add_line("input_h = h * strides + i;")
    cw.add_line("input_w = w * strides + j;")
    cw.add_line("data1 = "+load_func+"((const int64_t *) &inputs[(input_h * width * depth /"+str(vec_len)+"+ input_w * depth /"+str(vec_len)+") * "+str(vec_len)+" /64]);")

    if num_vec_op > 1:
        for n in range(num_vec_op):
            cw.add_line("data1.val["+str(n)+"] = "+operation_func+"(data1.val["+str(n)+"],data2.val["+str(n)+"]);")
    else:
        cw.add_line("data1 = "+operation_func+"(data1,data2);")

    if precision == 1:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
        if num_vec_op > 1:
            for n in range(num_vec_op):
                res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+getres_func_end
                if n < num_vec_op - 1:
                    res_string += "+"
        else:
            res_string += getres_func_start+"("+"data1"+getres_func_end
        res_string += ");"
    elif precision == 8:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
        if num_vec_op > 1:
            for n in range(num_vec_op):
                res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+ getres_func_end
                if n < num_vec_op - 1:
                    res_string += "+"
        else:
            res_string += getres_func_start+"("+"data1"+ getres_func_end
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
        cw.add_line("input_h = h * strides + i;")
        cw.add_line("input_w = w * strides + j;")
        if i < num_input_cache:
            input_var_name = "input_cache_"+str(i)
        else:
            input_var_name = "data1"
            cw.add_line("data1 = "+load_func+"((const int64_t *) &inputs[(input_h * width * depth /"+str(vec_len)+"+ input_w * depth /"+str(vec_len)+") * "+str(vec_len)+" /64]);")
        if num_vec_op > 1:
            for n in range(num_vec_op):
                cw.add_line("data1.val["+str(n)+"] = "+operation_func+"("+input_var_name+".val["+str(n)+"]"+", data2.val["+str(n)+"]);")
        else:
            cw.add_line("data1 = "+operation_func+"("+input_var_name+", data2);")

        if i < num_output_cache:
            if num_vec_op > 1:
                for n in range(num_vec_op):
                    cw.add_line("output_cache_"+str(i)+".val["+str(n)+"] = vaddq_u8(output_cache_"+str(i)+".val["+str(n)+"],data1.val["+str(n)+"]);")
            else:
                cw.add_line("output_cache_"+str(i)+" = vaddq_u8(output_cache_"+str(i)+",data1);")
        else:
            if precision == 1:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+getres_func_end
                        if n < num_vec_op - 1:
                            res_string += "+"
                else:
                    res_string += getres_func_start+"("+"data1"+getres_func_end
                res_string += ");"
            elif precision == 8:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+getres_func_end
                        if n < num_vec_op - 1:
                            res_string += "+"
                else:
                    res_string += getres_func_start+"("+"data1"+getres_func_end

                res_string += ";"

            cw.add_line(res_string)
            cw.add_line("")

    cw.add_line("")
    input_var_name = "data1"
    cw.add_line("for (h = 0; h < out_height; h++) {")
    cw.indent()
    cw.add_line("for (w = "+ str(should_unroll_num)+"; w < out_width; w++) {")
    cw.indent()
    cw.add_line("input_h = h * strides + i;")
    cw.add_line("input_w = w * strides + j;")
    cw.add_line("data1 = "+load_func+"((const int64_t *) &inputs[(input_h * width * depth /"+str(vec_len)+"+ input_w * depth /"+str(vec_len)+") * "+str(vec_len)+" /64]);")

    if num_vec_op > 1:
        for n in range(num_vec_op):
            cw.add_line("data1.val["+str(n)+"] = "+operation_func+"(data1.val["+str(n)+"],data2.val["+str(n)+"]);")
    else:
        cw.add_line("data1 = "+operation_func+"(data1,data2);")


    if precision == 1:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
        if num_vec_op > 1:
            for n in range(num_vec_op):
                res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+getres_func_end
                if n < num_vec_op - 1:
                    res_string += "+"
        else:
            res_string += getres_func_start+"("+"data1"+getres_func_end
        res_string += ");"
    elif precision == 8:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
        if num_vec_op > 1:
            for n in range(num_vec_op):
                res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+ getres_func_end
                if n < num_vec_op - 1:
                    res_string += "+"
        else:
            res_string += getres_func_start+"("+"data1"+ getres_func_end

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
        cw.add_line("input_h = h * strides + i;")
        cw.add_line("input_w = w * strides + j;")
        if i < num_input_cache:
            input_var_name = "input_cache_"+str(i)
        else:
            input_var_name = "data1"
            cw.add_line("data1 = "+load_func+"((const int64_t *) &inputs[(input_h * width * depth /"+str(vec_len)+"+ input_w * depth /"+str(vec_len)+") * "+str(vec_len)+" /64]);")
        if num_vec_op > 1:
            for n in range(num_vec_op):
                cw.add_line("data1.val["+str(n)+"] = "+operation_func+"("+input_var_name+".val["+str(n)+"]"+",data2.val["+str(n)+"]);")
        else:
            cw.add_line("data1 = "+operation_func+"("+input_var_name+",data2);")
        if i < num_output_cache:
            if num_vec_op > 1:
                for n in range(num_vec_op):
                    cw.add_line("output_cache_"+str(i)+".val["+str(n)+"] = vaddq_u8(output_cache_"+str(i)+".val["+str(n)+"],data1.val["+str(n)+"]);")
            else:
                cw.add_line("output_cache_"+str(i)+" = vaddq_u8(output_cache_"+str(i)+",data1);")
            if precision == 1:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        res_string += getres_func_start+"(output_cache_"+str(i)+".val["+str(n)+"]"+getres_func_end
                        if n < num_vec_op - 1:
                            res_string += "+"
                else:
                    res_string += getres_func_start+"(output_cache_"+str(i)+getres_func_end
                res_string += ");"
            elif precision == 8:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        res_string += getres_func_start+"(output_cache_"+str(i)+".val["+str(n)+"]"+getres_func_end
                        if n < num_vec_op - 1:
                            res_string += "+"
                else:
                    res_string += getres_func_start+"(output_cache_"+str(i)+getres_func_end

                res_string += ";"

            cw.add_line(res_string)
            cw.add_line("")
                
        else:
            if precision == 1:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+getres_func_end
                        if n < num_vec_op - 1:
                            res_string += "+"
                else:
                    res_string += getres_func_start+"("+"data1"+getres_func_end
                res_string += ");"
            elif precision == 8:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+getres_func_end
                        if n < num_vec_op - 1:
                            res_string += "+"
                else:
                    res_string += getres_func_start+"("+"data1"+getres_func_end

                res_string += ";"

            cw.add_line(res_string)
            cw.add_line("")
    input_var_name = "data1"
    cw.add_line("")
        
    cw.add_line("for (h = 0; h < out_height; h++) {")
    cw.indent()
    cw.add_line("for (w = "+ str(should_unroll_num)+"; w < out_width; w++) {")
    cw.indent()
    cw.add_line("input_h = h * strides + i;")
    cw.add_line("input_w = w * strides + j;")
    cw.add_line("data1 = "+load_func+"((const int64_t *) &inputs[(input_h * width * depth /"+str(vec_len)+"+ input_w * depth /"+str(vec_len)+") * "+str(vec_len)+" /64]);")

    if num_vec_op > 1:
        for n in range(num_vec_op):
            cw.add_line("data1.val["+str(n)+"] = "+operation_func+"(data1.val["+str(n)+"],data2.val["+str(n)+"]);")
    else:
        cw.add_line("data1 = "+operation_func+"(data1,data2);")

    if precision == 1:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
        if num_vec_op > 1:
            for n in range(num_vec_op):
                res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+getres_func_end
                if n < num_vec_op - 1:
                    res_string += "+"
        else:
            res_string += getres_func_start+"("+"data1"+getres_func_end

        res_string += ");"
    elif precision == 8:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
        if num_vec_op > 1:
            for n in range(num_vec_op):
                res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+ getres_func_end
                if n < num_vec_op - 1:
                    res_string += "+"
        else:
            res_string += getres_func_start+"("+"data1"+ getres_func_end


        res_string += ";"
    cw.add_line(res_string)
    cw.dedent()
    
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("m5_dump_reset_stats(0, 0);")
    cw.add_line("free(inputs);")
    cw.add_line("free(outputs);")
    cw.add_line("free(filters);")
    cw.add_line("}")




        







def gen_WS_anchored_program_block(precision, vec_len, aux_stationarity, block_scheme):
    num_input_cache = aux_stationarity["IS"]
    num_output_cache = aux_stationarity["OS"]

def gen_WS_anchored_program_real(cw: CodeWriter, precision, vec_len, fh, fw, aux_stationarity,stride):

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
    cw.add_line("double time_elapsed_ms;")
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
    cw.add_line("int h;")
    cw.add_line("int w;")
    cw.add_line("int input_h;")
    cw.add_line("int input_w;")
    cw.add_line("std::clock_t c_start;")
    cw.add_line("std::clock_t c_end;")
    cw.add_line("")

    cw.add_line(vec_type+" data1;")
    cw.add_line(vec_type+" data2;")

    for i in range(num_input_cache):
        cw.add_line(vec_type+" input_cache_"+str(i)+";")
    
    cw.add_line("")
    
    for i in range(num_output_cache):
        cw.add_line(vec_type+" output_cache_"+str(i)+";")
    cw.add_line("")
    cw.add_line("c_start = std::clock();")
    cw.add_line("")
    cw.add_line("for (int f = 0; f < num_filters; f++) {")
    cw.indent()

    for i in range(num_output_cache):
        if num_vec_op > 1:
            for n in range(num_vec_op):
                cw.add_line("output_cache_"+str(i)+".val["+str(n)+"]=vdupq_n_u64(0);")
        else:
            cw.add_line("output_cache_"+str(i)+"=vdupq_n_u64(0);")

    for i in range(num_input_cache):
        cw.add_line(vec_type+" input_cache_"+str(i)+" = " + load_func + "((const int64_t *) &inputs[("+ str(i // fw)+ " * width + "+ str(i % fw)+") * "+str(vec_len)+" /64]);")

    cw.add_line("int i;")
    cw.add_line("int j;")
    cw.add_line("for (i = 0; i < filter_height - 1; i ++) {")
    cw.indent()
    cw.add_line("for (j = 0; j < filter_width; j ++) {")
    cw.indent()
    cw.add_line("h = 0;")
    cw.add_line("w = 0;")
    cw.add_line("data2 = "+load_func+"((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*"+str(vec_len)+"/64]);")
    cw.add_line('')

    should_unroll_num = max(num_input_cache,num_output_cache)

    input_var_name = "data1"

    for i in range(should_unroll_num):
        cw.add_line("")
        if i > 0:
            cw.add_line("w++;")
        cw.add_line("input_h = h * strides + i;")
        cw.add_line("input_w = w * strides + j;")
        if i < num_input_cache:
            input_var_name = "input_cache_"+str(i)
        else:
            input_var_name = "data1"
            cw.add_line("data1 = "+load_func+"((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*"+str(vec_len)+"/64]);")
        if num_vec_op > 1:
            for n in range(num_vec_op):
                cw.add_line("data1.val["+str(n)+"] = "+operation_func+"("+input_var_name+".val["+str(n)+"]"+",data2.val["+str(n)+"]);")
        else:
            cw.add_line("data1 = "+operation_func+"("+input_var_name+",data2);")

        if i < num_output_cache:
            if num_vec_op > 1:
                for n in range(num_vec_op):
                    cw.add_line("output_cache_"+str(i)+".val["+str(n)+"] = vaddq_u8(output_cache_"+str(i)+".val["+str(n)+"],data1.val["+str(n)+"]);")
            else:
                cw.add_line("output_cache_"+str(i)+" = vaddq_u8(output_cache_"+str(i)+",data1);")
        else:
            if precision == 1:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        res_string += getres_func_start+"(data1.val["+str(n)+"]"+getres_func_end
                        if n < num_vec_op - 1:
                            res_string += "+"
                else:
                    res_string += getres_func_start+"("+"data1"+getres_func_end
                res_string += ");"
            elif precision == 8:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+getres_func_end
                        if n < num_vec_op - 1:
                            res_string += "+"
                else:
                    res_string += getres_func_start+"("+"data1"+getres_func_end

                res_string += ";"

            cw.add_line(res_string)
            cw.add_line("")
        
    cw.add_line("for (h = 0; h < out_height; h++) {")
    cw.indent()
    input_var_name = "data1"
    cw.add_line("for (w = "+ str(should_unroll_num)+"; w < out_width; w++) {")
    cw.indent()
    cw.add_line("input_h = h * strides + i;")
    cw.add_line("input_w = w * strides + j;")
    cw.add_line("data1 = "+load_func+"((const int64_t *) &inputs[(input_h * width * depth /"+str(vec_len)+"+ input_w * depth /"+str(vec_len)+") * "+str(vec_len)+" /64]);")

    if num_vec_op > 1:
        for n in range(num_vec_op):
            cw.add_line("data1.val["+str(n)+"] = "+operation_func+"(data1.val["+str(n)+"],data2.val["+str(n)+"]);")
    else:
        cw.add_line("data1 = "+operation_func+"(data1,data2);")

    if precision == 1:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
        if num_vec_op > 1:
            for n in range(num_vec_op):
                res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+getres_func_end
                if n < num_vec_op - 1:
                    res_string += "+"
        else:
            res_string += getres_func_start+"("+"data1"+getres_func_end
        res_string += ");"
    elif precision == 8:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
        if num_vec_op > 1:
            for n in range(num_vec_op):
                res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+ getres_func_end
                if n < num_vec_op - 1:
                    res_string += "+"
        else:
            res_string += getres_func_start+"("+"data1"+ getres_func_end
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
        cw.add_line("input_h = h * strides + i;")
        cw.add_line("input_w = w * strides + j;")
        if i < num_input_cache:
            input_var_name = "input_cache_"+str(i)
        else:
            input_var_name = "data1"
            cw.add_line("data1 = "+load_func+"((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*"+str(vec_len)+"/64]);")
        if num_vec_op > 1:
            for n in range(num_vec_op):
                cw.add_line("data1.val["+str(n)+"] = "+operation_func+"("+input_var_name+".val["+str(n)+"]"+", data2.val["+str(n)+"]);")
        else:
            cw.add_line("data1 = "+operation_func+"("+input_var_name+", data2);")

        if i < num_output_cache:
            if num_vec_op > 1:
                for n in range(num_vec_op):
                    cw.add_line("output_cache_"+str(i)+".val["+str(n)+"] = vaddq_u8(output_cache_"+str(i)+".val["+str(n)+"],data1.val["+str(n)+"]);")
            else:
                cw.add_line("output_cache_"+str(i)+" = vaddq_u8(output_cache_"+str(i)+",data1);")
        else:
            if precision == 1:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+getres_func_end
                        if n < num_vec_op - 1:
                            res_string += "+"
                else:
                    res_string += getres_func_start+"("+"data1"+getres_func_end
                res_string += ");"
            elif precision == 8:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+getres_func_end
                        if n < num_vec_op - 1:
                            res_string += "+"
                else:
                    res_string += getres_func_start+"("+"data1"+getres_func_end

                res_string += ";"

            cw.add_line(res_string)
            cw.add_line("")

    cw.add_line("")
    input_var_name = "data1"
    cw.add_line("for (h = 0; h < out_height; h++) {")
    cw.indent()
    cw.add_line("for (w = "+ str(should_unroll_num)+"; w < out_width; w++) {")
    cw.indent()
    cw.add_line("input_h = h * strides + i;")
    cw.add_line("input_w = w * strides + j;")
    cw.add_line("data1 = "+load_func+"((const int64_t *) &inputs[(input_h * width * depth /"+str(vec_len)+" + input_w * depth /"+str(vec_len)+") * "+str(vec_len)+" /64]);")

    if num_vec_op > 1:
        for n in range(num_vec_op):
            cw.add_line("data1.val["+str(n)+"] = "+operation_func+"(data1.val["+str(n)+"],data2.val["+str(n)+"]);")
    else:
        cw.add_line("data1 = "+operation_func+"(data1,data2);")


    if precision == 1:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
        if num_vec_op > 1:
            for n in range(num_vec_op):
                res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+getres_func_end
                if n < num_vec_op - 1:
                    res_string += "+"
        else:
            res_string += getres_func_start+"("+"data1"+getres_func_end
        res_string += ");"
    elif precision == 8:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
        if num_vec_op > 1:
            for n in range(num_vec_op):
                res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+ getres_func_end
                if n < num_vec_op - 1:
                    res_string += "+"
        else:
            res_string += getres_func_start+"("+"data1"+ getres_func_end

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
        cw.add_line("input_h = h * strides + i;")
        cw.add_line("input_w = w * strides + j;")
        if i < num_input_cache:
            input_var_name = "input_cache_"+str(i)
        else:
            input_var_name = "data1"
            cw.add_line("data1 = "+load_func+"((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*"+str(vec_len)+"/64]);")
        if num_vec_op > 1:
            for n in range(num_vec_op):
                cw.add_line("data1.val["+str(n)+"] = "+operation_func+"("+input_var_name+".val["+str(n)+"]"+",data2.val["+str(n)+"]);")
        else:
            cw.add_line("data1 = "+operation_func+"("+input_var_name+",data2);")
        if i < num_output_cache:
            if num_vec_op > 1:
                for n in range(num_vec_op):
                    cw.add_line("output_cache_"+str(i)+".val["+str(n)+"] = vaddq_u8(output_cache_"+str(i)+".val["+str(n)+"],data1.val["+str(n)+"]);")
            else:
                cw.add_line("output_cache_"+str(i)+" = vaddq_u8(output_cache_"+str(i)+",data1);")
            if precision == 1:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        res_string += getres_func_start+"(output_cache_"+str(i)+".val["+str(n)+"]"+getres_func_end
                        if n < num_vec_op - 1:
                            res_string += "+"
                else:
                    res_string += getres_func_start+"(output_cache_"+str(i)+getres_func_end
                res_string += ");"
            elif precision == 8:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        res_string += getres_func_start+"(output_cache_"+str(i)+".val["+str(n)+"]"+getres_func_end
                        if n < num_vec_op - 1:
                            res_string += "+"
                else:
                    res_string += getres_func_start+"(output_cache_"+str(i)+getres_func_end

                res_string += ";"

            cw.add_line(res_string)
            cw.add_line("")
                
        else:
            if precision == 1:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+getres_func_end
                        if n < num_vec_op - 1:
                            res_string += "+"
                else:
                    res_string += getres_func_start+"("+"data1"+getres_func_end
                res_string += ");"
            elif precision == 8:
                res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+getres_func_end
                        if n < num_vec_op - 1:
                            res_string += "+"
                else:
                    res_string += getres_func_start+"("+"data1"+getres_func_end

                res_string += ";"

            cw.add_line(res_string)
            cw.add_line("")
    input_var_name = "data1"
    cw.add_line("")
        
    cw.add_line("for (h = 0; h < out_height; h++) {")
    cw.indent()
    cw.add_line("for (w = "+ str(should_unroll_num)+"; w < out_width; w++) {")
    cw.indent()
    cw.add_line("input_h = h * strides + i;")
    cw.add_line("input_w = w * strides + j;")
    cw.add_line("data1 = "+load_func+"((const int64_t *) &inputs[(input_h * width * depth /"+str(vec_len)+"+ input_w * depth /"+str(vec_len)+") * "+str(vec_len)+" /64]);")

    if num_vec_op > 1:
        for n in range(num_vec_op):
            cw.add_line("data1.val["+str(n)+"] = "+operation_func+"(data1.val["+str(n)+"],data2.val["+str(n)+"]);")
    else:
        cw.add_line("data1 = "+operation_func+"(data1,data2);")

    if precision == 1:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * ("
        if num_vec_op > 1:
            for n in range(num_vec_op):
                res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+getres_func_end
                if n < num_vec_op - 1:
                    res_string += "+"
        else:
            res_string += getres_func_start+"("+"data1"+getres_func_end

        res_string += ");"
    elif precision == 8:
        res_string = "outputs[h * out_width * num_filters + w * num_filters + f] += "
        if num_vec_op > 1:
            for n in range(num_vec_op):
                res_string += getres_func_start+"("+"data1"+".val["+str(n)+"]"+ getres_func_end
                if n < num_vec_op - 1:
                    res_string += "+"
        else:
            res_string += getres_func_start+"("+"data1"+ getres_func_end


        res_string += ";"
    cw.add_line(res_string)
    cw.dedent()
    
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.add_line("c_end = std::clock();")
    cw.add_line("time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;")
    cw.add_line("printf(\"%lf\\n\", time_elapsed_ms);")
    cw.dedent()
    cw.add_line("free(inputs);")
    cw.add_line("free(outputs);")
    cw.add_line("free(filters);")
    cw.add_line("}")




        





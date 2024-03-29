from csnake import CodeWriter
from nn_ext_dataflows.arm_gen.src.utils import generate_inout_sequence

def gen_IS_anchored_program(cw: CodeWriter, precision, vec_len, fh, fw, aux_stationarity,stride):
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
    cw.add_line("int output_h;")
    cw.add_line("int output_w;")
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
    cw.add_line("out_height = ceil((height - filter_height + 2 * padding)  + 1);")
    cw.add_line("out_width = ceil((width - filter_width + 2 * padding)  + 1);")

    # Memory allocation for feature maps
    cw.add_line("inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);")
    cw.add_line("outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);")
    cw.add_line("filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);")
    cw.add_line("")

    # Declaration of Vector Variables for active computation
    # (i.e. for anchoring stationarity and uncached auxiliary stationarities)
    cw.add_line(vec_type+" input;")
    cw.add_line(vec_type+" data1;")
    cw.add_line(vec_type+" data2;")

    # Declaration of Vector Variables for auxiliary stationarities
    #   Vector Variables for auxiliary weight data
    for i in range(num_weight_cache):
        cw.add_line(vec_type+" weight_cache_"+str(i)+";")
    cw.add_line("")
    #   Vector Variables for auxiliary output data
    for i in range(num_output_cache):
        cw.add_line(vec_type+" output_cache_"+str(i)+";")
    
    #   Spacing
    cw.add_line("")
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

    # Initializations of the Vector Variables for auxiliary weight
    #   Weights are initialized by loading
    for i in range(num_weight_cache):
        cw.add_line("weight_cache_"+str(i)+" = "+load_func+"((const int64_t*) &filters[(f * filter_height * filter_width +"+ str(i) +")*"+str(vec_len)+"/64]);")

    # h and w loops
    cw.add_line("for (int h = 0; h < height; h++) {")
    cw.indent()
    cw.add_line("for (int w = 0; w < width; w ++) {")
    cw.indent()

    # Calculate the current input index
    cw.add_line("idx = h * width * depth / 64 + w * depth / 64;")

    # Assigns the value to the input vector variable by loading
    cw.add_line("input = "+load_func+"((const int64_t *)&inputs[idx]);")
    cw.add_line(" ")

    output_cache_indices = []
    num_ocache_byrow = {}
    curr_output_base = 0

    # We perform sequential horizontal allocation for auxiliary
    #   outputs under input-anchored dataflows
    count = 0
    for i in range(fh):
        num_ocache_byrow[i] = 0
        for j in range(fw-1):
            if count < num_output_cache:
                output_cache_indices.append(i*fw+j)
                num_ocache_byrow[i] = num_ocache_byrow[i] + 1
                count += 1

    # Calling util function to determine the sequence of output
    #   usage after we perform secondary unrolling to bypass
    #   data transfer among vector registers.
    ocache_unroll_sequence = generate_inout_sequence(fw,fh,1,num_ocache_byrow)

    # Secondary unrolling to create fw-1 units
    for a in range(fw-1):
        # Manually increment the iterator w between 
        #   two unrolled units
        if a > 0:
            cw.add_line("w ++;")

        # Now, for each anchored input, we will want to have
        #   loop through all corresponding filters to project onto
        #   outputs. We determine the sequence of output utilization
        #   based on previously obtained information.
        for i in range(fh):
            for j in range(fw):
                idx = (i * fw + j)

                cw.add_line("i = "+str(fh - 1 - i)+";")
                cw.add_line("j = "+str(fw - 1 - j)+";")
                cw.add_line("output_h = floor((h - i));")
                cw.add_line("output_w = floor((w - j));")
                cw.add_line("if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {")
                cw.indent()
                set_new_cache = False
                add_to_cache = False
                write_output = True
                
                if idx in output_cache_indices:
                    add_to_cache = True
                    output_var_name = "output_cache_"+str(ocache_unroll_sequence[a][i][j])
                    if idx % fw >= 1:
                        write_output = False
                else: 
                    if num_ocache_byrow[i] > 0 and (idx - 1) in output_cache_indices:
                        output_var_name = "output_cache_"+str(ocache_unroll_sequence[(a+1)%(fw-1)][i][j-1])

                        set_new_cache = True
                        write_output = False
                        
                    else:
                        output_var_name = "data1"
                        cw.add_line("output_h = (h + padding - i) ;")
                        cw.add_line("output_w = (w + padding - j) ;")

            
    
                if idx < num_weight_cache and idx >= 0:
                    weight_var_name = "weight_cache_"+str(idx)
                else:
                    weight_var_name = "data2"
                    cw.add_line("data2 = "+load_func+"((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);")
               
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        if set_new_cache:
                            cw.add_line(output_var_name+".val["+str(n)+"] = "+operation_func+"(input.val["+str(n)+"],"+weight_var_name+".val["+str(n)+"]);")
                        else:
                            cw.add_line("data1.val["+str(n)+"] = "+operation_func+"(input.val["+str(n)+"],"+weight_var_name+".val["+str(n)+"]);")
                else:
                    if set_new_cache:
                        cw.add_line(output_var_name+" = "+operation_func+"(input,"+weight_var_name+");")
                    else:
                        cw.add_line("data1 = "+operation_func+"(input,"+weight_var_name+");")

                # We accumulate to vector variables for auxiliary outputs
                #   whenever we can. The following code determines it.
                if add_to_cache:
                    if num_vec_op > 1:
                        for n in range(num_vec_op):
                            cw.add_line(output_var_name+".val["+str(n)+ "]= vaddq_u8("+output_var_name+".val["+str(n)+"],data1.val["+str(n)+"]);")
                    else:
                        cw.add_line(output_var_name+" = vaddq_u8("+output_var_name+",data1);")
                
                # We reduce and write the vector variables back to the output
                #   feature maps if the output has been completely reused.
                if write_output:
                    if precision == 1:
                        res_string = "outputs[output_h * out_width * num_filters + output_w * num_filters + f] += 256 - 2 * ("
                        if num_vec_op > 1:
                            for n in range(num_vec_op):
                                res_string += getres_func_start+"("+output_var_name+".val["+str(n)+"]"+getres_func_end
                                if n < num_vec_op - 1:
                                    res_string += "+"
                        else:
                            res_string += getres_func_start+"("+output_var_name+getres_func_end
                        res_string += ");"
                    elif precision == 8:
                        res_string = "outputs[output_h * out_width * num_filters + output_w * num_filters + f] += "
                        if num_vec_op > 1:
                            for n in range(num_vec_op):
                                res_string += getres_func_start+"("+output_var_name+".val["+str(n)+"]"+getres_func_end
                                if n < num_vec_op - 1 :
                                    res_string += "+"
                        else:
                            res_string += getres_func_start+"("+output_var_name+getres_func_end

                        res_string += ";"

                    cw.add_line(res_string)
                    cw.add_line("")

                cw.dedent()
                cw.add_line("}")

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
    cw.add_line("m5_dump_reset_stats(0, 0);")
    cw.add_line("std::free(inputs);")
    cw.add_line("std::free(outputs);")
    cw.add_line("std::free(filters);")
    cw.add_line("}")


# This is basically the same as gen_IS_anchored_program
# Nonetheless, the "real" function adds real timing
#   for the process running this script. Instead of
#   logging simulation stats to the /log folder,
#   this function logs real timing in milliseconds.
def gen_IS_anchored_program_real(cw: CodeWriter, precision, vec_len, fh, fw, aux_stationarity,stride):

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
    cw.add_line("int output_h;")
    cw.add_line("int output_w;")
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
    cw.add_line("out_height = ceil((height - filter_height + 2 * padding)  + 1);")
    cw.add_line("out_width = ceil((width - filter_width + 2 * padding)  + 1);")

    # Memory allocation for feature maps
    cw.add_line("inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);")
    cw.add_line("outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);")
    cw.add_line("filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);")
    cw.add_line("")

    # Declaration of Vector Variables for active computation
    # (i.e. for anchoring stationarity and uncahced auxiliary stationarities)
    cw.add_line(vec_type+" input;")
    cw.add_line(vec_type+" data1;")
    cw.add_line(vec_type+" data2;")

    # Declaration of Vector Variables for auxiliary stationarities
    #   Vector Variables for auxiliary weight data
    for i in range(num_weight_cache):
        cw.add_line(vec_type+" weight_cache_"+str(i)+";")
    cw.add_line("")
    #   Vector Variables for auxiliary output data
    for i in range(num_output_cache):
        cw.add_line(vec_type+" output_cache_"+str(i)+";")
    
    #   Spacing
    cw.add_line("")
    cw.add_line("")
    
    # Memorize start time
    cw.add_line("c_start = std::clock();")

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

    # Initializations of the Vector Variables for auxiliary weight
    #   Weights are initialized by loading
    for i in range(num_weight_cache):
        cw.add_line("weight_cache_"+str(i)+" = "+load_func+"((const int64_t*) &filters[(f * filter_height * filter_width +"+ str(i) +")*"+str(vec_len)+"/64]);")


    # h and w loops
    cw.add_line("for (int h = 0; h < height; h++) {")
    cw.indent()
    cw.add_line("for (int w = 0; w < width; w ++) {")
    cw.indent()
    # Calculate the current input index
    cw.add_line("idx = h * width * depth / 64 + w * depth / 64;")

    # Assigns the value to the input vector variable by loading
    cw.add_line("input = "+load_func+"((const int64_t *)&inputs[idx]);")
    cw.add_line(" ")

    output_cache_indices = []
    num_ocache_byrow = {}
    curr_output_base = 0

    # We perform sequential horizontal allocation for auxiliary
    #   outputs under input-anchored dataflows
    count = 0
    for i in range(fh):
        num_ocache_byrow[i] = 0
        for j in range(fw-1):
            if count < num_output_cache:
                output_cache_indices.append(i*fw+j)
                num_ocache_byrow[i] = num_ocache_byrow[i] + 1
                count += 1

    # Calling util function to determine the sequence of output
    #   usage after we perform secondary unrolling to bypass
    #   data transfer among vector registers.
    ocache_unroll_sequence = generate_inout_sequence(fw,fh,1,num_ocache_byrow)

    # Secondary unrolling to create fw-1 units
    for a in range(fw-1):

        # Manually increment the iterator w between 
        #   two unrolled units
        if a > 0:
            cw.add_line("w ++;")

        # Now, for each anchored input, we will want to have
        #   loop through all corresponding filters to project onto
        #   outputs. We determine the sequence of output utilization
        #   based on previously obtained information.
        for i in range(fh):
            for j in range(fw):
                
                idx = (i * fw + j)

                cw.add_line("i = "+str(fh - 1 - i)+";")
                cw.add_line("j = "+str(fw - 1 - j)+";")
                cw.add_line("output_h = floor((h - i) );")
                cw.add_line("output_w = floor((w - j) );")
                cw.add_line("if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {")
                cw.indent()
                set_new_cache = False
                add_to_cache = False
                write_output = True
                
                if idx in output_cache_indices:
                    add_to_cache = True
                    output_var_name = "output_cache_"+str(ocache_unroll_sequence[a][i][j])
                    if idx % fw >= 1:
                        write_output = False
                else: 
                    if num_ocache_byrow[i] > 0 and (idx - 1) in output_cache_indices:
                        output_var_name = "output_cache_"+str(ocache_unroll_sequence[(a+1)%(fw-1)][i][j-1])

                        set_new_cache = True
                        write_output = False
                        
                    else:
                        output_var_name = "data1"
                        cw.add_line("output_h = (h + padding - i) ;")
                        cw.add_line("output_w = (w + padding - j) ;")

            
    
                if idx < num_weight_cache and idx >= 0:
                    weight_var_name = "weight_cache_"+str(idx)
                else:
                    weight_var_name = "data2"
                    cw.add_line("data2 = "+load_func+"((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);")
               
                if num_vec_op > 1:
                    for n in range(num_vec_op):
                        if set_new_cache:
                            cw.add_line(output_var_name+".val["+str(n)+"] = "+operation_func+"(input.val["+str(n)+"],"+weight_var_name+".val["+str(n)+"]);")
                        else:
                            cw.add_line("data1.val["+str(n)+"] = "+operation_func+"(input.val["+str(n)+"],"+weight_var_name+".val["+str(n)+"]);")
                else:
                    if set_new_cache:
                        cw.add_line(output_var_name+" = "+operation_func+"(input,"+weight_var_name+");")
                    else:
                        cw.add_line("data1 = "+operation_func+"(input,"+weight_var_name+");")

                # We accumulate to vector variables for auxiliary outputs
                #   whenever we can. The following code determines it.
                if add_to_cache:
                    if num_vec_op > 1:
                        for n in range(num_vec_op):
                            cw.add_line(output_var_name+".val["+str(n)+ "]= vaddq_u8("+output_var_name+".val["+str(n)+"],data1.val["+str(n)+"]);")
                    else:
                        cw.add_line(output_var_name+" = vaddq_u8("+output_var_name+",data1);")
                
                # We reduce and write the vector variables back to the output
                #   feature maps if the output has been completely reused.
                if write_output:
                    if precision == 1:
                        res_string = "outputs[output_h * out_width * num_filters + output_w * num_filters + f] += 256 - 2 * ("
                        if num_vec_op > 1:
                            for n in range(num_vec_op):
                                res_string += getres_func_start+"("+output_var_name+".val["+str(n)+"]"+getres_func_end
                                if n < num_vec_op - 1:
                                    res_string += "+"
                        else:
                            res_string += getres_func_start+"("+output_var_name+getres_func_end
                        res_string += ");"
                    elif precision == 8:
                        res_string = "outputs[output_h * out_width * num_filters + output_w * num_filters + f] += "
                        if num_vec_op > 1:
                            for n in range(num_vec_op):
                                res_string += getres_func_start+"("+output_var_name+".val["+str(n)+"]"+getres_func_end
                                if n < num_vec_op - 1 :
                                    res_string += "+"
                        else:
                            res_string += getres_func_start+"("+output_var_name+getres_func_end

                        res_string += ";"

                    cw.add_line(res_string)
                    cw.add_line("")

                cw.dedent()
                cw.add_line("}")

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
    cw.add_line("c_end = std::clock();")
    cw.add_line("time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;")
    cw.add_line("printf(\"%lf\\n\", time_elapsed_ms);")
    cw.add_line("")
    cw.add_line("std::free(inputs);")
    cw.add_line("std::free(outputs);")
    cw.add_line("std::free(filters);")
    cw.add_line("}")

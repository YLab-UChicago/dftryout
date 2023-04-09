'''''
This file is for implementing details of the code generator.
'''''

from csnake import CodeWriter, Variable, FormattedLiteral, Function

# Generates all include statements necessary to run the program

def generate_include_libraries(cw: CodeWriter):
    cw.add_line("#include <stdio.h>")
    cw.add_line("#include <string.h>")
    cw.add_line("#include <math.h>")
    cw.add_line("#include <ctime>")
    cw.add_line("#include <immintrin.h>")
    cw.add_line("#include <iostream>")
    cw.add_line("#include <cstdint>")
    cw.add_line("using namespace std;")

# Generates the xnor popcount helper function for binarized operations
def generate_xnor_popcount_helper(cw: CodeWriter):
    cw.add_line("int xnor_popcount(int a, int b) {")
    cw.indent()
    cw.add_line("return __builtin_popcount(~(a^b));")
    cw.dedent()
    cw.add_line("}")
    cw.add_line("")

# Sets up the header of main function


def generate_main_func_setup(cw: CodeWriter):
    cw.add_line("int main (int argc, char *argv[]) {")
    cw.indent()

# Declares local variables for all convolution operations in the program


def generate_conv_params_init(cw: CodeWriter, name: str, precision: int):
    if precision == 1:
        _generate_bi_conv_params_init(cw, name)
    if precision == 8:
        _generate_8bit_conv_params_init(cw, name)


def _generate_8bit_conv_params_init(cw: CodeWriter, name: str):
    cw.add_line("FILE* pFile = fopen(\"" +
                "out/durations/"+name+".txt\", \"a\");")
    cw.add_line("int batch_size;")
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
    cw.add_line("int8_t* inputs;")
    cw.add_line("int8_t* outputs;")
    cw.add_line("int8_t* filters;")
    cw.add_line("")


def _generate_bi_conv_params_init(cw: CodeWriter, name: str):
    cw.add_line("FILE* pFile = fopen(\"" +
                "out/durations/"+name+".txt\", \"a\");")
    cw.add_line("int batch_size;")
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
    cw.add_line("int* inputs;")
    cw.add_line("int* outputs;")
    cw.add_line("int* filters;")
    cw.add_line("")

# Declares local variables for all pooling operations in the program
#
# Note: Currently, several variables are shared, may need to distinguish later


def generate_pool_params_init(cw: CodeWriter):
    cw.add_line("int output_depth;")
    cw.add_line("int pool_size;")
    cw.add_line("")

# Declares local variables for all binary-transformation operations in the program
#
# Note: Currently, several variables are shared, may need to distinguish later


def generate_bitrans_params_init(cw: CodeWriter):
    cw.add_line("int input_size;")
    cw.add_line("float* input_bitrans;")
    cw.add_line("")

# Declares local variables for all fully-connected operations in the program
#
# Note: Currently, several variables are shared, may need to distinguish later


def generate_fc_params_init(cw: CodeWriter):
    cw.add_line("int fc_input_size;")
    cw.add_line("int fc_output_size;")
    cw.add_line("")

# Declares local variables for calculating the time elapsed for each layer
# Also, initializes a layer_counter local variable for counting processed layers


def generate_clock_params_init(cw: CodeWriter):
    cw.add_line("std::clock_t c_start;")
    cw.add_line("std::clock_t c_end;")
    cw.add_line("int layer_counter = 0;")
    cw.add_line("double time_elapsed_ms;")
    cw.add_line("")


# Generates update statements for all pararmeters related to convolution operations
# This immediately precedes all convolution layers
def generate_conv_params_update(cw: CodeWriter, layer, blocking_scheme, precision: int):
    if precision == 1:
        _generate_bi_conv_params_update(cw, layer.batch_size, layer.input_height, layer.input_width,
                                        layer.input_depth, layer.filter_depth, layer.filter_height,
                                        layer.filter_width, layer.padding, layer.stride,
                                        blocking_scheme[0], blocking_scheme[1], blocking_scheme[2], blocking_scheme[3])
    if precision == 8:
        _generate_8bit_conv_params_update(cw, layer.batch_size, layer.input_height, layer.input_width,
                                          layer.input_depth, layer.filter_depth, layer.filter_height,
                                          layer.filter_width, layer.padding, layer.stride,
                                          blocking_scheme[0], blocking_scheme[1], blocking_scheme[2], blocking_scheme[3])


def _generate_8bit_conv_params_update(cw: CodeWriter, batch_size, height, width, depth,
                                      num_filters, filter_height, filter_width, padding, strides,
                                      h_block, w_block, d_block, f_block):
    cw.add_line('batch_size = '+str(batch_size)+';')
    cw.add_line('height = '+str(height)+';')
    cw.add_line('width = '+str(width)+';')
    cw.add_line('depth = '+str(depth)+';')
    cw.add_line('num_filters = '+str(num_filters)+';')
    cw.add_line('filter_height = '+str(filter_height)+';')
    cw.add_line('filter_width = '+str(filter_width)+';')
    cw.add_line('padding = '+str(padding)+';')
    cw.add_line('strides = '+str(strides)+';')
    cw.add_line('h_block = '+str(h_block)+';')
    cw.add_line('w_block = '+str(w_block)+';')
    cw.add_line('d_block = '+str(d_block)+";")
    cw.add_line('f_block = '+str(f_block)+';')
    cw.add_line(
        "int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);")
    cw.add_line(
        "int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);")
    cw.add_line(
        "inputs = (int8_t*)malloc(sizeof(int8_t)*batch_size*(height+2*padding)*(width+2*padding)*depth);")
    cw.add_line(
        "outputs = (int8_t*)malloc(sizeof(int8_t)*batch_size*out_height*out_width*num_filters);")
    cw.add_line(
        "filters = (int8_t*)malloc(sizeof(int8_t)*batch_size*filter_height*filter_width*num_filters*depth);")
    cw.add_line("")
# Implements the generate_conv_param_update function


def _generate_bi_conv_params_update(cw: CodeWriter, batch_size, height, width, depth,
                                    num_filters, filter_height, filter_width, padding, strides,
                                    h_block, w_block, d_block, f_block):
    cw.add_line('batch_size = '+str(batch_size)+';')
    cw.add_line('height = '+str(height)+';')
    cw.add_line('width = '+str(width)+';')
    cw.add_line('depth = '+str(depth)+';')
    cw.add_line('num_filters = '+str(num_filters)+';')
    cw.add_line('filter_height = '+str(filter_height)+';')
    cw.add_line('filter_width = '+str(filter_width)+';')
    cw.add_line('padding = '+str(padding)+';')
    cw.add_line('strides = '+str(strides)+';')
    cw.add_line('h_block = '+str(h_block)+';')
    cw.add_line('w_block = '+str(w_block)+';')
    cw.add_line('d_block = '+str(d_block)+";")
    cw.add_line('f_block = '+str(f_block)+';')
    cw.add_line(
        "int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);")
    cw.add_line(
        "int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);")
    cw.add_line(
        "inputs = (int*)malloc(sizeof(int)*batch_size*(height+2*padding)*(width+2*padding)*depth);")
    cw.add_line(
        "outputs = (int*)malloc(sizeof(int)*batch_size*out_height*out_width*num_filters);")
    cw.add_line(
        "filters = (int*)malloc(sizeof(int)*batch_size*filter_height*filter_width*num_filters*depth);")
    cw.add_line("")

# Generates update statements for all pararmeters related to pooling operations
# This immediately precedes all pooling layers


def generate_pool_params_update(cw: CodeWriter, layer, pooling_scheme):
    _generate_pool_params_update(cw, layer.input_width, layer.input_height, layer.input_depth,
                                 layer.pool_size, pooling_scheme)

# Implements the generate_pool_param_update function


def _generate_pool_params_update(cw: CodeWriter, input_width, input_height, input_depth, pool_size, pooling_scheme):
    pass

# TODO: Generates update statements for all pararmeters related to convolution operations
# This immediately precedes all binary transformation layers


def generate_bitrans_params_update(input_size):
    pass

# Generates code details of a output stationary convolution layer given a loop order
# TODO: currently, generated code with non-trivial (> 1) blocking sizes throw segfaults
#       need to fix this issue


def generate_os_convolution_code(cw: CodeWriter, loop_order: tuple, num_bit: int):
    if num_bit == 1:
        _generate_1bit_os_convolution_code(cw, loop_order)
    elif num_bit == 8:
        _generate_8bit_os_convolution_code(cw, loop_order)


def _generate_8bit_os_convolution_code(cw: CodeWriter, loop_order: tuple):
    cw.add_line("for (int b = 0; b < batch_size; b++) {")

    init_indent_level = cw._indent_level
    outer_looping = loop_order[0]
    inner_looping = loop_order[1]
    cw.indent()

    for lp in outer_looping:
        if lp == "h":
            cw.add_line("for (int h = 0; h < out_height; h += h_block) {")
        if lp == "w":
            cw.add_line("for (int w = 0; w < out_width; w += w_block) {")
        if lp == "f":
            cw.add_line("for (int f = 0; f < num_filters; f += f_block) {")
        if lp == "d":
            cw.add_line("for (int d = 0; d < depth; d += d_block) {")
        cw.indent()

    for lp in inner_looping:
        if lp == "h":
            cw.add_line("for (int h_ = 0; h_ < h_block; h_++) {")
        if lp == "w":
            cw.add_line("for (int w_ = 0; w_ < w_block; w_++) {")
        if lp == "f":
            cw.add_line("for (int f_ = 0; f_ < f_block; f_++) {")
        if lp == "d":
            cw.add_line("for (int d_ = 0; d_ < d_block; d_ ++) {")
        cw.indent()

    cw.add_line("float sum_block = 0;")
    cw.add_line("for (int i = 0; i < filter_height; i++) {")
    cw.indent()
    cw.add_line("for (int j = 0; j < filter_width; j++) {")
    cw.indent()
    cw.add_line("int input_h = (h+h_) * strides + i - padding;")
    cw.add_line("int input_w = (w+w_) * strides + j - padding;")
    cw.add_line(
        "if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {")
    cw.indent()
    cw.add_line("sum_block += inputs[b * height * width * depth + input_h * width * depth + input_w * depth + (d+d_)]* filters[(f+f_) * filter_height * filter_width * depth + (d + d_)* filter_height * filter_width + i * filter_width + j];")
    cw.dedent()

    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.add_line(
        "outputs[b * out_height * out_width * num_filters + (h+h_) * out_width * num_filters + (w+w_) * num_filters + (f+f_)] = sum_block;")
    cw.dedent()

    while cw._indent_level > init_indent_level:
        cw.add_line("}")
        cw.dedent()

    cw.add_line("}")
    cw.add_line("")


def _generate_1bit_os_convolution_code(cw: CodeWriter, loop_order: tuple):
    cw.add_line("for (int b = 0; b < batch_size; b++) {")

    init_indent_level = cw._indent_level
    outer_looping = loop_order[0]
    inner_looping = loop_order[1]
    cw.indent()

    for lp in outer_looping:
        if lp == "h":
            cw.add_line("for (int h = 0; h < out_height; h += h_block) {")
        if lp == "w":
            cw.add_line("for (int w = 0; w < out_width; w += w_block) {")
        if lp == "f":
            cw.add_line("for (int f = 0; f < num_filters; f += f_block) {")
        if lp == "d":
            cw.add_line("for (int d = 0; d < depth; d += d_block) {")
        cw.indent()

    for lp in inner_looping:
        if lp == "h":
            cw.add_line("for (int h_ = 0; h_ < h_block; h_++) {")
        if lp == "w":
            cw.add_line("for (int w_ = 0; w_ < w_block; w_++) {")
        if lp == "f":
            cw.add_line("for (int f_ = 0; f_ < f_block; f_++) {")
        if lp == "d":
            cw.add_line("for (int d_ = 0; d_ < d_block; d_++) {")
        cw.indent()

    cw.add_line("float sum_block = 0;")
    cw.add_line("for (int i = 0; i < filter_height; i++) {")
    cw.indent()
    cw.add_line("for (int j = 0; j < filter_width; j++) {")
    cw.indent()
    cw.add_line("int input_h = (h+h_) * strides + i - padding;")
    cw.add_line("int input_w = (w+w_) * strides + j - padding;")
    cw.add_line(
        "if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {")
    cw.indent()
    cw.add_line(
        "sum_block += xnor_popcount(inputs[b * height * width * depth + input_h * width * depth + input_w * depth + (d+d_)], filters[(f+f_) * filter_height * filter_width * depth + (d + d_)* filter_height * filter_width + i * filter_width + j]);")
    cw.dedent()

    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.add_line(
        "outputs[b * out_height * out_width * num_filters + (h+h_) * out_width * num_filters + (w+w_) * num_filters + (f+f_)] = sum_block;")
    cw.dedent()

    while cw._indent_level > init_indent_level:
        cw.add_line("}")
        cw.dedent()

    cw.add_line("}")
    cw.add_line("")


# TODO
def _generate_1bit_512_os_convolution_code(cw: CodeWriter, loop_order: tuple):
    cw.add_line("for (int b = 0; b < batch_size; b++) {")

    init_indent_level = cw._indent_level
    outer_looping = loop_order[0]
    inner_looping = loop_order[1]
    cw.indent()

    for lp in outer_looping:
        if lp == "h":
            cw.add_line("for (int h = 0; h < out_height; h += h_block) {")
        if lp == "w":
            cw.add_line("for (int w = 0; w < out_width; w += w_block) {")
        if lp == "f":
            cw.add_line("for (int f = 0; f < num_filters; f += f_block) {")
        if lp == "d":
            cw.add_line("for (int d = 0; d < depth; d += d_block) {")
        cw.indent()

    for lp in inner_looping:
        if lp == "h":
            cw.add_line("for (int h_ = 0; h_ < h_block; h_++) {")
        if lp == "w":
            cw.add_line("for (int w_ = 0; w_ < w_block; w_++) {")
        if lp == "f":
            cw.add_line("for (int f_ = 0; f_ < f_block; f_++) {")
        if lp == "d":
            cw.add_line("for (int d_ = 0; d_ < d_block; d_++) {")
        cw.indent()

    cw.add_line("float sum_block = 0;")
    cw.add_line("for (int i = 0; i < filter_height; i++) {")
    cw.indent()
    cw.add_line("for (int j = 0; j < filter_width; j++) {")
    cw.indent()
    cw.add_line("int input_h = (h+h_) * strides + i - padding;")
    cw.add_line("int input_w = (w+w_) * strides + j - padding;")
    cw.add_line(
        "if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {")
    cw.indent()
    cw.add_line(
        "sum_block += xnor_popcount(inputs[b * height * width * depth + input_h * width * depth + input_w * depth + (d+d_)], filters[(f+f_) * filter_height * filter_width * depth + (d + d_)* filter_height * filter_width + i * filter_width + j]);")
    cw.dedent()

    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.add_line(
        "outputs[b * out_height * out_width * num_filters + (h+h_) * out_width * num_filters + (w+w_) * num_filters + (f+f_)] = sum_block;")
    cw.dedent()

    while cw._indent_level > init_indent_level:
        cw.add_line("}")
        cw.dedent()

    cw.add_line("}")
    cw.add_line("")

# TODO
def _generate_1bit_256_os_convolution_code(cw: CodeWriter, loop_order: tuple):
    cw.add_line("for (int b = 0; b < batch_size; b++) {")

    init_indent_level = cw._indent_level
    outer_looping = loop_order[0]
    inner_looping = loop_order[1]
    cw.indent()

    for lp in outer_looping:
        if lp == "h":
            cw.add_line("for (int h = 0; h < out_height; h += h_block) {")
        if lp == "w":
            cw.add_line("for (int w = 0; w < out_width; w += w_block) {")
        if lp == "f":
            cw.add_line("for (int f = 0; f < num_filters; f += f_block) {")
        if lp == "d":
            cw.add_line("for (int d = 0; d < depth; d += d_block) {")
        cw.indent()

    for lp in inner_looping:
        if lp == "h":
            cw.add_line("for (int h_ = 0; h_ < h_block; h_++) {")
        if lp == "w":
            cw.add_line("for (int w_ = 0; w_ < w_block; w_++) {")
        if lp == "f":
            cw.add_line("for (int f_ = 0; f_ < f_block; f_++) {")
        if lp == "d":
            cw.add_line("for (int d_ = 0; d_ < d_block; d_++) {")
        cw.indent()

    cw.add_line("float sum_block = 0;")
    cw.add_line("for (int i = 0; i < filter_height; i++) {")
    cw.indent()
    cw.add_line("for (int j = 0; j < filter_width; j++) {")
    cw.indent()
    cw.add_line("int input_h = (h+h_) * strides + i - padding;")
    cw.add_line("int input_w = (w+w_) * strides + j - padding;")
    cw.add_line(
        "if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {")
    cw.indent()
    cw.add_line(
        "sum_block += xnor_popcount(inputs[b * height * width * depth + input_h * width * depth + input_w * depth + (d+d_)], filters[(f+f_) * filter_height * filter_width * depth + (d + d_)* filter_height * filter_width + i * filter_width + j]);")
    cw.dedent()

    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.add_line(
        "outputs[b * out_height * out_width * num_filters + (h+h_) * out_width * num_filters + (w+w_) * num_filters + (f+f_)] = sum_block;")
    cw.dedent()

    while cw._indent_level > init_indent_level:
        cw.add_line("}")
        cw.dedent()

    cw.add_line("}")
    cw.add_line("")


# TODO
def _generate_1bit_128_os_convolution_code(cw: CodeWriter, loop_order: tuple):
    cw.add_line("for (int b = 0; b < batch_size; b++) {")

    init_indent_level = cw._indent_level
    outer_looping = loop_order[0]
    inner_looping = loop_order[1]
    cw.indent()

    for lp in outer_looping:
        if lp == "h":
            cw.add_line("for (int h = 0; h < out_height; h += h_block) {")
        if lp == "w":
            cw.add_line("for (int w = 0; w < out_width; w += w_block) {")
        if lp == "f":
            cw.add_line("for (int f = 0; f < num_filters; f += f_block) {")
        if lp == "d":
            cw.add_line("for (int d = 0; d < depth; d += d_block) {")
        cw.indent()

    for lp in inner_looping:
        if lp == "h":
            cw.add_line("for (int h_ = 0; h_ < h_block; h_++) {")
        if lp == "w":
            cw.add_line("for (int w_ = 0; w_ < w_block; w_++) {")
        if lp == "f":
            cw.add_line("for (int f_ = 0; f_ < f_block; f_++) {")
        if lp == "d":
            cw.add_line("for (int d_ = 0; d_ < d_block; d_++) {")
        cw.indent()

    cw.add_line("float sum_block = 0;")
    cw.add_line("for (int i = 0; i < filter_height; i++) {")
    cw.indent()
    cw.add_line("for (int j = 0; j < filter_width; j++) {")
    cw.indent()
    cw.add_line("int input_h = (h+h_) * strides + i - padding;")
    cw.add_line("int input_w = (w+w_) * strides + j - padding;")
    cw.add_line(
        "if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {")
    cw.indent()
    cw.add_line(
        "sum_block += xnor_popcount(inputs[b * height * width * depth + input_h * width * depth + input_w * depth + (d+d_)], filters[(f+f_) * filter_height * filter_width * depth + (d + d_)* filter_height * filter_width + i * filter_width + j]);")
    cw.dedent()

    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.add_line(
        "outputs[b * out_height * out_width * num_filters + (h+h_) * out_width * num_filters + (w+w_) * num_filters + (f+f_)] = sum_block;")
    cw.dedent()

    while cw._indent_level > init_indent_level:
        cw.add_line("}")
        cw.dedent()

    cw.add_line("}")
    cw.add_line("")

# Generates code details of a input stationary convolution layer given a loop order
# TODO: currently, generated code with non-trivial (> 1) blocking sizes throw segfaults
#       need to fix this issue
# TODO: Need to consider how to specify loop order formats
def generate_is_convolution_code(loop_order: tuple, cw: CodeWriter):
    cw.add_line("for (int b = 0; b < batch_size; b++) {")

    init_indent_level = cw._indent_level
    outer_looping = loop_order[0]
    inner_looping = loop_order[1]
    cw.indent()

    for lp in outer_looping:
        if lp == "h":
            cw.add_line("for (int h = 0; h < out_height; h += h_block) {")
        if lp == "w":
            cw.add_line("for (int w = 0; w < out_width; w += w_block) {")
        if lp == "f":
            cw.add_line("for (int f = 0; f < num_filters; f += f_block) {")
        cw.indent()

    for lp in inner_looping:
        if lp == "h":
            cw.add_line("for (int h_ = 0; h_ < h_block; h_++) {")
        if lp == "w":
            cw.add_line("for (int w_ = 0; w_ < w_block; w_++) {")
        if lp == "f":
            cw.add_line("for (int f_ = 0; f_ < f_block; f_++) {")
        cw.indent()

    cw.add_line("float sum_block = 0;")
    cw.add_line("for (int i = 0; i < filter_height; i++) {")
    cw.indent()
    cw.add_line("for (int j = 0; j < filter_width; j++) {")
    cw.indent()
    cw.add_line("int input_h = (h+h_) * strides + i - padding;")
    cw.add_line("int input_w = (w+w_) * strides + j - padding;")
    cw.add_line(
        "if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {")
    cw.indent()
    cw.add_line(
        "output[b * out_height * out_width * num_filters + h * out_width * num_filters + w * num_filters + f] += xnor_popcount(inputs[b * height * width * depth + input_h * width * depth + input_w * depth + d],filter);")
    cw.dedent()

    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()

    while cw._indent_level > init_indent_level:
        cw.add_line("}")
        cw.dedent()
    cw.add_line("}")
    cw.add_line("")

# Generates code details of a weight stationary convolution layer given a loop order
# TODO: currently, generated code with non-trivial (> 1) blocking sizes throw segfaults
#       need to fix this issue
# TODO: Need to consider how to specify loop order formats
def generate_ws_convolution_code(cw: CodeWriter, loop_order: tuple,precision: int):
    if precision == 1:
        _generate_1bit_ws_convolution_code(cw, loop_order)
    if precision == 8:
        _generate_8bit_ws_convolution_code(cw, loop_order)

def _generate_1bit_ws_convolution_code(cw: CodeWriter, loop_order: tuple):
    cw.add_line("for (int b = 0; b < batch_size; b++) {")

    init_indent_level = cw._indent_level
    outer_looping = loop_order[0]
    inner_looping = loop_order[1]
    cw.indent()

    # In this case, prior to getting the filter value, we have
    #   batch_size, num_filters, filter_height, filter_width, and depth
    # After getting the filter value, we have out_height and out_width

    # 'Prior' attributes
    #   nf - num_filters
    #   fh - filter_height
    #   fw - filter_width
    #   d - depth
    #
    # 'Post' attributes
    #   oh - out_height,
    #   ow - out_width
    #   These two we may want to block
    #   For now, we haven't yet implemented bocking

    for lp in outer_looping:
        if lp == "f":
            cw.add_line("for (int f = 0; f < num_filters; f += f_block) {")
        if lp == "fh":
            cw.add_line("for (int i = 0; i < filter_height; i += i_block) {")
        if lp == "fw":
            cw.add_line("for (int j = 0; j < filter_width; j += j_block) {")
        if lp == "d":
            cw.add_line("for (int d = 0; d < depth; d += d_block) {")
        cw.indent()

    for lp in inner_looping:
        if lp == "f":
            cw.add_line("for (int f_ = 0; f < f_block; f_++) {")
        if lp == "fh":
            cw.add_line("for (int i_ = 0; i < i_block; i_++) {")
        if lp == "fw":
            cw.add_line("for (int j_ = 0; j < j_block; j_++) {")
        if lp == "d":
            cw.add_line("for (int d_ = 0; d < d_block; d_++) {")
        cw.indent()

    cw.add_line(
        "int filter = filters[(f+f_) * filter_height * filter_width * depth + (d + d_)* filter_height * filter_width + (i+i_) * filter_width + (j+j_)]];")

    cw.add_line("for (int h = 0; h < out_height; h++) {")
    cw.indent()
    cw.add_line("for (int w = 0; w < out_weight; w+=) {")
    cw.indent()

    cw.add_line("int input_h = h * strides + (i+i_) - padding;")
    cw.add_line("int input_w = w * strides + (j+j_) - padding;")
    cw.add_line(
        "if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {")
    cw.indent()
    cw.add_line("output[b * out_height * out_width * num_filters + h * out_width * num_filters + w * num_filters + f] += xnor_popcount(inputs[b * height * width * depth + input_h * width * depth + input_w * depth + (d+d_)],filter);")

    while cw._indent_level > init_indent_level:
        cw.add_line("}")
        cw.dedent()
    cw.add_line("}")
    cw.add_line("")


def _generate_8bit_ws_convolution_code(cw: CodeWriter, loop_order: tuple):
    cw.add_line("for (int b = 0; b < batch_size; b++) {")

    init_indent_level = cw._indent_level
    outer_looping = loop_order[0]
    inner_looping = loop_order[1]
    cw.indent()

    # In this case, prior to getting the filter value, we have
    #   batch_size, num_filters, filter_height, filter_width, and depth
    # After getting the filter value, we have out_height and out_width

    # 'Prior' attributes
    #   nf - num_filters
    #   fh - filter_height
    #   fw - filter_width
    #   d - depth
    #
    # 'Post' attributes
    #   oh - out_height,
    #   ow - out_width
    #   These two we may want to block
    #   For now, we haven't yet implemented bocking

    for lp in outer_looping:
        if lp == "f":
            cw.add_line("for (int f = 0; f < num_filters; f += f_block) {")
        if lp == "fh":
            cw.add_line("for (int i = 0; i < filter_height; i += i_block) {")
        if lp == "fw":
            cw.add_line("for (int j = 0; j < filter_width; j += j_block) {")
        if lp == "d":
            cw.add_line("for (int d = 0; d < depth; d += d_block) {")
        cw.indent()

    for lp in inner_looping:
        if lp == "f":
            cw.add_line("for (int f_ = 0; f < f_block; f_++) {")
        if lp == "fh":
            cw.add_line("for (int i_ = 0; i < i_block; i_++) {")
        if lp == "fw":
            cw.add_line("for (int j_ = 0; j < j_block; j_++) {")
        if lp == "d":
            cw.add_line("for (int d_ = 0; d < d_block; d_++) {")
        cw.indent()

    cw.add_line(
        "int filter = filters[(f+f_) * filter_height * filter_width * depth + (d + d_)* filter_height * filter_width + (i+i_) * filter_width + (j+j_)]];")

    cw.add_line("for (int h = 0; h < out_height; h++) {")
    cw.indent()
    cw.add_line("for (int w = 0; w < out_weight; w+=) {")
    cw.indent()

    cw.add_line("int input_h = h * strides + (i+i_) - padding;")
    cw.add_line("int input_w = w * strides + (j+j_) - padding;")
    cw.add_line(
        "if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {")
    cw.indent()
    cw.add_line("output[b * out_height * out_width * num_filters + h * out_width * num_filters + w * num_filters + f] += inputs[b * height * width * depth + input_h * width * depth + input_w * depth + (d+d_)] * filter;")

    while cw._indent_level > init_indent_level:
        cw.add_line("}")
        cw.dedent()
    cw.add_line("}")
    cw.add_line("")

# Generates code details of a max-pooling layer
#   TODO: may test if this can also benefit from blocking


def generate_max_pooling_code(cw: CodeWriter):
    cw.add_line("output_width = input_width / pool_size")
    cw.add_line("output_height = input_height / pool_size")
    cw.add_line("output_depth = input_depth")
    cw.add_line("")
    cw.add_line("for (int d = 0; d < output_depth; d++) {")
    cw.indent()
    cw.add_line("for (int i = 0; i < output_height; i++) {")
    cw.indent()
    cw.add_line("for (int j = 0; j < output_width; j++) {")
    cw.indent()
    cw.add_line("curr = 0;")
    cw.add_line("for (int k = 0; k < pool_size; k++) {")
    cw.indent()
    cw.add_line("for (int l = 0; l < pool_size; l++) {")
    cw.indent()
    cw.add_line(
        "curr = curr | inputs[d * input_height * input_width + (i * pool_size + k) * input_width + j * pool_size + l];")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.add_line(
        "output[d * output_height * output_width + i * output_width + j] = curr;")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.add_line("")

# Generates code details of a average-pooling layer
#   TODO: may test if this can also benefit from blocking


def generate_avg_pooling_code(cw: CodeWriter):
    cw.add_line("output_width = input_width / pool_size")
    cw.add_line("output_height = input_height / pool_size")
    cw.add_line("output_depth = input_depth")
    cw.add_line("")
    cw.add_line("for (int d = 0; d < output_depth; d++) {")
    cw.indent()
    cw.add_line("for (int i = 0; i < output_height; i++) {")
    cw.indent()
    cw.add_line("for (int j = 0; j < output_width; j++) {")
    cw.indent()
    cw.add_line("curr = 0;")
    cw.add_line("for (int k = 0; k < pool_size; k++) {")
    cw.indent()
    cw.add_line("for (int l = 0; l < pool_size; l++) {")
    cw.indent()
    cw.add_line(
        "curr = curr + inputs[d * input_height * input_width + (i * pool_size + k) * input_width + j * pool_size + l];")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.add_line(
        "output[d * output_height * output_width + i * output_width + j] = curr / pool_size / pool_size;")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.add_line("")


# Generates a helper function for binary transformation
#   TODO: may test if this can also benefit from blocking
def generate_binary_trans_func(cw: CodeWriter):
    cw.add_line(
        "void binarize(float* inputMatrix, int input_size, int* binarizedMatrix) {")
    cw.indent()
    cw.add_line("for (int i = 0; i < input_size; i++) {")
    cw.indent()
    cw.add_line(
        "binarizedMatrix[i] = (int)((unsigned int)inputMatrix[i] >> 31);")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.add_line("")

# Generates code details of a binary-transformation layer
#   This implementation simply calls the corresponding helper function
#   TODO: may test if this can also benefit from blocking


def generate_binary_trans_layer(cw: CodeWriter):
    cw.add_line(
        "binarize(float* inputs_bitrans, int input_size, float threshold, int* output)")
    cw.add_line("")

# Generates code details of a fully-connected layer


def generate_fc_code(cw: CodeWriter):
    cw.add_line("for (int i = 0; i < fc_output_size; i++) {")
    cw.indent()
    cw.add_line("int activation = 0;")
    cw.add_line("for (int j = 0; j < fc_input_size; j++) {")
    cw.indent()
    cw.add_line("activation += ~(inputs[j]^weights[i][j];")
    cw.dedent()
    cw.add_line("outputs[i]=1.0/(1.0+exp(-activation)")
    cw.dedent()
    cw.add_line("}")
    cw.dedent()
    cw.add_line("}")
    cw.add_line("")


# Inject the required timing operations preceding a layer
#   note that this comes later than the update
def inject_clock_start(cw: CodeWriter):
    cw.add_line("c_start = std::clock();")
    cw.add_line("")

# Inject the required timing operations after a layer
# This includes take another timing, calculate the difference
#   as well as outputing data to the corresponding log file


def inject_clock_end(cw: CodeWriter, file_name):
    cw.add_line("c_end = std::clock();")
    cw.add_line("time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;")
    cw.add_line("std::fprintf(pFile, \"%lf\\n\",time_elapsed_ms);")
    cw.add_line("")

# Generates code for freeing allocated memories


def write_conv_free_mallocs(cw: CodeWriter):
    cw.add_line("std::free(inputs);")
    cw.add_line("std::free(outputs);")
    cw.add_line("std::free(filters);")

# Closes the main function


def generate_main_closure(cw: CodeWriter):
    cw.dedent()
    cw.add_line("}")

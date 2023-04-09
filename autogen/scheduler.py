'''''
This is the scheduler class that calls the codewriter and "orchestrates"
    all activities, read json inputs, generate codes, compile outputted
    files with SIMD enabled, and run compiled executables while dumping
    platform information and scheduling history.
'''''

import datetime
from codewriter import (generate_os_convolution_code, generate_max_pooling_code, generate_avg_pooling_code, generate_binary_trans_func, generate_binary_trans_layer, generate_conv_params_update,
                        generate_pool_params_update, generate_conv_params_init, generate_pool_params_init, generate_bitrans_params_init, generate_fc_code, inject_clock_start, inject_clock_end, generate_clock_params_init, generate_xnor_popcount_helper, write_conv_free_mallocs, generate_include_libraries, generate_main_func_setup, generate_main_closure,generate_bitrans_params_update)
from csnake import CodeWriter
import platform
import psutil
import logging
from layer import ConvLayer, PoolLayer, BitransLayer, FCLayer
import os
import json
import math


class Scheduler:

    # Initializes a scheduler by taking in the path to json
    #   and a boolean specifying whether we want this neural network
    #   to be binarized
    def __init__(self, path_to_json: str, precision: int, \
                 stationarities: list, loop_orders: list, blocking_schemes: list):
        self._code_writer = CodeWriter()
        self.precision = precision
        self.name = "TBD"
        self._num_layers = 0
        self.read_json("inputs/"+path_to_json,stationarities, loop_orders, blocking_schemes)
        self._schedule_scheme = None
        self._scheme_history = []
        self._get_sys_info()

    # Printing a scheduler simply gets the current code writer content
    def __str__(self) -> str:
        return self._code_writer.__str__()

    # A program uses this function to set the scheduling scheme
    #   and add the new scheduling scheme to the end of history
    def set_schedule_scheme(self, schedule_scheme):
        self._schedule_scheme = schedule_scheme
        self._scheme_history += schedule_scheme

    # Reads inputs from json files and properties related to layer characteristics
    def read_json(self, path, stationarities: list, loop_orders: list, blocking_schemes: list):
        fp = open(path)
        data = json.load(fp)

        nn_name = data["name"]
        self.name = nn_name+datetime.datetime.now().strftime("__%d_%m_%Y_%H_%M_%S")
        self._layers = []

        for i in data["body"].values():
            
            if i["type"] == "convolution":
                simd_length = None
                bs = i["batch_size"]
                ih = i["input_height"]
                iw = i["input_width"]
                if self.precision == 1:
                    idep = str(int(int(i["input_depth"])/32))
                else:
                    idep = i["input_depth"]
                fd = i["filter_depth"]
                fw = i["filter_width"]
                fh = i["filter_height"]
                pd = i["padding"]
                st = i["stride"]
                self._layers.append(ConvLayer(bs, iw, ih, idep, fw, fh, fd, pd, st,\
                                              stationarities[self._num_layers],
                                              loop_orders[self._num_layers],
                                              blocking_schemes[self._num_layers]))


            elif i["type"] == "pooling":
                pt = i["pool_type"]
                bs = i["batch_size"]
                iw = i["input_width"]
                ih = i["input_height"]
                if self.precision == 1:
                    idep = str(int(int(i["input_depth"])/32))
                else:
                    idep = i["input_depth"]
                ps = i["input_size"]
                self._layers.append(PoolLayer(pt, bs, iw, ih, idep, ps))

            elif i["type"] == "binary transformation":
                bs = i["batch_size"]
                iw = i["input_width"]
                ih = i["input_height"]
                idep = i["input_depth"]
                self._layers.append(BitransLayer(bs, iw, ih, idep))

            elif i["type"] == "fully connected":
                bs = i["batch_size"]
                isize = i["input_size"]
                osize = i["output_size"]
                self._layers.append(FCLayer(bs, isize, osize))

            self._num_layers += 1

    # This function replaces the original code write instance with a new one

    def reset_code_writer(self):
        self._code_writer = CodeWriter()

    '''''
    This function estimates the simd operations based on tile sizes
    We will move it to other classes/programs

    def estimate_SIMD_ops(width, height, in_depth, out_depth, tile_width, tile_height, tile_depth):
        return math.ceil(width/tile_width) * math.ceil(tile_width/2) * math.ceil(height/tile_height) * math.ceil(tile_height/2) * \
            math.ceil(out_depth/tile_depth) * \
            math.ceil(tile_depth/64) * 9 * in_depth

    '''''

    # This is the ultimate boss function for generating the whole bnn program based
    # on information of layers and configurations
    def generate_whole_program(self):

        # Set up
        generate_include_libraries(self._code_writer)

        # Declare variables
        generate_binary_trans_func(self._code_writer)
        generate_xnor_popcount_helper(self._code_writer)
        generate_main_func_setup(self._code_writer)
        generate_conv_params_init(self._code_writer, self.name, self.precision)
        generate_pool_params_init(self._code_writer)
        generate_bitrans_params_init(self._code_writer)
        generate_clock_params_init(self._code_writer)

        # Generate Layers and Assign Variable Values
        for layer in self._layers:

            if isinstance(layer, BitransLayer):
                generate_bitrans_params_update(layer.batch_size*layer.input_width*\
                                               layer.input_height*layer.input_depth)
                inject_clock_start(self._code_writer)
                generate_binary_trans_layer(self._code_writer)
                inject_clock_end(self._code_writer)

            elif isinstance(layer, ConvLayer):
                loop_order, blocking_scheme = self._schedule_scheme
                generate_conv_params_update(self._code_writer,layer, blocking_scheme,self.precision)
                inject_clock_start(self._code_writer)
                generate_os_convolution_code(self._code_writer, loop_order,self.precision)
                inject_clock_end(self._code_writer, self.name)
                write_conv_free_mallocs(self._code_writer)

            elif isinstance(layer, PoolLayer):
                pooling_scheme = self._schedule_scheme
                inject_clock_start(self._code_writer)
                generate_pool_params_update(self._code_writer, layer, pooling_scheme)
                if layer.pool_type == "MAX":
                    generate_max_pooling_code(self._code_writer)
                else:
                    generate_avg_pooling_code(self._code_writer)
                inject_clock_end(self._code_writer, self.name)

            elif isinstance(layer, FCLayer):
                inject_clock_start(self._code_writer)
                generate_conv_params_update(layer.input_size, layer.output_size)
                generate_fc_code(self._code_writer)
                inject_clock_end(self._code_writer, self.name)

        generate_main_closure(self._code_writer)


    # This function writes all codes cumulated in the code writer to
    # the file specified a given path
    def output_code_to_file(self):
        self._code_writer.write_to_file(("out/programs/"+self.name+".cpp"))

    # This function runs the files in the path parameter with the arguments provided
    def run_file(self):
        os.system("./out/compiled/"+self.name)

    # Compiles files with SIMD enabled
    def compile_file(self):
        os.system(
            "g++ -I -O3 -march=native -ftree-vectorize -msse4.2 -o out/compiled/"+self.name+" out/programs/"+self.name+".cpp")

    # This function gets the system information and stores it in the system_info field
    #       This includes the platform, the platform-release, the platform-version,
    #       the architecture, the processor, and the ram size.
    def _get_sys_info(self):
        try:
            info = {}
            info['platform'] = platform.system()
            info['platform-release'] = platform.release()
            info['platform-version'] = platform.version()
            info['architecture'] = platform.machine()
            info['processor'] = platform.processor()
            info['ram'] = str(
                round(psutil.virtual_memory().total / (1024.0 ** 3)))+" GB"
            self.system_info = info
        except Exception as e:
            print("error getting sys info")
            logging.exception(e)

    def output_sys_info_to_file(self):
        sys_info_json = self.name
        with open("out/"+"platforms/"+sys_info_json+".json", "w") as outfile:
            json.dump(self.system_info, outfile, indent=4)

    def output_scheme_history_to_file(self):
        scheme_hist_json = self.name
        file = open("out/"+"schemes/"+scheme_hist_json+".json", 'w')
        for hist in self._scheme_history:
            file.write(str(hist)+"\n")
        file.close()

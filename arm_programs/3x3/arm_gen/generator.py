
import datetime

from csnake import CodeWriter
import platform
import psutil
import logging
from is_anchored import gen_IS_anchored_program, gen_IS_anchored_program_block
from os_anchored import gen_OS_anchored_program, gen_OS_anchored_program_block
from ws_anchored import gen_WS_anchored_program, gen_WS_anchored_program_block
import os
import json
import math


class CodeGenerator:
    def __init__():
        pass

    def generate_whole_program(precision, vec_len, anchor_stationarity, aux_stationarity, with_blocking: bool = False, block_scheme = None):
        if with_blocking:
            if anchor_stationarity == "IS":
                gen_IS_anchored_program_block(precision, vec_len, aux_stationarity, block_scheme)
            elif anchor_stationarity == "WS":
                gen_WS_anchored_program_block(precision, vec_len, aux_stationarity, block_scheme)
            elif anchor_stationarity == "OS":
                gen_OS_anchored_program_block(precision, vec_len, aux_stationarity, block_scheme)
            else:
                raise ValueError("anchoring stationarity not found")
            
        else:
            if anchor_stationarity == "IS":
                gen_IS_anchored_program(precision, vec_len, aux_stationarity)
            elif anchor_stationarity == "WS":
                gen_WS_anchored_program(precision, vec_len, aux_stationarity)
            elif anchor_stationarity == "OS":
                gen_OS_anchored_program(precision, vec_len, aux_stationarity)
            else:
                raise ValueError("anchoring stationarity not found")


    


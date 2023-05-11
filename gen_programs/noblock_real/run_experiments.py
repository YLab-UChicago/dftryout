import os
import subprocess
import shutil
from datetime import datetime

start_time = datetime.now()
directory = '/home/zhouzikai/nn_ext_dataflows/gen_programs/noblock_real/out'
log_dir = '/home/zhouzikai/nn_ext_dataflows/gen_programs/noblock_real/log'

size_configs = [56, 112, 224]
num_filter_configs = [8, 16, 32, 64]

cpu_list = ["ArmO3CPU"]
fixed_stat_pos = "/home/zhouzikai/gem5/m5out/stats.txt"
num_files = len(os.listdir(directory))

file_counter = 0
for filename in os.listdir(directory):
    print(">>>>>>>>>>Progress: " + str(file_counter) + "/" + str(num_files) + "<<<<<<<<<<<<")
    print('------Duration: {}'.format(datetime.now() - start_time) + "------")
    print(filename)
    file_counter += 1
    for size in size_configs:
        for nf in num_filter_configs:
            for cpu in cpu_list:
                for iter in range(20):
                    log_name = filename + "_hw_" + str(size) + "_nf_" + str(nf) + "_" + cpu + "_" + str(iter) + "_stats.txt"
                    log_path = os.path.join(log_dir, log_name)
                    f = os.path.join(directory, filename)

                    cmd = f"{f} {size} {size} {nf} > {log_path}"
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

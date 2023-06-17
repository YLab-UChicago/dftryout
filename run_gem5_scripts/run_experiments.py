# import required module
import os
import subprocess
import shutil
from datetime import datetime
start_time = datetime.now()
# assign directory
directory = '/home/zhouzikai/nn_ext_dataflows/gen_programs/noblock/out'
log_dir = '/home/zhouzikai/nn_ext_dataflows/gen_programs/noblock/log'


size_configs = [56,112]
num_filter_configs = [8,16,32]

cpu_list = ["ArmO3CPU"]
fixed_stat_pos = "/home/zhouzikai/gem5/m5out/stats.txt"
num_files = len(os.listdir(directory))


file_counter = 0
# iterate over files in
# that directory
for filename in os.listdir(directory):
    print(">>>>>>>>>>Progress: "+str(file_counter)+"/"+str(num_files)+"<<<<<<<<<<<<")
    print('------Duration: {}'.format(datetime.now() - start_time)+"------")
    print(filename)
    file_counter += 1
    if ("is_ws_" in filename) and ("os_ws_" in filename):
        continue
    for size in size_configs:
        for nf in num_filter_configs:
            for cpu in cpu_list:
                
                log_name = filename + "_hw_"+str(size)+ "_nf_"+str(nf)+"_"+cpu+"_"+"_stats.txt"
                log_path = os.path.join(log_dir,log_name)
                f = os.path.join(directory, filename)
                # checking if it is a file
                
                cmd = [
                    "sudo",
                    "./build/ARM/gem5.opt",
                    "configs/example/se.py",
                    "--cmd="+f,
                    "--options="+str(size)+" " +str(size)+" "+str(nf), #note the 256 here does not matter
                    "--cpu-type="+cpu,
                    "--caches",
                    "--l2cache",
                    "--l1i_size=16kB",
                    "--l1d_size=64kB",
                    "--l2_size=256kB",
                    "--l1i_assoc=4",
                    "--l1d_assoc=4",
                    "--l2_assoc=8",
                    "--cacheline_size=64",
                ]

                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                shutil.copy(fixed_stat_pos,log_path)
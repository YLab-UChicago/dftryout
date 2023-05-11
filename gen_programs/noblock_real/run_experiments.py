# import required module
import os
import subprocess
import shutil
from datetime import datetime
start_time = datetime.now()
# assign directory
directory = '/home/zhouzikai/nn_ext_dataflows/gen_programs/noblock_real/out'
log_dir = '/home/zhouzikai/nn_ext_dataflows/gen_programs/noblock_real/log'
#directory = '/home/zhouzikai/nn_ext_dataflows/gen_programs/block/out'

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
    for size in size_configs:
        for nf in num_filter_configs:
            for cpu in cpu_list:
                for iter in range(20):
                
                    log_name = filename + "_hw_"+str(size)+ "_nf_"+str(nf)+"_"+cpu+"_"+str(iter)+"_stats.txt"
                    log_path = os.path.join(log_dir,log_name)
                    f = os.path.join(directory, filename)
                    # checking if it is a file
                    
                    cmd = [
                        "sudo",
                        f,
                        str(size)+" " +str(size)+" "+str(nf),
                        "> "+log_path
                    ]

                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
import os
import subprocess
import shutil
from datetime import datetime
# assign directory
directory = '/home/zhouzikai/nn_ext_dataflows/baselines/out'
log_dir = '/home/zhouzikai/nn_ext_dataflows/baselines/log'
start_time=datetime.now()
size_configs = [56,112]
filter_sizes = [3,4,5]
strides = [2]
num_filter_configs = [32,64]

cpu_list = ["ArmO3CPU"]
fixed_stat_pos = "/home/zhouzikai/gem5/m5out/stats.txt"
num_files = len(os.listdir(directory))
file_counter = 0
for filename in os.listdir(directory):
    print(">>>>>>>>>>Progress: "+str(file_counter)+"/"+str(num_files)+"<<<<<<<<<<<<")
    print('------Duration: {}'.format(datetime.now() - start_time)+"------")
    print(filename)
    print('')
    file_counter += 1
    if "is_" not in filename:
        continue
    # if "os_" not in filename[24:27]:
    #     continue
    for size in size_configs:
        for nf in num_filter_configs:
            for cpu in cpu_list:
                for filter_size in filter_sizes:
                    for stride in strides:
                        log_name = filename + "_hw_"+str(size)+'_f_'+ str(filter_size)+"_nf_"+str(nf)+"_s_"+str(stride)+"_"+cpu+"_"+"_stats.txt"
                        log_path = os.path.join(log_dir,log_name)
                        f = os.path.join(directory, filename)
                        # checking if it is a file
                        
                        cmd = [
                            "sudo",
                            "./build/ARM/gem5.opt",
                            "configs/example/se.py",
                            "--cmd="+f,
                            "--options="+str(size)+" " +str(size)+" "+str(nf)+" "+str(filter_size) + ' '+str(filter_size)+' '+str(stride), #note the 256 here does not matter
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
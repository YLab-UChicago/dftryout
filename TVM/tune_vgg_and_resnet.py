"""

Please read:

-TO RUN THIS FILE:  “python tuneNetwork.py <network name>”
	-use a lowercase spelling for the network name
	-use a hyphen between the model name and size
	- examples: “resnet-18”, “vgg-11”

-I am setting the number of trials for each layer to the size of the configuration space
	-This can be changed manually if necessary by adjusting “n_trial” in the code. 

-Our target it “target = tvm.target.Target('llvm -mattr=+neon,+neoversen1,+i8mm')” 
this is the most specific I could make it without causing the search space for each 
layer of the network to become massive (112,000 for example)

-Breakdown of the flags in the target:
	+neon is for SIMD operations
	+neoversen1 is for our specific cpu
	+i8mm is to enable 8 bit operations, although, this does not provide much of a 
    benefit. I have left it in for good measure.

-I plan to spend a bit of time further playing around with the target to see if I 
can make it more specific without massively increasing the search space. 
    -The current target results in 300-1000 configurations for each layer. 
    -Each program takes ~2-3 hours to run at this size so it is necessary to minimize 
    the amount of configurations. 

-I have set “dtype = "float32"” because this is before quantization is performed.
I do not believe there is a way to specify int8 right from the beginning

-To iterate over the search space, I am using a tuner that performs a grid search. 
Other available tuners are:
	-XGBTuner which uses “XGBoost” which is a “gradient boosting framework”
	-GATuner which uses “Genetic Algorithms”
	-RandomTuner which randomly selects configurations

-The type of tuner we use is potentially an area of further exploration, 
although the grid search tuner (that we are using) seems to be the most comprehensive

-Once the program is done running:
	-execution times are saved to “<model_name>_eval_times.txt”
	-optimal configurations are saved to “<model_name>_graph_opt.log”
    -configurations tried are recorded in "<model_name>.log"
    -"graph_tuner.log" will track the tasks that were successfully completed

"""


import os
import sys
import numpy as np
import tvm
from tvm import relay, autotvm
from tvm.relay import testing
from tvm.autotvm.tuner import GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_executor as runtime


# Set up
target = tvm.target.Target('llvm -mattr=+neon,+neoversen1,+i8mm')
batch_size = 1
dtype = "float32"
model_name = sys.argv[1]
log_file = "%s.log" % model_name
graph_opt_sch_file = "%s_graph_opt.log" % model_name
input_name = "data"

# Use 1 thread
num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)


def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)
    n_layer = int(name.split("-")[1])

    if "resnet" in name:
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "vgg" in name:
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


tuning_option = {
    "log_filename": log_file,
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True
        ),
    ),
}


# Tune kernels using a GridSearchTuner
def tune_kernels(
    tasks, measure_option, early_stopping=None, log_filename=f"{model_name}_tuning.log"
):
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner_obj = GridSearchTuner(task)

        # perform tuning
        n_trial = len(task.config_space)
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )


# Use graph tuner to achieve graph level optimal schedules
def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
    target_op = [
        relay.op.get("nn.conv2d"),
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)


def get_performance(lib, data_shape):
    # upload parameters to device
    dev = tvm.cpu()
    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    module = runtime.GraphModule(lib["default"](dev))
    module.set_input(input_name, data_tvm)
    
    # evaluate
    return str((module.benchmark(dev, number=100, repeat=3)))


def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, data_shape, _ = get_network(model_name, batch_size)

    # Quantize the model to 8-bit
    with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
        mod = relay.quantize.quantize(mod, params)

    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    # run tuning tasks
    tune_kernels(tasks, **tuning_opt)
    tune_graph(mod["main"], data_shape, log_file, graph_opt_sch_file)

    # compile, time, and record results
    with open(f'{model_name}_eval_times.txt', 'w') as f:

        # default
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
            default_perf = get_performance(lib, data_shape)

            # save to file
            f.write("Evaluation of the network compiled in 'default' mode without auto tune:")
            f.write(default_perf)
            f.write("\n")

            # print to terminal
            print("Evaluation of the network compiled in 'default' mode without auto tune:")
            print(default_perf)
            print("\n")

        # kernel tuned
        with autotvm.apply_history_best(log_file):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params=params)
            kernel_perf = get_performance(lib, data_shape)

            # save to file
            f.write("Evaluation of the network been tuned on kernel level:")
            f.write(kernel_perf)
            f.write("\n")

            # print to terminal
            print("Evaluation of the network been tuned on kernel level:")
            print(kernel_perf)
            print("\n")

        # graph tuned
        with autotvm.apply_graph_best(graph_opt_sch_file):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build_module.build(mod, target=target, params=params)
            graph_perf = get_performance(lib, data_shape)

            # save to file
            f.write("Evaluation of the network been tuned on graph level:")
            f.write(graph_perf)
            f.write("\n")

            # print to terminal
            print("Evaluation of the network been tuned on graph level:")
            print(graph_perf)


tune_and_evaluate(tuning_option)

import sys
import os
import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.auto_scheduler import TuningOptions
from tvm.contrib import graph_executor
import numpy as np

# TO RUN THIS FILE:  “python tuneNetwork.py <(densenet) network name>”
# names can be "densenet-121", "densenet-161", "densenet-169", "densenet-201"

# Use 1 thread
num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)

name = sys.argv[1]
n_layer = int(name.split("-")[1])
batch_size = 1
dtype = "float32"
input_shape = (1, 3, 224, 224)

mod, params = relay.testing.densenet.get_workload(
    densenet_size=n_layer, batch_size=batch_size, dtype=dtype)

# Quantize the model
with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
    mod = relay.quantize.quantize(mod, params)

# Define Target Hardware
target = tvm.target.Target('llvm -mattr=+neon,+neoversen1,+i8mm')

json_name = f"{name}_densenet_tuning.json"
# Extract Tasks
tasks, task_weights = tvm.auto_scheduler.extract_tasks(mod["main"], params, target)

# Set Tuning Options
tune_option = TuningOptions(
    num_measure_trials=2000,
    measure_callbacks=[auto_scheduler.RecordToFile(json_name)],
    verbose=2,
)

# Tune the Model
tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
tuner.tune(tune_option)

# Compile the Model with Tuned Parameters
with auto_scheduler.ApplyHistoryBest(json_name):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

# Run the Model
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
input_data = np.random.uniform(size=input_shape).astype("float32") # Using the custom shape
module.set_input("data", input_data)

# Create a time evaluator function
timer = module.module.time_evaluator("run", dev, number=10) # You can adjust the number of repetitions

# Run the model and measure the time
timing = timer()

# Print the results
time = timing.mean * 1000
print("Execution time:", time, "ms")

#Save end-to-end time
with open(f'{name}_avg_inference_time.txt', 'w') as f:
    f.write(f'Average Inference Time: {time} ms\n')

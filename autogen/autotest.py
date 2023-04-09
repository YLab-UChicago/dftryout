from scheduler import Scheduler

all_files_to_run = ["single2.json","single3.json","single4.json","single5.json"]


loop_orders = [["h", "w", "f"],
               ["h", "f", "w"],
               ["f", "w", "h"],
               ["f", "h", "w"],
               ["w", "h", "f"],
               ["w", "f", "h"]]
all_block_sizes = [1,7,14,28]


for file_name in all_files_to_run:
    for outer_order in loop_orders:
        for inner_order in loop_orders:
            for block_size in all_block_sizes:
                scheme = ((outer_order, inner_order),[block_size,block_size,1])
                s = Scheduler(file_name,True)
                s.set_schedule_scheme(scheme)
                s.generate_whole_bnn_program()
                s.output_code_to_file()
                s.output_sys_info_to_file()
                s.output_scheme_history_to_file()
                s.compile_file()
                for i in range(10):
                    s.run_file()
                del s
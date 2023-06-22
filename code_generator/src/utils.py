def generate_inout_sequence(filter_width,filter_height,stride,num_cache_byrow):
    '''
    Function:
        This method generates the sequence of auxiliary cache
        usage for both input auxiliary caches under output-anchored
        dataflows and output auxiliary caches under input-anchored
        dataflow. This sequence is used assuming secondary unrolling.
    '''

    unroll_num = filter_width - stride
    sequence_list = []
    for un in range(unroll_num):
        this_unroll_iter = []
        for fh in range(filter_height):
            this_height_iter = []
            if num_cache_byrow[fh] <= stride:
                for nc in range(num_cache_byrow[fh]):
                    this_height_iter.append(nc + fh*(filter_width-stride))
            else:
                for nc in range(num_cache_byrow[fh]):
                    this_height_iter.append((un + nc)%num_cache_byrow[fh]+ fh*(filter_width-stride))
            this_unroll_iter.append(this_height_iter)
        sequence_list.append(this_unroll_iter)
    return sequence_list


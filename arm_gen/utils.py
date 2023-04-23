def generate_inout_sequence(filter_width,filter_height,stride,num_cache_byrow):

    unroll_num = filter_width - stride
    sequence_list = []
    num_cache_count = 0
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


#test
print(generate_inout_sequence(3,3,1,[2,2,1]))
print(generate_inout_sequence(4,4,2,[2,2,2,1]))
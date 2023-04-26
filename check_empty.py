import os
import pandas as pd
import sys 
sys.path.append('/home/zhouzikai/nn_ext_dataflows/data_analysis')
from scripts import build_dataframe

def get_file_size(file_path):
    return os.path.getsize(file_path)

directory = '/home/zhouzikai/nn_ext_dataflows/gen_programs/noblock/log'


def get_data_size_df(directory):
    file_data = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if (get_file_size(f)) == 0:
            print("found one empty")
        file_data.append({'file_path': f, 'size': get_file_size(f)})

    df = pd.DataFrame(file_data).set_index('file_path')
    return df

def append_dataframe(df1, df2, join_keyword='filename'):
    '''
    add the 'size' column from df1 to df2, assuming their indices are the same (file path)
    '''
    #merged_df = df2.merge(df1[[join_keyword, 'size']], on=join_keyword, how='left')
    merged_df = df2.join(df1)

    return merged_df

#df = get_data_size_df(directory)
#empty_files_df = df[df['size'] == 0]
#print("empty stats:", empty_files_df)
#print("all df:", df)
#file_list=[]
#for filename in os.listdir(directory):
#    f = os.path.join(directory, filename)
#    file_list.append(f)
#testing appending dataframe
#first off calling Zack's df function 
#df_files = build_dataframe(file_list)
#print(df_files)
#merged = append_dataframe(df,df_files)
#print(merged)

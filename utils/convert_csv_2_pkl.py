import pandas as pd
import time
import os

def read_csv_file(folder_path, file_name_list, new_file_name_list):
    t1 = time.time()
    for old_file, new_file in zip(file_name_list, new_file_name_list):
        data = pd.read_csv(os.path.join(folder_path, old_file))
        data.to_pickle(os.path.join(folder_path, new_file))
    t2 = time.time()
    print("Converting CSV to PKL Using %.2f s" % (t2-t1))

if __name__ == '__main__':
    folder_path = 'demo/credit_data'
    file_name_list = ['train.csv', 'train_target.csv', 'test.csv']
    new_file_name_list = ['train.pkl', 'train_target.pkl', 'test.pkl']
    read_csv_file(folder_path, file_name_list, new_file_name_list)
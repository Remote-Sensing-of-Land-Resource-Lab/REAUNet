import os
import random
from data_crop import listdir


def create_txt(txt_name, txt_list):
    if os.path.exists(txt_name):
        os.remove(txt_name)
    with open(txt_name, 'w') as f:
        for i in txt_list:
            f.write(i + '\n')
    f.close()


def list2txt(folder, txt_name):
    list_all = listdir(folder)
    file_number = len(list_all)
    print('file number:', file_number)
    random.shuffle(list_all)
    create_txt(txt_name, list_all)


def train_val_split(folder, train_percentage=0.8):
    txt_name_train = 'data_train.txt'
    txt_name_val = 'data_val.txt'

    list_all = listdir(folder)
    file_number = len(list_all)
    print('file number:', file_number)
    random.shuffle(list_all)

    split_number = int(file_number * train_percentage)
    print('split', split_number)

    list_train = list_all[:split_number]
    list_val = list_all[split_number:]

    create_txt(txt_name_train, list_train)
    create_txt(txt_name_val, list_val)
    print('train:', len(list_train), 'val:', len(list_val))

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pathlib
import re

import utils

# class Task1aDataSet2018(Dataset):

#     def __init__(self, db_path, class_map, feature_folder, mode="train"):
#         #mode: train | test | eval

#         #read the txt to the data_list
#         self.db_path = db_path
#         self.class_map = class_map
#         df = pd.read_csv("{}/evaluation_setup/fold1_{}.txt".format(db_path, mode), sep="\t", header=None)

#         self.X_filepaths = df[0].str.replace("audio", pathlib.Path(feature_folder).name).str.replace(".wav", "-A.npy")

#         #TODO: maybe refer to meta.csv
#         if mode == "test":
#             self.y_classnames = df[0].str.split("/", expand=True)[1].str.split("-", n=1, expand=True)[0]
#         else:
#             self.y_classnames = df[1]


#         self.data_list = []

#     def __len__(self):
#         return len(self.X_filepaths)

#     def __getitem__(self, idx):

#         path = self.X_filepaths[idx]
#         f = open("{}/{}".format(self.db_path, path), 'rb')
#         feature = np.load(f)
#         label = self.class_map[self.y_classnames[idx]]

#         return feature.T, label

# class Task1aDataSet2019(Dataset):
    
#     def __init__(self, db_path, class_map, feature_folder, mode="train"):
#         #mode: train | test | eval

#         self.db_path = db_path
#         self.class_map = class_map
#         df = pd.read_csv("{}/evaluation_setup/fold1_{}.csv".format(db_path, mode), sep="\t")

#         self.X_filepaths = feature_folder + "/" + df['filename'].str.replace('audio/','').str.replace('.wav','.npy')
#         self.y_classnames = df['scene_label']

#     def __len__(self):
#         return len(self.X_filepaths)

#     def __getitem__(self, idx):

#         path = self.X_filepaths[idx]
#         f = open(path, 'rb')
#         feature = np.load(f)
#         label = self.class_map[self.y_classnames[idx]]

#         return feature.T, label
        
class ASCDataSet(Dataset):
    
    def __init__(self, db_path, class_map, feature_folder, mode="train"):
        #mode: train | test | eval

        self.db_path = db_path
        self.class_map = class_map

        datalist="{}/fold1_{}.csv".format(db_path, mode)
        
        df = pd.read_csv(datalist, sep="\t")

        self.filenames = df['filename']

        self.X_filepaths = feature_folder + "/" + df['filename'].str.replace('audio/','').str.replace('.wav','.npy')
        self.y_classnames = df['scene_label']

        print('{:^9s} {:>6d} {}'.format(mode, len(self.X_filepaths), datalist))

    def __len__(self):
        return len(self.X_filepaths)

    def __getitem__(self, idx):

        path = self.X_filepaths[idx]
        f = open(path, 'rb')
        feature = np.swapaxes(np.load(f), 1, 2)
        scene = self.class_map[self.y_classnames[idx]]
        label_str = pathlib.Path(path).name.replace('.npy','')
        device = utils.get_indexOfDevice(utils.getDevice(label_str))

        return feature, scene, device
import sys
import os
import re

import torch
from torch.utils.data import Dataset
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from utils import ReadUtt2Lang, ReadFeatsScp

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.shape[dim]
    return np.concatenate([vec, np.zeros(pad_size)], axis=dim)
    #print(np.concatenate([vec, np.zeros(pad_size)], axis=dim).shape)


class PadCollate(object):
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        #batch = map(lambda x:
        #                print(x[1]), batch) 
        batch = map(lambda x:
                (pad_tensor(x[0], pad=max_len, dim=self.dim), x[1]), batch)
        batch = list(batch)
        # stack all
        xs = np.stack(list(map(lambda x: x[0], batch)))
        ys = np.stack(list(map(lambda x: x[1], batch)))
        #xs = list(map(lambda x: x[0], batch))
        #ys = list(map(lambda x: x[1], batch))
        #print(xs.shape)
        #print(ys.shape)
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)

class KaldiDataSet(Dataset):
    
    def __init__(self, dataset_dir, mode="train"):
        #mode: train | test | eval

        utt2lang_path = os.path.join(dataset_dir, "utt2lang")
        utt2feat_path = os.path.join(dataset_dir, "feats.scp")

        utt2lang, utt2lang_id = ReadUtt2Lang(utt2lang_path)
        utt2feats = ReadFeatsScp(utt2feat_path)

        self.X_feature = []
        self.Y_lang_id = []

        for utt,lang_id in utt2lang_id.items():
            self.X_feature.append(utt2feats[utt])
            self.Y_lang_id.append(lang_id)

    def __len__(self):
        return len(self.X_feature)

    def __getitem__(self, idx):

        path = self.X_feature[idx]
        f = open(path, 'rb')
        feature = np.swapaxes(np.load(f), 1, 2)
        lang_id = self.Y_lang_id[idx]

        return feature, lang_id

def TestFeatureLength():
    utt2feats = ReadFeatsScp("data/train/feats.scp")
    for utt, feats in utt2feats.items():
        F = np.load(feats)
        print(utt, F.shape)

if __name__=="__main__":
    from hparams import hparams
    from torch.utils.data import Dataset, DataLoader

    dataset = KaldiDataSet("data/train", mode="train")
    dataloader = DataLoader(dataset, collate_fn=PadCollate(dim=1), 
                    batch_size=hparams.batch_size, shuffle=True)
    #TestFeatureLength()

    for x,y in dataloader:
        print("Targets", x.shape,y.shape)

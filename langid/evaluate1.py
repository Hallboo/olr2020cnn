from hparams import hparams, hparams_debug_string

import os, sys
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models import Cnn_9layers_AvgPooling
from dataset import KaldiDataSet, PadCollate
import computeCavg as cavg
import utils

device = torch.device("cuda" if hparams.use_cuda else "cpu")

def ComputeAccuracy(all_outputs, all_predict, all_targets):

    if not (all_predict.shape == all_targets.shape):
        raise ValueError("The shape of all_predict and all_targets must be same.")
 
    # compute the confusion matrix
    confu_mat = confusion_matrix(all_targets, all_predict)
    confu_mat_norm = confu_mat.astype('float') / confu_mat.sum(axis=1)[:, np.newaxis]

    class_total = np.sum(confu_mat, axis=1)
    acc = np.trace(confu_mat) / np.sum(class_total)
    # accuracy is the sum of confusion matrix diagonal divide total number of sample in evalset

    class_acc = np.diagonal(confu_mat) / class_total
    # accuracy for each class

    return acc, class_acc, confu_mat

# def ComputeCavg(all_outputs, all_predict, all_targets, spk2utt_path, utt2lang_path):
#     lang2lang_id, utt2lang_id, lang_num, trial_list = GetLangIDDict(spk2utt_path, utt2lang_path)
#     pairs, min_score, max_score = process_pair_scores(all_outputs, all_targets, lang2lang_id, utt2lang_id, lang_num, trial_list) 
#     threshhold_bins = 20
#     p_target = 0.5
#     cavgs, min_cavg = cavg.get_cavg(pairs, lang_num, min_score, max_score, threshhold_bins, p_target)
#     print(round(min_cavg, 4))

def ComputeCavg(all_outputs, all_targets):

    pairs = []
    lang_num = all_outputs.shape[1]
    for u in range(len(all_outputs)):
        for l in range(len(all_outputs[u])):
            pairs.append([l, all_targets[u], all_outputs[u][l]])

    min_score = np.min(all_outputs)
    max_score = np.max(all_outputs)

    threshhold_bins = 20
    p_target = 0.5
    cavgs, min_cavg = cavg.get_cavg(pairs, lang_num, min_score, max_score, threshhold_bins, p_target)
    print(round(min_cavg, 4))


# def process_pair_scores(outputs, all_targets, lang2lang_id, utt2lang_id, lang_num, trial_list):
#   pairs = []
#   stats = []

#   for line in lines:
#     lang, utt, score = line.strip().split()
#     if trial_list.has_key(lang + utt):
#       if utt2lang_id.has_key(utt):
#         pairs.append([lang2lang_id[lang], utt2lang_id[utt], float(score)])
#       else:
#         pairs.append([lang2lang_id[lang], -1, float(score)])
#       stats.append(float(score))
#   return pairs, min(stats), max(stats)

# def GetLangIDDict(spk2utt_path, utt2lang_path):
#     # get lang2lang_id
#     lang2lang_id, langs = utils.ReadLang2UttGetLangLabel(spk2utt_path)
#     utt2lang = utils.Read2Column(utt2lang_path)
#     # get utt2lang_id
#     utt2lang_id = {utt:lang2lang_id[lang] for utt, lang in utt2lang.items()}
#     trial_list = dict()
#     for utt, label in utt2lang.items(): 
#         for lang in langs:
#             target = ''
#             if label == lang:
#                 target = 'target'
#             else:
#                 target = 'nontarget'
#             trial_list[lang + utt] = target

#     return lang2lang_id, utt2lang_id, len(langs), trial_list

def ComputeEER(all_outputs, all_predict, all_targets):
    pass

if __name__=="__main__":

    all_outputs = np.load("outputs.npy", allow_pickle=True)
    all_targets = np.load("targets.npy", allow_pickle=True)
    all_predict = np.argmax(all_outputs, axis = 1)
    #ComputeAccuracy(all_outputs, all_predict, all_targets)
    #ComputeCavg(all_outputs, all_predict, all_targets, 'data/dev_all/spk2utt', 'data/dev_all/utt2lang')
    ComputeCavg(all_outputs, all_targets)

 
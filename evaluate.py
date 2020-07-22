# coding: utf-8
"""
Preprocess dataset

usage: preprocess.py [options] <db_path> <feat_path> <exp/task> 

options:
    -h, --help               Show help message.
"""

from docopt import docopt
from utils import get_label2index, getClassName
from hparams import hparams, hparams_debug_string

import os, sys
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from asc.models import Cnn_9layers_AvgPooling
# from asc.models import (Cnn_5layers_AvgPooling, Cnn_9layers_MaxPooling, Cnn_13layers_AvgPooling)
from asc.dataset import ASCDataSet

device = torch.device("cuda" if hparams.use_cuda else "cpu")

# def evaluate_andy(model, criterion, dataloader, exp_dir):
#     model.eval()

#     total_loss = 0
#     batch = 0

#     class_correct = list(0. for i in range(10))
#     class_total = list(0. for i in range(10))
#     confusion_matrix = np.zeros((10, 10), dtype=int) # 2D, [actual_cls][predicted_cls]

#     with torch.no_grad():
#         for x, targets in dataloader:
#             x = torch.FloatTensor(x).to(device)
#             targets = torch.LongTensor(targets).to(device)
#             outputs = model(x)
#             loss = criterion(outputs, targets)
#             total_loss += loss.item()

#             _, predicted = torch.max(outputs, 1)

#             c = (predicted == targets).squeeze()
#             for i in range(len(targets)):
#                 label = targets[i]
#                 class_correct[label] += c[i].item()
#                 class_total[label] += 1

#                 confusion_matrix[label.item()][predicted[label.item()].item()] += 1

#             batch += 1
#     acc = np.array(class_correct).sum() / np.array(class_total).sum()

#     for i in range(10):
#         if class_total[i] == 0:
#             continue
#         print(': Accuracy of {:18s} : {:2.2f}%'.format(getClassName(i), 100 * class_correct[i] / class_total[i]))

#     print("Accuracy: %f%%"%(acc*100))
#     print(confusion_matrix)

#     return (total_loss / batch), acc, class_correct, class_total, confusion_matrix

def evaluate(model, criterion, dataloader, exp_dir):
    model.eval()

    total_loss = 0
    batch = 0

    all_predicted = []
    all_targets   = []
    all_devices   = []

    with torch.no_grad():
        for x, targets, devices in dataloader:
            x = torch.FloatTensor(x).to(device)
            targets = torch.LongTensor(targets).to(device)
            outputs = model(x)
            loss = criterion(outputs, targets)
            total_loss += loss

            _, predicted = torch.max(outputs, 1)

            all_predicted.append(predicted)
            all_targets.append(targets)
            all_devices.append(devices)

            batch += 1
    # forward done, got all prediction of evaluate data, Start statistic

    eval_loss = total_loss / batch
    all_predicted = torch.cat(all_predicted).cpu().data.numpy()
    all_targets = torch.cat(all_targets).cpu().data.numpy()
    all_devices = torch.cat(all_devices).numpy()

    assert(all_predicted.shape == all_targets.shape == all_devices.shape)

    # np.save("prediction.npy", all_predicted)
    # np.save("targets.npy", all_targets)

    # all_predicted = np.load("prediction.npy")
    # all_targets = np.load("targets.npy")

    # compute the confusion matrix
    confu_mat = confusion_matrix(all_targets, all_predicted)
    confu_mat_norm = confu_mat.astype('float') / confu_mat.sum(axis=1)[:, np.newaxis]

    class_total = np.sum(confu_mat, axis=1)
    acc = np.trace(confu_mat) / np.sum(class_total)
    # accuracy is the sum of confusion matrix diagonal divide total number of sample in evalset

    for i in range(len(class_total)):
        print('* Accuracy of {:18s} : {:2.2f}% {:4d}/{:<4d}'.format(
            getClassName(i), 100 * confu_mat[i][i] / class_total[i], confu_mat[i][i], class_total[i]))

    dev_res = deviceStat(all_predicted, all_targets, all_devices)
    for i in range(len(dev_res)):
        print(": Accuracy of Device {:>2s}: {:>4.2f}%".format(hparams.devices[i], dev_res[i]*100))

    print(": Accuracy All: %f%%"%(acc*100))

    return acc, eval_loss, confu_mat

def deviceStat(all_predicted, all_targets, all_devices):

    res = (all_predicted == all_targets)
    dev_counter = np.zeros((len(hparams.devices)), dtype=int)
    dev_correct = np.zeros((len(hparams.devices)), dtype=int)

    for i in range(len(all_predicted)):
        dev_counter[all_devices[i]] += 1
        if res[i]:
            dev_correct[all_devices[i]] += 1

    dev_counter += (dev_counter == 0)
    
    return dev_correct/dev_counter

if __name__=="__main__":
    args = docopt(__doc__)
    db_path = args['<db_path>']
    feat_path = args['<feat_path>']
    exp_dir = args['<exp/task>']

    with open(os.path.join(exp_dir, "config.json")) as f:
        hparams.parse_json(f.read())

    print(hparams_debug_string())

    Model = eval(hparams.model_type)
    model = Model(len(hparams.labels), activation='logsoftmax')

    print("Load the model: %s"%os.path.join(exp_dir, 'best.pth'))
    checkpoint = torch.load(os.path.join(exp_dir, 'best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    ##### optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #epoch = checkpoint['epoch']
    epoch = checkpoint['ep']
    #loss = checkpoint['losses']

    print("The epoch of the best model is {}".format(epoch))

    criterion = nn.NLLLoss()

    if hparams.use_cuda: model.cuda()

    data_set_eval = ASCDataSet(db_path, get_label2index(), feature_folder=feat_path, mode="evaluate")
    dataloader_eval = DataLoader(data_set_eval, batch_size=hparams.batch_size, shuffle=True)

    evaluate(model, criterion, dataloader_eval, exp_dir)

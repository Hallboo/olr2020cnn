# coding: utf-8
"""
Preprocess dataset

usage: preprocess.py [options] <dev_dir> <exp_dir> 

options:
    -h, --help               Show help message.
"""

from docopt import docopt
from hparams import hparams, hparams_debug_string

import os, sys
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models import Cnn_9layers_AvgPooling
from src.dataset import KaldiDataSet

device = torch.device("cuda" if hparams.use_cuda else "cpu")

def evaluate(model, criterion, dataloader, exp_dir):
    model.eval()

    total_loss = 0
    batch = 0

    all_predicted = []
    all_targets   = []

    with torch.no_grad():
        for x, targets in dataloader:
            x = torch.FloatTensor(x).to(device)
            targets = torch.LongTensor(targets).to(device)
            outputs = model(x)
            loss = criterion(outputs, targets)
            total_loss += loss

            _, predicted = torch.max(outputs, 1)

            all_predicted.append(predicted)
            all_targets.append(targets)

            batch += 1
    # forward done, got all prediction of evaluate data, Start statistic

    eval_loss = total_loss / batch
    all_predicted = torch.cat(all_predicted).cpu().data.numpy()
    all_targets = torch.cat(all_targets).cpu().data.numpy()

    assert(all_predicted.shape == all_targets.shape)

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
            hparams.lang[i], 100 * confu_mat[i][i] / class_total[i], confu_mat[i][i], class_total[i]))

    print(": Accuracy All: %f%%"%(acc*100))

    return acc, eval_loss, confu_mat

if __name__=="__main__":
    args = docopt(__doc__)
    dev_dir = args['<dev_dir>']
    exp_dir = args['<exp_dir>']

    with open(os.path.join(exp_dir, "config.json")) as f:
        hparams.parse_json(f.read())

    print(hparams_debug_string())

    Model = eval(hparams.model_type)
    model = Model(len(hparams.labels), activation='logsoftmax')

    print("Load the model: %s"%os.path.join(exp_dir, 'best.pth'))
    checkpoint = torch.load(os.path.join(exp_dir, 'best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint['epoch']

    print("The epoch of the best model is {}".format(epoch))

    criterion = nn.NLLLoss()

    if hparams.use_cuda: model.cuda()

    data_set_dev   = KaldiDataSet(dev_dir)
    dataloader_dev = DataLoader(data_set_dev, collate_fn=PadCollate(dim=1),
                                batch_size=hparams.batch_size, shuffle=True)

    evaluate(model, criterion, dataloader_eval, exp_dir)

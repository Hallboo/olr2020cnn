# coding: utf-8
"""
Preprocess dataset

usage: preprocess.py [options] <db_path> <feat_path> <exp/task> 

options:
    --preset=<json>          Path of preset parameters (json).
    -h, --help               Show help message.
"""

from docopt import docopt
from utils import get_label2index, getClassName
from hparams import hparams, hparams_debug_string

import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from asc.models import Cnn_9layers_AvgPooling
from asc.models import (Cnn_5layers_AvgPooling, Cnn_9layers_MaxPooling, Cnn_13layers_AvgPooling)
from asc.dataset import ASCDataSet

from evaluate import evaluate

device = torch.device("cuda" if hparams.use_cuda else "cpu")

def train(db_path, feat_path, exp_dir):

    in_domain_classes_num = len(hparams.labels)

    Model = eval(hparams.model_type)

    model = Model(in_domain_classes_num, activation='logsoftmax')

    print(model)

    if hparams.use_cuda:
        model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)
    
    # Data generator
    data_set_train = ASCDataSet(db_path, get_label2index(), feature_folder=feat_path, mode="train")
    data_set_eval  = ASCDataSet(db_path, get_label2index(), feature_folder=feat_path, mode="evaluate")
    dataloader = DataLoader(data_set_train, batch_size=hparams.batch_size, shuffle=True)
    dataloader_eval = DataLoader(data_set_eval, batch_size=hparams.batch_size, shuffle=True)
    criterion = nn.NLLLoss()
    losses = []
    log_interval = 10
    best_acc = 0.0

    for current_epoch in range(hparams.max_epoch):
        total_loss = 0
        batch = 0
        model.train()

        for x, targets, _ in dataloader:
            x = torch.FloatTensor(x).to(device)
            targets = torch.LongTensor(targets).to(device)

            batch_output = model(x)
            loss = criterion(batch_output, targets)

            losses.append(loss.item())
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batch % log_interval == 0 and batch > 0:
            #     avg_loss = total_loss / batch
            #     print('| Epoch {:3d} | Batch {:3d} | Loss {:0.8f}'.format(current_epoch, batch, avg_loss))

            batch += 1
        
        acc, eval_loss, confusion_matrix = evaluate(model, criterion, dataloader_eval, exp_dir)
        if best_acc < acc:
            best_acc = acc
            torch.save({
                "epoch": current_epoch,
                "losses": losses,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(exp_dir, 'best.pth'))

        print("| Epoch: {:3d}, Eval loss: {:0.4f}, current acc: {:2.3f}%, the best: {:2.3f}%".format(current_epoch, eval_loss, acc*100, best_acc*100))

if __name__=="__main__":
    args = docopt(__doc__)
    db_path = args['<db_path>']
    feat_path = args['<feat_path>']
    exp_dir = args['<exp/task>']
    preset = args["--preset"]

    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())

    # make a dir, like exp/2019task1b
    os.makedirs(exp_dir, exist_ok=True)

    fp = open(os.path.join(exp_dir, "config.json"),'w')
    fp.write(hparams.to_json(indent=' '*4))
    fp.close()

    print(hparams_debug_string())

    train(db_path, feat_path, exp_dir)
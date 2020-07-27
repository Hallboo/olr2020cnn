# coding: utf-8
"""
Preprocess dataset

usage: preprocess.py [options] <trn_dir> <dev_dir> <exp_dir> 

options:
    --preset=<json>          Path of preset parameters (json).
    -h, --help               Show help message.
"""

from docopt import docopt
import utils
from hparams import hparams, hparams_debug_string

import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models import Cnn_9layers_AvgPooling
from src.dataset import KaldiDataSet, PadCollate

device = torch.device("cuda" if hparams.use_cuda else "cpu")

def train(trn_dir, dev_dir, exp_dir):

    lang_dict, lang_list = utils.ReadLang2UttGetLangLabel(
                                os.path.join(trn_dir, "spk2utt"))
    hparams.lang = lang_list

    in_domain_classes_num = len(lang_list)

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
    data_set_trn = KaldiDataSet(trn_dir)
    data_set_dev = KaldiDataSet(dev_dir)
    dataloader_trn = DataLoader(data_set_trn, collate_fn=PadCollate(dim=1),
                                batch_size=hparams.batch_size, shuffle=True)
    dataloader_dev = DataLoader(data_set_dev, collate_fn=PadCollate(dim=1),
                                batch_size=hparams.batch_size, shuffle=True)

    criterion = nn.NLLLoss()
    losses = []
    log_interval = 10
    best_acc = 0.0

    for current_epoch in range(hparams.max_epoch):
        total_loss = 0
        batch = 0
        model.train()

        for x,targets in dataloader_trn:
            x = torch.FloatTensor(x).to(device)
            targets = torch.LongTensor(targets).to(device)

            batch_output = model(x)
            loss = criterion(batch_output, targets)

            losses.append(loss.item())
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch += 1
        
        acc, eval_loss, confusion_matrix = evaluate(model, criterion,
                                                    dataloader_dev, exp_dir)
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
    trn_dir = args['<trn_dir>']
    dev_dir = args['<dev_dir>']
    exp_dir = args['<exp_dir>']
    preset = args["--preset"]

    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())

    # make a dir, like exp/2019task1b
    os.makedirs(exp_dir, exist_ok=True)

    fp = open(os.path.join(exp_dir, "config.json"),'w')
    fp.write(hparams.to_json(indent=' '*2))
    fp.close()

    print(hparams_debug_string())

    train(trn_dir, dev_dir, exp_dir)


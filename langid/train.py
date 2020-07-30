# coding: utf-8
"""
Preprocess dataset

usage: train.py [options] <trn_dir> <dev_dir> <exp_dir> 

options:
    --preset=<json>          Path of preset parameters (json).
    --resume=""                 Seed Model
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

from models import Cnn_9layers_AvgPooling
from models import Cnn_13layers_AvgPooling
from dataset import KaldiDataSet, PadCollate
from evaluate import Evaluate
device = torch.device("cuda" if hparams.use_cuda else "cpu")

def train(trn_dir, dev_dir, exp_dir, resume):

    lang_dict, lang_list = utils.ReadLang2UttGetLangLabel(
                                os.path.join(trn_dir, "spk2utt"))
    hparams.lang = lang_list

    in_domain_classes_num = len(lang_list)

    Model = eval(hparams.model_type)

    model = Model(in_domain_classes_num, activation='logsoftmax')

    if hparams.use_cuda:
        model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)

    best_cavg = 9999.9
    best_cavg_acc = "UNK"
    best_cavg_eer = "UNK"
    best_cavg_epo = 0
    best_cavg_loss = 999.9

    current_epoch = 0
    if resume != None:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        current_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        losses = checkpoint['losses']
        if 'best_cavg' in checkpoint:
            best_cavg = checkpoint['best_cavg']

    print(model)

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

    while (current_epoch < hparams.max_epoch):
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
        
        acc, eval_loss, confusion_matrix, cavg, eer, thd = Evaluate(
                                model, criterion, dataloader_dev, exp_dir)
        if best_cavg > cavg:
            best_cavg = cavg
            best_cavg_acc = acc
            best_cavg_eer = eer
            best_cavg_epo = current_epoch
            best_cavg_loss = eval_loss
            torch.save({
                "epoch" : current_epoch,
                "cavg": cavg,
                "acc" : acc,
                "eer" : eer,
                "losses" : losses,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
            }, os.path.join(exp_dir, 'bestcavg.pth'))
        print(": Epoch {} Best(Cavg:{} Acc:{} Epoch:{} Loss:{})".format(current_epoch,
                                                                        best_cavg,
                                                                        best_cavg_acc,
                                                                        best_cavg_epo,
                                                                        best_cavg_loss))
        current_epoch += 1

if __name__=="__main__":
    args = docopt(__doc__)
    trn_dir = args['<trn_dir>']
    dev_dir = args['<dev_dir>']
    exp_dir = args['<exp_dir>']
    preset = args["--preset"]
    resume = args['--resume']

    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())

    # make a dir, like exp/2019task1b
    os.makedirs(exp_dir, exist_ok=True)

    fp = open(os.path.join(exp_dir, "config.json"),'w')
    fp.write(hparams.to_json(indent=' '*2))
    fp.close()

    print(hparams_debug_string())

    train(trn_dir, dev_dir, exp_dir, resume)

# coding: utf-8
"""
Preprocess dataset

usage: preprocess.py [options] <dataset_dir> <feats_path>

options:
    --num_workers=<n>        Num workers.
    --hparams=<parmas>       Hyper parameters [default: ].
    --preset=<json>          Path of preset parameters (json).
    -h, --help               Show help message.
"""

from docopt import docopt
import time
import sys
import glob
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import librosa
import librosa.display
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from hparams import hparams, hparams_debug_string
import utils

def deltas(X_in):
    assert (len(X_in.shape)==3)
    # X_in [ channel : num_feats : time_freams ]
    X_out = (X_in[:,:,2:]-X_in[:,:,:-2])/10.0
    X_out = X_out[:,:,1:-1]+(X_in[:,:,4:]-X_in[:,:,:-4])/5.0
    return X_out

def logmeldeltasx3(logmel):
    
    # compute deltas
    logmel_deltas = deltas(logmel)
    logmel_deltas_deltas = deltas(logmel_deltas)

    # concatenate
    return np.concatenate(
        (logmel[:,:,4:-4],logmel_deltas[:,:,2:-2],logmel_deltas_deltas),axis=0)

class PreProcessBase(object):

    def __init__(self, dataset_dir, feats_dir):
        self.dataset_dir = dataset_dir
        self.feats_dir = feats_dir

        if not os.path.exists(os.path.join(dataset_dir, "wav.scp")):
            raise RuntimeError("wav.scp does not exist.")

        utt2wav,wav2utt = utils.ReadWavScp(os.path.join(dataset_dir, "wav.scp"))
        self.utt2wav = utt2wav
        self.wav2utt = wav2utt

        self.feats = utils.GetFeatScp(self.utt2wav, self.feats_dir)

        if not os.path.exists(feats_path):
            os.makedirs(feats_path,exist_ok=True)
            print("Creat a folder: %s"%feats_path)

    def extract_feature(self, wavefile_path):
        raise Exception("Please implement this function")

    def process(self, num_workers):

        executor = ProcessPoolExecutor(max_workers=num_workers)
        futures = []

        print('Find %d wave files'%len(self.utt2wav))

        for utt,wav in self.utt2wav.items():
            futures.append(
                executor.submit(partial(self.extract_feature, utt, wav)))

        return [future.result() for future in tqdm(futures)]

    def write_feats_scp(self, feats_scp_path):
        feats_scp_content = ""
        for k,v in self.feats.items():
            feats_scp_content += "{} {}\n".format(k, v)
        with open(feats_scp_path, 'w') as fp:
            fp.write(feats_scp_content)
            fp.close()
        print("Write Done %s"%feats_scp_path)

def LogMelOneChannelSave(x, the_feats_path):
    F = librosa.feature.melspectrogram(x,
                        sr=hparams.sample_rate,
                        n_fft=hparams.n_fft,
                        hop_length=hparams.hop_length,
                        win_length=hparams.win_length,
                        n_mels=hparams.num_mels)

    F = np.log10(np.maximum(F, 1e-10))
    F = np.expand_dims(np.flip(F), axis=0)

    if hparams.deltas:
        F = logmeldeltasx3(F)
    np.save(the_feats_path, F.astype(np.float32), allow_pickle=False)

class LogMelPreProcess(PreProcessBase):

    def __init__(self, wavefiles_path, feats_path): 
        super().__init__(wavefiles_path, feats_path)

    def extract_feature(self, utt_id, wavefile_path):
    
        x , sr = librosa.load(wavefile_path, hparams.sample_rate)
        LogMelOneChannelSave(x, self.feats[utt_id])
        return


if __name__ == "__main__":
    args = docopt(__doc__)
    dataset_dir = args["<dataset_dir>"]
    feats_path   = args["<feats_path>"]
    num_workers  = args["--num_workers"]

    if num_workers is None: num_workers = 1
    preset = args["--preset"]

    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    
    print(hparams_debug_string())

    print("The feature file will be located in {}".format(feats_path))
    time_tag = time.time()

    preProcess = LogMelPreProcess(dataset_dir, feats_path)
    preProcess.process(int(num_workers))
    preProcess.write_feats_scp(os.path.join(dataset_dir, "feats.scp"))
    print("Done. {}".format(time.time() - time_tag))



###################################################################

def PlotAFigure(S, file_path):
    plt.matshow(S)
    plt.savefig(file_path, format="png")
    plt.close()


def PlotToCheck(feats_path, num):
    from pathlib import Path
    import random

    plots_path = os.path.join(Path(feats_path).parent, 'plot2check')
    os.system('rm -rf %s/*.png'%plots_path)
    os.makedirs(plots_path, exist_ok=True)

    all_file = os.listdir(feats_path)

    all_utt_ids = list(set([ utils.GetUttID(x) for x in all_file ]))
    num_all = len(all_utt_ids)

    utt_ids = None
    if num < num_all:
        random.shuffle(all_utt_ids)
        utt_ids = all_utt_ids[:num]
    else:
        utt_ids = all_utt_ids

    for i in utt_ids:
        R=np.load(os.path.join(feats_path, "%s-R.npy"%i))
        L=np.load(os.path.join(feats_path, "%s-L.npy"%i))
        A=np.load(os.path.join(feats_path, "%s-A.npy"%i))
        D=np.load(os.path.join(feats_path, "%s-D.npy"%i))
        
        print("PlotToCheck:", "R", R.shape, "L",L.shape, "A", A.shape, "D", D.shape, i)

        # PlotAFigure(R, os.path.join(feats_path, "%s-R.png"%i))
        # PlotAFigure(L, os.path.join(feats_path, "%s-L.png"%i))
        # PlotAFigure(A, os.path.join(feats_path, "%s-A.png"%i))
        # PlotAFigure(D, os.path.join(feats_path, "%s-D.png"%i))

        # fig, ax = plt.subplots(4, 1, figsize=(16,8))
        fig, ax = plt.subplots(4, 1, figsize=(16,int(hparams.num_mels/40*6.5)))

        fig.suptitle(i)

        # plt.figure(figsize=(128, 128))
        # fig.set_size_inches(12,12)

        ax[0].matshow(R)
        ax[0].set_title("Right")
        ax[1].matshow(L)
        ax[1].set_title("Left")
        ax[2].matshow(A)
        ax[2].set_title("Average")
        ax[3].matshow(D)
        ax[3].set_title("Difference")

        # plt.tight_layout()
        plt_path=os.path.join(plots_path, "%s.png"%i)
        
        fig.savefig(plt_path, format="png")



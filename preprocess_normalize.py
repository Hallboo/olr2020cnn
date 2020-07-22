# coding: utf-8
"""Perform meanvar normalization to preprocessed features.

usage: preprocess_normalize.py [options] <in_dir> <out_dir>

options:
    --inverse                Inverse transform.
    --num_workers=<n>        Num workers.
    -h, --help               Show help message.
"""
from docopt import docopt
import sys
import os
from os.path import join, exists, basename, splitext

from sklearn.preprocessing import StandardScaler
from multiprocessing import cpu_count
from tqdm import tqdm
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from shutil import copyfile

import joblib
from glob import glob
from itertools import zip_longest


def get_paths_by_glob(in_dir, filt):
    return glob(join(in_dir, filt))


def _process_utterance(out_dir, feat_path, scaler, inverse):

    # [Required] apply normalization for features
    assert exists(feat_path)
    x = np.load(feat_path)[0]
    if inverse:
        y = scaler.inverse_transform(x)
    else:
        y = scaler.transform(x)
    assert x.dtype == y.dtype
    name = splitext(basename(feat_path))[0]
    np.save(join(out_dir, name), np.expand_dims(y, axis=0), allow_pickle=False)


def apply_normalization_dir2dir(in_dir, out_dir, scaler, inverse, num_workers):

    feature_paths = get_paths_by_glob(in_dir, "*.npy")
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    for feature_path in feature_paths:
        futures.append(executor.submit(
            partial(_process_utterance, out_dir, feature_path, scaler, inverse)))
    for future in tqdm(futures):
        future.result()


if __name__ == "__main__":
    args = docopt(__doc__)
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    inverse = args["--inverse"]
    num_workers = args["--num_workers"]
    num_workers = cpu_count() // 2 if num_workers is None else int(num_workers)

    os.makedirs(out_dir, exist_ok=True)

    scaler = StandardScaler()
    lines = get_paths_by_glob(in_dir, "*.npy")
    assert len(lines) > 0
    for path in tqdm(lines):
        c = np.load(path.strip())
        scaler.partial_fit(c[0])

    print("mean:\n{}".format(scaler.mean_))
    print("var:\n{}".format(scaler.var_))

    apply_normalization_dir2dir(in_dir, out_dir, scaler, inverse, num_workers)

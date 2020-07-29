import os, sys, re
import numpy as np
from sklearn.metrics import confusion_matrix
import computeCavg as cavg
import utils

def ComputeAccuracy(all_outputs, all_predict, all_targets):

    if not (all_predict.shape == all_targets.shape):
        raise ValueError("The shape of all_predict and all_targets must be the same.")
 
    # compute the confusion matrix
    # TODO 未测试在有未知语言时，计算混淆矩阵是否正确。未知语言的target = -1
    confu_mat = confusion_matrix(all_targets, all_predict)
    confu_mat_norm = confu_mat.astype('float') / confu_mat.sum(axis=1)[:, np.newaxis]

    class_total = np.sum(confu_mat, axis=1)
    acc = np.trace(confu_mat) / np.sum(class_total)
    # accuracy is the sum of confusion matrix diagonal divide total number of sample
    # in evalset

    class_accs = np.diagonal(confu_mat) / class_total
    # accuracy for each class

    return round(acc, 4), class_accs, confu_mat

def GetPair(all_outputs, all_targets):
    pairs = []
    lang_num = all_outputs.shape[1]
    for u in range(len(all_outputs)):
        for l in range(len(all_outputs[u])):
            pairs.append([l, all_targets[u], all_outputs[u][l]])

    min_score = np.min(all_outputs)
    max_score = np.max(all_outputs)

    return pairs, lang_num, min_score, max_score

def ComputeCavg(all_outputs, all_targets):

    pairs, lang_num, min_score, max_score = GetPair(all_outputs, all_targets) 
    threshhold_bins = 20
    p_target = 0.5
    cavgs, min_cavg = cavg.get_cavg(pairs, lang_num, min_score, max_score, threshhold_bins, p_target)
    return round(min_cavg, 4)

def ComputeEER(all_outputs, all_targets):
    import subprocess

    pair, _, _, _, = GetPair(all_outputs, all_targets)
    score2target = ''

    # Get <score, target/nontarget> for compute-eer in Kaldi
    for it in pair:
        lang  = it[0]
        lalab = it[1]
        score = it[2]
        if lang == lalab:
            score2target += "{}\ttarget\n".format(score)
        else:
            score2target += "{}\tnontarget\n".format(score)
    score2target = score2target[:-1]

    tempfile = '/tmp/olr-compute-eer-score'
    with open(tempfile, 'w') as fp:
        fp.write(score2target)

    # Compute EER
    cmd = 'compute-eer {}'.format(tempfile).split()
    pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    out, err = pro.communicate()

    # Get the result
    err = err.decode('utf-8').replace('\n', ' ')
    pat = r".*Equal error rate is ([0-9\.]+)%, at threshold ([-0-9\.]+).*"
    eer = round(float(re.sub(pat, r'\1', err)), 4)
    thd = round(float(re.sub(pat, r'\2', err)), 4)

    return eer, thd

if __name__=="__main__":

    output_egs  = 'egs/outputs.npy'
    target_egs  = 'egs/targets.npy'

    all_outputs = np.load(output_egs, allow_pickle=True)
    all_targets = np.load(target_egs, allow_pickle=True)

    all_predict = np.argmax(all_outputs, axis = 1)
    acc, class_accs, confu_mat = ComputeAccuracy(all_outputs, all_predict, all_targets)
    cavg = ComputeCavg(all_outputs, all_targets)
    eer, thd = ComputeEER(all_outputs, all_targets)

    from hparams import hparams

    class_total = np.sum(confu_mat, axis=1)
    for i in range(len(class_accs)):
        print('* Accuracy of {:6s} ........... {:6.2f}% {:4d}/{:<4d}'.format(
            hparams.lang[i], 100*class_accs[i], confu_mat[i][i], class_total[i]))

    print("Acc:{} Cavg: {}  EER: {}%  threshold: {}".format(acc, cavg, eer, thd))

# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2018/2/9 下午9:22
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : Common.py
# @Software: PyCharm Community Edition


import numpy as np


unk_key = '-unk-'
padding_key = '-padding-'
predicate_key = '-predicate-'

seed = 666

srl_labels = ["R-ARGM-COM", "C-ARGM-NEG", "C-ARGM-TMP", "R-ARGM-DIR", "ARGM-LOC", "R-ARG2", "ARGM-GOL", "ARG5", "ARGM-EXT", "R-ARGM-ADV", "C-ARGM-MNR", "ARGA", "C-ARG4", "C-ARG2", "C-ARG3", "C-ARG0", "C-ARG1", "ARGM-ADV", "ARGM-NEG", "R-ARGM-MNR", "C-ARGM-EXT", "R-ARGM-PRP", "C-ARGM-ADV", "R-ARGM-MOD", "C-ARGM-ADJ", "ARGM-LVB", "R-ARGM-PRD", "ARGM-MNR", "ARGM-ADJ", "C-ARGM-CAU", "ARGM-CAU", "C-ARGM-MOD", "R-ARGM-EXT", "C-ARGM-COM", "ARGM-COM", "R-ARGM-GOL", "R-ARGM-TMP", "R-ARG4", "ARGM-MOD", "R-ARG1", "R-ARG0", "R-ARG3", "V", "ARGM-REC", "C-ARGM-DSP", "R-ARG5", "ARGM-DIS", "ARGM-DIR", "R-ARGM-LOC", "C-ARGM-DIS", "ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARGM-TMP", "C-ARGM-DIR", "ARGM-PRD", "R-ARGM-PNC", "ARGM-PRX", "ARGM-PRR", "R-ARGM-CAU", "C-ARGM-LOC", "ARGM-PNC", "ARGM-PRP", "C-ARGM-PRP", "ARGM-DSP"]

def get_idx(elem_list, alpha):
    '''
    :param words: [i like it .]
    :param alpha: Alphabet()
    :return: indexs -> [23, 65, 7]
    '''
    indexs = []
    for elem in elem_list:
        if type(elem) is list:
            idx = []
            for char in elem:
                cid = alpha.get(char, -1)
                if cid == -1:
                    cid = alpha[unk_key]
                idx.append(cid)
            indexs.append(idx)
        else:
            idx = alpha.from_string(elem)
            if idx == -1:
                idx = alpha.from_string(unk_key)
            indexs.append(idx)
    return indexs

death = "Death"
life = "Life"


def getMaxindex(model_out, label_size, args):
    max = model_out.data[0]
    maxIndex = 0
    for idx in range(1, label_size):
        if model_out.data[idx] > max:
            max = model_out.data[idx]
            maxIndex = idx
    return maxIndex


def getMaxindex_np(model_out):
    model_out_list = model_out.data.tolist()
    maxIndex = model_out_list.index(np.max(model_out_list))
    return maxIndex


def getMaxindex_batch(model_out):
    model_out_list = model_out.data.tolist()
    maxIndex_batch = []
    for list in model_out_list:
        maxIndex_batch.append(list.index(np.max(list)))
    return maxIndex_batch

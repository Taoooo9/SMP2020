import torch.nn as nn
import codecs
import pickle
import torch
import numpy as np
import random


seed_num = 666
unkkey = "<unk>"
paddingkey = "<pad>"
cpu_device = "cpu"


def create_embedding(src_vocab, config):
    embedding_dim = 0
    embedding_num = 0
    find_count = 0
    embedding = np.zeros((src_vocab.getsize, 1), dtype='float64')
    with open(config.embedding_file, encoding='utf-8') as f:
        for vec in f.readlines():
            vec = vec.strip()
            vec = vec.split()
            if embedding_num == 0:
                embedding_dim = len(vec) - 1
                embedding = np.zeros((src_vocab.getsize, embedding_dim), dtype='float64')
            if vec[0] in src_vocab.i2w:
                find_count += 1
                vector = np.array(vec[1:], dtype='float64')
                embedding[src_vocab.word2id(vec[0])] = vector
                embedding[src_vocab.UNK] += vector
            embedding_num += 1
        not_find = src_vocab.getsize - find_count
        oov_ration = float(not_find / src_vocab.getsize)
        embedding[src_vocab.UNK] = embedding[src_vocab.UNK] / find_count
        embedding = embedding / np.std(embedding)
        embedding = torch.from_numpy(embedding).float()
        print('Total word:', str(embedding_num))
        print('The dim of pre_embedding:' + str(embedding_dim) + '\n')
        print('oov ratio is: {:.4f}'.format(oov_ration))
        return embedding


def torch_max(output):
    """
    :param output: batch * seq_len * label_num
    :return:
    """
    _, arg_max = torch.max(output, dim=2)
    label = arg_max.view(arg_max.size(0) * arg_max.size(1)).cpu().tolist()
    return label


def print_common():
    print("unkkey", unkkey)
    print("paddingkey", paddingkey)
    print("seed_num", seed_num)


def stop_word(file):
    words = []
    with open(file, encoding="GBK") as f:
        for word in f.readlines():
            words.append(word.strip())
    return words


def clear_data(data, stop_words):
    if isinstance(data, list):
        for word in data[:]:
            if word in stop_words:
                data.remove(word)
            else:
                word.strip().lower()
        return data


def read_pkl(pkl):
    file_pkl = codecs.open(pkl, 'rb')
    return pickle.load(file_pkl)


def random_seed(hope):
    # This is the seed of hope.
    torch.manual_seed(hope)
    torch.cuda.manual_seed(hope)
    np.random.seed(hope)
    random.seed(hope)


def decay_learning_rate(config, optimizer, epoch):
    lr = config.lr / (1 + config.lr_rate_decay * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def correct_num(logit, target, accusation_num):
    correct = 0
    a = torch.max(logit, 1)[1].view(-1, accusation_num)
    b = target.view(-1, accusation_num)
    for i in range(a.size()[0]):
        if torch.equal(a[i], b[i]):
            correct += 1
    return correct


def find_maxlen(batch):
    max_premise_len = 0
    max_hypothesis_len = 0
    for line in batch:
        if len(line.sentence1_list) > max_premise_len:
            max_premise_len = len(line.sentence1_list)
        if len(line.sentence2_list) > max_hypothesis_len:
            max_hypothesis_len = len(line.sentence2_list)
    return max_premise_len, max_hypothesis_len


def find_single_maxlen(batch):
    max_len = 0
    for line in batch:
        if len(line.sentence1_list) > max_len:
            max_len = len(line.sentence1_list)
    return max_len


def find_distinguish_maxlen(batch):
    max_premise_len = 0
    max_hypothesis_len = 0
    for line in batch:
        if len(line[0].sentence1_list) > max_premise_len:
            max_premise_len = len(line[0].sentence1_list)
        if len(line[1].sentence1_list) > max_hypothesis_len:
            max_hypothesis_len = len(line[1].sentence1_list)
    return max_premise_len, max_hypothesis_len


def find_char_maxlen(batch):
    max_premise_len = 0
    max_hypothesis_len = 0
    max_premise_char_len = 0
    max_hypothesis_char_len = 0
    for one_batch in batch:
        if len(one_batch[0]) > max_premise_len:
            max_premise_len = len(one_batch[0])
        for word in one_batch[0]:
            if len(word) > max_premise_char_len:
                max_premise_char_len = len(word)
        for hypothesis in one_batch[1:3]:
            if len(hypothesis) > max_hypothesis_len:
                max_hypothesis_len = len(hypothesis)
            for word in hypothesis:
                if len(word) > max_hypothesis_char_len:
                    max_hypothesis_char_len = len(word)
    return max_premise_len, max_premise_char_len, max_hypothesis_len, max_hypothesis_char_len


def find_maxlennum(batch):
    max_len = 0
    sen_num = 0
    for line in batch:
        if line[3] > max_len:
            max_len = line[3]
        if line[4] > sen_num:
            sen_num = line[4]
    return max_len, sen_num


def split_tensor(origin, word_length, config):
    premise_len = []
    hypothesis_len = []
    batch_size = origin.size(0)
    max_len = origin.size(2)
    dim = origin.size(3)
    premise = torch.zeros((batch_size * 2, max_len, dim), dtype=torch.float)
    hypothesis = torch.zeros((batch_size * 2, max_len, dim), dtype=torch.float)
    for i in range(0, origin.size(0) * 2, 2):
        premise[i] = origin[int(i/2)][0]
        premise[i+1] = origin[int(i/2)][0]
        hypothesis[i] = origin[int(i/2)][1]
        hypothesis[i+1] = origin[int(i/2)][2]
    for j in range(0, len(word_length), 3):
        premise_len.append(word_length[j])
        premise_len.append(word_length[j])
        hypothesis_len.append(word_length[j+1])
        hypothesis_len.append(word_length[j+2])
    premise_len = sorted(premise_len, reverse=True)
    premise_len = torch.tensor(premise_len)
    hypothesis_len = torch.tensor(hypothesis_len)
    hypothesis_len = sorted(hypothesis_len, reverse=True)
    if config.use_cuda:
        premise = premise.cuda()
        hypothesis = hypothesis.cuda()
        premise_len = premise_len.cuda()
        hypothesis_len = hypothesis_len.cuda()
    return premise, hypothesis, premise_len, hypothesis_len

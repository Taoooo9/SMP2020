import os
from collections import Counter
from Units.units import *
UNK, PAD = 0, 1
PAD_S, UNK_S = '<pad>', '<unk>'


class FactWordSrcVocab(object):

    def __init__(self, all_data, config):
        self.w2i = {}
        self.UNK = 1
        word_counter = Counter()
        for data in all_data:
            for line in data:
                for word in line.sentence1:
                    word_counter[word] += 1
                for word in line.sentence2:
                    word_counter[word] += 1
        most_word = [k for k, v in word_counter.most_common()]
        self.i2w = [PAD_S, UNK_S] + most_word
        for idx, word in enumerate(self.i2w):
            self.w2i[word] = idx

        config.add_args('Model', 'embedding_word_num', str(self.getsize))

    def word2id(self, xx):
        if isinstance(xx, list):
            return [self.w2i.get(word, UNK) for word in xx]
        else:
            return self.w2i.get(xx)

    def id2word(self, xx):
        if isinstance(xx, list):
            return [self.i2w[idx] for idx in xx]
        else:
            return self.i2w[xx]

    @property
    def getsize(self):
        return len(self.i2w)


class FactWordTagVocab(object):

    def __init__(self, all_data):
        self.w2i = {}
        word_counter = Counter()
        for data in all_data:
            for line in data:
                # print(line.gold_label)
                word_counter[line.gold_label] += 1
        most_word = [k for k, v in word_counter.most_common()]
        self.i2w = most_word
        for idx, word in enumerate(self.i2w):
            self.w2i[word] = idx

    def word2id(self, xx):
        if isinstance(xx, list):
            return [self.w2i.get(word, UNK) for word in xx]
        else:
            return self.w2i.get(xx)

    def id2word(self, xx):
        if isinstance(xx, list):
            return [self.i2w[idx] for idx in xx]
        else:
            return self.i2w[xx]

    @property
    def getsize(self):
        return len(self.i2w)


class DistinguishVocab(object):

    def __init__(self):
        self.w2i = {"unique": 0, "mix": 1}
        self.i2w = ['unique', 'mix']

    def word2id(self, xx):
        if isinstance(xx, list):
            return [self.w2i.get(word, UNK) for word in xx]
        else:
            return self.w2i.get(xx)

    def id2word(self, xx):
        if isinstance(xx, list):
            return [self.i2w[idx] for idx in xx]
        else:
            return self.i2w[xx]

    @property
    def getsize(self):
        return len(self.i2w)



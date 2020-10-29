import torch
import math
import numpy as np
import random
from data.Common import *
import copy
random.seed(seed)


class buildBatch():
    def __init__(self, insts, config, padding_id, char_padding_id, batch_size):
        self.insts = insts
        self.batch_size = batch_size
        self.eval_batch_size = config.eval_batch_size
        self.padding_id = padding_id
        self.char_padding_id = char_padding_id
        self.shuffle = config.shuffle
        self.sort = config.sort
        self.batchs = math.ceil(len(self.insts) / self.batch_size)
        self.eval_batchs = math.ceil(len(self.insts) / self.eval_batch_size)
        self.max_tokens_per_batch = config.max_tokens_per_batch
        self.inst_num = 0

    def batchIter(self):
        insts = self.insts
        if self.sort:
            insts = sorted(self.insts, key=lambda inst: len(inst.sentence_id_list))
        if self.shuffle:
            insts = self.insts
            random.shuffle(insts)

        start = 0
        tokens = 0
        inst_num = 0
        batch_insts = []
        for idx, inst in enumerate(insts):
            tokens += len(inst.sentence_list)
            inst_num += 1
            batch_insts.append(inst)
            if tokens > self.max_tokens_per_batch or inst_num == self.batch_size or idx == len(insts)-1:
                if tokens > self.max_tokens_per_batch:
                    info_dic = self.getBatch(batch_insts[:-1])
                    batch_insts = []
                    batch_insts.append(inst)
                    tokens = len(inst.sentence_list)
                    inst_num = 1
                else:
                    info_dic = self.getBatch(batch_insts)
                    batch_insts = []
                    inst_num = 0
                    tokens = 0
                yield info_dic

    def getBatch(self, batch_insts):
        batch_insts = sorted(batch_insts, key=lambda inst: len(inst.sentence_id_list), reverse=True)
        # word max length
        length_list = [inst.sentence_len for inst in batch_insts]
        max_len = max(length_list)
        # char max length
        max_char_len = max([len(char_list) for inst in batch_insts for char_list in inst.char_id_list])

        info_dic = {}
        word_id_tensor = torch.zeros((len(batch_insts), max_len)).fill_(self.padding_id).long()
        char_id_tensor = torch.zeros((len(batch_insts), max_len, max_char_len)).fill_(self.char_padding_id).long()
        sentence_list = []
        pre_start_list = []
        pre_end_list = []
        arg_start_list = []
        arg_end_list = []
        arg_label_list = []
        for i, inst in enumerate(batch_insts):
            sentence_list.append(inst.sentence_list)
            pre_start_list.append(inst.pre_starts)
            pre_end_list.append(inst.pre_ends)
            arg_start_list.append(inst.arg_starts)
            arg_end_list.append(inst.arg_ends)
            arg_label_list.append(inst.arg_labels_list)
            for j, word_id in enumerate(inst.sentence_id_list):
                word_id_tensor[i][j] = word_id
                for k, char_id in enumerate(inst.char_id_list[j]):
                    char_id_tensor[i][j][k] = char_id

        srl_len_list = [len(pre_start) for pre_start in pre_start_list]
        pre_start_list = self.pad(pre_start_list, padid=0)
        pre_end_list = self.pad(pre_end_list, padid=0)
        arg_start_list = self.pad(arg_start_list, padid=0)
        arg_end_list = self.pad(arg_end_list, padid=0)
        arg_label_list = self.pad(arg_label_list, padid=0)

        srl_len_tensor = torch.LongTensor(srl_len_list)
        pre_start_tensor = torch.LongTensor(pre_start_list)
        pre_end_tensor = torch.LongTensor(pre_end_list)
        arg_start_tensor = torch.LongTensor(arg_start_list)
        arg_end_tensor = torch.LongTensor(arg_end_list)
        arg_label_tensor = torch.LongTensor(arg_label_list)

        info_dic['srl_len_tensor'] = srl_len_tensor
        info_dic['word_id_tensor'] = word_id_tensor
        info_dic['length_list'] = length_list
        info_dic['char_id_tensor'] = char_id_tensor
        info_dic['pre_start_tensor'] = pre_start_tensor
        info_dic['pre_end_tensor'] = pre_end_tensor
        info_dic['arg_start_tensor'] = arg_start_tensor
        info_dic['arg_end_tensor'] = arg_end_tensor
        info_dic['arg_label_tensor'] = arg_label_tensor
        info_dic['sentence_list'] = sentence_list

        return info_dic

    def evalBatchIter(self):
        insts = self.insts
        if self.sort:
            insts = sorted(self.insts, key=lambda inst: len(inst.sentence_id_list))
        if self.shuffle:
            insts = self.insts
            random.shuffle(insts)

        batch_insts = []
        info_dic_list = []
        for idx, inst in enumerate(insts):
            batch_insts.append(inst)
            if idx == self.batch_size-1:
                info_dic = self.getEvalBatch(batch_insts)
                batch_insts = []
                info_dic_list.append(info_dic)
        if len(batch_insts) > 0:
            info_dic = self.getEvalBatch(batch_insts)
            info_dic_list.append(info_dic)
        return info_dic_list
        # print(self.inst_num)

    def getEvalBatch(self, batch_insts):
        self.inst_num += len(batch_insts)
        # print(len(batch_insts))
        #batch_instss = sorted(batch_insts, key=lambda inst: len(inst.sentence_id_list), reverse=True)
        # word max length
        length_list = [inst.sentence_len for inst in batch_insts]
        max_len = max(length_list)
        # char max length
        max_char_len = max([len(char_list) for inst in batch_insts for char_list in inst.char_id_list])

        info_dic = {}
        word_id_tensor = torch.zeros((len(batch_insts), max_len)).fill_(self.padding_id).long()
        char_id_tensor = torch.zeros((len(batch_insts), max_len, max_char_len)).fill_(self.char_padding_id).long()
        sentence_list = []
        pre_start_list = []
        pre_end_list = []
        arg_start_list = []
        arg_end_list = []
        arg_label_list = []
        srl_rels_list = []
        for i, inst in enumerate(batch_insts):
            sentence_list.append(inst.sentence_list)
            pre_start_list.append(inst.pre_starts)
            pre_end_list.append(inst.pre_ends)
            arg_start_list.append(inst.arg_starts)
            arg_end_list.append(inst.arg_ends)
            arg_label_list.append(inst.arg_labels_list)
            srl_rels_list.append(inst.srl_rels)
            for j, word_id in enumerate(inst.sentence_id_list):
                word_id_tensor[i][j] = word_id
                for k, char_id in enumerate(inst.char_id_list[j]):
                    char_id_tensor[i][j][k] = char_id

        srl_len_list = [len(pre_start) for pre_start in pre_start_list]
        pre_start_list = self.pad(pre_start_list, padid=0)
        pre_end_list = self.pad(pre_end_list, padid=0)
        arg_start_list = self.pad(arg_start_list, padid=0)
        arg_end_list = self.pad(arg_end_list, padid=0)
        arg_label_list = self.pad(arg_label_list, padid=0)

        srl_len_tensor = torch.LongTensor(srl_len_list)
        pre_start_tensor = torch.LongTensor(pre_start_list)
        pre_end_tensor = torch.LongTensor(pre_end_list)
        arg_start_tensor = torch.LongTensor(arg_start_list)
        arg_end_tensor = torch.LongTensor(arg_end_list)
        arg_label_tensor = torch.LongTensor(arg_label_list)

        info_dic['srl_len_tensor'] = srl_len_tensor
        info_dic['word_id_tensor'] = word_id_tensor
        info_dic['length_list'] = length_list
        info_dic['char_id_tensor'] = char_id_tensor
        info_dic['pre_start_tensor'] = pre_start_tensor
        info_dic['pre_end_tensor'] = pre_end_tensor
        info_dic['arg_start_tensor'] = arg_start_tensor
        info_dic['arg_end_tensor'] = arg_end_tensor
        info_dic['arg_label_tensor'] = arg_label_tensor
        info_dic['sentence_list'] = sentence_list
        info_dic['srl_rels_list'] = srl_rels_list

        return info_dic

    # def evalBatchIter(self):
    #     insts = self.insts
    #     if self.sort:
    #         insts = sorted(self.insts, key=lambda inst: len(inst.sentence_id_list))
    #     if self.shuffle:
    #         insts = self.insts
    #         random.shuffle(insts)
    #     start = 0
    #     for i in range(self.eval_batchs):
    #         batch_insts = insts[start:start+self.batch_size]
    #         start += self.eval_batch_size
    #         batch_insts = sorted(batch_insts, key=lambda inst: len(inst.sentence_id_list), reverse=True)
    #         # word max length
    #         length_list = [inst.sentence_len for inst in batch_insts]
    #         max_len = max(length_list)
    #         # char max length
    #         max_char_len = max([len(char_list) for inst in batch_insts for char_list in inst.char_id_list])
    #
    #         info_dic = {}
    #         word_id_tensor = torch.zeros((len(batch_insts), max_len)).fill_(self.padding_id).long()
    #         char_id_tensor = torch.zeros((len(batch_insts), max_len, max_char_len)).fill_(self.char_padding_id).long()
    #         sentence_list = []
    #         predicate_list = []
    #         arg_start_list = []
    #         arg_end_list = []
    #         arg_label_list = []
    #         srl_rels_list = []
    #         for i, inst in enumerate(batch_insts):
    #             sentence_list.append(inst.sentence_list)
    #             predicate_list.append(inst.predicates)
    #             arg_start_list.append(inst.arg_starts)
    #             arg_end_list.append(inst.arg_ends)
    #             arg_label_list.append(inst.arg_labels_list)
    #             srl_rels_list.append(inst.srl_rels)
    #             for j, word_id in enumerate(inst.sentence_id_list):
    #                 word_id_tensor[i][j] = word_id
    #                 for k, char_id in enumerate(inst.char_id_list[j]):
    #                     char_id_tensor[i][j][k] = char_id
    #
    #         srl_len_list = [len(predicate) for predicate in predicate_list]
    #         predicate_list = self.pad(predicate_list, padid=0)
    #         arg_start_list = self.pad(arg_start_list, padid=0)
    #         arg_end_list = self.pad(arg_end_list, padid=0)
    #         arg_label_list = self.pad(arg_label_list, padid=0)
    #
    #         srl_len_tensor = torch.LongTensor(srl_len_list)
    #         predicate_tensor = torch.LongTensor(predicate_list)
    #         arg_start_tensor = torch.LongTensor(arg_start_list)
    #         arg_end_tensor = torch.LongTensor(arg_end_list)
    #         arg_label_tensor = torch.LongTensor(arg_label_list)
    #
    #         info_dic['srl_len_tensor'] = srl_len_tensor
    #         info_dic['word_id_tensor'] = word_id_tensor
    #         info_dic['length_list'] = length_list
    #         info_dic['char_id_tensor'] = char_id_tensor
    #         info_dic['predicate_tensor'] = predicate_tensor
    #         info_dic['arg_start_tensor'] = arg_start_tensor
    #         info_dic['arg_end_tensor'] = arg_end_tensor
    #         info_dic['arg_label_tensor'] = arg_label_tensor
    #         info_dic['sentence_list'] = sentence_list
    #         info_dic['srl_rels_list'] = srl_rels_list


            # yield info_dic



    def subPad(self, l, max_len):
        for i in range(len(l)):
            for j in range(len(l[i])):
                l[i][j] = l[i][j] + [l[i][j][-1]]*(max_len - len(l[i][j]))
        return l

    def pad(self, l: [[list]], padid=-1):
        max_len = max([len(sub_l) for sub_l in l])
        for i in range(len(l)):
            l[i].extend([padid]*(max_len - len(l[i])))
        return l
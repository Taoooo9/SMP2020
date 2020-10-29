# Version python3.6
# -*- coding: utf-8 -*-

from collections import OrderedDict
from data.Instance import Instance
from data.Alphabet import Alphabet
from data.Common import *
import json


class Dataloader:
    def __init__(self, kd_train_file_p, kd_train_file_h, ca_train_file_p, ca_train_file_h, kd_dev_file_p,
                 kd_dev_file_h, ca_dev_file_p, ca_dev_file_h, config):
        self.kd_p_train_insts = []
        self.kd_h_train_insts = []
        self.ca_p_train_insts = []
        self.ca_h_train_insts = []

        self.kd_p_dev_insts = []
        self.kd_h_dev_insts = []
        self.ca_p_dev_insts = []
        self.ca_h_dev_insts = []

        self.config = config
        self.words = []

        self.word_alphabet = None
        self.char_alphabet = {}

        # self.adjunct_roles, self.core_roles = self.split_srl_labels(srl_labels, config.include_c_v)
        # self.srl_labels_inv = [""] + self.adjunct_roles + self.core_roles
        # self.srl_labels = {l: i for i, l in enumerate(self.srl_labels_inv)}
        self.srl_labels_dic = {'': 0}
        srl_label_dic_index = 1
        # build char dict
        with open(config.char_vocab_path, 'r', encoding='utf8') as f:
            self.char_alphabet[unk_key] = 0
            self.char_alphabet[padding_key] = 1
            for idx, char in enumerate(f.readlines()):
                char = char.strip()
                if char not in self.char_alphabet:
                    self.char_alphabet[char] = idx+1

        # read json into inst
        for idx, path in enumerate([kd_train_file_p, kd_train_file_h, ca_train_file_p, ca_train_file_h, kd_dev_file_p,
                 kd_dev_file_h, ca_dev_file_p, ca_dev_file_h]):
            with open(path, 'r', encoding='utf8') as f:
                doc_num = 0
                sent_num = 0
                info_list = []
                for line in f.readlines():
                    line = line.strip()
                    if len(line) == 0 and len(info_list) != 0:
                        srl, sentence = self.build_inst(info_list)
                        info_list = []
                        doc_num += 1
                        sent_num += 1
                        char_list = []
                        for word in sentence:
                            char_list.append(list(word))
                        gold_predicates = self.get_all_predicates(srl)

                        # label
                        pre_starts, pre_ends, arg_starts, arg_ends, arg_labels = self.tensorize_srl_relations(srl)

                        srl_rels = {}
                        for r in srl:
                            pred_id = str([r[0], r[1]])
                            if pred_id not in srl_rels:
                                srl_rels[pred_id] = []
                            srl_rels[pred_id].append((r[2], r[3], r[4]))

                        if idx == 0:
                            for arg_label in arg_labels:
                                if arg_label not in self.srl_labels_dic:
                                    self.srl_labels_dic[arg_label] = srl_label_dic_index
                                    srl_label_dic_index += 1
                        inst = Instance(sentence, srl, char_list, gold_predicates, pre_starts, pre_ends, arg_starts, arg_ends, arg_labels, srl_rels)
                        inst.default()
                        self.words.extend(sentence)
                        if idx == 0:
                            self.kd_p_train_insts.append(inst)
                        elif idx == 1:
                            self.kd_h_train_insts.append(inst)
                        elif idx == 2:
                            self.ca_p_train_insts.append(inst)
                        elif idx == 3:
                            self.ca_h_train_insts.append(inst)
                        elif idx == 4:
                            self.kd_p_dev_insts.append(inst)
                        elif idx == 5:
                            self.kd_h_dev_insts.append(inst)
                        elif idx == 6:
                            self.ca_p_dev_insts.append(inst)
                        elif idx == 7:
                            self.ca_h_dev_insts.append(inst)
                    else:
                        info_list.append(line)
                if len(info_list) != 0:
                    srl, sentence = self.build_inst(info_list)
                    info_list = []
                    doc_num += 1
                    sent_num += 1
                    char_list = []
                    for word in sentence:
                        char_list.append(list(word))
                    gold_predicates = self.get_all_predicates(srl)

                    # label
                    pre_starts, pre_ends, arg_starts, arg_ends, arg_labels = self.tensorize_srl_relations(srl)

                    srl_rels = {}
                    for r in srl:
                        pred_id = str([r[0], r[1]])
                        if pred_id not in srl_rels:
                            srl_rels[pred_id] = []
                        srl_rels[pred_id].append((r[2], r[3], r[4]))

                    if idx == 0:
                        for arg_label in arg_labels:
                            if arg_label not in self.srl_labels_dic:
                                self.srl_labels_dic[arg_label] = srl_label_dic_index
                                srl_label_dic_index += 1
                    inst = Instance(sentence, srl, char_list, gold_predicates, pre_starts, pre_ends, arg_starts,
                                    arg_ends, arg_labels, srl_rels)
                    inst.default()
                    self.words.extend(sentence)
                    if idx == 0:
                        self.kd_p_train_insts.append(inst)
                    elif idx == 1:
                        self.kd_h_train_insts.append(inst)
                    elif idx == 2:
                        self.ca_p_train_insts.append(inst)
                    elif idx == 3:
                        self.ca_h_train_insts.append(inst)
                    elif idx == 4:
                        self.kd_p_dev_insts.append(inst)
                    elif idx == 5:
                        self.kd_h_dev_insts.append(inst)
                    elif idx == 6:
                        self.ca_p_dev_insts.append(inst)
                    elif idx == 7:
                        self.ca_h_dev_insts.append(inst)
                print('{} has {} docs, {} sentences.'.format(path, doc_num, sent_num))

            if idx == 7:
                self.dict = self.buildDictionary(self.words)
                self.word_alphabet = self.build_vab(self.dict, need_pad_unk=True)
            if idx == 7:
                self.wordChar2Id(self.kd_p_train_insts)
                self.wordChar2Id(self.kd_h_train_insts)
                self.wordChar2Id(self.ca_p_train_insts)
                self.wordChar2Id(self.ca_h_train_insts)
                self.wordChar2Id(self.kd_p_dev_insts)
                self.wordChar2Id(self.kd_h_dev_insts)
                self.wordChar2Id(self.ca_p_dev_insts)
                self.wordChar2Id(self.ca_h_dev_insts)
        # self.save_inst_sentence(self.p_train_insts, 'orl/train.json')
        # self.save_inst_sentence(self.h_train_insts, 'orl/train.json')
        # self.save_inst_sentence(self.p_dev_insts, 'orl/dev.json')
        # self.save_inst_sentence(self.h_dev_insts, 'orl/dev.json')
        # self.save_inst_sentence(self.p_test_insts, 'orl/test.json')
        # self.save_inst_sentence(self.h_test_insts, 'orl/test.json')
        print('word dictionary size:{} label size:{}'.format(len(self.word_alphabet.id2string), len(self.srl_labels_dic)))

    def save_inst_sentence(self, insts, save_path):
        # {"doc_key": "S6", "sentences": [["The", "new", "rate", "will", "be", "payable", "Feb.", "15", "."]],
        #  "srl": [[[0, 0, 0, "A1"]]]}
        with open(save_path, 'w', encoding='utf8') as f:
            for idx, inst in enumerate(insts):
                inst_dic = {}
                inst_dic["doc_key"] = str(idx)
                inst_dic["sentences"] = [inst.sentence_list]
                inst_dic["srl"] = [[[0, 0, 0, "A1"]]]
                json_str = json.dumps(inst_dic)
                f.write(json_str + '\n')

    def build_inst(self, info_list):
        dic_start = {}
        sentence = []
        start_idx = -1
        srl = []
        for idx, line_info in enumerate(info_list):
            line_info_list = line_info.split()
            if line_info_list[0] == 'token':
                # token Out IN -1 none o
                word = line_info_list[1]
                if self.config.lower:
                    word = word.lower()
                sentence.append(word)
                # (s|b|m|e)-(TARGET|DSE|AGENT)
                label = line_info_list[-1].split('-')
                if label[0] == 's':
                    dic_start[idx] = idx
                elif label[0] == 'b':
                    start_idx = idx
                elif label[0] == 'm':
                    continue
                elif label[0] == 'e':
                    if start_idx == -1:
                        raise RuntimeError
                    dic_start[idx] = start_idx
                    start_idx = -1
            elif line_info_list[0] == 'rel':
                # rel 4 8 -1 TARGET-DSE
                rel_0, rel_1 = line_info_list[1], line_info_list[2]
                if line_info_list[3] == '-1':
                    rel_0, rel_1 = rel_1, rel_0
                rel_0, rel_1 = int(rel_0), int(rel_1)
                label = line_info_list[-1]
                srl.append((dic_start[rel_1], rel_1, dic_start[rel_0], rel_0, label))
            else:
                print('wrong: first word is ', line_info_list[0])
                raise RuntimeError
        return srl, sentence


    def split_srl_labels(self, srl_labels, include_c_v):
        adjunct_role_labels = []
        core_role_labels = []
        for label in srl_labels:
            if "AM" in label or "ARGM" in label:
                adjunct_role_labels.append(label)
            elif label != "V" and (include_c_v or label != "C-V"):
                core_role_labels.append(label)
        return adjunct_role_labels, core_role_labels

    def get_all_predicates(self, tuples):
        if len(tuples) > 0:
            pre_starts, pre_ends, _, _, _ = zip(*tuples)
            predicates = []
            for pre_start, pre_end in zip(pre_starts, pre_ends):
                predicates.append((pre_start, pre_end))
        else:
            predicates = []
        return list(set(predicates))

    def tensorize_srl_relations(self, tuples):
        if len(tuples) > 0:
            head_starts, head_ends, starts, ends, labels = zip(*tuples)
        else:
            head_starts, head_ends, starts, ends, labels = [], [], [], [], []
        # return (list(heads), list(starts), list(ends), [label_dict.get(c, 0) for c in labels])
        return (list(head_starts), list(head_ends), list(starts), list(ends), labels)

    def buildOnehot(self, accu_alphabet, article_alphabet, insts):
        max_accu_size = accu_alphabet.m_size
        max_article_size = article_alphabet.m_size
        for inst in insts:
            inst.accu_label_id_list = [0 for _ in range(max_accu_size)]
            inst.article_label_id_list = [0 for _ in range(max_article_size)]
            for idx in inst.accu_id_list:
                inst.accu_label_id_list[idx] = 1
            for idx in inst.article_id_list:
                inst.article_label_id_list[idx] = 1

    def wordChar2Id(self, insts):
        for inst in insts:
            inst.sentence_id_list = get_idx(inst.sentence_list, self.word_alphabet)
            inst.char_id_list = get_idx(inst.char_list, self.char_alphabet)
            inst.arg_labels_list = [self.srl_labels_dic .get(label, 0) for label in inst.arg_labels]

    def buildDictionary(self, words):
        print('build vacab...')
        dict = OrderedDict()
        # char_dict = OrderedDict()
        for word in words:
            if word not in dict:
                dict[word] = 1
            else:
                dict[word] += 1
            # for char in word:
            #     if char not in char_dict:
            #         char_dict[char] = 1
            #     else:
            #         char_dict[char] += 1
        return dict

    def build_vab(self, dict, need_pad_unk=True):
        '''
        :param dict: OrderedDict() -> freq:word
        :param cutoff: frequence's smaller than cutoff will be deleted.
        :return: alphabet class
        '''
        if need_pad_unk:
            dict[unk_key] = 100
            dict[padding_key] = 100
        alphabet = Alphabet(cutoff=self.config.cutoff, max_cap=self.config.max_cap)
        alphabet.initial(dict)
        alphabet.m_b_fixed = True

        # char_alphabet = None
        # if self.char_dict != None:
        #     self.char_dict[unk_key] = 100
        #     self.char_dict[padding_key] = 100
        #     char_alphabet = Alphabet(cutoff=self.config.cutoff, max_cap=self.config.max_cap)
        #     char_alphabet.initial(self.char_dict)
        #     char_alphabet.m_b_fixed = True

        return alphabet





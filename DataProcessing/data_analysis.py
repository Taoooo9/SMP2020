import json
import os

from collections import Counter
from Units.units import *


class DataAnalysis(object):

    def __init__(self, config, file):
        self.config = config
        if not os.path.isdir(config.save_dir):
            os.makedirs(config.save_dir)
        if not os.path.isdir(config.save_analysis_pkl_path):
            os.makedirs(config.save_analysis_pkl_path)
        if config.stop_word:
            self.stop_words = stop_word(config.stop_words)
        if os.path.isfile(self.config.train_data_analysis_pkl):
            information = read_pkl(self.config.train_data_analysis_pkl)
        else:
            information = self.load_data(file, self.stop_words)
            pickle.dump(information, open(self.config.train_data_analysis_pkl, 'wb'))
        self.fact_len = information[0]
        self.punish = information[1]
        self.accusation = information[2]
        self.relevant_articles = information[3]
        self.accusation_radio = {}

    @staticmethod
    def select(line):
        return len(line[0]) > 1500

    def analysis_k(self):
        file = 'Dataset/greater_k.txt'
        if os.path.exists(file):
            os.remove(file)
        writer = open(file, encoding='utf-8', mode='w')
        counter_k = Counter()
        data = []
        for i, k in zip(self.fact_len, self.accusation):
            data.append([i, k])
        src = list(filter(self.select, data))
        for idx, line in enumerate(src):
            counter_k[','.join(line[1])] += 1
        for k, v in counter_k.most_common():
            value = self.accusation_radio.get(k)
            value = (value / 100) * 154256
            writer.write('罪名：' + str(k) + '--->' + '数量：' + str(v) + '\t\t' + '犯罪数/总量：' + str(int(value)) + "\n")
        writer.close()

    def analysis_accusation(self):
        file = 'Dataset/accusation.txt'
        accusation = {}
        if os.path.exists(file):
            os.remove(file)
        writer = open(file, encoding='utf-8', mode='w')
        len_counter = Counter()
        for line in self.accusation:
            len_counter[','.join(line)] += 1
        for k, v in len_counter.most_common():
            radio = 100 * v / len(self.fact_len)
            writer.write('罪名：' + str(k) + '--->' + '数量：' + str(v) + '\t\t' + '犯罪数/总量：' + str(radio) + '%' + "\n")
            accusation[k] = radio
        self.accusation_radio = accusation
        writer.close()

    def analysis_fact_len(self):
        file = 'Dataset/length.txt'
        if os.path.exists(file):
            os.remove(file)
        writer = open(file, encoding='utf-8', mode='w')
        src_ids = sorted(range(len(self.fact_len)), key=lambda src_id: len(self.fact_len[src_id]), reverse=True)
        src = [self.fact_len[src_id] for src_id in src_ids]
        len_counter = Counter()
        for line in self.fact_len:
            len_counter[str(len(line))] += 1
        for k, v in len_counter.most_common():
            writer.write('长度：' + str(k) + '--->' + '数量：' + str(v) + "\n")
        writer.close()

    def unequal_data(self):
        pass

    def load_data(self, file, stop_words):
        fac_len_list = []
        punish_list = []
        accusation_list = []
        relevant_articles_list = []
        with open(file, encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line)
                meta = line['meta']
                criminals = meta['criminals']  # list
                term_of_imprisonment = meta['term_of_imprisonment']  # three children have
                death_penalty = term_of_imprisonment['death_penalty']  # bool
                imprisonment = term_of_imprisonment['imprisonment']  # int
                life_imprisonment = term_of_imprisonment['life_imprisonment']  # bool
                punish_of_money = meta['punish_of_money']  # int
                punish_list.append(punish_of_money)
                accusation = meta['accusation']  # list
                accusation_list.append(accusation)
                relevant_articles = meta['relevant_articles']  # list
                relevant_articles_list.append(relevant_articles)
                fact = line['fact']  # str
                fact = fact.split(' ')
                fact = clear_data(fact, stop_words)
                fac_len_list.append(fact)
        return [fac_len_list, punish_list, accusation_list, relevant_articles_list]







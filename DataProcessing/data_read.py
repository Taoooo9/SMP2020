import os
import jieba
import re

from Units.units import *


class SMP_Data(object):

    def __init__(self, sentence_id, topic, gold_label, sentence1, sentence1_list):
        self.sentence_id = sentence_id
        self.topic = topic
        self.gold_label = gold_label
        self.sentence1 = sentence1
        self.sentence1_list = sentence1_list


class ReadData(object):

    def __init__(self, config):
        self.config = config
        if not os.path.isdir(config.save_dir):
            os.makedirs(config.save_dir)
        if not os.path.isdir(config.save_pkl_path):
            os.makedirs(config.save_pkl_path)

    def read_data(self, tokenizer):
        if os.path.isfile(self.config.usual_train_data_word_pkl):
            usual_tra_data = read_pkl(self.config.usual_train_data_word_pkl)
            virus_tra_data = read_pkl(self.config.virus_train_data_word_pkl)
            usual_vat_data = read_pkl(self.config.usual_vat_data_word_pkl)
            virus_vat_data = read_pkl(self.config.virus_vat_data_word_pkl)
        else:
            usual_tra_data = self.load_data(self.config.usual_train_file, tokenizer, topic='usual')
            virus_tra_data = self.load_data(self.config.virus_train_file, tokenizer, topic='virus')
            usual_vat_data = self.load_eval_data(self.config.usual_vat_file, tokenizer, topic='usual')
            virus_vat_data = self.load_eval_data(self.config.virus_vat_file, tokenizer, topic='virus')
            pickle.dump(usual_tra_data, open(self.config.usual_train_data_word_pkl, 'wb'))
            pickle.dump(usual_vat_data, open(self.config.usual_vat_data_word_pkl, 'wb'))
            pickle.dump(virus_tra_data, open(self.config.virus_train_data_word_pkl, 'wb'))
            pickle.dump(virus_vat_data, open(self.config.virus_vat_data_word_pkl, 'wb'))
        if os.path.isfile(self.config.usual_dev_data_word_pkl):
            usual_dev_data = read_pkl(self.config.usual_dev_data_word_pkl)
            virus_dev_data = read_pkl(self.config.virus_dev_data_word_pkl)
        else:
            usual_dev_data = self.load_data(self.config.usual_dev_file, tokenizer, topic='usual')
            virus_dev_data = self.load_data(self.config.virus_dev_file, tokenizer, topic='virus')
            pickle.dump(usual_dev_data, open(self.config.usual_dev_data_word_pkl, 'wb'))
            pickle.dump(virus_dev_data, open(self.config.virus_dev_data_word_pkl, 'wb'))
        return usual_tra_data, virus_tra_data, usual_dev_data, virus_dev_data, usual_vat_data, virus_vat_data

    def read_eval_data(self, tokenizer):
        usual_eval_data = self.load_eval_data(self.config.usual_eval_file, tokenizer, topic='usual')
        virus_eval_data = self.load_eval_data(self.config.virus_eval_file, tokenizer, topic='virus')
        return usual_eval_data, virus_eval_data

    def load_data(self, file, tokenizer, topic=''):
        data = []
        with open(file, encoding='utf-8') as f:
            for line in f.readlines():
                all_data = eval(line)
                for value in all_data:
                    if value['content']:
                        content = re.sub(r'//@.*?:', '', value['content'])
                        content = re.sub(
                            r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?',
                            '', content)
                        sentence_list = jieba.lcut(content)
                        bert_piece = tokenizer.encode_plus(text=content, add_special_tokens=True, return_tensors='pt')
                        bert_len = bert_piece["input_ids"].size()[1]
                        while bert_len > self.config.max_length:
                            content = content[bert_len - self.config.max_length + 1:]
                            bert_piece = tokenizer.encode_plus(text=content, add_special_tokens=True, return_tensors='pt')
                            bert_len = bert_piece["input_ids"].size()[1]
                        data.append(SMP_Data(value['id'], topic, value['label'], content, sentence_list))
            return data

    def load_eval_data(self, file, tokenizer, topic=''):
        data = []
        with open(file, encoding='utf-8') as f:
            for line in f.readlines():
                all_data = eval(line)
                for value in all_data:
                    if value['content']:
                        content = re.sub(r'//@.*?:', '', value['content'])
                        content = re.sub(
                            r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?',
                            '', content)
                        sentence_list = jieba.lcut(content)
                        bert_piece = tokenizer.encode_plus(text=content, add_special_tokens=True, return_tensors='pt')
                        bert_len = bert_piece["input_ids"].size()[1]
                        while bert_len > self.config.max_length:
                            content = content[bert_len - self.config.max_length + 1:]
                            bert_piece = tokenizer.encode_plus(text=content, add_special_tokens=True, return_tensors='pt')
                            bert_len = bert_piece["input_ids"].size()[1]
                        data.append(SMP_Data(value['id'], topic, None, content, sentence_list))
            return data




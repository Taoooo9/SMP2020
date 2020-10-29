import torch

from data.Dataloader import Dataloader
from data.buildBatch import buildBatch
from driver.SRLModel import SRLModel
from srl_config.config import Configurable
import torch.nn as nn
from data.Embedding import load_predtrained_emb_zero
import argparse
from data.Common import *


class Decoder():
    def __init__(self, new_word_alphabet, config, use_cuda):
        self.config = config
        self.model_path = self.config.load_model_path
        self.alphabet_dic = torch.load(self.config.load_vocab_path)
        self.use_cuda = use_cuda
        self.model = SRLModel(self.alphabet_dic, self.config, self.use_cuda)

        if self.use_cuda:
            self.model.load_state_dict(torch.load(self.config.load_model_path))
        else:
            self.model.load_state_dict(torch.load(self.config.load_model_path, map_location='cpu'))
        print('load historic model from {} successfully !'.format(self.config.load_model_path))

        print('word alphabet:', new_word_alphabet.m_size)

        # change the embedding
        self.model.context_embeddings = nn.Embedding(new_word_alphabet.m_size, self.config.word_embed_dim)
        nn.init.xavier_uniform_(self.model.context_embeddings.weight.data)
        self.model.head_embeddings = nn.Embedding(new_word_alphabet.m_size, self.config.word_embed_dim)
        nn.init.xavier_uniform_(self.model.head_embeddings.weight.data)
        embedding = load_predtrained_emb_zero(self.config.context_embedding_path, new_word_alphabet.string2id)
        self.model.context_embeddings.weight.data.copy_(embedding)
        self.model.context_embeddings.weight.requires_grad = False
        embedding = load_predtrained_emb_zero(self.config.head_embedding_path, new_word_alphabet.string2id)
        self.model.head_embeddings.weight.data.copy_(embedding)
        self.model.head_embeddings.weight.requires_grad = False

    def decode(self, word_id_tensor, char_id_tensor, length_list, info_dic):

        word_id_tensor = info_dic['word_id_tensor']
        length_list = info_dic['length_list']
        char_id_tensor = info_dic['char_id_tensor']

        if self.use_cuda:
            word_id_tensor = word_id_tensor.cuda()
            char_id_tensor = char_id_tensor.cuda()

        hidden = self.model(word_id_tensor, char_id_tensor, length_list, info_dic)
        return hidden


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser1 = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='config/config1.cfg')
    argparser1.add_argument('--config_file', default='SRLModel/config')
    args, extra_args = argparser.parse_known_args()
    args1, extra_args1 = argparser1.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    config1 = Configurable(args1.config_file, extra_args1)
    dataloder = Dataloader(config.train_path, config.dev_path, config.test_path, config)
    test_insts = dataloder.test_insts
    word_alphabet = dataloder.word_alphabet
    char_alphabet = dataloder.char_alphabet
    test_batch = buildBatch(test_insts, config, word_alphabet.from_string(padding_key), char_alphabet[padding_key])
    test_batch_iter = test_batch.evalBatchIter()
    srl_model_decoder = Decoder(word_alphabet, config1)

    for i, info_dic in enumerate(test_batch_iter):
        word_id_tensor = info_dic['word_id_tensor']
        length_list = info_dic['length_list']
        char_id_tensor = info_dic['char_id_tensor']
        pre_start_tensor = info_dic['pre_start_tensor']
        arg_start_tensor = info_dic['arg_start_tensor']
        arg_end_tensor = info_dic['arg_end_tensor']
        arg_label_tensor = info_dic['arg_label_tensor']
        srl_len = info_dic['srl_len_tensor']
        sentence_list = info_dic['sentence_list']
        srl_rels_list = info_dic['srl_rels_list']
        # if config.use_cuda:
        #     word_id_tensor = word_id_tensor.cuda()
        #     char_id_tensor = char_id_tensor.cuda()
        #     info_dic['pre_start_tensor'] = info_dic['pre_start_tensor'].cuda()
        #     info_dic['pre_end_tensor'] = info_dic['pre_end_tensor'].cuda()
        #     info_dic['arg_start_tensor'] = info_dic['arg_start_tensor'].cuda()
        #     info_dic['arg_end_tensor'] = info_dic['arg_end_tensor'].cuda()
        #     info_dic['arg_label_tensor'] = info_dic['arg_label_tensor'].cuda()
        #     info_dic['srl_len_tensor'] = info_dic['srl_len_tensor'].cuda()

        srl_hiddens = srl_model_decoder.model(word_id_tensor, char_id_tensor, length_list, info_dic)
        print(srl_hiddens)

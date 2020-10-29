import time
import numpy as np
import torch

from Units.units import find_maxlen, find_maxlennum, find_char_maxlen
from data.buildBatch import buildBatch, padding_key


def create_tra_batch(tra_data, usual_vat_data, tag_vocab, batch_size, config, tokenizer, distinguish_vocab, shuffle=False):
    if shuffle:
        state = np.random.get_state()
        np.random.shuffle(tra_data)
        np.random.set_state(state)
        np.random.shuffle(usual_vat_data)

    unit = []
    vat_unit = []
    instances = []
    vat_instances = []
    for instance, vat_instance in zip(tra_data, usual_vat_data):
        instances.append(instance)
        vat_instances.append(vat_instance)
        if len(instances) == batch_size:
            unit.append(instances)
            instances = []
        if len(vat_instances) == batch_size:
            vat_unit.append(vat_instances)
            vat_instances = []

    if len(instances) > 0:
        unit.append(instances)
    if len(vat_instances) > 0:
        vat_unit.append(instances)

    for batch, vat_batch in zip(unit, vat_unit):
        one_word_data = smp_single_word_data_variable(batch, tag_vocab, tokenizer, config)
        vat_word_data = smp_single_word_data_variable(vat_batch, tag_vocab, tokenizer, config)
        two_word_data = smp_distinguish_word_data_variable(batch, distinguish_vocab, tokenizer, config)
        yield one_word_data, two_word_data, vat_word_data


def create_eval_batch(tra_data, tag_vocab, batch_size, config, tokenizer, shuffle=False):

    if shuffle:
        np.random.shuffle(tra_data)

    unit = []
    instances = []
    for instance in tra_data:
        instances.append(instance)
        if len(instances) == batch_size:
            unit.append(instances)
            instances = []

    if len(instances) > 0:
        unit.append(instances)

    for batch in unit:
        one_word_data = smp_single_word_data_variable(batch, tag_vocab, tokenizer, config)
        yield one_word_data


def smp_single_word_data_variable(batch, tag_vocab, tokenizer, config):
    batch_size = len(batch)
    src_premise_bert_indice = []
    src_premise_segments_id = []
    data_ids = []
    tag_matrix = np.zeros(batch_size)
    for idx, instance in enumerate(batch):
        data_ids.append([instance.sentence_id, instance.topic])
        premise = tokenizer.encode_plus(text=instance.sentence1, add_special_tokens=True, return_tensors='pt')
        premise_bert_indice = premise["input_ids"].squeeze()
        premise_segments_id = premise["token_type_ids"].squeeze()
        premise_list_bert_indice = premise_bert_indice.tolist()
        premise_list_segments_id = premise_segments_id.tolist()

        tag_matrix[idx] = tag_vocab.word2id(instance.gold_label)

        src_premise_bert_indice.append(premise_list_bert_indice)
        src_premise_segments_id.append(premise_list_segments_id)

    premise_find_max_len = sorted(src_premise_bert_indice, key=lambda k: len(k), reverse=True)
    max_premise_bert_len = len(premise_find_max_len[0])

    premise_bert_indices = np.zeros((batch_size, max_premise_bert_len))
    premise_bert_segments = np.ones((batch_size, max_premise_bert_len))

    for idx in range(batch_size):
        for jdx, premise_indice in enumerate(src_premise_bert_indice[idx]):
            premise_bert_indices[idx][jdx] = premise_indice
        for jdx, premise_segments in enumerate(src_premise_segments_id[idx]):
            premise_bert_segments[idx][jdx] = premise_segments

    premise_bert_matrix = torch.from_numpy(premise_bert_indices).long()
    premise_bert_segments = torch.from_numpy(premise_bert_segments).long()
    tag_matrix = torch.from_numpy(tag_matrix).long()
    if config.use_cuda:
        premise_bert_matrix = premise_bert_matrix.cuda()
        premise_bert_segments = premise_bert_segments.cuda()
        tag_matrix = tag_matrix.cuda()
    premise_bert_iuput = (premise_bert_matrix, premise_bert_segments)

    return [premise_bert_iuput, tag_matrix, data_ids]


def smp_distinguish_word_data_variable(batch, distinguish_vocab, tokenizer, config):
    batch_size = len(batch) // 2
    new_batch = [[batch[2*i], batch[2*i+1]] for i in range(batch_size)]
    src_premise_bert_indice = []
    src_premise_segments_id = []
    tag_matrix = np.zeros(batch_size)
    for idx, instance in enumerate(new_batch):
        premise = tokenizer.encode_plus(text=instance[0].sentence1, text_pair=instance[1].sentence1,
                                        add_special_tokens=True, return_tensors='pt', max_length=config.max_length)
        premise_bert_indice = premise["input_ids"].squeeze()
        premise_segments_id = premise["token_type_ids"].squeeze()
        premise_list_bert_indice = premise_bert_indice.tolist()
        premise_list_segments_id = premise_segments_id.tolist()
        premise_bert_tokens = tokenizer.convert_ids_to_tokens(premise_list_bert_indice)
        prmisr_tokens = tokenizer.convert_tokens_to_string(premise_bert_tokens)

        if instance[0].topic == instance[1].topic:
            tag_matrix[idx] = distinguish_vocab.word2id('unique')
        else:
            tag_matrix[idx] = distinguish_vocab.word2id('mix')

        src_premise_bert_indice.append(premise_list_bert_indice)
        src_premise_segments_id.append(premise_list_segments_id)

    premise_find_max_len = sorted(src_premise_bert_indice, key=lambda k: len(k), reverse=True)
    max_premise_bert_len = len(premise_find_max_len[0])

    premise_bert_indices = np.zeros((batch_size, max_premise_bert_len))
    premise_bert_segments = np.ones((batch_size, max_premise_bert_len))

    for idx in range(batch_size):
        for jdx, premise_indice in enumerate(src_premise_bert_indice[idx]):
            premise_bert_indices[idx][jdx] = premise_indice
        for jdx, premise_segments in enumerate(src_premise_segments_id[idx]):
            premise_bert_segments[idx][jdx] = premise_segments

    premise_bert_matrix = torch.from_numpy(premise_bert_indices).long()
    premise_bert_segments = torch.from_numpy(premise_bert_segments).long()
    tag_matrix = torch.from_numpy(tag_matrix).long()
    if config.use_cuda:
        premise_bert_matrix = premise_bert_matrix.cuda()
        premise_bert_segments = premise_bert_segments.cuda()
        tag_matrix = tag_matrix.cuda()
    premise_bert_iuput = (premise_bert_matrix, premise_bert_segments)

    return [premise_bert_iuput, tag_matrix]


def create_batch(tra_data, p_train_insts, h_train_insts, tag_vocab, parser_vocab, word_alphabet, char_alphabet,
                 batch_size, config, srl_config, tokenizer, shuffle=False):

    start_time = time.time()
    batch_word_data = []
    if shuffle:
        state = np.random.get_state()
        np.random.shuffle(tra_data)
        np.random.set_state(state)
        np.random.shuffle(p_train_insts)
        np.random.set_state(state)
        np.random.shuffle(h_train_insts)

    unit = []
    unit_p = []
    unit_h = []
    instances = []
    instances_p = []
    instances_h = []
    for instance, instance1, instance2 in zip(tra_data, p_train_insts, h_train_insts):
        instances.append(instance)
        instances_p.append(instance1)
        instances_h.append(instance2)
        if len(instances) == batch_size:
            unit.append(instances)
            unit_p.append(instances_p)
            unit_h.append(instances_h)
            instances = []
            instances_p = []
            instances_h = []

    if len(instances) > 0:
        unit.append(instances)
    if len(instances_p) > 0:
        unit_p.append(instances_p)
    if len(instances_h) > 0:
        unit_h.append(instances_h)

    for batch, batch_p, batch_h in zip(unit, unit_p, unit_h):
        one_word_data = pair_word_data_variable(batch, tag_vocab, parser_vocab, tokenizer, config)
        p_batch_tensor = buildBatch(batch_p, srl_config, word_alphabet.from_string(padding_key), char_alphabet[padding_key], batch_size)
        h_batch_tensor = buildBatch(batch_h, srl_config, word_alphabet.from_string(padding_key), char_alphabet[padding_key], batch_size)
        p_batch_iter = p_batch_tensor.evalBatchIter()[0]
        h_batch_iter = h_batch_tensor.evalBatchIter()[0]
        batch_word_data.append([one_word_data, p_batch_iter, h_batch_iter])

    print('the create_batch use:{:.2f} S'.format(time.time() - start_time))
    return batch_word_data


def pair_word_data_variable(batch, tag_vocab, tokenizer, config):
    max_premise_len, max_hypothesis_len = find_maxlen(batch)
    batch_size = len(batch)
    premise_length = [len(batch[b].sentence1_list) for b in range(batch_size)]
    hypothesis_length = [len(batch[b].sentence2_list) for b in range(batch_size)]
    src_premise_bert_indice = []
    src_premise_segments_id = []
    src_premise_piece_id = []
    wiki_ids = []
    p_mask = np.zeros((batch_size, max_premise_len))
    h_mask = np.zeros((batch_size, max_hypothesis_len))
    tag_matrix = np.zeros(batch_size)
    for idx, instance in enumerate(batch):
        wiki_ids.append([instance.sentence_ID, instance.number_ID])
        premise = tokenizer.encode_plus(text=instance.sentence1, text_pair=instance.sentence2,
                                        add_special_tokens=True, return_tensors='pt')
        premise_bert_indice = premise["input_ids"].squeeze()
        premise_segments_id = premise["token_type_ids"].squeeze()
        premise_list_bert_indice = premise_bert_indice.tolist()
        premise_list_segments_id = premise_segments_id.tolist()
        premise_bert_tokens = tokenizer.convert_ids_to_tokens(premise_list_bert_indice)
        prmisr_tokens = tokenizer.convert_tokens_to_string(premise_bert_tokens)

        tag_matrix[idx] = tag_vocab.word2id(instance.gold_label)

        premise_piece_id = []
        for idx, bpe_u in enumerate(premise_bert_tokens):
            if bpe_u.startswith("##"):
                tok_len = len(premise_piece_id)
                premise_piece_id[tok_len - 1].append(idx)
            else:
                premise_piece_id.append([idx])

        src_premise_bert_indice.append(premise_list_bert_indice)
        src_premise_segments_id.append(premise_list_segments_id)
        src_premise_piece_id.append(premise_piece_id)

    premise_find_max_len = sorted(src_premise_bert_indice, key=lambda k: len(k), reverse=True)
    max_premise_bert_len = len(premise_find_max_len[0])

    premise_bert_indices = np.zeros((batch_size, max_premise_bert_len))
    premise_bert_segments = np.ones((batch_size, max_premise_bert_len))
    premise_bert_pieces = np.zeros((batch_size, max_premise_len + max_hypothesis_len, max_premise_bert_len))
    premise_parser_pieces = np.zeros((batch_size, max_premise_bert_len, max_premise_len + max_hypothesis_len))

    shift_pos = 1
    for idx in range(batch_size):
        for jdx, premise_indice in enumerate(src_premise_bert_indice[idx]):
            premise_bert_indices[idx][jdx] = premise_indice
        for jdx, premise_segments in enumerate(src_premise_segments_id[idx]):
            premise_bert_segments[idx][jdx] = premise_segments
        for jdx in range(premise_length[idx]):
            p_mask[idx][jdx] = 1
        for jdx in range(hypothesis_length[idx]):
            h_mask[idx][jdx] = 1

        for word_len in range(premise_length[idx] + hypothesis_length[idx]):
            avg_score = 1.0 / len(src_premise_piece_id[idx][word_len + shift_pos])
            for jdx, tindex in enumerate(src_premise_piece_id[idx][word_len + shift_pos], 1):
                if jdx == premise_length[idx] + 1:
                    pass
                else:
                    premise_bert_pieces[idx, word_len, tindex] = avg_score

        for word_len in range(premise_length[idx] + hypothesis_length[idx]):
            avg_score = 1.0
            for jdx, tindex in enumerate(src_premise_piece_id[idx][word_len]):
                premise_parser_pieces[idx, tindex, word_len] = avg_score

    premise_bert_matrix = torch.from_numpy(premise_bert_indices).long()
    premise_bert_segments = torch.from_numpy(premise_bert_segments).long()
    premise_bert_pieces = torch.from_numpy(premise_bert_pieces).float()
    premise_parser_pieces = torch.from_numpy(premise_parser_pieces).float()
    p_mask = torch.from_numpy(p_mask).float()
    h_mask = torch.from_numpy(h_mask).float()
    tag_matrix = torch.from_numpy(tag_matrix).long()
    if config.use_cuda:
        premise_bert_matrix = premise_bert_matrix.cuda()
        premise_bert_segments = premise_bert_segments.cuda()
        premise_bert_pieces = premise_bert_pieces.cuda()
        premise_parser_pieces = premise_parser_pieces.cuda()
        p_mask = p_mask.cuda()
        h_mask = h_mask.cuda()
        tag_matrix = tag_matrix.cuda()
    premise_bert_iuput = (premise_bert_matrix, premise_bert_segments, premise_bert_pieces, premise_parser_pieces)

    return [premise_bert_iuput, p_mask, h_mask, tag_matrix, wiki_ids]


def pair_char_data_variable(batch, tra_char_vocab, config):
    batch_size = len(batch)
    max_premise_len, max_premise_char_len, max_hypothesis_len, max_hypothesis_char_len = find_char_maxlen(batch)
    src_premise_matrix = np.zeros((batch_size * 2, max_premise_len, max_premise_char_len))
    src_hypothesis_matrix = np.zeros((batch_size * 2, max_hypothesis_len, max_hypothesis_char_len))
    for idx, instance in enumerate(batch):
        for kdx, sentence in enumerate(instance[0:3]):
            if kdx == 0:
                for jdx, word in enumerate(sentence):
                    for gdx, char in enumerate(word):
                        src_premise_matrix[idx * 2][jdx][gdx] = tra_char_vocab.word2id(char)
                        src_premise_matrix[idx * 2 + 1][jdx][gdx] = tra_char_vocab.word2id(char)
            elif kdx == 1:
                for jdx, word in enumerate(sentence):
                    for gdx, char in enumerate(word):
                        src_hypothesis_matrix[idx * 2][jdx][gdx] = tra_char_vocab.word2id(char)
            else:
                for jdx, word in enumerate(sentence):
                    for gdx, char in enumerate(word):
                        src_hypothesis_matrix[idx * 2 + 1][jdx][gdx] = tra_char_vocab.word2id(char)
    src_premise_matrix = torch.from_numpy(src_premise_matrix).long()
    src_hypothesis_matrix = torch.from_numpy(src_hypothesis_matrix).long()
    if config.use_cuda:
        src_premise_matrix = src_premise_matrix.cuda()
        src_hypothesis_matrix = src_hypothesis_matrix.cuda()
    return [src_premise_matrix, src_hypothesis_matrix]


def create_sen_batch(tra_data, tra_fact_vocab, config, shuffle=True):
    print('create_batch is start')
    start_time = time.time()
    batch_data = []
    if shuffle:
        np.random.shuffle(tra_data)

    unit = []
    instances = []
    for instance in tra_data:
        instances.append(instance)
        if len(instances) == config.batch_size:
            unit.append(instances)
            instances = []

    if len(instances) > 0:
        unit.append(instances)

    for batch in unit:
        one_data = pair_sen_data_variable(batch, tra_fact_vocab, config)
        batch_data.append(one_data)

    print('the create_batch use:{:.2f} S'.format(time.time() - start_time))
    return batch_data


def pair_sen_data_variable(batch, tra_fact_vocab, config):
    length = []
    max_data_length, max_sen_num = find_maxlennum(batch)
    batch_size = len(batch)
    src_matrix = np.zeros((batch_size, 3, max_sen_num, max_data_length))
    tag_matrix = np.ones(batch_size)
    for idx, instance in enumerate(batch):
        for kdx, sentences in enumerate(instance[0:3]):
            for wdx, sentence in enumerate(sentences):
                length.append(len(sentence))
                sentence = tra_fact_vocab.word2id(sentence)
                for jdx, value in enumerate(sentence):
                    src_matrix[idx][kdx][wdx][jdx] = value
                if wdx + 1 == len(sentences):
                    for i, k in enumerate(range(len(sentences), max_sen_num), 1):
                        src_matrix[idx][kdx][wdx+i][0] = 1
                        length.append(1)
    src_matrix = torch.from_numpy(src_matrix).long()
    tag_matrix = torch.from_numpy(tag_matrix).float()
    length = torch.tensor(length)
    if config.use_cuda:
        src_matrix = src_matrix.cuda()
        tag_matrix = tag_matrix.cuda()
        length = length.cuda()
    return [src_matrix, length, tag_matrix]

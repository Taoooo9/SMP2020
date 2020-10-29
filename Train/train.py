import json
import numpy as np
import os
import time
import torch.nn as nn

from collections import Counter
from torch import optim
from transformers import AdamW, get_linear_schedule_with_warmup
from DataProcessing.data_batchiter import create_tra_batch, create_eval_batch
from Model.loss import *
from Model.vat import VATLoss


def train(bert_model, distinguish_model, usual_tra_data_set, virus_tra_data_set, usual_dev_data_set, virus_dev_data_set, tag_vocab, config,
          distinguish_vocab, tokenizer, usual_eval_data, virus_eval_data, usual_vat_data, virus_vat_data):

    usual_tra_data_set.extend(virus_tra_data_set)
    usual_vat_data.extend(virus_vat_data)

    batch_num = int(np.ceil(len(usual_tra_data_set) / float(config.tra_batch_size)))

    no_decay = ['bias', 'LayerNorm.weight']
    bert_optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in bert_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer_bert = AdamW(bert_optimizer_grouped_parameters, lr=config.bert_lr, eps=config.epsilon)
    scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, num_warmup_steps=0, num_training_steps=config.epoch * batch_num)

    dis_parameters = filter(lambda p: p.requires_grad, distinguish_model.parameters())
    dis_optimizer = optim.Adam(dis_parameters, lr=config.dis_lr, weight_decay=config.weight_decay)
    decay, decay_step = config.decay, config.decay_steps
    rate = lambda epochss: decay ** (epochss // decay_step)
    dis_scheduler = torch.optim.lr_scheduler.LambdaLR(dis_optimizer, lr_lambda=rate)

    # Get start!
    global_step = 0

    best_tra_f1 = 0
    best_cla_acc = 0
    best_dis_acc = 0
    best_dev_usual = 0
    best_dev_virus = 0

    epoch_accuracy = 0

    critierion = LabelSmoothing(config)

    vat_loss = VATLoss(config)

    for epoch in range(0, config.epoch):
        gold_label = []
        predict_ids = []
        cla_score = 0
        dis_score = 0
        print('\nThe epoch is starting.')
        epoch_start_time = time.time()
        batch_iter = 0
        print('The epoch is :', str(epoch))
        for all_batch in create_tra_batch(usual_tra_data_set, usual_vat_data, tag_vocab, config.tra_batch_size, config, tokenizer,
                                          distinguish_vocab, shuffle=True):
            start_time = time.time()
            word_batch = all_batch[0]
            distinguish_batch = all_batch[1]
            vat_word_batch = all_batch[2]
            bert_model.train()
            distinguish_model.train()

            batch_size = word_batch[0][0].size(0)
            input_tensor = word_batch[0]
            target = word_batch[1]
            gold_label.extend(target)

            distinguish_input = distinguish_batch[0]
            distinguish_target = distinguish_batch[1]

            ul_input_tensor = vat_word_batch[0]
            lds = vat_loss(bert_model, ul_input_tensor)

            logits, last_distinguish = bert_model(input_tensor, distinguish_input=distinguish_input)
            dis_logits = distinguish_model(last_distinguish)
            _, correct, predict_id, accuracy = class_loss(logits, target)
            cla_loss = critierion(logits, target)
            dis_loss, dis_correct, dis_accuracy = distinguish_loss(dis_logits, distinguish_target)
            predict_ids.extend(predict_id)
            loss = (cla_loss + dis_loss + config.vat_alpha * lds) / config.update_every
            loss.backward()
            cla_loss_value = cla_loss.item()
            dis_loss_value = dis_loss.item()
            vat_loss_value = (config.vat_alpha * lds).item()
            during_time = float(time.time() - start_time)
            print('Step:{}, Epoch:{}, batch_iter:{}, cla_accuracy:{:.4f}({}/{}), dis_accuracy:{:.4f}({}/{}),'
                  'time:{:.2f}, cla_loss:{:.6f}, cla_loss:{:.6f}, vat_loss:{:.6f}'.format(global_step, epoch, batch_iter, accuracy,
                                                                         correct, batch_size, dis_accuracy, dis_correct,
                                                                         batch_size / 2, during_time, cla_loss_value,
                                                                         dis_loss_value, vat_loss_value))
            batch_iter += 1

            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                if config.clip_max_norm_use:
                    nn.utils.clip_grad_norm_(bert_model.parameters(), max_norm=config.clip)
                    nn.utils.clip_grad_norm_(distinguish_model.parameters(), max_norm=config.clip)
                scheduler_bert.step()
                dis_scheduler.step(epoch_accuracy)
                optimizer_bert.step()
                dis_optimizer.step()
                bert_model.zero_grad()
                dis_optimizer.zero_grad()
                global_step += 1
            cla_score += correct
            dis_score += dis_correct

            if batch_iter % config.test_interval == 0 or batch_iter == batch_num:
                print("now bert lr is {}".format(optimizer_bert.param_groups[0].get("lr")), '\n')
                dev_usual_score = evaluate(bert_model, usual_dev_data_set, config, tag_vocab, tokenizer)
                if best_dev_usual < dev_usual_score:
                    print('the best usual_dev score is: acc:{}'.format(dev_usual_score))
                    best_dev_usual = dev_usual_score
                    decoder(bert_model, usual_eval_data, config, tag_vocab, tokenizer)
                    if os.path.exists(config.save_model_path):
                        torch.save(bert_model.state_dict(), config.usual_model_pkl)
                    else:
                        os.makedirs(config.save_model_path)
                        torch.save(bert_model.state_dict(), config.usual_model_pkl)

                dev_virus_score = evaluate(bert_model, virus_dev_data_set, config, tag_vocab, tokenizer, test=True)
                if best_dev_virus < dev_virus_score:
                    print('the best virus_dev score is: acc:{}'.format(dev_virus_score) + '\n')
                    best_dev_virus = dev_virus_score
                    decoder(bert_model, virus_eval_data, config, tag_vocab, tokenizer, test=True)
                    if os.path.exists(config.save_model_path):
                        torch.save(bert_model.state_dict(), config.virus_model_pkl)
                    else:
                        os.makedirs(config.save_model_path)
                        torch.save(bert_model.state_dict(), config.virus_model_pkl)
        epoch_time = float(time.time() - epoch_start_time)
        tra_score = get_Macro_F1_score(gold_label, predict_ids, tag_vocab)
        all_cla_score = 100.0 * cla_score / len(usual_tra_data_set)
        all_dis_score = 100.0 * dis_score / (len(usual_tra_data_set) / 2)
        if tra_score > best_tra_f1:
            best_tra_f1 = tra_score
            print('the best_train F1 is:{:.2f}'.format(best_tra_f1))
        if all_cla_score > best_cla_acc:
            best_cla_acc = all_cla_score
            print('the best_train cla_score is:{}({}/{})'.format(best_cla_acc, cla_score, len(usual_tra_data_set)))
        if all_dis_score > best_dis_acc:
            best_dis_acc = all_dis_score
            print('the best_train dis_score is:{}({}/{})'.format(best_dis_acc, dis_score, len(usual_tra_data_set)/2))
        print("epoch_time is:", epoch_time)


def evaluate(bert_model, dev_data, config, tag_vocab, tokenizer, test=False):
    bert_model.eval()
    get_score = 0
    start_time = time.time()
    gold_label = []
    predict_ids = []
    for word_batch in create_eval_batch(dev_data, tag_vocab, config.test_batch_size, config, tokenizer):
        batch_size = word_batch[0][0].size(0)
        input_tensor = word_batch[0]
        target = word_batch[1]
        gold_label.extend(target)
        logits, _ = bert_model(input_tensor)
        loss, correct, predict_id, accuracy = class_loss(logits, target)
        predict_ids.extend(predict_id)
        get_score += correct
    if test:
        dev_score = get_Macro_F1_score(gold_label, predict_ids, tag_vocab)
        print('the current_test virus_score is: F1:{:.2f}'.format(dev_score))
    else:
        dev_score = get_Macro_F1_score(gold_label, predict_ids, tag_vocab)
        print('the current_dev usual_score is: F1:{:.2f}'.format(dev_score))
    during_time = float(time.time() - start_time)
    print('spent time is:{:.4f}'.format(during_time))
    return dev_score


def get_F1_score(gold_num, pred_num, correct_num):
    p = float(correct_num) / pred_num if pred_num != 0 else 0
    r = float(correct_num) / gold_num if gold_num != 0 else 0
    f1 = 200.0 * correct_num / (gold_num + pred_num) if gold_num + pred_num != 0 else 0.
    return p, r, f1


def get_Macro_F1_score(gold_label, predict_ids, tag_vocab):

    total_f1 = 0

    gold_counter = Counter()
    pre_counter = Counter()
    correct_counter = Counter()

    if len(gold_label) != len(predict_ids):
        print('Error!!!!!!!!!')

    for gold, pre in zip(gold_label, predict_ids):
        gold_counter[int(gold)] += 1
        pre_counter[int(pre)] += 1
        if gold == pre:
            correct_counter[tag_vocab.id2word(gold)] += 1

    gold_dict = dict(gold_counter)
    pre_dict = dict(pre_counter)
    correct_dict = dict(correct_counter)

    for key in gold_dict.keys():
        if tag_vocab.id2word(key) not in correct_dict.keys():
            correct_dict[tag_vocab.id2word(key)] = 0
        if key not in pre_dict.keys():
            pre_dict[key] = 0

    print('--------------------evaluate------------------------')
    for key in gold_dict.keys():
        p, r, f1 = get_F1_score(gold_dict[key], pre_dict[key], correct_dict[tag_vocab.id2word(key)])
        print('{}:\t\tP:{:.2f}%\tR:{:.2f}%\tF1:{:.2f}%'.format(tag_vocab.id2word(key), 100 * p, 100 * r, f1))
        total_f1 += f1
    macro_f1 = total_f1 / len(gold_dict)
    return macro_f1


def decoder(bert_model, eval_data, config, tag_vocab, tokenizer, test=False):
    bert_model.eval()
    predict_ids = []
    data_ids = []
    for word_batch in create_eval_batch(eval_data, tag_vocab, config.test_batch_size, config, tokenizer):
        batch_size = word_batch[0][0].size(0)
        input_tensor = word_batch[0]
        target = word_batch[1]
        data_id = word_batch[2]
        data_ids.extend(data_id)
        logits, _ = bert_model(input_tensor)
        predict_id = smp_eval(logits)
        predict_ids.extend(predict_id)
    if test:
        path = config.save_virus_path
    else:
        path = config.save_usual_path

    json_list = []
    for index, predict_id in zip(data_ids, predict_ids):
        submit_dic = {}
        submit_dic["id"] = index[0]
        submit_dic["label"] = tag_vocab.id2word(predict_id)
        json_list.append(submit_dic)
    json_list = sorted(json_list, key=lambda d: d['id'])
    json_str = json.dumps(json_list)
    with open(path, 'w', encoding='utf8') as f:
        f.write(json_str)
    print('Write over.')


class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()








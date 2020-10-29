import torch
import numpy as np
import time
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from driver.SRLModel import SRLModel
from driver.ORLModel import SRLModel as ORLModel
from driver.orl_lsgn_evaluator import LSGNEvaluator
from torch.nn.utils import clip_grad_norm_
from data.Common import seed
from driver.cail_eval import Eval, F1_measure, getFscore_Avg
from driver.decode import Decoder

np.random.seed(seed)
torch.manual_seed(seed)

best_dev_F1 = 0
best_dev_test_F1 = 0
best_epoch = 0

# train_eval_micro = Eval()
# dev_eval_micro = Eval()
# test_eval_micro = Eval()
eval_micro = Eval()
eval_macro = []
# train_eval_macro = []
# dev_eval_macro = []
# test_eval_macro = []


def list2D21D(l):
    new = []
    for elem in l:
        new.extend(elem)
    return new

def get_time():
    # tm_year=2018, tm_mon=10, tm_mday=28, tm_hour=10, tm_min=32, tm_sec=14, tm_wday=6, tm_yday=301, tm_isdst=0
    cur_time = time.localtime(time.time())
    dic = dict()
    dic['year'] = cur_time.tm_year
    dic['month'] = cur_time.tm_mon
    dic['day'] = cur_time.tm_mday
    dic['hour'] = cur_time.tm_hour
    dic['min'] = cur_time.tm_min
    dic['sec'] = cur_time.tm_sec
    time_str = '{}.{:0>2d}.{:0>2d} {:0>2d}:{:0>2d}:{:0>2d}'.format(dic['year'], dic['month'], dic['day'], dic['hour'], dic['min'], dic['sec'])
    return time_str


def adjust_learning_rate(optim, lr_decay_rate):
    lr = None
    for param_group in optim.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay_rate
        lr = param_group['lr']
    return lr


def trainer(iter_dic, alphabet_dic, config, args):
    if args.use_cuda:
        torch.cuda.manual_seed(666)
        print('cuda state:', torch.cuda.is_available())
        torch.backends.cudnn.deterministic = True

    global best_dev_F1
    global best_dev_test_F1
    global best_epoch

    max_dev_f1 = -1
    max_test_f1 = -1
    train_batch = iter_dic['train']
    dev_batch = iter_dic['dev']
    test_batch = iter_dic['test']

    torch.save(alphabet_dic, config.save_vocab_path)
    print('save [word_alphabet, char_alphabet, srl_labels_dic] in ' + config.save_vocab_path + 'successfully !')

    word_alphabet = alphabet_dic['word_alphabet']
    char_alphabet = alphabet_dic['char_alphabet']
    srl_labels_dic = alphabet_dic['srl_labels_dic']
    # for i in range(_alphabet.m_size):
    #     eval_macro.append(Eval())
    # for i in range(accu_alphabet.m_size):
    #     train_eval_macro.append(Eval())
    # for i in range(accu_alphabet.m_size):
    #     dev_eval_macro.append(Eval())
    # for i in range(accu_alphabet.m_size):
    #     test_eval_macro.append(Eval())


    print('init ORL model')
    model = ORLModel(alphabet_dic, config, args.use_cuda)
    print('init SRL model')
    srl_model_decoder = Decoder(args.config_srl_path, [], word_alphabet, args.use_cuda)

    evaluator = LSGNEvaluator(config)
    print('init optimizer')
    Optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    srl_model_decoder.model.eval()

    if args.use_cuda:
        print('###use cuda!')
        model = model.cuda()
        srl_model_decoder.model.cuda()
    print('start training...')

    train_batchs = train_batch.batchs

    batchs_num = 1
    step = 1
    fine_tune_flag = False
    for epoch_i in range(config.epoch):
        model.train()
        if not fine_tune_flag and config.fine_tune_from_epoch <= epoch_i:
            print('fine_tune is open !')
            srl_model_decoder.model.train()
            fine_tune_flag = True
        total_loss = 0
        train_batch_iter = train_batch.batchIter()
        for info_dic in train_batch_iter:
            time_start = time.time()
            word_id_tensor = info_dic['word_id_tensor']
            length_list = info_dic['length_list']
            char_id_tensor = info_dic['char_id_tensor']
            pre_start_tensor = info_dic['pre_start_tensor']
            pre_end_tensor = info_dic['pre_end_tensor']
            arg_start_tensor = info_dic['arg_start_tensor']
            arg_end_tensor = info_dic['arg_end_tensor']
            arg_label_tensor = info_dic['arg_label_tensor']
            srl_len = info_dic['srl_len_tensor']

            if args.use_cuda:
                word_id_tensor = word_id_tensor.cuda()
                char_id_tensor = char_id_tensor.cuda()
                info_dic['pre_start_tensor'] = info_dic['pre_start_tensor'].cuda()
                info_dic['pre_end_tensor'] = info_dic['pre_end_tensor'].cuda()
                info_dic['arg_start_tensor'] = info_dic['arg_start_tensor'].cuda()
                info_dic['arg_end_tensor'] = info_dic['arg_end_tensor'].cuda()
                info_dic['arg_label_tensor'] = info_dic['arg_label_tensor'].cuda()
                info_dic['srl_len_tensor'] = info_dic['srl_len_tensor'].cuda()

            srl_hiddens = srl_model_decoder.model(word_id_tensor, char_id_tensor, length_list, info_dic)
            _, loss = model(word_id_tensor, char_id_tensor, length_list, info_dic, srl_hiddens)

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=config.max_gradient_norm)
            Optimizer.step()
            total_loss += loss.item()
            Optimizer.zero_grad()
            time_end = time.time()
            time_last = time_end - time_start
            if step % config.decay_frequency == 0:
                lr = adjust_learning_rate(Optimizer, config.decay_rate)
                print('### adjust lr to ', lr)

            if step % config.report_frequency == 0:
                time_str = get_time()
                print('[{}] Step {} Epoch_{} loss={:.5f} last:{:.1f}s'.format(time_str, step, epoch_i, loss.item(), time_last))

            if step % config.eval_frequency == 0:
                dev_batch_iter = dev_batch.evalBatchIter()
                test_batch_iter = test_batch.evalBatchIter()
                # model = model.cpu()
                # flag = False
                # model.use_cuda = flag
                # dev
                time_start = time.time()
                dev_gold_num, dev_predict_num, dev_correct_num = evaluator.evalu
                ate(model, srl_model_decoder, dev_batch_iter, use_cuda=args.use_cuda)
                time_end = time.time()
                time_last = time_end - time_start
                dev_score = 200.0 * dev_correct_num / (dev_gold_num + dev_predict_num) if dev_correct_num > 0 else 0.0
                time_str = get_time()
                print("[%s] Dev: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f last:%.1fs" % \
                      (time_str, dev_correct_num, dev_gold_num, 100.0 * dev_correct_num / dev_gold_num if dev_correct_num > 0 else 0.0, \
                       dev_correct_num, dev_predict_num, 100.0 * dev_correct_num / dev_predict_num if dev_correct_num > 0 else 0.0, \
                       dev_score, time_last))
                # test
                time_start = time.time()
                test_gold_num, test_predict_num, test_correct_num = evaluator.evaluate(model, srl_model_decoder, test_batch_iter, use_cuda=args.use_cuda)
                time_end = time.time()
                time_last = time_end - time_start
                test_score = 200.0 * test_correct_num / (test_gold_num + test_predict_num) if test_correct_num > 0 else 0.0
                time_str = get_time()
                print("[%s] Test:Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f last:%.1fs" % \
                      (time_str, test_correct_num, test_gold_num, 100.0 * test_correct_num / test_gold_num if test_correct_num > 0 else 0.0, \
                       test_correct_num, test_predict_num, 100.0 * test_correct_num / test_predict_num if test_correct_num > 0 else 0.0, \
                       test_score, time_last))
                model.train()
                # if args.use_cuda:
                #     model = model.cuda()
                #     model.use_cuda = True
                if dev_score > max_dev_f1:
                    max_dev_f1 = dev_score
                    max_test_f1 = test_score
                    if config.save_after >= 0 and epoch_i >= config.save_after:
                        torch.save(model.state_dict(), config.save_model_path)
                        print('save model in {} successfully !'.format(config.save_model_path))
                print("Current max Test Exact F1: {:.2f}".format(max_test_f1))

            step += 1
        print('###mean loss:', total_loss)


def evaluate(model, eval_batch, args):
    batch_iter = eval_batch.evalBatchIter()
    # model = model.cpu()
    flag = True
    model.use_cuda = flag
    # dev
    # dev_gold_num, dev_predict_num, dev_correct_num = evaluator.evaluate(model, dev_batch_iter, use_cuda=flag)


def decode(model, sess, batch_insts, config):
    model.dropout = 0
    print('start decode...')
    batch_iter = batch_insts.batchIter()
    with open(config.decode_path, 'w', encoding='utf8') as f:
        for batch_inst_tensor, length_list, aspect, e_exp, gold, a_lengths, e_lengths, word_list, pure_label_list, infos in batch_iter:
            batch_size = len(batch_inst_tensor)
            logits = sess.run(model.logits,
                             feed_dict={
                                 model.batch_size: batch_size,
                                 model.word_ids: batch_inst_tensor,
                                 model.a_index: aspect,
                                 model.e_index: e_exp,
                                 model.a_lengths: a_lengths,
                                 model.e_lengths: e_lengths,
                                 model.labels: gold,
                                 model.is_training: False
                             })
            predicts = np.argmax(logits, axis=1).tolist()
            # print(predicts)
            pred_inst_list = []
            start = 0
            for a, e in zip(a_lengths, e_lengths):
                end = start + a*e
                pred_inst_list.append(predicts[start:end])
                start = end
            have_rel_tuple = [[] for _ in range(len(e_exp))]
            idx = 0
            for a_inst_list, e_inst_list in zip(aspect, e_exp):
                index = 0
                for i in range(a_lengths[idx]):
                    for j in range(e_lengths[idx]):
                        if pred_inst_list[idx][index] == 1:
                            have_rel_tuple[idx].append((list(set(a_inst_list[i])), list(set(e_inst_list[j]))))
                        index += 1
                idx += 1
            # print(have_rel_tuple)
            for idx, sub_label_list in enumerate(pure_label_list):
                for pair in have_rel_tuple[idx]:
                    if pair == []:
                        break
                    a, e = pair # a:[1,2] e:[6,7,8]
                    a_tag = sub_label_list[a[0]].split('-')[-1]
                    for ide in e:
                        sub_label_list[ide] += ('-' + str(a_tag))

            # save file
            for idx in range(len(pure_label_list)):
                f.write(infos[idx][0] + '\n')
                f.write(infos[idx][1] + '\n')
                f.write(infos[idx][2] + '\n')
                f.write(infos[idx][3] + '\n')
                for word, label in zip(word_list[idx], pure_label_list[idx]):
                    f.write(word + '\t' + label + '\n')
                f.write('\n')



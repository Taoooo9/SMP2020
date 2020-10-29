#!/usr/bin/env python
# encoding: utf-8

import datetime
import time
import numpy as np
import operator
from collections import Counter
import os
import codecs
import subprocess


_CORE_ARGS = {"ARG0": 1, "ARG1": 2, "ARG2": 4, "ARG3": 8, "ARG4": 16, "ARG5": 32, "ARGA": 64,
               "A0": 1, "A1": 2, "A2": 4, "A3": 8, "A4": 16, "A5": 32, "AA": 64}
_SRL_CONLL_EVAL_SCRIPT = "scripts/run_conll_eval.sh"

def maybe_divide(x, y):
  return 0 if y == 0 else x / float(y)

def pad_batch_tensors(tensor_dicts, tensor_name):
    """
    Args:
      tensor_dicts: List of dictionary tensor_name: numpy array of length B.
      tensor_name: String name of tensor.

    Returns:
      Numpy array of (B, ?)
    """
    batch_size = len(tensor_dicts)
    tensors = [np.expand_dims(td[tensor_name], 0) for td in tensor_dicts]
    shapes = [t.shape for t in tensors]
    # Take max shape along each dimension.
    max_shape = np.max(zip(*shapes), axis=1)
    # print tensor_name, batch_size, tensors[0].shape, max_shape
    zeros = np.zeros_like(max_shape)
    padded_tensors = [np.pad(t, zip(zeros, max_shape - t.shape), "constant") for t in tensors]
    return np.concatenate(padded_tensors, axis=0)

def _dp_decode_non_overlapping_spans(starts, ends, scores, max_len, labels_inv, pred_id, u_constraint=False):
  num_roles = scores.shape[1]
  labels = np.argmax(scores, axis=1)
  spans = zip(starts, ends, range(len(starts)))
  spans = sorted(spans, key=lambda x: (x[0], x[1]))

  if u_constraint:
    f = np.zeros([max_len + 1, 128], dtype=float) - 0.1
  else:
    f = np.zeros([max_len + 1, 1], dtype=float) - 0.1
  f[0, 0] = 0
  states = { 0: set([0]) }  # A dictionary from id to list of binary core-arg states.
  pointers = {}  # A dictionary from states to (arg_id, role, prev_t, prev_rs)
  best_state = [(0, 0)]

  def _update_state(t0, rs0, t1, rs1, delta, arg_id, role):
    if f[t0][rs0] + delta > f[t1][rs1]:
      f[t1][rs1] = f[t0][rs0] + delta
      if t1 not in states:
        states[t1] = set()
      states[t1].update([rs1])
      pointers[(t1, rs1)] = (arg_id, role, t0, rs0)
      if f[t1][rs1] > f[best_state[0][0]][best_state[0][1]]:
        best_state[0] = (t1, rs1)

  for start, end, i in spans:
    assert scores[i][0] == 0
    # The extra dummy score should be same for all states, so we can safely skip arguments overlap
    # with the predicate.
    if pred_id is not None and start <= pred_id and pred_id <= end:
      continue
    r0 = labels[i]  # Locally best role assignment.
    # Strictly better to incorporate a dummy span if it has the highest local score.
    if r0 == 0:
      continue
    r0_str = labels_inv[int(r0)]
    # Enumerate explored states.
    t_states = [t for t in states.keys() if t <= start]
    for t in t_states:
      role_states = states[t]
      # Update states if best role is not a core arg.
      if not u_constraint or not r0_str in _CORE_ARGS:
        for rs in role_states:
          _update_state(t, rs, end+1, rs, scores[i][r0], i, r0)
      else:
        for rs in role_states:
          for r in range(1, num_roles):
            if scores[i][r] > 0:
              r_str = labels_inv[r]
              core_state = _CORE_ARGS.get(r_str, 0)
              #print start, end, i, r_str, core_state, rs
              if core_state & rs == 0:
                _update_state(t, rs, end+1, rs|core_state, scores[i][r], i, r)
  # Backtrack to decode.
  new_spans = []
  t, rs = best_state[0]
  while (t, rs) in pointers:
    i, r, t0, rs0 = pointers[(t, rs)]
    new_spans.append((starts[i], ends[i], labels_inv[r]))
    t = t0
    rs = rs0

  return new_spans[::-1]

def srl_decode(length_list, predict_dict, srl_labels_inv, config):
    predictions = {}

    # Decode sentence-level tasks.
    num_sentences = len(length_list)
    predictions["srl"] = [{} for i in range(num_sentences)]

    # Sentence-level predictions.
    for i in range(num_sentences):
        if "srl" in predictions:
            num_args = int(predict_dict["num_args"][i])
            num_preds = int(predict_dict["num_preds"][i])
            for j, pred_id in enumerate(predict_dict["predicates"][i][:num_preds]):
                # arg_spans = _decode_non_overlapping_spans(
                arg_spans = _dp_decode_non_overlapping_spans(
                    predict_dict["arg_starts"][i][:num_args],
                    predict_dict["arg_ends"][i][:num_args],
                    predict_dict["srl_scores"][i, :num_args, j, :],
                    # len(sentences[i]), srl_labels_inv, pred_id, config.enforce_srl_constraint)
                    length_list[i], srl_labels_inv, pred_id, config.enforce_srl_constraint)
                # To avoid warnings in the eval script.
                if config.use_gold_predicates:
                    arg_spans.append((pred_id, pred_id, "V"))
                if arg_spans:
                    predictions["srl"][i][pred_id] = sorted(arg_spans, key=lambda x: (x[0], x[1]))

    return predictions

def evaluate_retrieval(span_starts, span_ends, span_scores, pred_starts, pred_ends, gold_spans,
                       text_length, evaluators, debugging=False):
  """
  Evaluation for unlabeled retrieval.

  Args:
    gold_spans: Set of tuples of (start, end).
  """
  if len(span_starts) > 0:
    sorted_starts, sorted_ends, sorted_scores = zip(*sorted(
        zip(span_starts, span_ends, span_scores),
        key=operator.itemgetter(2), reverse=True))
  else:
    sorted_starts = []
    sorted_ends = []
  for k, evaluator in evaluators.items():
    if k == -3:
      predicted_spans = set(zip(span_starts, span_ends)) & gold_spans
    else:
      if k == -2:
        predicted_starts = pred_starts
        predicted_ends = pred_ends
        if debugging:
          print("Predicted", zip(sorted_starts, sorted_ends, sorted_scores)[:len(gold_spans)])
          print("Gold", gold_spans)
     # FIXME: scalar index error
      elif k == 0:
        is_predicted = span_scores > 0
        predicted_starts = span_starts[is_predicted]
        predicted_ends = span_ends[is_predicted]
      else:
        if k == -1:
          num_predictions = len(gold_spans)
        else:
          num_predictions = int((k * text_length) / 100)
        predicted_starts = sorted_starts[:num_predictions]
        predicted_ends = sorted_ends[:num_predictions]
      predicted_spans = set(zip(predicted_starts, predicted_ends))
    evaluator.update(gold_set=gold_spans, predicted_set=predicted_spans)

def srl_constraint_tracker(pred_to_args):
  unique_core_role_violations = 0
  continuation_role_violations = 0
  reference_role_violations = 0
  for pred_ids, args in pred_to_args.items():
    # Sort by span start, assuming they are not overlapping.
    sorted_args = sorted(args, key=lambda x: x[0], reverse=True)
    core_args = set()
    base_args = set()
    for start, end, role in sorted_args:
      if role in _CORE_ARGS:
        if role in core_args:
          unique_core_role_violations += 1
        core_args.update([role])
      elif role.startswith("C-") and not role[2:] in base_args:
        continuation_role_violations += 1
      if not role.startswith("C-") and not role.startswith("R-"):
        base_args.update(role)
    for start, end, role in sorted_args:
      if role.startswith("R-") and not role[2:] in base_args:
        reference_role_violations += 1
  return unique_core_role_violations, continuation_role_violations, reference_role_violations

def _print_f1(total_gold, total_predicted, total_matched, message=""):
  precision = 100.0 * total_matched / total_predicted if total_predicted > 0 else 0
  recall = 100.0 * total_matched / total_gold if total_gold > 0 else 0
  f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
  print ("{}: Precision: {}, Recall: {}, F1: {}".format(message, precision, recall, f1))
  return precision, recall, f1

def print_sentence_to_conll(fout, tokens, labels):
  """Print a labeled sentence into CoNLL format.
  """
  for label_column in labels:
    assert len(label_column) == len(tokens)
  for i in range(len(tokens)):
    fout.write(tokens[i].ljust(15))
    for label_column in labels:
      fout.write(label_column[i].rjust(15))
    fout.write("\n")
  fout.write("\n")

def print_to_conll(sentences, srl_labels, output_filename, gold_predicates):
  fout = codecs.open(output_filename, "w", "utf8")
  for sent_id, words in enumerate(sentences):
    if gold_predicates:
      assert len(gold_predicates[sent_id]) == len(words)
    pred_to_args = srl_labels[sent_id]
    props = ["-" for _ in words]
    col_labels = [["*" for _ in words] for _ in range(len(pred_to_args))]
    for i, pred_id in enumerate(sorted(pred_to_args.keys())):
      if type(pred_id) is not int:
          pred_id = int(pred_id)
      # To make sure CoNLL-eval script count matching predicates as correct.
      if gold_predicates and gold_predicates[sent_id][pred_id] != "-":
        props[pred_id] = gold_predicates[sent_id][pred_id]
      else:
        props[pred_id] = "P" + words[pred_id]
      flags = [False for _ in words]
      for start, end, label in pred_to_args[pred_id]:
        if not max(flags[start:end+1]):
          col_labels[i][start] = "(" + label + col_labels[i][start]
          col_labels[i][end] = col_labels[i][end] + ")"
          for j in range(start, end+1):
            flags[j] = True
      # Add unpredicted verb (for predicted SRL).
      if not flags[pred_id]:
        col_labels[i][pred_id] = "(V*)"
    print_sentence_to_conll(fout, props, col_labels)
  fout.close()

def compute_srl_f1(sentences, gold_srl, predictions, srl_conll_eval_path):
    assert len(gold_srl) == len(predictions)
    total_gold = 0
    total_predicted = 0
    total_matched = 0
    total_unlabeled_matched = 0
    comp_sents = 0
    label_confusions = Counter()

    # Compute unofficial F1 of SRL relations.
    for gold, prediction in zip(gold_srl, predictions):
        gold_rels = 0
        pred_rels = 0
        matched = 0
        for pred_id, gold_args in gold.items():
            filtered_gold_args = [a for a in gold_args if a[2] not in ["V", "C-V"]]
            total_gold += len(filtered_gold_args)
            gold_rels += len(filtered_gold_args)
            if pred_id not in prediction:
                continue
            for a0 in filtered_gold_args:
                for a1 in prediction[pred_id]:
                    if a0[0] == a1[0] and a0[1] == a1[1]:
                        total_unlabeled_matched += 1
                        label_confusions.update([(a0[2], a1[2]), ])
                        if a0[2] == a1[2]:
                            total_matched += 1
                            matched += 1
        for pred_id, args in prediction.items():
            filtered_args = [a for a in args if a[2] not in ["V"]]  # "C-V"]]
            total_predicted += len(filtered_args)
            pred_rels += len(filtered_args)

        if gold_rels == matched and pred_rels == matched:
            comp_sents += 1

    precision, recall, f1 = _print_f1(total_gold, total_predicted, total_matched, "SRL (unofficial)")
    ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_predicted, total_unlabeled_matched,
                                          "Unlabeled SRL (unofficial)")

    # Prepare to compute official F1.
    # if not srl_conll_eval_path:
    # print("No gold conll_eval data provided. Recreating ...")
    gold_path = "tmp/srl_pred_%d.gold" % os.getpid()
    print_to_conll(sentences, gold_srl, gold_path, None)
    gold_predicates = None
    # else:
    #     raise RuntimeError

    temp_output = "tmp/srl_pred_%d.tmp" % os.getpid()
    print_to_conll(sentences, predictions, temp_output, gold_predicates)

    # Evalute twice with official script.
    child = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, gold_path, temp_output), shell=True, stdout=subprocess.PIPE)
    eval_info = child.communicate()[0]
    child2 = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, temp_output, gold_path), shell=True, stdout=subprocess.PIPE)
    eval_info2 = child2.communicate()[0]
    try:
        conll_recall = float(eval_info.strip().split("\n".encode(encoding='utf8'))[6].strip().split()[5])
        conll_precision = float(eval_info2.strip().split("\n".encode(encoding='utf8'))[6].strip().split()[5])
        if conll_recall + conll_precision > 0:
            conll_f1 = 2 * conll_recall * conll_precision / (conll_recall + conll_precision)
        else:
            conll_f1 = 0
        # print(eval_info)
        # print(eval_info2)
        print("Official CoNLL Precision={}, Recall={}, Fscore={}".format(
            conll_precision, conll_recall, conll_f1))
    except IndexError:
        conll_recall = 0
        conll_precision = 0
        conll_f1 = 0
        print("Unable to get FScore. Skipping.")

    return precision, recall, f1, conll_precision, conll_recall, conll_f1, ul_prec, ul_recall, ul_f1, label_confusions, comp_sents

class RetrievalEvaluator(object):
  def __init__(self):
    self._num_correct = 0
    self._num_gold = 0
    self._num_predicted = 0

  def update(self, gold_set, predicted_set):
    self._num_correct += len(gold_set & predicted_set)
    self._num_gold += len(gold_set)
    self._num_predicted += len(predicted_set)

  def recall(self):
    return maybe_divide(self._num_correct, self._num_gold)

  def precision(self):
    return maybe_divide(self._num_correct, self._num_predicted)

  def metrics(self):
    recall = self.recall()
    precision = self.precision()
    f1 = maybe_divide(2 * recall * precision, precision + recall)
    return recall, precision, f1

def compute_span_f1(gold_data, predictions, task_name):
  assert len(gold_data) == len(predictions)
  total_gold = 0
  total_predicted = 0
  total_matched = 0
  total_unlabeled_matched = 0
  label_confusions = Counter()  # Counter of (gold, pred) label pairs.

  for i in range(len(gold_data)):
    gold = gold_data[i]
    pred = predictions[i]
    total_gold += len(gold)
    total_predicted += len(pred)
    for a0 in gold:
      for a1 in pred:
        if a0[0] == a1[0] and a0[1] == a1[1]:
          total_unlabeled_matched += 1
          label_confusions.update([(a0[2], a1[2]),])
          if a0[2] == a1[2]:
            total_matched += 1
  prec, recall, f1 = _print_f1(total_gold, total_predicted, total_matched, task_name)
  ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_predicted, total_unlabeled_matched, "Unlabeled " + task_name)
  return prec, recall, f1, ul_prec, ul_recall, ul_f1, label_confusions

class LSGNEvaluator(object):
    def __init__(self, config):
        self.config = config
        self.eval_data = None

    # TODO: Split to multiple functions.
    def evaluate(self, model, dev_batch, use_cuda, decode=False, official_stdout=False):
        model.eval()
        total_sent_num = 0
        total_batch_num = 0
        self.eval_data = dev_batch

        def _k_to_tag(k):
            if k == -3:
                return "oracle"
            elif k == -2:
                return "actual"
            elif k == -1:
                return "exact"
            elif k == 0:
                return "threshold"
            else:
                return "{}%".format(k)

            # Retrieval evaluators.

        arg_evaluators = {k: RetrievalEvaluator() for k in [-3, -2, -1, 30, 40, 50, 80, 100, 120, 150]}
        predicate_evaluators = {k: RetrievalEvaluator() for k in [-3, -2, -1, 10, 20, 30, 40, 50, 70]}
        total_loss = 0
        total_num_predicates = 0
        total_gold_predicates = 0
        srl_comp_sents = 0
        srl_predictions = []
        all_gold_predicates = []
        all_guessed_predicates = []
        start_time = time.time()
        sent_id = 0

        # Simple analysis.
        unique_core_role_violations = 0
        continuation_role_violations = 0
        reference_role_violations = 0
        gold_u_violations = 0
        gold_c_violations = 0
        gold_r_violations = 0

        # Go through document-level predictions.
        total_sentence_list = []
        total_srl_rels_list = []
        for i, info_dic in enumerate(self.eval_data):
            word_id_tensor = info_dic['word_id_tensor']
            length_list = info_dic['length_list']
            char_id_tensor = info_dic['char_id_tensor']
            predicate_tensor = info_dic['predicate_tensor']
            arg_start_tensor = info_dic['arg_start_tensor']
            arg_end_tensor = info_dic['arg_end_tensor']
            arg_label_tensor = info_dic['arg_label_tensor']
            srl_len = info_dic['srl_len_tensor']
            sentence_list = info_dic['sentence_list']
            srl_rels_list = info_dic['srl_rels_list']

            if use_cuda:
                word_id_tensor = word_id_tensor.cuda()
                char_id_tensor = char_id_tensor.cuda()
                info_dic['predicate_tensor'] = info_dic['predicate_tensor'].cuda()
                info_dic['arg_start_tensor'] = info_dic['arg_start_tensor'].cuda()
                info_dic['arg_end_tensor'] = info_dic['arg_end_tensor'].cuda()
                info_dic['arg_label_tensor'] = info_dic['arg_label_tensor'].cuda()
                info_dic['srl_len_tensor'] = info_dic['srl_len_tensor'].cuda()

            total_sentence_list.extend(sentence_list)
            total_srl_rels_list.extend(srl_rels_list)

            total_sent_num += len(length_list)
            total_batch_num += 1

            predict_dict, loss = model(word_id_tensor, char_id_tensor, length_list, info_dic)

            predict_dict['loss'] = loss.item()

            decoded_predictions = srl_decode(length_list, predict_dict, list(model.srl_labels.keys()), self.config)

            if "srl" in decoded_predictions:
                srl_predictions.extend(decoded_predictions["srl"])
                # Evaluate retrieval.
                word_offset = 0
                for j in range(len(length_list)):
                    text_length = length_list[j]
                    na = predict_dict["num_args"][j]
                    np = predict_dict["num_preds"][j]
                    # sent_example = self.eval_data[sent_id]  # sentence, srl, ner

                    gold_args = set([])
                    gold_preds = set([])
                    guessed_preds = set([])
                    for pred, args in srl_rels_list[j].items():
                        filtered_args = [(a[0], a[1]) for a in args if a[2] not in ["V", "C-V"]]
                        if len(filtered_args) > 0:
                            gold_preds.add((pred, pred))
                            gold_args.update(filtered_args)
                    for pred, args in decoded_predictions["srl"][j].items():
                        guessed_preds.add((pred, pred, "V"))
                    all_gold_predicates.append([(p[0], p[1], "V") for p in gold_preds])
                    all_guessed_predicates.append(guessed_preds)

                    evaluate_retrieval(
                        predict_dict["candidate_starts"][j], predict_dict["candidate_ends"][j],
                        predict_dict["candidate_arg_scores"][j], predict_dict["arg_starts"][j][:na],
                        predict_dict["arg_ends"][j][:na],
                        gold_args, text_length, arg_evaluators)
                    evaluate_retrieval(
                        range(text_length), range(text_length), predict_dict["candidate_pred_scores"][j],
                        predict_dict["predicates"][j][:np], predict_dict["predicates"][j][:np], gold_preds, text_length,
                        predicate_evaluators)

                    # TODO: Move elsewhere.
                    u_violations, c_violations, r_violations = srl_constraint_tracker(
                        decoded_predictions["srl"][j])
                    unique_core_role_violations += u_violations
                    continuation_role_violations += c_violations
                    reference_role_violations += r_violations
                    total_num_predicates += len(decoded_predictions["srl"][j].keys())
                    u_violations, c_violations, r_violations = srl_constraint_tracker(srl_rels_list[j])
                    gold_u_violations += u_violations
                    gold_c_violations += c_violations
                    gold_r_violations += r_violations
                    total_gold_predicates += len(srl_rels_list[j].keys())
                    sent_id += 1
                    word_offset += text_length

            total_loss += predict_dict["loss"]

        summary_dict = {}
        task_to_f1 = {}  # From task name to F1.
        elapsed_time = time.time() - start_time

        # sentences, gold_srl, gold_ner = zip(*self.eval_data)
        sentences = total_sentence_list
        gold_srl = total_srl_rels_list

        # Summarize results, evaluate entire dev set.
        precision, recall, f1, conll_precision, conll_recall, conll_f1, ul_prec, ul_recall, ul_f1, srl_label_mat, comp = (
            compute_srl_f1(sentences, gold_srl, srl_predictions, self.config.srl_conll_eval_path))
        pid_precision, pred_recall, pid_f1, _, _, _, _ = compute_span_f1(
            all_gold_predicates, all_guessed_predicates, "Predicate ID")
        task_to_f1["srl"] = conll_f1
        summary_dict["PAS F1"] = f1
        summary_dict["PAS precision"] = precision
        summary_dict["PAS recall"] = recall
        summary_dict["Unlabeled PAS F1"] = ul_f1
        summary_dict["Unlabeled PAS precision"] = ul_prec
        summary_dict["Unlabeled PAS recall"] = ul_recall
        summary_dict["CoNLL F1"] = conll_f1
        summary_dict["CoNLL precision"] = conll_precision
        summary_dict["CoNLL recall"] = conll_recall
        if total_num_predicates > 0:
            summary_dict["Unique core violations/Predicate"] = 1.0 * unique_core_role_violations / total_num_predicates
            summary_dict[
                "Continuation violations/Predicate"] = 1.0 * continuation_role_violations / total_num_predicates
            summary_dict["Reference violations/Predicate"] = 1.0 * reference_role_violations / total_num_predicates
        # print("Completely correct sentences: {}/{}".format(comp, 100.0 * comp / len(srl_predictions)))

        # for k, evaluator in sorted(arg_evaluators.items(), key=operator.itemgetter(0)):
        #     tags = ["{} {} @ {}".format("Args", t, _k_to_tag(k)) for t in ("R", "P", "F")]
        #     results_to_print = []
        #     for t, v in zip(tags, evaluator.metrics()):
        #         results_to_print.append("{:<10}: {:.4f}".format(t, v))
        #         summary_dict[t] = v
        #     print(", ".join(results_to_print))

        # for k, evaluator in sorted(predicate_evaluators.items(), key=operator.itemgetter(0)):
        #     tags = ["{} {} @ {}".format("Predicates", t, _k_to_tag(k)) for t in ("R", "P", "F")]
        #     results_to_print = []
        #     for t, v in zip(tags, evaluator.metrics()):
        #         results_to_print.append("{:<10}: {:.4f}".format(t, v))
        #         summary_dict[t] = v
        #     print(", ".join(results_to_print))
        #
        # if total_num_predicates > 0:
        #     print("Constraint voilations: U: {} ({}), C: {} ({}), R: {} ({})".format(
        #         1.0 * unique_core_role_violations / total_num_predicates, unique_core_role_violations,
        #         1.0 * continuation_role_violations / total_num_predicates, continuation_role_violations,
        #         1.0 * reference_role_violations / total_num_predicates, reference_role_violations))
        # if total_gold_predicates > 0:
        #     print("Gold constraint voilations: U: {} ({}), C: {} ({}), R: {} ({})".format(
        #         1.0 * gold_u_violations / total_gold_predicates, gold_u_violations,
        #         1.0 * gold_c_violations / total_gold_predicates, gold_c_violations,
        #         1.0 * gold_r_violations / total_gold_predicates, gold_r_violations))
        # # for label_pair, freq in srl_label_mat.most_common():
        # #  if label_pair[0] != label_pair[1] and freq > 10:
        # #    print ("{}\t{}\t{}".format(label_pair[0], label_pair[1], freq))
        #
        # summary_dict["Dev Loss"] = total_loss / total_sent_num
        # print("Decoding took {}.".format(str(datetime.timedelta(seconds=int(elapsed_time)))))
        # print("Decoding speed: {}/sentence, or {}/batch.".format(
        #     str(datetime.timedelta(seconds=int(elapsed_time / total_sent_num))),
        #     str(datetime.timedelta(seconds=int(elapsed_time / total_batch_num)))
        # ))
        metric_names = self.config.main_metrics.split("_")
        main_metric = sum([task_to_f1[t] for t in metric_names]) / len(metric_names)
        print("Combined metric ({}): {}".format(self.config.main_metrics, main_metric))
        if not decode:
            return main_metric, task_to_f1
        else:
            return total_sentence_list, srl_predictions
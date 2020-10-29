import torch
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import copy
from driver.HighWayLSTM import Highway_Concat_BiLSTM
from data.Embedding import load_predtrained_emb_zero, load_predtrained_emb_avg


def initializer_1d(input_tensor, initializer):
    assert len(input_tensor.size()) == 1
    input_tensor = input_tensor.view(-1, 1)
    input_tensor = initializer(input_tensor)
    return input_tensor.view(-1)


class SRLModel(nn.Module):
    def __init__(self, alphabet_dic, config, use_cuda):
        super(SRLModel, self).__init__()
        self.config = config
        self.srl_labels = alphabet_dic['srl_labels_dic']
        self.word_num = alphabet_dic['word_alphabet'].m_size
        self.word_string2id = alphabet_dic['word_alphabet'].string2id
        self.char_num = len(alphabet_dic['char_alphabet'].keys())
        self.use_cuda = use_cuda
        self.use_char = False
        self.word_embed_dim = config.word_embed_dim
        self.contextualization_size = config.contextualization_size
        self.contextualization_layers = config.contextualization_layers

        self.context_embeddings = nn.Embedding(self.word_num, config.word_embed_dim)
        nn.init.xavier_uniform_(self.context_embeddings.weight.data)
        self.head_embeddings = nn.Embedding(self.word_num, config.word_embed_dim)
        nn.init.xavier_uniform_(self.head_embeddings.weight.data)
        self.char_embedding = nn.Embedding(self.char_num, config.char_embedding_size)
        nn.init.xavier_uniform_(self.char_embedding.weight.data)
        self.span_width_embeddings = nn.Embedding(config.max_arg_width, config.feature_size)
        nn.init.xavier_uniform_(self.span_width_embeddings.weight.data)

        if config.context_embedding_path != '':
            embedding = load_predtrained_emb_zero(config.context_embedding_path, self.word_string2id)
            self.context_embeddings.weight.data.copy_(embedding)
            self.context_embeddings.weight.requires_grad = False
        if config.head_embedding_path != '':
            embedding = load_predtrained_emb_zero(config.head_embedding_path, self.word_string2id)
            self.head_embeddings.weight.data.copy_(embedding)
            self.head_embeddings.weight.requires_grad = False

        cnn_output_size = len(config.filter_widths) * config.filter_size
        lstm_input_size = self.word_embed_dim + cnn_output_size
        # self.lstmlist = nn.ModuleList(
        #     [nn.LSTM(
        #         input_size=lstm_input_size,
        #         hidden_size=self.contextualization_size,
        #         num_layers=1,
        #         bidirectional=True,
        #         batch_first=True)
        #         for _ in range(self.contextualization_layers)]
        # )
        self.bilstm = Highway_Concat_BiLSTM(
            input_size=lstm_input_size,
            hidden_size=self.contextualization_size,  # // 2 for MyLSTM
            num_layers=self.contextualization_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=0,
            dropout_out=config.lstm_dropout_rate
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.filter_size, (K, config.char_embedding_size), stride=1, padding=(K // 2, 0)) for K in
             config.filter_widths])

        # highway
        self.gate_linear = nn.Linear(lstm_input_size, lstm_input_size, bias=True)
        self.transform_linear = nn.Linear(self.contextualization_size*2, lstm_input_size)

        # ffnn output linear
        self.output_linear = nn.Linear(self.contextualization_size*2, config.num_attention_heads, bias=True)
        # 1
        self.linear0 = nn.Linear(self.contextualization_size*4+lstm_input_size+config.feature_size, config.ffnn_size, bias=True)
        self.linear1 = nn.Linear(config.ffnn_size, config.ffnn_size, bias=True)
        self.output_linear1 = nn.Linear(config.ffnn_size, config.num_attention_heads, bias=True)
        # 2
        self.linear2 = nn.Linear(self.contextualization_size*2, config.ffnn_size, bias=True)
        self.linear3 = nn.Linear(config.ffnn_size, config.ffnn_size, bias=True)
        self.output_linear2 = nn.Linear(config.ffnn_size, config.num_attention_heads, bias=True)

        # 3
        self.linear4 = nn.Linear(self.contextualization_size*6+lstm_input_size+config.feature_size, config.ffnn_size, bias=True)
        self.linear5 = nn.Linear(config.ffnn_size, config.ffnn_size, bias=True)
        self.output_linear3 = nn.Linear(config.ffnn_size, len(self.srl_labels)-1, bias=True)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.lexical_dropout = nn.Dropout(config.lexical_dropout_rate)
        self.lstm_dropout = nn.Dropout(config.lstm_dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.convs:
            init.xavier_uniform_(layer.weight)
            initializer_1d(layer.bias, init.xavier_uniform_)

        init.xavier_uniform_(self.gate_linear.weight)
        initializer_1d(self.gate_linear.bias, init.xavier_uniform_)

        init.xavier_uniform_(self.transform_linear.weight)

        init.xavier_uniform_(self.output_linear.weight)
        initializer_1d(self.output_linear.bias, init.xavier_uniform_)

        init.xavier_uniform_(self.linear0.weight)
        initializer_1d(self.linear0.bias, init.xavier_uniform_)

        init.xavier_uniform_(self.linear1.weight)
        initializer_1d(self.linear1.bias, init.xavier_uniform_)

        init.xavier_uniform_(self.output_linear1.weight)
        initializer_1d(self.output_linear1.bias, init.xavier_uniform_)

        init.xavier_uniform_(self.linear2.weight)
        initializer_1d(self.linear2.bias, init.xavier_uniform_)

        init.xavier_uniform_(self.linear3.weight)
        initializer_1d(self.linear3.bias, init.xavier_uniform_)

        init.xavier_uniform_(self.output_linear2.weight)
        initializer_1d(self.output_linear2.bias, init.xavier_uniform_)

        init.xavier_uniform_(self.linear4.weight)
        initializer_1d(self.linear4.bias, init.xavier_uniform_)

        init.xavier_uniform_(self.linear5.weight)
        initializer_1d(self.linear5.bias, init.xavier_uniform_)

        init.xavier_uniform_(self.output_linear3.weight)
        initializer_1d(self.output_linear3.bias, init.xavier_uniform_)

    def forward(self, word_id_tensor, char_id_tensor, length_list, info_dic):
        length_tensor = torch.LongTensor(length_list)
        if self.use_cuda:
            length_tensor = length_tensor.cuda()
        context_word_emb = self.context_embeddings(word_id_tensor)
        head_word_emb = self.context_embeddings(word_id_tensor)
        context_emb, head_emb = self.getEmbeddings(char_id_tensor, context_word_emb, head_word_emb)
        # context_outputs = self.highwayLSTM(context_emb, length_list)
        # xqr highwaylstm
        masks = self.init_masks(len(length_list), length_tensor)
        context_outputs, _ = self.bilstm(context_emb, masks)
        return context_outputs
        ###
        # [sent_num, max_arg_width, max_len] ...
        candidate_starts, candidate_ends, candidate_mask = self.getSpanCandidates(length_tensor, self.config.max_arg_width)
        candidate_mask_shape = candidate_mask.size()
        # [sent_num, max_arg_width * max_len]
        flat_candidate_mask = candidate_mask.view(candidate_mask_shape[0], -1)

        cumsum = torch.cumsum(length_tensor, dim=0)  # [sent_num]
        zeros = torch.zeros(1).long()
        if self.use_cuda:
            zeros = zeros.cuda()
        cumsum = torch.cat((zeros, cumsum[:-1]), 0)  # [sent_num]
        batch_word_offset = torch.unsqueeze(cumsum, 1)  # [sent_num, 1]
        # [num of total batch words]
        flat_candidate_starts = torch.masked_select(candidate_starts + batch_word_offset, flat_candidate_mask)
        flat_candidate_ends = torch.masked_select(candidate_ends + batch_word_offset, flat_candidate_mask)
        # [sent_num, max_len]
        text_len_mask = self.sequence_mask(length_tensor, int(max(length_tensor).item()))

        flat_context_outputs = self.flatten_emb_by_sentence(context_outputs, text_len_mask)  # [num of total batch words, emb]
        flat_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask)  # [num of total batch words, emb]

        candidate_span_emb, head_scores, span_head_emb, head_indices, head_indices_log_mask = self.getSpanEmb(
            flat_head_emb, flat_context_outputs, flat_candidate_starts, flat_candidate_ends,
        )  # [num_candidates, emb], [num_candidates, max_span_width, emb], [num_candidates, max_span_width]

        # sparse_to_dense
        num_candidates = self.shape(candidate_span_emb, 0)
        max_num_candidates_per_sentence = self.shape(candidate_mask, 1)
        candidate_ids = torch.arange(0, num_candidates).long()
        # [num_sentences, max_num_candidates]
        # candidate_span_id = self.getSpanIds(candidate_mask, candidate_ids).long()

        # xqr code
        sparse_indices = (candidate_mask == 1).nonzero()
        sparse_values = torch.arange(0, num_candidates)
        if self.use_cuda:
            sparse_values = sparse_values.cuda()
        candidate_span_ids = torch.sparse.FloatTensor(sparse_indices.t(), sparse_values,
                                                      torch.Size([len(length_list),
                                                                  max_num_candidates_per_sentence])).to_dense()
        candidate_span_ids = candidate_span_ids.long()
        ###

        spans_log_mask = torch.log(candidate_mask.float())  # [num_sentences, max_num_candidates]

        # Compute SRL representation.
        # [num_candidates,]
        flat_candidate_arg_scores = self.get_unary_scores(candidate_span_emb, self.dropout, 1)
        # [num_sents, max_num_candidates]
        # candidate_arg_score = self.getSpanIds(candidate_mask, flat_candidate_arg_scores) + spans_log_mask
        # xqr
        candidate_arg_scores = flat_candidate_arg_scores.index_select(0, candidate_span_ids.view(-1)) \
            .view(candidate_span_ids.size()[0], candidate_span_ids.size()[1])
        candidate_arg_scores = candidate_arg_scores + spans_log_mask
        ###

        # [num_sentences, max_num_args], ... [num_sentences,], [num_sentences, max_num_args]
        max_len = int(max(length_tensor).item())
        arg_starts, arg_ends, arg_scores, num_args, top_arg_indices = self.get_batch_topk(
            candidate_starts, candidate_ends, candidate_arg_scores, self.config.argument_ratio, length_tensor,
            max_len, sort_spans=False, enforce_non_crossing=False)

        max_len = torch.arange(0, max_len)
        if self.use_cuda:
            max_len = max_len.cuda()
        candidate_pred_ids = max_len.unsqueeze(0).expand(len(length_list), -1)  # [num_sentences, max_sentence_length]
        candidate_pred_emb = context_outputs  # [num_sentences, max_sentence_length, emb]
        candidate_pred_scores = self.get_unary_scores(
            candidate_pred_emb, self.dropout, 1, flag='candidate_pred_scores') + torch.log(text_len_mask.float())  # [num_sentences, max_sentence_length]

        if self.config.use_gold_predicates:
            # predicates = inputs["gold_predicates"]
            # num_preds = inputs["num_gold_predicates"]
            # pred_scores = tf.zeros_like(predicates, dtype=tf.float32)
            # top_pred_indices = predicates
            raise RuntimeError
        else:
            # [num_sentences, max_num_preds] ... [num_sentences,]
            predicates, _, pred_scores, num_preds, top_pred_indices = self.get_batch_topk(
                candidate_pred_ids, candidate_pred_ids, candidate_pred_scores, self.config.predicate_ratio,
                length_tensor, max_len, sort_spans=False, enforce_non_crossing=False)

        #arg_span_indices = self.batchGather(candidate_span_ids, top_arg_indices)  # [num_sentences, max_num_args]
        if self.use_cuda:
            top_arg_indices = top_arg_indices.cuda()
        arg_span_indices = torch.gather(candidate_span_ids, 1, top_arg_indices)
        arg_span_indices_size = arg_span_indices.size()
        arg_span_indices = arg_span_indices.view(-1)
        arg_emb = torch.index_select(candidate_span_emb, 0, arg_span_indices.long())\
            .view(arg_span_indices_size[0], arg_span_indices_size[1], -1)  # [num_sentences, max_num_args, emb]
        pred_emb = self.batchGather(candidate_pred_emb, top_pred_indices)  # [num_sentences, max_num_preds, emb]
        # max_num_args = self.shape(arg_scores, 1)
        # max_num_preds = self.shape(pred_scores, 1)
        # output_dic = {}
        # output_dic['arg_starts'] = arg_starts
        # output_dic['arg_ends'] = arg_ends
        # output_dic['predicates'] = predicates
        # output_dic['arg_emb'] = arg_emb
        # output_dic['pred_emb'] = pred_emb

        # [num_sentences, max_num_args, max_num_preds]
        srl_labels = self.get_srl_labels(arg_starts, arg_ends, predicates, info_dic, max(length_list))

        # [num_sentences, max_num_args, max_num_preds, num_labels]
        srl_scores = self.get_srl_scores(
            arg_emb, pred_emb, arg_scores, pred_scores, len(self.srl_labels), self.config, self.dropout
        )

        srl_loss = self.get_srl_softmax_loss(
            srl_scores, srl_labels, num_args, num_preds)  # [num_sentences, max_num_args, max_num_preds]
        # srl_loss1 = self.get_srl_softmax_loss1(
        #     srl_scores, srl_labels, num_args, num_preds)

        srl_scores_argmax = torch.max(srl_scores, dim=-1)[1]

        predict_dict = {
            "candidate_starts": candidate_starts.long().cpu().numpy(),
            "candidate_ends": candidate_ends.long().cpu().numpy(),
            "head_scores": head_scores.detach().cpu().numpy(),
            "candidate_arg_scores": candidate_arg_scores.detach().cpu().numpy(),
            "candidate_pred_scores": candidate_pred_scores.detach().cpu().numpy(),
            "arg_starts": arg_starts.long().cpu().numpy(),
            "arg_ends": arg_ends.long().cpu().numpy(),
            "predicates": predicates.cpu().numpy(),
            "arg_scores": arg_scores.detach().cpu().numpy(),  # New ...
            "pred_scores": pred_scores.detach().cpu().numpy(),
            "num_args": num_args.long().cpu().numpy(),
            "num_preds": num_preds.long().cpu().numpy(),
            "arg_labels": srl_scores_argmax.cpu().numpy(),  # [num_sentences, num_args, num_preds]
            "srl_scores": srl_scores.detach().cpu().numpy()
        }

        return predict_dict, srl_loss

    def init_masks(self, batch_size, lengths):
        max_sent_length = max(lengths)
        num_sentences = lengths.size()[0]
        indices = torch.arange(0, max_sent_length).unsqueeze(0).expand(num_sentences, -1)
        masks = indices < lengths.unsqueeze(1).cpu()
        masks = masks.type(torch.FloatTensor)
        # masks = torch.zeros(batch_size, max_length)s
        # masks.requires_grad = False
        # for i, length in enumerate(lengths):
        #     masks.data[i][:length] += 1.0
        if self.use_cuda:
            masks = masks.cuda()
        return masks

    def getSpanIds(self, candidate_mask, ids):
        new_tensor = copy.deepcopy(candidate_mask).float()
        start, end = 0, 0
        for idx in range(len(new_tensor)):
            length = int(torch.sum(new_tensor[idx]).item())
            end += length
            new_tensor[idx][:length] = ids[start:end]
            start = end
        return new_tensor

    def getSpanEmb(self, head_emb, context_outputs, span_starts, span_ends):
        """Compute span representation shared across tasks.
          Args:
            head_emb: Tensor of [num_words, emb]
            context_outputs: Tensor of [num_words, emb]
            span_starts: [num_spans]
            span_ends: [num_spans]
          """
        sent_len = context_outputs.size(0)
        num_span =span_starts.size(0)

        span_start_emb = torch.index_select(context_outputs, 0, span_starts.long())  # [num_words, emb]
        span_end_emb = torch.index_select(context_outputs, 0, span_ends.long())  # [num_words, emb]
        span_emb_list = [span_start_emb, span_end_emb]

        span_width = 1 + span_ends - span_starts  # [num_spans]
        max_arg_width = self.config.max_arg_width
        num_heads = self.config.num_attention_heads

        if self.config.use_features:
            span_width_index = span_width - 1  # [num_spans]
            span_width_emb = self.span_width_embeddings(span_width_index.long())  # [num_spans, emb]
            span_width_emb = self.dropout(span_width_emb)
            span_emb_list.append(span_width_emb)

        head_scores = None
        span_text_emb = None
        span_indices = None
        span_indices_log_mask = None

        if self.config.model_heads:
            threshold = torch.LongTensor([sent_len - 1])
            max_arg_width_range = torch.arange(0, max_arg_width)
            # if self.use_cuda:
            #     threshold = threshold.cuda()
            #     max_arg_width_range = max_arg_width_range.cuda()
            # [num_spans, max_span_width]
            span_starts = span_starts.cpu()
            span_indices = torch.min(max_arg_width_range.unsqueeze(0) + span_starts.unsqueeze(1), other=threshold)
            span_indices_size = span_indices.size()
            flat_span_indices = span_indices.long().view(span_indices_size[0] * span_indices_size[1])
            if self.use_cuda:
                flat_span_indices = flat_span_indices.cuda()
            span_text_emb = torch.index_select(head_emb,
                                               0,
                                               flat_span_indices)\
                   .view(span_indices.size(0), span_indices.size(1), head_emb.size(1))  # [num_spans, max_arg_width, emb]

            span_indices_mask = self.sequence_mask(span_width, max_arg_width, dtype=torch.float).float()
            # [num_spans, max_arg_width]
            span_indices_log_mask = torch.log(span_indices_mask)
            if self.use_cuda:
                span_indices_log_mask = span_indices_log_mask.cuda()
            # head_scores
            head_scores = self.projection(context_outputs, num_heads)  # [num_words, num_heads]

            # [num_spans, max_arg_width, num_heads]
            span_attention = F.softmax(
                torch.index_select(head_scores, 0, flat_span_indices).view(span_indices.size(0),
                                                                           span_indices.size(1),
                                                                           head_scores.size(1)
                                                                           ) + span_indices_log_mask.unsqueeze(2),
                dim=1)
            span_head_emb = torch.sum(span_attention * span_text_emb, dim=1)  # [num_spans, emb]
            span_emb_list.append(span_head_emb)
        #
        span_emb = torch.cat(span_emb_list, 1)  # [num_spans, emb]
        return span_emb, head_scores, span_text_emb, span_indices, span_indices_log_mask

    def sequence_mask(self, lengths, maxlen, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()
        ones = torch.ones((len(lengths), maxlen)).long()
        mask = ~(ones.cumsum(dim=1).t() > lengths.cpu()).t()
        mask.type(dtype)
        if self.use_cuda:
            mask = mask.cuda()
        return mask

    def getEmbeddings(self, char_id_tensor, context_word_emb, head_word_emb):
        context_emb_list = [context_word_emb]
        head_emb_list = [head_word_emb]
        char_embed = self.char_embedding(char_id_tensor)
        shape = char_embed.size()
        char_embed = char_embed.view(shape[0]*shape[1], shape[2], shape[3])
        char_embed = torch.unsqueeze(char_embed, 1)
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(torch.relu(conv(char_embed)).squeeze(3))
        pool_outputs = []
        for output in conv_outputs:
            pool_outputs.append(F.max_pool1d(output, kernel_size=output.size(2)).squeeze(2))
        outputs = torch.cat(pool_outputs, 1)
        outputs = outputs.view(shape[0], shape[1], -1)
        context_emb_list.append(outputs)
        head_emb_list.append(outputs)
        context_emb = torch.cat(context_emb_list, 2)
        head_emb = torch.cat(head_emb_list, 2)
        context_emb = self.lexical_dropout(context_emb)
        head_emb = self.lexical_dropout(head_emb)
        return context_emb, head_emb

    def highwayLSTM(self, context_emb, length_list):
        x = context_emb
        hidden = None
        for idx, lstm in enumerate(self.lstmlist):
            source_x = x
            x = pack_padded_sequence(x, lengths=length_list, batch_first=True)
            if idx == 0:
                x, hidden = lstm(x)
            else:
                x, hidden = lstm(x, hidden)
            x = pad_packed_sequence(x, batch_first=True)[0]
            x = self.lstm_dropout(x)
            x = self.transform_linear(x)
            output_x = x
            highway_gates = F.sigmoid(self.gate_linear(output_x))
            x = highway_gates * output_x + (1 - highway_gates) * source_x
        return x

    def getSpanCandidates(self, length_tensor, max_arg_width):
        max_len = int(max(length_tensor).item())
        sent_num = len(length_tensor)
        '''
            candidate_starts: [sent_num, max_arg_width, max_len]
            [[[ 0.,  1.,  2.,  ..., 47., 48., 49.],
              [ 0.,  1.,  2.,  ..., 47., 48., 49.],
              [ 0.,  1.,  2.,  ..., 47., 48., 49.],
              ...,
              ]]
        '''
        candidate_starts_range = torch.arange(0, max_len)
        if self.use_cuda:
            candidate_starts_range = candidate_starts_range.cuda()
        candidate_starts = torch.unsqueeze(torch.unsqueeze(candidate_starts_range, 0), 1).expand(sent_num, max_arg_width, max_len)
        '''
            candidate_widths: [1, max_arg_width, 1]
            [[[ 0.],
              [ 1.],
              [ 2.],
              [ 3.],
              ...
            ]]
        '''
        candidate_widths_range = torch.arange(0, max_arg_width)
        if self.use_cuda:
            candidate_widths_range = candidate_widths_range.cuda()
        candidate_widths = torch.unsqueeze(torch.unsqueeze(candidate_widths_range, 0), 2)
        '''
            candidate_ends: [sent_num, max_arg_width, max_len] 
            [[[ 0.,  1.,  2.,  ..., 47., 48., 49.],
              [ 1.,  2.,  3.,  ..., 48., 49., 50.],
              [ 2.,  3.,  4.,  ..., 49., 50., 51.],
              ...,
            ]]
        '''
        candidate_ends = candidate_starts + candidate_widths
        # [sent_num, max_arg_width * max_len]
        candidate_starts = candidate_starts.contiguous().view(sent_num, max_arg_width * max_len)

        # [sent_num, max_arg_width * max_len]
        candidate_ends = candidate_ends.contiguous().view(sent_num, max_arg_width * max_len)

        # [sent_num, max_arg_width * max_len]
        candidate_mask = torch.lt(candidate_ends,
                                  torch.unsqueeze(length_tensor.long(), 1).expand(sent_num, max_arg_width * max_len))

        # Mask to avoid indexing error.
        candidate_starts = candidate_starts * candidate_mask.long()
        candidate_ends = candidate_ends * candidate_mask.long()

        return candidate_starts, candidate_ends, candidate_mask

    def flatten_emb_by_sentence(self, emb, text_len_mask):
        emb_size = emb.size()
        sent_num = emb_size[0]
        max_sent_len = emb_size[1]
        flattened_emb = self.flatten_emb(emb)
        text_len_mask_boardcast = text_len_mask.view(sent_num * max_sent_len).unsqueeze(1).\
            expand(sent_num * max_sent_len, emb_size[2])
        return torch.masked_select(flattened_emb, text_len_mask_boardcast).view(-1, emb_size[2])

    def flatten_emb(self, emb):
        emb_size = emb.size()
        num_sentences = emb_size[0]
        max_sentence_length = emb_size[1]
        emb_rank = len(emb.size())
        if emb_rank == 2:
            flattened_emb = emb.contiguous().view(num_sentences * max_sentence_length)
        elif emb_rank == 3:
            flattened_emb = emb.contiguous().view([num_sentences * max_sentence_length, self.shape(emb, 2)])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        return flattened_emb

    def shape(self, x, dim):
        return x.size(dim) or x.size(dim)

    def projection(self, inputs, output_size, initializer=None):
        return self.ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)

    def ffnn(self, inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
        if len(inputs.size()) > 2:
            current_inputs = inputs.view(-1, self.shape(inputs, -1))
        else:
            current_inputs = inputs

        if num_hidden_layers > 0:
            output0 = torch.relu(self.linear0(current_inputs))
            if dropout is not None:
                output0 = dropout(output0)
            output1 = torch.relu(self.linear1(output0))
            if dropout is not None:
                output1 = dropout(output1)
            current_inputs = output1

        outputs = self.output_linear(current_inputs)

        if len(inputs.size()) == 3:
            outputs = outputs.view(self.shape(inputs, 0), self.shape(inputs, 1), output_size)
        elif len(inputs.size()) > 3:
            raise ValueError("FFNN with rank {} not supported".format(len(inputs.size())))
        return outputs

    def ffnn_(self, inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
        if len(inputs.size()) > 2:
            current_inputs = inputs.view(-1, self.shape(inputs, -1))
        else:
            current_inputs = inputs
        # outputs = None
        if num_hidden_layers == 2:
            output0 = torch.relu(self.linear0(current_inputs))
            if dropout is not None:
                output0 = dropout(output0)
            output1 = torch.relu(self.linear1(output0))
            if dropout is not None:
                output1 = dropout(output1)
            current_inputs = output1

        outputs = self.output_linear1(current_inputs)

        # if num_hidden_layers == 1:
        #     output2 = torch.relu(self.linear2(current_inputs))
        #     outputs = self.output_linear2(output2)

        if len(inputs.size()) == 3:
            outputs = outputs.view(self.shape(inputs, 0), self.shape(inputs, 1), output_size)
        elif len(inputs.size()) > 3:
            raise ValueError("FFNN with rank {} not supported".format(len(inputs.size())))
        return outputs

    def ffnn__(self, inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
        if len(inputs.size()) > 2:
            current_inputs = inputs.contiguous().view(-1, self.shape(inputs, -1))
        else:
            current_inputs = inputs
        # outputs = None
        if num_hidden_layers == 2:
            output2 = torch.relu(self.linear2(current_inputs))
            if dropout is not None:
                output2 = dropout(output2)
            output3 = torch.relu(self.linear3(output2))
            if dropout is not None:
                output3 = dropout(output3)
            current_inputs = output3

        outputs = self.output_linear2(current_inputs)

        # if num_hidden_layers == 1:
        #     output2 = torch.relu(self.linear2(current_inputs))
        #     outputs = self.output_linear2(output2)


        if len(inputs.size()) == 3:
            outputs = outputs.view(self.shape(inputs, 0), self.shape(inputs, 1), output_size)
        elif len(inputs.size()) > 3:
            raise ValueError("FFNN with rank {} not supported".format(len(inputs.size())))
        return outputs

    def ffnn___(self, inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
        if len(inputs.size()) > 2:
            current_inputs = inputs.view(-1, self.shape(inputs, -1))
        else:
            current_inputs = inputs
        # outputs = None
        if num_hidden_layers == 2:
            output2 = torch.relu(self.linear4(current_inputs))
            if dropout is not None:
                output2 = dropout(output2)
            output3 = torch.relu(self.linear5(output2))
            if dropout is not None:
                output3 = dropout(output3)
            current_inputs = output3

        outputs = self.output_linear3(current_inputs)

        if len(inputs.size()) == 3:
            outputs = outputs.view(self.shape(inputs, 0), self.shape(inputs, 1), output_size)
        elif len(inputs.size()) > 3:
            raise ValueError("FFNN with rank {} not supported".format(len(inputs.size())))
        return outputs

    def get_unary_scores(self, span_emb, dropout, num_labels=1, flag=''):
        """Compute span score with FFNN(span embedding).
        Args:
          span_emb: Tensor of [num_sentences, num_spans, emb].
        """
        # span_scores
        # [num_sentences, num_spans, num_labels] or [k, num_labels]
        # scores = self.ffnn_(span_emb, self.config.ffnn_depth, self.config.ffnn_size, num_labels, dropout)
        if flag == 'candidate_pred_scores':
            scores = self.ffnn__(span_emb, self.config.ffnn_depth, self.config.ffnn_size, num_labels, dropout)
        elif flag == 'predicate_argument_scores':
            scores = self.ffnn___(span_emb, self.config.ffnn_depth, self.config.ffnn_size, num_labels, dropout)
        else:
            scores = self.ffnn_(span_emb, self.config.ffnn_depth, self.config.ffnn_size, num_labels, dropout)
        if num_labels == 1:
            scores = scores.squeeze(-1)  # [num_sentences, num_spans] or [k]
        return scores

    def get_batch_topk(self, candidate_starts, candidate_ends, candidate_scores, topk_ratio, text_len,
                       max_sentence_length, sort_spans=False, enforce_non_crossing=True):
        """
        Args:
          candidate_starts: [num_sentences, max_num_candidates]
          candidate_mask: [num_sentences, max_num_candidates]
          topk_ratio: A float number.
          text_len: [num_sentences,]
          max_sentence_length:
          enforce_non_crossing: Use regular top-k op if set to False.
       """
        num_sentences = self.shape(candidate_starts, 0)
        max_num_candidates = self.shape(candidate_starts, 1)

        # [num_sentences]
        floor = torch.floor(text_len.float() * topk_ratio)
        ones = torch.ones([num_sentences, ])
        if self.use_cuda:
            ones = ones.cuda()
        topk = torch.max(floor, other=ones).long()

        predicted_indices = self.extractSpans(
            candidate_scores, candidate_starts, candidate_ends, topk, max_sentence_length,
            sort_spans, enforce_non_crossing)  # [num_sentences, max_num_predictions]
        # predicted_indices.set_shape([None, None])

        # predicted_start = self.batchGather(candidate_starts, predicted_indices)  # [num_sentences, max_num_predictions]
        # # predicted_ends = self.batchGather(candidate_ends, predicted_indices)  # [num_sentences, max_num_predictions]
        # # predicted_scores = self.batchGather(candidate_scores, predicted_indices)  # [num_sentences, max_num_predictions]
        # xqr
        candidate_starts = candidate_starts.cpu()
        candidate_ends = candidate_ends.cpu()
        candidate_scores = candidate_scores.cpu()
        predicted_starts = torch.gather(candidate_starts, 1, predicted_indices)
        predicted_ends = torch.gather(candidate_ends, 1, predicted_indices)
        predicted_scores = torch.gather(candidate_scores, 1, predicted_indices)
        ###
        return predicted_starts, predicted_ends, predicted_scores, topk, predicted_indices

    def extractSpans(self, candidate_scores, candidate_starts, candidate_ends, topk, max_sentence_length,
            sort_spans, enforce_non_crossing):
        _sort_spans = sort_spans
        _suppress_crossing = enforce_non_crossing  # do not it is right !!!

        span_scores = candidate_scores
        candidate_starts = candidate_starts
        candidate_ends = candidate_ends
        num_output_spans = topk
        max_sentence_length = max_sentence_length


        # xqr
        max_num_output_spans = int(torch.max(topk))
        indices = [score.topk(k)[1] for score, k in zip(candidate_scores, topk)]
        output_span_indices_tensor = [F.pad(item, [0, max_num_output_spans - item.size()[0]], value=item[-1]) for item in indices]
        output_span_indices = torch.stack(output_span_indices_tensor).cpu()
        ###
        # num_sentences = span_scores.size(0)
        # num_input_spans = span_scores.size(1)
        # max_num_output_spans = 0
        # for i in range(num_sentences):
        #     if num_output_spans[i] > max_num_output_spans:
        #         max_num_output_spans = num_output_spans[i]
        #
        # output_span_indices = torch.zeros((num_sentences, int(max_num_output_spans.item()))).int()
        # sorted_input_span_indices = torch.zeros((num_sentences, num_input_spans))
        # if self.use_cuda:
        #     output_span_indices = output_span_indices.cuda()
        #     sorted_input_span_indices = sorted_input_span_indices.cuda()
        # for i in range(num_sentences):
        #     for j in range(num_input_spans):
        #         sorted_input_span_indices[i][j] = j
        #     _, index = torch.sort(span_scores[i], descending=True)
        #     # index = index.tolist()
        #     sorted_input_span_indices[i] = sorted_input_span_indices[i][index]
        #
        # for l in range(num_sentences):
        #     top_span_indices = []
        #     # end_to_earliest_start = {}
        #     # start_to_latest_end = {}
        #     current_span_index, num_selected_spans = 0, 0
        #     while num_selected_spans < num_output_spans[l] and current_span_index < num_input_spans:
        #         i = sorted_input_span_indices[l][current_span_index]
        #         any_crossing = False
        #         if not any_crossing:
        #             if sort_spans:
        #                 top_span_indices.append(i)
        #             else:
        #                 output_span_indices[l][num_selected_spans] = i
        #             num_selected_spans += 1
        #         current_span_index += 1
        #     last_selected = num_selected_spans - 1
        #     if last_selected >= 0:
        #         for i in range(num_selected_spans, int(max_num_output_spans.item())):
        #             output_span_indices[l][i] = output_span_indices[l][last_selected]

        return output_span_indices

    def batchGather(self, emb, indices):
        # TODO: Merge with util.batch_gather.
        """
        Args:
          emb: Shape of [num_sentences, max_sentence_length, (emb)]
          indices: Shape of [num_sentences, k, (l)]
        """
        num_sentences = emb.size(0)
        max_sentence_length = emb.size(1)
        emb_size = emb.size()
        flattened_emb = self.flatten_emb(emb)  # [num_sentences * max_sentence_length, emb]
        num_sentences_range = torch.arange(0, num_sentences)
        # if self.use_cuda:
        #     num_sentences_range = num_sentences_range.cuda()
        offset = (num_sentences_range * max_sentence_length).unsqueeze(1)  # [num_sentences, 1]
        if len(indices.size()) == 3:
            offset = offset.unsqueeze(2)  # [num_sentences, 1, 1]
        indice_offset = indices.long() + offset.long()
        indice_offset_size = indice_offset.size()
        indice_offset = indice_offset.view(-1)
        selected = torch.index_select(flattened_emb.cpu(), 0, indice_offset)
        if len(emb_size) > 2:
            selected = selected.view(indice_offset_size[0], indice_offset_size[1], emb_size[-1])
        else:
            selected = selected.view(indice_offset_size)
        if self.use_cuda:
            selected = selected.cuda()
        return selected

    def get_dense_span_labels(self, span_starts, span_ends, span_labels, num_spans, max_sentence_length, span_parents=None):
        """Utility function to get dense span or span-head labels.
        Args:
          span_starts: [num_sentences, max_num_spans]
          span_ends: [num_sentences, max_num_spans]
          span_labels: [num_sentences, max_num_spans]
          num_spans: [num_sentences,]
          max_sentence_length:
          span_parents: [num_sentences, max_num_spans]. Predicates in SRL.
        """
        num_sentences = span_starts.size(0)
        max_num_spans = span_starts.size(1)
        if self.use_cuda:
            span_starts = span_starts.cuda()
        # For padded spans, we have starts = 1, and ends = 0, so they don't collide with any existing spans.
        sequence_mask_result = self.sequence_mask(num_spans, maxlen=int(max(num_spans).item()), dtype=torch.int32).long()
        span_starts += (1 - sequence_mask_result)  # [num_sentences, max_num_spans]
        num_sentences_range = torch.arange(0, num_sentences)
        if self.use_cuda:
            num_sentences_range = num_sentences_range.cuda()
        sentence_indices = num_sentences_range.unsqueeze(1).expand(num_sentences, max_num_spans)  # [num_sentences, max_num_spans]
        sparse_indices = torch.cat([
            sentence_indices.unsqueeze(2),
            span_starts.unsqueeze(2),
            span_ends.unsqueeze(2)], dim=2)  # [num_sentences, max_num_spans, 3]
        if span_parents is not None:
            sparse_indices = torch.cat([sparse_indices, span_parents.unsqueeze(2)], dim=2)  # [num_sentenes, max_num_spans, 4]

        rank = 3 if (span_parents is None) else 4
        # (sent_id, span_start, span_end) -> span_label
        # sparse to dense
        sparse_indices = sparse_indices.view(num_sentences * max_num_spans, rank)
        # output_shape = [num_sentences, max_sentence_length, max_sentence_length, max_sentence_length]
        # default_value = 0
        # sparse_values = span_labels.view(-1)
        # xqr
        dense_labels = torch.sparse.FloatTensor(sparse_indices.cpu().view(num_sentences * max_num_spans, rank).t(),
                                                span_labels.view(-1).type(torch.FloatTensor),
                                                torch.Size(
                                                    [num_sentences] + [max_sentence_length] * (rank - 1))).to_dense()
        ###
        # dense_label = torch.zeros(num_sentences, max_sentence_length, max_sentence_length, max_sentence_length)
        # if self.use_cuda:
        #     dense_label = dense_label.cuda()
        # for idx, value in enumerate(sparse_values):
        #     i, j, k, t = sparse_indices[idx]
        #     dense_label[i][j][k][t] = value

        return dense_labels

    def gather_4d(self, params, indices):  # ugly
        assert len(params.size()) == 4 and len(indices.size()) == 4
        # print params.size(), indices.size()
        # exit()
        params = params.type(torch.LongTensor)
        indices_a, indices_b, indices_c, indices_d = indices.chunk(4, dim=3)
        result = params[indices_a, indices_b, indices_c, indices_d]
        # result = torch.zeros(indices.size()[:3]).type(torch.LongTensor)
        #
        # for i in range(indices.size()[0]):
        #     for j in range(indices.size()[1]):
        #         for k in range(indices.size()[2]):
        #             result[i][j][k] = params[indices[i][j][k][0]][indices[i][j][k][1]][indices[i][j][k][2]][indices[i][j][k][3]]
        return result.unsqueeze(3)

    def get_srl_labels(self, arg_starts, arg_ends, predicates, labels, max_sentence_length):
        """
        Args:
          arg_starts: [num_sentences, max_num_args]
          arg_ends: [num_sentences, max_num_args]
          predicates: [num_sentences, max_num_predicates]
          labels: Dictionary of label tensors.
          max_sentence_length: An integer scalar.
        """
        # arg_starts = output_dic['arg_starts']
        # arg_ends = output_dic['arg_ends']
        # predicates = output_dic['predicates']
        num_sentences = arg_starts.size(0)
        max_num_args = arg_starts.size(1)
        max_num_preds = predicates.size(1)
        num_sentences_range = torch.arange(0, num_sentences)
        # if self.use_cuda:
        #     num_sentences_range = num_sentences_range.cuda()
        sentence_indices_2d = num_sentences_range. \
            unsqueeze(1).unsqueeze(2). \
            expand(num_sentences, max_num_args, max_num_preds)  # [num_sentences, max_num_args, max_num_preds]
        tiled_arg_starts = arg_starts.unsqueeze(2). \
            expand(arg_ends.size(0), arg_ends.size(1), max_num_preds)  # [num_sentences, max_num_args, max_num_preds]
        tiled_arg_ends = arg_ends.unsqueeze(2). \
            expand(arg_ends.size(0), arg_ends.size(1), max_num_preds)  # [num_sentences, max_num_args, max_num_preds]
        tiled_predicates = predicates.unsqueeze(1). \
            expand(predicates.size(0), max_num_args, predicates.size(1))  # [num_sentences, max_num_args, max_num_preds]
        tiled_arg_starts = tiled_arg_starts.cpu()
        tiled_arg_ends = tiled_arg_ends.cpu()
        tiled_predicates = tiled_predicates.cpu()
        pred_indices = torch.cat([
            sentence_indices_2d.unsqueeze(3),
            tiled_arg_starts.unsqueeze(3),
            tiled_arg_ends.unsqueeze(3),
            tiled_predicates.unsqueeze(3)], dim=3)  # [num_sentences, max_num_args, max_num_preds, 4]

        dense_srl_labels = self.get_dense_span_labels(
            labels["arg_start_tensor"], labels["arg_end_tensor"], labels["arg_label_tensor"], labels["srl_len_tensor"],
            max_sentence_length,
            span_parents=labels["predicate_tensor"])  # [num_sentences, max_sent_len, max_sent_len, max_sent_len]

        # srl_labels = torch.index_select(dense_srl_labels, 3, pred_indices.long())  # [num_sentences, max_num_args]
        # srl_labels = torch.zeros((num_sentences, max_num_args, max_num_preds))
        # if self.use_cuda:
        #     srl_labels = srl_labels.cuda()
        # pred_indices = pred_indices.long()  # [num_sentences, max_num_args, max_num_preds]
        # for i in range(num_sentences):
        #     for j in range(max_num_args):
        #         for k in range(max_num_preds):
        #             index = pred_indices[i][j][k]
        #             srl_labels[i][j][k] = dense_srl_labels[index[0]][index[1]][index[2]][index[3]]

        srl_labels = self.gather_4d(dense_srl_labels, pred_indices.type(torch.LongTensor))

        return srl_labels

    def get_srl_scores(self, arg_emb, pred_emb, arg_scores, pred_scores, num_labels, config, dropout):
        num_sentences = self.shape(arg_emb, 0)
        num_args = self.shape(arg_emb, 1)
        num_preds = self.shape(pred_emb, 1)

        arg_emb_expanded = arg_emb.unsqueeze(2)  # [num_sents, num_args, 1, emb]
        pred_emb_expanded = pred_emb.unsqueeze(1)  # [num_sents, 1, num_preds, emb]

        arg_emb_size = arg_emb_expanded.size()
        # [num_sentences, num_args, num_preds, emb]
        arg_emb_tiled = arg_emb_expanded.expand(arg_emb_size[0], arg_emb_size[1], num_preds, arg_emb_size[3])

        pred_emb_size = pred_emb_expanded.size()
        # [num_sents, num_args, num_preds, emb]
        pred_emb_tiled = pred_emb_expanded.expand(pred_emb_size[0], num_args, pred_emb_size[2], pred_emb_size[3])

        pair_emb_list = [arg_emb_tiled, pred_emb_tiled]
        pair_emb = torch.cat(pair_emb_list, 3)  # [num_sentences, num_args, num_preds, emb]
        pair_emb_size = self.shape(pair_emb, 3)
        flat_pair_emb = pair_emb.view(num_sentences * num_args * num_preds, pair_emb_size)

        flat_srl_scores = self.get_unary_scores(flat_pair_emb, dropout, num_labels - 1,
                                           "predicate_argument_scores")  # [num_sentences * num_args * num_predicates, 1]
        srl_scores = flat_srl_scores.view(num_sentences, num_args, num_preds, num_labels - 1)
        if self.use_cuda:
            arg_scores = arg_scores.cuda()
            pred_scores = pred_scores.cuda()
        srl_scores = srl_scores + arg_scores.unsqueeze(2).unsqueeze(3) + pred_scores.unsqueeze(1).unsqueeze(3)  # [num_sentences, 1, max_num_preds, num_labels-1]

        dummy_scores = torch.zeros([num_sentences, num_args, num_preds, 1]).float()
        if self.use_cuda:
            dummy_scores = dummy_scores.cuda()
        srl_scores = torch.cat([dummy_scores, srl_scores],
                               3)  # [num_sentences, max_num_args, max_num_preds, num_labels]
        return srl_scores  # [num_sentences, num_args, num_predicates, num_labels]

    def get_srl_softmax_loss(self, srl_scores, srl_labels, num_predicted_args, num_predicted_preds):
        max_num_arg = srl_scores.size()[1]
        max_num_pred = srl_scores.size()[2]
        num_labels = srl_scores.size()[3]

        args_mask = self.sequence_mask(num_predicted_args, max_num_arg)
        # print args_mask.sum()
        pred_mask = self.sequence_mask(num_predicted_preds, max_num_pred)
        # print pred_mask.sum()
        srl_loss_mask = Variable((args_mask.unsqueeze(2) == 1) & (pred_mask.unsqueeze(1) == 1))

        srl_scores = srl_scores.view(-1, num_labels)
        # print "srl labels", torch.sum(srl_labels)
        srl_labels  = Variable(srl_labels.view(-1, 1))
        if self.use_cuda:
            srl_labels = srl_labels.cuda()
        # print srl_scores, srl_labels.size()
        output = F.log_softmax(srl_scores, 1)  #* Variable(srl_labels.view(-1)).cuda()

        negative_log_likelihood_flat = -torch.gather(output, dim=1, index=srl_labels).view(-1)
        # print torch.sum(srl_loss_mask.type(torch.FloatTensor))
        srl_loss_mask = (srl_loss_mask.view(-1) == 1).nonzero().view(-1)
        # print srl_loss_mask.size()
        # print negative_log_likelihood_flat
        negative_log_likelihood = torch.gather(negative_log_likelihood_flat, dim=0, index=srl_loss_mask)
        # negative_log_likelihood = negative_log_likelihood_flat.view(*srl_labels.size())  # [B, T]
        # negative_log_likelihood = negative_log_likelihood_flat * srl_loss_mask.view(-1).float()
        # print negative_log_likelihood
        # z = ((torch.ones_like(negative_log_likelihood) * 75217952 - negative_log_likelihood) != 0).nonzero()
        # print z
        # print negative_log_likelihood.view(-1).index_select(dim=0, index=z.view(-1))
        loss = negative_log_likelihood.sum()
        return loss

    def get_srl_softmax_loss1(self, srl_scores, srl_labels, num_predicted_args, num_predicted_preds):
        """Softmax loss with 2-D masking (for SRL).
        Args:
          srl_scores: [num_sentences, max_num_args, max_num_preds, num_labels]
          srl_labels: [num_sentences, max_num_args, max_num_preds]
          num_predicted_args: [num_sentences]
          num_predicted_preds: [num_sentences]
        """

        max_num_args = self.shape(srl_scores, 1)
        max_num_preds = self.shape(srl_scores, 2)
        num_labels = self.shape(srl_scores, 3)
        args_mask = self.sequence_mask(num_predicted_args, max_num_args)  # [num_sentences, max_num_args]
        preds_mask = self.sequence_mask(num_predicted_preds, max_num_preds)  # [num_sentences, max_num_preds]
        # args_mask.unsqueeze(2)   # [num_sentences, max_num_args, 1]
        # preds_mask.unsqueeze(1)  # [num_sentences, 1, max_num_preds]
        srl_loss_mask = args_mask.unsqueeze(2) & preds_mask.unsqueeze(1) # [num_sentences, max_num_args, max_num_preds]
        logp = torch.nn.functional.log_softmax(srl_scores.view(-1, num_labels)[srl_loss_mask.view(-1)])
        srl_labels_masked = Variable(srl_labels.view(-1)[srl_loss_mask.view(-1)].view(-1, 1))
        if self.use_cuda:
            srl_labels_masked = srl_labels_masked.cuda()
        logpy = torch.gather(logp, 1, srl_labels_masked.long())
        loss = -(logpy).sum()
        # loss = F.cross_entropy(
        #     input=srl_scores.view(-1, num_labels),
        #     target=srl_labels.view(-1).long()
        # )  # [num_sentences * max_num_args * max_num_preds]
        # loss = tf.boolean_mask(loss, srl_loss_mask.view(-1))
        # loss.set_shape([None])
        # loss = tf.reduce_sum(loss)
        return loss
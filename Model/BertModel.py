import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, XLNetModel


class MyBertModel(nn.Module):

    def __init__(self, config):
        super(MyBertModel, self).__init__()
        self.config = config
        self.Roberta = BertModel.from_pretrained('E:\my_nlp\SMP\SMP-4.1\RoBERTa_zh_Large_PyTorch/.')
        self.Roberta_config = self.Roberta.config
        self.bert_embedding = nn.Embedding(self.Roberta_config.vocab_size, self.Roberta_config.hidden_size,
                                           padding_idx=self.Roberta_config.pad_token_id)
        self.activation = nn.Tanh()
        self.pool = nn.Linear(self.config.bert_size, self.config.bert_size)

        self._classification = nn.Sequential(nn.Dropout(p=self.config.dropout),
                                             nn.Linear(self.config.bert_size,
                                             self.config.class_num))

        self._dis_classification = nn.Sequential(nn.Dropout(p=self.config.dropout),
                                                 nn.Linear(self.config.bert_size,
                                                 self.config.dis_class_num))

    def forward(self, input_tensor, distinguish_input=None, vat=None):
        input_id = input_tensor[0]
        if vat is not None:
            bert_emb = self.bert_embedding(input_id)
            bert_emb = bert_emb + vat
        segment = input_tensor[1]
        attention_mask = torch.zeros_like(input_id)

        for idx in range(attention_mask.size(0)):
            for jdx, value in enumerate(input_id[idx]):
                if value > 0:
                    attention_mask[idx][jdx] = 1

        #  return last_hidden_state pooler_output hidden_states
        if vat is not None:
            _, poor_out, hidden_states = self.Roberta(attention_mask=attention_mask, token_type_ids=segment,
                                                      inputs_embeds=bert_emb)
        else:
            _, poor_out, hidden_states = self.Roberta(input_id, attention_mask=attention_mask, token_type_ids=segment)

        last_hidden = hidden_states[-1]
        first_token_tensor = last_hidden[:, 0]
        pooled_output = self.pool(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        logits = self._classification(pooled_output)

        if distinguish_input:
            distinguish_input_id = distinguish_input[0]
            distinguish_segment = distinguish_input[1]
            distinguish_attention_mask = torch.zeros_like(distinguish_input_id)
            for idx in range(distinguish_attention_mask.size(0)):
                for jdx, value in enumerate(input_id[idx]):
                    if value > 0:
                        distinguish_attention_mask[idx][jdx] = 1
            _, poor_out, distinguish_hidden_states = self.Roberta(distinguish_input_id,
                                                                  attention_mask=distinguish_attention_mask,
                                                                  token_type_ids=distinguish_segment)
            return logits, distinguish_hidden_states[-1]

        return logits, last_hidden





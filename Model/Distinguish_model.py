import torch.nn as nn


class DistinguishModel(nn.Module):

    def __init__(self, config):
        super(DistinguishModel, self).__init__()
        self.config = config
        self.activation = nn.Tanh()
        self.pool = nn.Linear(self.config.bert_size, self.config.bert_size)

        self._dis_classification = nn.Sequential(nn.Dropout(p=self.config.dropout),
                                                 nn.Linear(self.config.bert_size,
                                                 self.config.dis_class_num))

    def forward(self, distinguish_input):
        first_token_tensor = distinguish_input[:, 0]
        pooled_output = self.pool(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        logits = self._dis_classification(pooled_output)
        return logits


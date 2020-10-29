import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNLI(nn.Module):

    def __init__(self, config):
        super(MyNLI, self).__init__()
        self.config = config
        self._srl_projection = nn.Linear(400, config.srl_dim, bias=True)
        self._parser_projection = nn.Linear(768, config.parser_dim, bias=True)
        self._classification_srl = nn.Sequential(nn.Dropout(p=self.config.dropout),
                                                 nn.Linear(self.config.bert_size + config.srl_dim,
                                                           self.config.class_num))
        self._classification_parser = nn.Sequential(nn.Dropout(p=self.config.dropout),
                                                    nn.Linear(self.config.bert_size + config.parser_dim,
                                                              self.config.class_num))
        self._classification_sp = nn.Sequential(nn.Dropout(p=self.config.dropout),
                                                nn.Linear(self.config.bert_size + config.parser_dim + config.srl_dim,
                                                          self.config.class_num))
        self._classification = nn.Sequential(nn.Dropout(p=self.config.dropout),
                                             nn.Linear(self.config.bert_size,
                                                       self.config.class_num))

    def forward(self, last_hidden, p_srl_hiddens, h_srl_hiddens, p_parser_output, h_parser_output, p_mask, h_mask):
        if self.config.SRL:
            srl_hiddens = torch.cat([p_srl_hiddens, h_srl_hiddens], dim=1)
            srl_hiddens = self._srl_projection(srl_hiddens)
            class_hidden = torch.cat([last_hidden, srl_hiddens], dim=-1)
            class_hidden = class_hidden.transpose(1, 2)
            class_hidden = F.avg_pool1d(class_hidden, class_hidden.size(2)).squeeze(2)
            logits = self._classification_srl(class_hidden)
        elif self.config.parser:
            p_parser_output = p_parser_output[:, 1:, :]
            h_parser_output = h_parser_output[:, 1:, :]
            parser_hiddens = torch.cat([p_parser_output, h_parser_output], dim=1)
            parser_hiddens = self._parser_projection(parser_hiddens)
            class_hidden = torch.cat([last_hidden, parser_hiddens], dim=-1)
            class_hidden = class_hidden.transpose(1, 2)
            class_hidden = F.avg_pool1d(class_hidden, class_hidden.size(2)).squeeze(2)
            logits = self._classification_parser(class_hidden)
        elif self.config.s_p:
            p_parser_output = p_parser_output[:, 1:, :]
            h_parser_output = h_parser_output[:, 1:, :]
            srl_hiddens = torch.cat([p_srl_hiddens, h_srl_hiddens], dim=1)
            srl_hiddens = self._srl_projection(srl_hiddens)
            parser_hiddens = torch.cat([p_parser_output, h_parser_output], dim=1)
            parser_hiddens = self._parser_projection(parser_hiddens)
            class_hidden = torch.cat([last_hidden, parser_hiddens, srl_hiddens], dim=-1)
            class_hidden = class_hidden.transpose(1, 2)
            class_hidden = F.avg_pool1d(class_hidden, class_hidden.size(2)).squeeze(2)
            logits = self._classification_sp(class_hidden)
        else:
            class_hidden = last_hidden.transpose(1, 2)
            class_hidden = F.avg_pool1d(class_hidden, class_hidden.size(2)).squeeze(2)
            logits = self._classification(class_hidden)

        return logits

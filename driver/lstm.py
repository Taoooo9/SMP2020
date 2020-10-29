import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class lstmModel(nn.Module):
    def __init__(self, alphabet_dic, config, use_cuda):
        super(lstmModel, self).__init__()
        self.config = config
        self.accu_num = alphabet_dic['accu_alphabet'].m_size
        self.use_cuda = use_cuda
        self.use_char = False
        self.word_embed_dim = config.word_embed_dim
        self.hidden_size = config.hidden_size
        self.hidden_layer = config.hidden_layer
        self.bidirectional = config.bidirectional
        self.num_direction = 2 if self.bidirectional else 1

        self.word_embedding = nn.Embedding(alphabet_dic['alphabet'].m_size, config.word_embed_dim)
        nn.init.uniform_(self.word_embedding.weight.data, -0.1, 0.1)

        lstm_input_size = self.word_embed_dim
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.hidden_layer,
                            bidirectional=self.bidirectional,
                            batch_first=True)
        self.dropout = nn.Dropout(config.dropout)

        self.weight = nn.Parameter(torch.randn(self.accu_num, config.hidden_size * 2))

        in_feature = config.hidden_size
        if config.bidirectional:
            in_feature *= 2
        self.linear = nn.Linear(in_feature, 2)

    def forward(self, batch_inst_tensor, length_list):
        word_embed = self.word_embedding(batch_inst_tensor)
        if self.use_char:
            char_embed = self.char_embedding(batch_inst_tensor)
            char_embed_pooled = torch.max(char_embed, dim=2)
            embed = torch.cat((word_embed, char_embed_pooled), dim=1)
        else:
            embed = word_embed
        embed = self.dropout(embed)

        embed_pack = pack_padded_sequence(embed, lengths=length_list, batch_first=True)
        output_pack, _ = self.lstm(embed_pack)
        output = pad_packed_sequence(output_pack, batch_first=True)[0]

        out = torch.transpose(output, 1, 2)  # [batch_size, embed_size, max_length]
        out = torch.tanh(out)
        out = F.max_pool1d(out, out.size(2))  # [batch_size, embed_size, 1]
        batch_size = out.size(0)
        out = out.view(batch_size, 1, self.hidden_size * self.num_direction).expand(batch_size, self.accu_num,
                                                                                    self.hidden_size * self.num_direction)
        out = out * self.weight
        out = out.squeeze(2)  # [1, 400]
        out = self.dropout(out)
        output = self.linear(F.relu(out))

        return output

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GateModel(nn.Module):
    def __init__(self, alphabet_dic, config, use_cuda):
        super(GateModel, self).__init__()
        self.config = config
        self.use_cuda = use_cuda
        self.use_char = False

        self.word_embedding = nn.Embedding(alphabet_dic['alphabet'].m_size, config.word_embed_dim)
        nn.init.uniform_(self.embeddings.weight.data, -0.1, 0.1)

        lstm_input_size = config.word_embed_dim
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=config.hidden_size,
                            num_layers=config.hidden_layer,
                            bidirectional=config.bidirectional,
                            batch_first=True)
        self.dropout = nn.Dropout(config.dropout)

        in_feature = config.hidden_size
        if config.bidirectional:
            in_feature *= 4
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

        output = self.dropout(output)

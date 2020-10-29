from transformers.modeling_bert import *


class InitBert(BertModel):
    def __init__(self, config):
        super(InitBert, self).__init__(config)
        self.config = config

    def forward(self, bert_indices, bert_segments, tune_start_layer):
        attention_mask = torch.zeros_like(bert_indices)
        for idx in range(attention_mask.size(0)):
            for jdx, value in enumerate(bert_indices[idx]):
                if value > 0:
                    attention_mask[idx][jdx] = 1
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.config.num_hidden_layers

        if tune_start_layer == 0:
            embedding_output = self.embeddings(bert_indices, token_type_ids=bert_segments)
            last_output, encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask)
            return encoder_outputs
        else:
            with torch.no_grad():
                embedding_output = self.embeddings(bert_indices, token_type_ids=bert_segments)
                all_hidden_states = (embedding_output,)
                for i in range(tune_start_layer):
                    layer_module = self.encoder.layer[i]
                    layer_outputs = layer_module(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask[i])

                    embedding_output = layer_outputs[0]
                    all_hidden_states = all_hidden_states + (embedding_output,)

            for i in range(tune_start_layer, self.config.num_hidden_layers):
                layer_module = self.encoder.layer[i]
                layer_outputs = layer_module(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask[i])

                embedding_output = layer_outputs[0]
                all_hidden_states = all_hidden_states + (embedding_output,)

            return all_hidden_states
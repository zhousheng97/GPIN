import torch
from torch import nn
import torch.nn.functional as F

from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)

class Q(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, fwd_results):
        txt_emb = fwd_results['txt_emb']
        txt_mask = fwd_results['txt_mask']
        encoder_inputs = txt_emb
        attention_mask = txt_mask

        txt_max_num = txt_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num

        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        fwd_results['txt_emb'] = fwd_results['txt_emb'] + torch.tanh(mmt_seq_output)


class QT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, fwd_results):
        txt_emb = fwd_results['txt_emb']
        txt_mask = fwd_results['txt_mask']
        obj_emb = fwd_results['ocr_mmt_in']
        obj_mask = fwd_results['ocr_mask']

        encoder_inputs = torch.cat(
            [txt_emb, obj_emb],
            dim=1
        )
        attention_mask = torch.cat(
            [txt_mask, obj_mask],
            dim=1
        )

        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num

        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        fwd_results['txt_emb'] = fwd_results['txt_emb'] + torch.tanh(mmt_seq_output[:, txt_begin:txt_end])
        fwd_results['ocr_mmt_in'] = fwd_results['ocr_mmt_in'] + torch.tanh(mmt_seq_output[:, txt_end:])


class QTV(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, fwd_results):
        txt_emb = fwd_results['txt_emb']
        txt_mask = fwd_results['txt_mask']
        obj_emb = fwd_results['obj_mmt_in']
        obj_mask = fwd_results['obj_mask']
        ocr_emb = fwd_results['ocr_mmt_in']
        ocr_mask = fwd_results['ocr_mask']

        encoder_inputs = torch.cat(
            [txt_emb, obj_emb, ocr_emb],
            dim=1
        )
        attention_mask = torch.cat(
            [txt_mask, obj_mask, ocr_mask],
            dim=1
        )

        # offsets of each modality in the joint embedding space
        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num
        ocr_begin = txt_max_num + obj_max_num
        ocr_end = ocr_begin + ocr_max_num

        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        fwd_results['txt_emb'] = fwd_results['txt_emb'] + torch.tanh(mmt_seq_output[:, txt_begin:txt_end])
        fwd_results['obj_mmt_in'] = fwd_results['obj_mmt_in'] + torch.tanh(mmt_seq_output[:, txt_end:ocr_begin])
        fwd_results['ocr_mmt_in'] = fwd_results['ocr_mmt_in'] + torch.tanh(mmt_seq_output[:, ocr_begin:ocr_end])

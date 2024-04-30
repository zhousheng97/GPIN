
import torch
import torch.nn as nn

import torch.nn.functional as F

import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_

import numpy as np
from collections import OrderedDict
import json
import os


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_sim(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class ITM(nn.Module):
    """
    Compute the image-text similarity by SGR, SAF, AVE
    Args: - img_emb: local region embeddings, shape: (batch_size, 100, 1024)
          - cap_emb: local word embeddings, shape: (batch_size, L, 1024)
    Returns:
        - sim_all: final image-text similarities, shape: (batch_size, batch_size).
    """
    def __init__(self, config):
        super(ITM, self).__init__()
        self.sgr_step = config.sgr_step
        self.embed_size = config.hidden_size
        self.sim_dim = config.sim_dim
        self.module_name = config.module_name

        self.sim_tranloc_w = nn.Linear(self.embed_size, self.sim_dim)
        self.sim_tranglo_w = nn.Linear(self.embed_size, self.sim_dim)

        self.sim_eval_w = nn.Linear(self.sim_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def similirity_score(self, obj_emb, q_emb, question_id):

        # local-global alignment construction
        Context_img = SCAN_attention(question_id, obj_emb, q_emb, smooth=9.0) # 句子单词所关注到的视觉特征
        sim_loc = torch.pow(torch.sub(Context_img, obj_emb), 2)
        sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)

        # concat the global and local alignments
        # sim_emb = torch.cat([sim_glo.unsqueeze(1), sim_loc], 1)
        sim_emb = sim_loc

        # compute the final similarity score
        sim_i = self.sigmoid(self.sim_eval_w(sim_emb)).squeeze()

        return sim_i

    def forward(self, fwd_results, sample_list):
        v_emb = fwd_results['obj_mmt_in']
        q_emb = fwd_results['txt_emb']

        question_id  = sample_list['question_id'].detach().cpu().numpy()

        obj_sim = self.similirity_score(v_emb, q_emb, question_id)

        fwd_results['obj_similarity'] = obj_sim


    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def SCAN_attention(question_id, query, context, smooth, eps=1e-8): # question, image
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT) # [b, word_num, 768] X [b, 768, obj_num] = [b, word_num, obj_num]

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, queryL, sourceL)
    attn = F.softmax(attn*smooth, dim=2)

    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)c
    weightedContext = torch.bmm(contextT, attnT) # [b, 768, word_num] X [b, word_num, obj_num] = [b, obj_num, 768] 
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    weightedContext = l2norm(weightedContext, dim=-1)

    return weightedContext
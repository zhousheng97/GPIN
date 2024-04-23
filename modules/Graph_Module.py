import functools
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import json
import os


class Graph_Module(nn.Module):  # mimic the GRU of the GGNN
    def __init__(self, config):
        super().__init__()
        self._build_common_layer(module_name='t1', hidden_size=config.hidden_size)
        self._build_common_layer(module_name='t2', hidden_size=config.hidden_size)
        self._build_common_layer(module_name='t3', hidden_size=config.hidden_size)

        self._build_common_layer(module_name='v1', hidden_size=config.hidden_size)
        self._build_common_layer(module_name='v2', hidden_size=config.hidden_size)
        self._build_common_layer(module_name='v3', hidden_size=config.hidden_size)
        self.ques_norm = nn.LayerNorm(config.hidden_size)
        self.t_node_attn1 = nn.Linear(config.hidden_size, 1)
        self.t_node_attn2 = nn.Linear(config.hidden_size, 1)
        self.t_node_attn3 = nn.Linear(config.hidden_size, 1)
        self.v_node_attn1 = nn.Linear(config.hidden_size, 1)
        self.v_node_attn2 = nn.Linear(config.hidden_size, 1)
        self.v_node_attn3 = nn.Linear(config.hidden_size, 1)

        self.t_lin1 = torch.nn.Linear(config.hidden_size*2, config.hidden_size)
        self.v_lin1 = torch.nn.Linear(config.hidden_size*2, config.hidden_size)

        self.t_graph_drop = nn.Dropout(0.1)
        self.v_graph_drop = nn.Dropout(0.1)
     

    def _build_common_layer(self, module_name, hidden_size, edge_dim=5):
        self_attn = nn.Linear(hidden_size, 1)
        transform_edge = nn.Sequential(nn.Linear(edge_dim, hidden_size // 2), 
                                        nn.ELU(),
                                        nn.Linear(hidden_size // 2, hidden_size),
                                        )

        embeded = nn.Linear(hidden_size, hidden_size)
        edge_attn = nn.Linear(hidden_size, 1)


        setattr(self, '{}_self_attn'.format(module_name), self_attn)
        setattr(self, '{}_transform_edge'.format(module_name), transform_edge)
        setattr(self, '{}_embeded'.format(module_name), embeded)
        setattr(self, '{}_edge_attn'.format(module_name), edge_attn)

        # build_fc_with_layernorm
        feat_layer_0 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                     nn.LayerNorm(hidden_size))
        feat_layer_1 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                     nn.LayerNorm(hidden_size))
        self.qt_drop = nn.Dropout(0.1)
        self.qv_drop = nn.Dropout(0.1)

        setattr(self, '{}_feat_layer_0'.format(module_name), feat_layer_0)
        setattr(self, '{}_feat_layer_1'.format(module_name), feat_layer_1)

    def _calculate_self_attn(self, ques, mask, module_name):
        attn = getattr(self, module_name + '_self_attn')(ques).squeeze(-1)  
        attn = F.softmax(attn, dim=-1)
        attn = attn * mask
        attn = attn / (attn.sum(1, keepdim=True)+ 1e-12)
        question_feature = torch.bmm(attn.unsqueeze(1), ques) 
        return question_feature


    def _build_compute_graph(self, edge_feat, ques, input_mask, ques_mask, module_name):
        batch, num_obj, num_subobj = edge_feat.size()[:3]

        edge_feat = getattr(self, module_name + '_transform_edge')(edge_feat) 

        # conduct edge additive attetnion
        ques_feature = self._calculate_self_attn(ques, ques_mask, module_name)  
        ques_feature_edge = getattr(self, module_name + '_embeded')(ques_feature).unsqueeze(1).expand(-1, num_obj, num_subobj, -1)                                 
        ques_guided_edge_attn = (getattr(self, module_name + '_edge_attn')(torch.tanh(ques_feature_edge * edge_feat))).squeeze(-1)  

        A_edge_attn = F.softmax(ques_guided_edge_attn, -1) 
        A_edge_attn = A_edge_attn * input_mask.unsqueeze(-1)
        A_edge_attn = A_edge_attn / (A_edge_attn.sum(dim=-1, keepdim=True) + 1e-12)

        return A_edge_attn

    def forward(self, sample_list, fwd_results): # sample_list, fwd_results
        v_feat = fwd_results['obj_mmt_in']
        t_feat = fwd_results['ocr_mmt_in']
        t2t_edge = sample_list['ocr_ocr_edge_feat']
        v2v_edge = fwd_results['obj_obj_edge_feat'] # fwd_results['obj_obj_edge_feat']

        q_emb = fwd_results['txt_emb']
        v_mask = fwd_results['obj_mask']
        t_mask = fwd_results['ocr_mask']
        q_mask = fwd_results['txt_mask']

        q_emb = self.ques_norm(q_emb)

        '''Pooling Ratio = 40% (50-20-8-1) (w/o ITM 100-40-16-1)''' 
        obj_layer1 = 20
        obj_layer2 = 8
        obj_layer3 = 3

        token_layer1 = 20
        token_layer2 = 8
        token_layer3 = 3

        ''' Token Graph Pooling Layer-1:'''
        # # aggregation
        t2t_attn1 = self._build_compute_graph(t2t_edge, q_emb, t_mask, q_mask, module_name='t1')
        t2t_mask1 = torch.bmm(t_mask.unsqueeze(-1), t_mask.unsqueeze(1))
        t2t_attn1 = t2t_attn1 * t2t_mask1
        new_t_feat1 = torch.bmm(t2t_attn1.transpose(1, 2), t_feat)  
        t_feat_1 = self.t1_feat_layer_0(t_feat) + self.t1_feat_layer_1(new_t_feat1)

        # softmax
        t_node_attn1 = self.t_node_attn1(t_feat_1).squeeze(-1) 
        t_score_graph1 = F.softmax(t_node_attn1, -1) 

        # pooling  50->20
        t_indices_graph1 = get_indices(t_score_graph1, k = token_layer1)  # pooling ratio = 60% 
        t_mask_graph1 = _get_mask(t_feat_1, token_layer1)
        t_new_node_feat_1 = extract_nodes_from_indices(t_feat_1, t_indices_graph1)
        t_new_edge_feat_1 = extract_edges_from_indices(t2t_edge, t_indices_graph1,  module = 't2t')

        # embedding (for skip connect)
        t_graph1_embedding = torch.cat([torch.mean(t_feat_1, dim=1), torch.max(t_feat_1, dim=1).values], dim=-1)

        # # ''' Obejct Graph Pooling Layer-1:'''
        # # aggregation
        v2v_attn1 = self._build_compute_graph(v2v_edge, q_emb, v_mask, q_mask, module_name='v1')
        v2v_mask1 = torch.bmm(v_mask.unsqueeze(-1), v_mask.unsqueeze(1))
        v2v_attn1 = v2v_attn1 * v2v_mask1
        new_v_feat1 = torch.bmm(v2v_attn1.transpose(1, 2), v_feat)  
        v_feat_1 = self.v1_feat_layer_0(v_feat) + self.v1_feat_layer_1(new_v_feat1)

        # # softmax
        v_node_attn1 = self.v_node_attn1(v_feat_1).squeeze(-1) 
        v_score_graph1 = F.softmax(v_node_attn1, -1) 

        # pooling  50->20
        v_indices_graph1 = get_indices(v_score_graph1, k = obj_layer1)  # pooling ratio = 60% 
        v_mask_graph1 = _get_mask(v_feat_1, obj_layer1)
        v_new_node_feat_1 = extract_nodes_from_indices(v_feat_1, v_indices_graph1)
        v_new_edge_feat_1 = extract_edges_from_indices(v2v_edge, v_indices_graph1,  module = 'v2v')

        # embedding (for skip connect)
        v_graph1_embedding = torch.cat([torch.mean(v_feat_1, dim=1), torch.max(v_feat_1, dim=1).values], dim=-1)

        # ''' Token Graph Pooling Layer-2:'''
        # # aggregation
        t2t_attn2 = self._build_compute_graph(t_new_edge_feat_1, q_emb, t_mask_graph1, q_mask, module_name='t2')
        t2t_mask2 = torch.bmm(t_mask_graph1.unsqueeze(-1), t_mask_graph1.unsqueeze(1))
        t2t_attn2 = t2t_attn2 * t2t_mask2
        new_t_feat2 = torch.bmm(t2t_attn2.transpose(1, 2), t_new_node_feat_1)  
        t_feat_2 = self.t2_feat_layer_0(t_new_node_feat_1) + self.t2_feat_layer_1(new_t_feat2)

        # softmax
        t_node_attn2 = self.t_node_attn2(t_feat_2).squeeze(-1) 
        t_score_graph2 = F.softmax(t_node_attn2, -1) 

        # pooling  20->8
        t_indices_graph2 = get_indices(t_score_graph2, k = token_layer2)  # pooling ratio = 60% 
        t_mask_graph2 = _get_mask(t_feat_2, token_layer2)
        t_new_node_feat_2 = extract_nodes_from_indices(t_feat_2, t_indices_graph2)
        t_new_edge_feat_2 = extract_edges_from_indices(t_new_edge_feat_1, t_indices_graph2,  module = 't2t')

        # embedding (for skip connect)
        t_graph2_embedding = torch.cat([torch.mean(t_feat_2, dim=1), torch.max(t_feat_2, dim=1).values], dim=-1)

        # ''' Obejct Graph Pooling Layer-2:'''
        # aggregation
        v2v_attn2 = self._build_compute_graph(v_new_edge_feat_1, q_emb, v_mask_graph1, q_mask, module_name='v2')
        v2v_mask2 = torch.bmm(v_mask_graph1.unsqueeze(-1), v_mask_graph1.unsqueeze(1))
        v2v_attn2 = v2v_attn2 * v2v_mask2
        v_new_v_feat2 = torch.bmm(v2v_attn2.transpose(1, 2), v_new_node_feat_1)  
        v_feat_2 = self.v2_feat_layer_0(v_new_node_feat_1) + self.v2_feat_layer_1(v_new_v_feat2)

        # softmax
        v_node_attn2 = self.v_node_attn2(v_feat_2).squeeze(-1) 
        v_score_graph2 = F.softmax(v_node_attn2, -1) 

        # # pooling  20->8
        v_indices_graph2 = get_indices(v_score_graph2, k = obj_layer2)  # pooling ratio = 60% 
        v_mask_graph2 = _get_mask(v_feat_2, obj_layer2)
        v_new_node_feat_2 = extract_nodes_from_indices(v_feat_2, v_indices_graph2)
        v_new_edge_feat_2 = extract_edges_from_indices(v_new_edge_feat_1, v_indices_graph2,  module = 'v2v')

        # embedding (for skip connect)
        v_graph2_embedding = torch.cat([torch.mean(v_feat_2, dim=1), torch.max(v_feat_2, dim=1).values], dim=-1)


        # ''' Token Graph Pooling Layer-3:'''
        # # aggregation
        t2t_attn3 = self._build_compute_graph(t_new_edge_feat_2, q_emb, t_mask_graph2, q_mask, module_name='t3')
        t2t_mask3 = torch.bmm(t_mask_graph2.unsqueeze(-1), t_mask_graph2.unsqueeze(1))
        t2t_attn3 = t2t_attn3 * t2t_mask3
        t_new_t_feat3 = torch.bmm(t2t_attn3.transpose(1, 2), t_new_node_feat_2)  
        t_feat_3 = self.t3_feat_layer_0(t_new_node_feat_2) + self.t3_feat_layer_1(t_new_t_feat3)

        # softmax
        t_node_attn3 = self.t_node_attn3(t_feat_3).squeeze(-1) 
        t_score_graph3 = F.softmax(t_node_attn3, -1)

        # pooling  20->8
        t_indices_graph3 = get_indices(t_score_graph3, k = 1)


        # # embedding (for skip connect)
        t_graph3_embedding = torch.cat([torch.mean(t_feat_3, dim=1), torch.max(t_feat_3, dim=1).values], dim=-1)

    
        ''' Obejct Graph Pooling Layer-3:'''
        # aggregation
        v2v_attn3 = self._build_compute_graph(v_new_edge_feat_2, q_emb, v_mask_graph2, q_mask, module_name='v3')
        v2v_mask3 = torch.bmm(v_mask_graph2.unsqueeze(-1), v_mask_graph2.unsqueeze(1))
        v2v_attn3 = v2v_attn3 * v2v_mask3
        v_new_v_feat3 = torch.bmm(v2v_attn3.transpose(1, 2), v_new_node_feat_2)  
        v_feat_3 = self.v3_feat_layer_0(v_new_node_feat_2) + self.v3_feat_layer_1(v_new_v_feat3)

        # softmax
        v_node_attn3 = self.v_node_attn3(v_feat_3).squeeze(-1) 
        v_score_graph3 = F.softmax(v_node_attn3, -1)

        # # pooling  20->8
        v_indices_graph3 = get_indices(v_score_graph3, k = 1)

        # # # embedding (for skip connect)
        v_graph3_embedding = torch.cat([torch.mean(v_feat_3, dim=1), torch.max(v_feat_3, dim=1).values], dim=-1)

        t_mask_graph = _get_mask(t_feat_1, 1)
        v_mask_graph = _get_mask(v_feat_1, 1)

        # # final token graph embedding
        t_graph_embedding = t_graph1_embedding  + t_graph2_embedding #+ t_graph3_embedding
        t_graph_embedding = self.t_lin1(t_graph_embedding).unsqueeze(1)
        t_graph_embedding = self.t_graph_drop(t_graph_embedding)

        # final object graph embedding
        v_graph_embedding = v_graph1_embedding + v_graph2_embedding + v_graph3_embedding
        v_graph_embedding = self.v_lin1(v_graph_embedding).unsqueeze(1)
        v_graph_embedding = self.v_graph_drop(v_graph_embedding)


        # final graph embedding
        graph_embedding = torch.cat([t_graph_embedding, v_graph_embedding], dim=1)
        mask_graph = torch.cat([t_mask_graph, v_mask_graph], dim=1)


        t_feat = self.qt_drop(t_feat_1)
        v_feat = self.qv_drop(v_feat_1)
        fwd_results['graph_embedding'] = graph_embedding
        fwd_results['graph_mask'] = mask_graph
        fwd_results['obj_mmt_in'] = v_feat
        fwd_results['ocr_mmt_in'] = t_feat


def HardTopK_large(k, x):
    topk_results = torch.topk(x, k=k, dim=-1, largest=True, sorted=False)
    indices = topk_results.indices # b, k
    indices = torch.sort(indices, dim=-1).values
    return indices

def get_indices(scores, k):
    indices = HardTopK_large(k, scores)
    return indices

def batched_node_index_select(input, dim, index):
    if len(index.shape) == 1: 
        index = index.view(1,-1)
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1 
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def batched_edge_index_select(input, dim, index, module):
    if len(index.shape) == 1: 
        index = index.view(1,-1)

    # 先按dim=1取特征
    index1 = torch.clone(index)
    for i in range(1, len(input.shape)):
        if i != dim[0]:  
            index1 = index1.unsqueeze(i)
    expanse1 = list(input.shape)
    expanse1[0] = -1
    expanse1[dim[0]] = -1 
    index1 = index1.expand(expanse1)
    feat1 =  torch.gather(input, dim[0], index1)

    # 再按dim=2取特征
    index2 = torch.clone(index)
    for i in range(1, len(feat1.shape)):
        if i != dim[1]:  
            index2 = index2.unsqueeze(i)
    expanse2 = list(feat1.shape)
    expanse2[0] = -1
    expanse2[dim[1]] = -1 
    index2 = index2.expand(expanse2)
    return  torch.gather(feat1, dim[1], index2)


def extract_nodes_from_indices(x, indices):  # [b, N, dim]
    batch_size, _, channels = x.shape
    k = indices.shape[-1]
    patches = x
    patches = batched_node_index_select(patches, 1, indices)
    patches = patches.contiguous().view(batch_size, k, channels)
    return patches

def extract_edges_from_indices(edge, indices, module):  # [b, N, N, dim]
    batch_size, x, y, channels = edge.shape
    k = indices.shape[-1]
    patches = edge
    patches = batched_edge_index_select(patches, [1,2], indices, module = 't2t')
    patches = patches.contiguous().view(batch_size, k, k, channels)
    return patches

def _get_mask(v_feat, nums):
    batch_size = v_feat.size(0)
    topk_mask = torch.ones([batch_size, nums])
    non_pad_mask = topk_mask.to(v_feat.device)
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask



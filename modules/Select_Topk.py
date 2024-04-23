import functools
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from pythia.modules.PerturbedTopKFunction import PerturbedTopKFunction
import einops
from einops import rearrange
import numpy as np
import json
import os


class Select_Topk(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sigma = config.sigma
        self.topk_obj = config.obj_k
        self.topk_ocr = config.ocr_k
        self.topk_type = config.topk_type
    
    def forward(self, fwd_results, sample_list):
        obj_scores = fwd_results['obj_similarity']
        v_feat = fwd_results['obj_mmt_in']

        # ocr_scores = fwd_results['ocr_similarity']
        v2v_edge = sample_list['obj_obj_edge_feat']
        
        indices_obj = get_indices(obj_scores, self.topk_obj)

        #### Screen out object node features ###
        patches = extract_nodes_from_indices(v_feat, indices_obj)
        fwd_results['obj_mmt_in'] = patches
        fwd_results['obj_mask'] = _get_mask(v_feat, self.topk_obj)
        
        patches = extract_edges_from_indices(v2v_edge, indices_obj,  module = 'v2v')
        fwd_results['obj_obj_edge_feat'] = patches


def HardTopK(k, x):
    topk_results = torch.topk(x, k=k, dim=-1, largest=False, sorted=False)
    indices = topk_results.indices # b, k
    indices = torch.sort(indices, dim=-1).values
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

    index1 = torch.clone(index)
    for i in range(1, len(input.shape)):
        if i != dim[0]:  
            index1 = index1.unsqueeze(i)
    expanse1 = list(input.shape)
    expanse1[0] = -1
    expanse1[dim[0]] = -1 
    index1 = index1.expand(expanse1)
    feat1 =  torch.gather(input, dim[0], index1)


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
    if  module == 't2v':
        patches = batched_edge_index_select(patches, 2, indices, module = 't2v')
        patches = patches.contiguous().view(batch_size, x, k, channels)
    elif module == 'v2t':
        patches = batched_edge_index_select(patches, 1, indices, module = 'v2t')
        patches = patches.contiguous().view(batch_size, k, y, channels)
    elif module == 't2t':
        patches = batched_edge_index_select(patches, [1,2], indices, module = 't2t')
        patches = patches.contiguous().view(batch_size, k, k, channels)
    elif module == 'v2v':
        patches = batched_edge_index_select(patches, [1,2], indices, module = 'v2v')
        patches = patches.contiguous().view(batch_size, k, k, channels)
    return patches

def get_indices(scores, k):
    indices = HardTopK(k, scores)
    return indices

def _get_mask(v_feat, nums):
    batch_size = v_feat.size(0)
    topk_mask = torch.ones([batch_size, nums])
    non_pad_mask = topk_mask.to(v_feat.device)
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask

    #### Screen out object edge features ###
    # patches = self.extract_edges_from_indices(t2v_edge, indices_obj, module = 't2v')
    # fwd_results['ocr_obj_edge_feat'] = patches

    

    #### Screen out OCR node features ###
    # indices_ocr = self.get_indices(ocr_scores, self.topk_ocr)

    # patches = self.extract_nodes_from_indices(t_feat, indices_ocr)
    # fwd_results['topk_ocr_mmt_in'] = patches
    # fwd_results['topk_ocr_mask'] = self._get_mask(t_feat, self.topk_ocr)

    # # #### Screen out object edge features ###
    # patches = self.extract_edges_from_indices(t2t_edge, indices_ocr, module = 't2t')
    # fwd_results['topk_ocr_ocr_edge_feat'] = patches

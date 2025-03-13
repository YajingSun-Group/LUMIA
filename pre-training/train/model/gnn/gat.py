# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Graph Attention Networks
#
# pylint: disable= no-member, arguments-differ, invalid-name

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling
from torch_scatter import scatter_mean
from .base_gnn import BaseGNN

__all__ = ["GAT"]

class GATLayer(nn.Module):
    """Single GAT layer"""
    def __init__(self, in_feats, out_feats, num_heads=4, feat_drop=0.0, 
                 attn_drop=0.0, residual=True, batch_norm=True, activation=None):
        super(GATLayer, self).__init__()
        
        self.gat_conv = GATConv(
            in_feats=in_feats,
            out_feats=out_feats // num_heads,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            residual=False,
            activation=None
        )
        
        self.activation = activation
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)
            
        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feats)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """重新初始化模型参数"""
        self.gat_conv.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()
            
    def forward(self, g, feats):
        """更新节点特征"""
        new_feats = self.gat_conv(g, feats).flatten(1)  # 合并多头注意力
        
        if self.residual:
            res_feats = self.res_connection(feats)
            new_feats = new_feats + res_feats
            
        if self.bn is not None:
            new_feats = self.bn(new_feats)
            
        if self.activation is not None:
            new_feats = self.activation(new_feats)
            
        return new_feats


class GAT(BaseGNN):
    """GAT model for contrastive learning"""
    def __init__(self, ffn_hidden_feats, gat_hidden_feats, num_heads=4,
                 feat_drop=0.1, attn_drop=0.1, ffn_dropout=0.1, 
                 classification=False):
        super(GAT, self).__init__(
            gnn_out_feats=gat_hidden_feats[-1],
            ffn_hidden_feats=ffn_hidden_feats,
            ffn_dropout=ffn_dropout,
            classification=classification
        )
        
        # 构建GAT层
        self.gnn_layers = nn.ModuleList()
        in_feats = gat_hidden_feats[0]
        for out_feats in gat_hidden_feats[1:]:
            self.gnn_layers.append(
                GATLayer(
                    in_feats=in_feats,
                    out_feats=out_feats,
                    num_heads=num_heads,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    residual=True,
                    batch_norm=True,
                    activation=F.relu
                )
            )
            in_feats = out_feats

    def forward(self, g):
        """前向传播，返回节点特征、全局特征和子结构特征"""
        # 获取输入特征
        node_feats = g.ndata.pop('node').float()
        
        # GAT层更新节点特征
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
            
        # 计算全局特征
        graph_feats = self.readout(g, node_feats)
        feats_global = self.feat_lin(graph_feats)
        out_global = self.out_lin(feats_global)
        
        # 计算子结构特征
        motif_batch = g.motif_batch.to(g.device)
        h_sub = scatter_mean(node_feats, motif_batch, dim=0)
        h_sub = h_sub[1:,:]  # 去除第一个虚拟节点
        feats_sub = self.feat_lin(h_sub)
        out_sub = self.out_lin(feats_sub)
        
        return graph_feats, out_global, out_sub

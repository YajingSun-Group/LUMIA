# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# MPNN
# pylint: disable= no-member, arguments-differ, invalid-name

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import NNConv
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling
from torch_scatter import scatter_mean
from .base_gnn import BaseGNN

__all__ = ['MPNN']

class MPNNLayer(nn.Module):
    """Single MPNN layer"""
    def __init__(self, node_feats, edge_feats, out_feats, 
                 batch_norm=True, activation=None, residual=True, dropout=0.0):
        super(MPNNLayer, self).__init__()
        
        # 边网络
        self.edge_network = nn.Sequential(
            nn.Linear(edge_feats, out_feats * node_feats),
            nn.ReLU(),
            nn.Linear(out_feats * node_feats, out_feats * out_feats)
        )
        
        # 消息传递层
        self.conv = NNConv(
            in_feats=node_feats,
            out_feats=out_feats,
            edge_func=self.edge_network,
            aggregator_type='sum'
        )
        
        # GRU更新
        self.gru = nn.GRU(out_feats, out_feats)
        
        self.activation = activation
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(node_feats, out_feats)
            
        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feats)
            
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
        
    def reset_parameters(self):
        """重新初始化模型参数"""
        for layer in self.edge_network:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.conv.reset_parameters()
        self.gru.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()
            
    def forward(self, g, node_feats, edge_feats):
        """更新节点特征"""
        # 消息传递
        new_feats = self.conv(g, node_feats, edge_feats)
        
        # GRU更新
        new_feats = F.relu(new_feats)
        new_feats, _ = self.gru(new_feats.unsqueeze(0))
        new_feats = new_feats.squeeze(0)
        
        # 残差连接
        if self.residual:
            res_feats = self.res_connection(node_feats)
            new_feats = new_feats + res_feats
            
        # Dropout
        new_feats = self.dropout(new_feats)
        
        # Batch Normalization
        if self.bn is not None:
            new_feats = self.bn(new_feats)
            
        if self.activation is not None:
            new_feats = self.activation(new_feats)
            
        return new_feats


class MPNN(BaseGNN):
    """MPNN model for contrastive learning"""
    def __init__(self, ffn_hidden_feats, mpnn_node_feats, mpnn_hidden_feats,
                 edge_hidden_feats=128, dropout=0.1, ffn_dropout=0.1, 
                 classification=False):
        super(MPNN, self).__init__(
            gnn_out_feats=mpnn_hidden_feats[-1],
            ffn_hidden_feats=ffn_hidden_feats,
            ffn_dropout=ffn_dropout,
            classification=classification
        )
        
        # 构建MPNN层
        self.gnn_layers = nn.ModuleList()
        in_node_feats = mpnn_node_feats
        for out_feats in mpnn_hidden_feats:
            self.gnn_layers.append(
                MPNNLayer(
                    node_feats=in_node_feats,
                    edge_feats=edge_hidden_feats,
                    out_feats=out_feats,
                    batch_norm=True,
                    activation=F.relu,
                    residual=True,
                    dropout=dropout
                )
            )
            in_node_feats = out_feats

    def forward(self, g):
        """前向传播，返回节点特征、全局特征和子结构特征"""
        # 获取输入特征
        node_feats = g.ndata.pop('node').float()
        edge_feats = g.edata.pop('edge').float()
        
        # MPNN层更新节点特征
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats, edge_feats)
            
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

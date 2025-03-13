# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Graph Convolutional Networks
# pylint: disable= no-member, arguments-differ, invalid-name

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling
from torch_scatter import scatter_mean

__all__ = ["GCN"]

# pylint: disable=W0221, C0103
class GCNLayer(nn.Module):
    r"""Single GCN layer from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    out_feats : int
        Number of output node features.
    gnn_norm : str
        The message passing normalizer, which can be `'right'`, `'both'` or `'none'`. The
        `'right'` normalizer divides the aggregated messages by each node's in-degree.
        The `'both'` normalizer corresponds to the symmetric adjacency normalization in
        the original GCN paper. The `'none'` normalizer simply sums the messages.
        Default to be 'none'.
    activation : activation function
        Default to be None.
    residual : bool
        Whether to use residual connection, default to be True.
    batchnorm : bool
        Whether to use batch normalization on the output,
        default to be True.
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    allow_zero_in_degree: bool
        Whether to allow zero in degree nodes in graph. Defaults to False.
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        gnn_norm="none",
        activation=None,
        residual=True,
        batchnorm=True,
        dropout=0.0,
        allow_zero_in_degree=False,
    ):
        super(GCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv = GraphConv(
            in_feats=in_feats,
            out_feats=out_feats,
            norm=gnn_norm,
            activation=activation,
            allow_zero_in_degree=allow_zero_in_degree,
        )
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.graph_conv.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, g, feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match in_feats in initialization

        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output node feature size, which must match out_feats in initialization
        """
        new_feats = self.graph_conv(g, feats)
        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats


class BaseGNN(nn.Module):
    """Base GNN class for contrastive learning"""
    def __init__(self, gnn_out_feats, ffn_hidden_feats, ffn_dropout=0.25, 
                 classification=False, readout='mean'):
        super(BaseGNN, self).__init__()
        
        self.classification = classification
        self.gnn_layers = nn.ModuleList()
        
        # 选择readout方法
        if readout == 'mean':
            self.readout = AvgPooling()
        elif readout == 'max':
            self.readout = MaxPooling()
        else:
            self.readout = SumPooling()
            
        # 特征转换层
        self.feat_lin = nn.Linear(gnn_out_feats, ffn_hidden_feats)
        self.out_lin = nn.Sequential(
            nn.Linear(ffn_hidden_feats, ffn_hidden_feats),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_hidden_feats, ffn_hidden_feats//2)
        )

    def forward(self, g):
        raise NotImplementedError


class GCN(BaseGNN):
    """GCN model for contrastive learning"""
    def __init__(self, ffn_hidden_feats, gcn_hidden_feats, 
                 gcn_dropout=0.25, ffn_dropout=0.25, classification=False):
        super(GCN, self).__init__(
            gnn_out_feats=gcn_hidden_feats[-1],
            ffn_hidden_feats=ffn_hidden_feats,
            ffn_dropout=ffn_dropout,
            classification=classification
        )
        
        # 构建GCN层
        in_feats = gcn_hidden_feats[0]  # 输入特征维度
        for out_feats in gcn_hidden_feats:
            self.gnn_layers.append(
                GCNLayer(
                    in_feats=in_feats,
                    out_feats=out_feats,
                    gnn_norm='both',
                    activation=F.relu,
                    residual=True,
                    batchnorm=True,
                    dropout=gcn_dropout
                )
            )
            in_feats = out_feats

    def forward(self, g):
        """前向传播，返回节点特征、全局特征和子结构特征"""
        # 获取输入特征
        node_feats = g.ndata.pop('node').float()
        
        # GNN层更新节点特征
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

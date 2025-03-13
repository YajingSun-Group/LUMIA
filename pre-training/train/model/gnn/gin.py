# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Graph Isomorphism Networks.
# pylint: disable= no-member, arguments-differ, invalid-name

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling
from torch_scatter import scatter_mean

__all__ = ['GIN']

# pylint: disable=W0221, C0103
class GINLayer(nn.Module):
    """Single Layer GIN for updating node features"""
    def __init__(self, num_edge_emb_list, emb_dim, batch_norm=True, 
                 activation=None, dropout=0.0, residual=True):
        super(GINLayer, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        
        # 边特征嵌入
        self.edge_embeddings = nn.ModuleList()
        for num_emb in num_edge_emb_list:
            emb_module = nn.Embedding(num_emb, emb_dim)
            self.edge_embeddings.append(emb_module)

        # 添加残差连接
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(emb_dim, emb_dim)
            
        # Batch Normalization
        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm1d(emb_dim)
            
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        """重新初始化模型参数"""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        
        for emb in self.edge_embeddings:
            nn.init.xavier_uniform_(emb.weight.data)
            
        if self.residual:
            self.res_connection.reset_parameters()
            
        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, g, node_feats, categorical_edge_feats):
        """更新节点特征"""
        # 边特征嵌入
        edge_embeds = []
        for i, feats in enumerate(categorical_edge_feats):
            edge_embeds.append(self.edge_embeddings[i](feats))
        edge_embeds = torch.stack(edge_embeds, dim=0).sum(0)
        
        # 消息传递
        g = g.local_var()
        g.ndata['feat'] = node_feats
        g.edata['feat'] = edge_embeds
        g.update_all(fn.u_add_e('feat', 'feat', 'm'), fn.sum('m', 'feat'))
        
        # MLP变换
        new_feats = self.mlp(g.ndata.pop('feat'))
        
        # 残差连接
        if self.residual:
            res_feats = self.res_connection(node_feats)
            if self.activation is not None:
                res_feats = self.activation(res_feats)
            new_feats = new_feats + res_feats
            
        # Dropout
        new_feats = self.dropout(new_feats)
        
        # Batch Normalization
        if self.bn is not None:
            new_feats = self.bn(new_feats)
            
        if self.activation is not None:
            new_feats = self.activation(new_feats)
            
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


class GIN(BaseGNN):
    """GIN model for contrastive learning"""
    def __init__(self, ffn_hidden_feats, num_node_emb_list, num_edge_emb_list,
                 gin_hidden_feats, num_layers=5, dropout=0.5, JK='last',
                 classification=False):
        super(GIN, self).__init__(
            gnn_out_feats=gin_hidden_feats,
            ffn_hidden_feats=ffn_hidden_feats,
            ffn_dropout=dropout,
            classification=classification
        )
        
        self.JK = JK
        self.num_layers = num_layers
        
        # 节点特征嵌入
        self.node_embeddings = nn.ModuleList()
        for num_emb in num_node_emb_list:
            emb_module = nn.Embedding(num_emb, gin_hidden_feats)
            self.node_embeddings.append(emb_module)
        
        # GIN层
        for layer in range(num_layers):
            if layer == num_layers - 1:
                self.gnn_layers.append(
                    GINLayer(
                        num_edge_emb_list=num_edge_emb_list,
                        emb_dim=gin_hidden_feats,
                        batch_norm=True,
                        dropout=dropout,
                        residual=True
                    )
                )
            else:
                self.gnn_layers.append(
                    GINLayer(
                        num_edge_emb_list=num_edge_emb_list,
                        emb_dim=gin_hidden_feats,
                        batch_norm=True,
                        activation=F.relu,
                        dropout=dropout,
                        residual=True
                    )
                )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """重新初始化模型参数"""
        for emb in self.node_embeddings:
            nn.init.xavier_uniform_(emb.weight.data)
        for layer in self.gnn_layers:
            layer.reset_parameters()

    def forward(self, g):
        """前向传播，返回节点特征、全局特征和子结构特征"""
        # 获取输入特征并确保类型正确
        categorical_node_feats = g.ndata.pop('node')
        categorical_edge_feats = g.edata.pop('edge')
        
        # 确保输入类型为 Long
        if isinstance(categorical_node_feats, torch.Tensor):
            categorical_node_feats = categorical_node_feats.long()
        else:
            categorical_node_feats = [feat.long() for feat in categorical_node_feats]
        
        if isinstance(categorical_edge_feats, torch.Tensor):
            categorical_edge_feats = categorical_edge_feats.long()
        else:
            categorical_edge_feats = [feat.long() for feat in categorical_edge_feats]
        
        # 节点特征嵌入
        node_embeds = []
        for i, feats in enumerate(categorical_node_feats):
            node_embeds.append(self.node_embeddings[i](feats))
        node_feats = torch.stack(node_embeds, dim=0).sum(0)
        
        # 存储所有层的节点特征
        all_layer_node_feats = [node_feats]
        
        # GIN层更新节点特征
        for layer in range(self.num_layers):
            node_feats = self.gnn_layers[layer](g, all_layer_node_feats[-1], categorical_edge_feats)
            all_layer_node_feats.append(node_feats)
        
        # 根据JK方式选择最终节点特征
        if self.JK == 'concat':
            node_feats = torch.cat(all_layer_node_feats, dim=1)
        elif self.JK == 'last':
            node_feats = all_layer_node_feats[-1]
        elif self.JK == 'max':
            all_layer_node_feats = [h.unsqueeze(0) for h in all_layer_node_feats]
            node_feats = torch.max(torch.cat(all_layer_node_feats, dim=0), dim=0)[0]
        elif self.JK == 'sum':
            all_layer_node_feats = [h.unsqueeze(0) for h in all_layer_node_feats]
            node_feats = torch.sum(torch.cat(all_layer_node_feats, dim=0), dim=0)
            
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

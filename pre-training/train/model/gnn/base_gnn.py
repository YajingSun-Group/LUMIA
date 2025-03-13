import torch.nn as nn
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling

class BaseGNN(nn.Module):
    """Base GNN class for contrastive learning"""
    def __init__(self, gnn_out_feats, ffn_hidden_feats, ffn_dropout=0.1, 
                 classification=False, readout='mean'):
        super(BaseGNN, self).__init__()
        
        self.classification = classification
        
        # 选择readout方法
        if readout == 'mean':
            self.readout = AvgPooling()
        elif readout == 'max':
            self.readout = MaxPooling()
        else:
            self.readout = SumPooling()
            
        # 特征转换层
        self.feat_lin = nn.Sequential(
            nn.Linear(gnn_out_feats, ffn_hidden_feats),
            nn.ReLU(inplace=True),
            nn.Dropout(ffn_dropout)
        )
        
        # 输出层
        self.out_lin = nn.Sequential(
            nn.Linear(ffn_hidden_feats, ffn_hidden_feats),
            nn.ReLU(inplace=True),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_hidden_feats, ffn_hidden_feats//2)
        )
        
        # GNN层列表，由子类实现
        self.gnn_layers = nn.ModuleList() 
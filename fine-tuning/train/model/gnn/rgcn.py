import torch.nn.functional as F
from dgl.nn.pytorch.conv import RelGraphConv
from torch import nn
import torch as th
from dgl.readout import sum_nodes
import torch
from ..readout import WeightAndSum


class RGCNLayer(nn.Module):
    """Single layer RGCN for updating node features
    Parameters
    ----------
    in_feats : int
        Number of input atom features
    out_feats : int
        Number of output atom features
    num_rels: int
        Number of bond type
    activation : activation function
        Default to be ReLU
    loop: bool:
        Whether to use self loop
        Default to be False
    residual : bool
        Whether to use residual connection, default to be True
    batchnorm : bool
        Whether to use batch normalization on the output,
        default to be True
    rgcn_drop_out : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    """
    
    def __init__(self, in_feats, out_feats, num_rels=65, activation=F.relu, loop=False,
                 residual=True, batchnorm=True, rgcn_drop_out=0.5):
        super(RGCNLayer, self).__init__()
        
        self.activation = activation
        self.graph_conv_layer = RelGraphConv(in_feats, out_feats, num_rels=num_rels, regularizer='basis',
                                             num_bases=None, bias=True, activation=activation,
                                             self_loop=loop, dropout=rgcn_drop_out)
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)
        
        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)
    
    def forward(self, bg, node_feats, etype, norm=None):
        """Update atom representations
        Parameters
        ----------
        bg : BatchedDGLGraph
            Batched DGLGraphs for processing multiple molecules in parallel
        node_feats : FloatTensor of shape (N, M1)
            * N is the total number of atoms in the batched graph
            * M1 is the input atom feature size, must match in_feats in initialization
        etype: int
            bond type
        norm: th.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`
        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output atom feature size, must match out_feats in initialization
        """
        new_feats = self.graph_conv_layer(bg, node_feats, etype, norm)
        if self.residual:
            res_feats = self.activation(self.res_connection(node_feats))
            new_feats = new_feats + res_feats
        if self.bn:
            new_feats = self.bn_layer(new_feats)
        del res_feats
        th.cuda.empty_cache()
        return new_feats


class BaseGNN(nn.Module):
    """HRGCN based predictor for multitask prediction on molecular graphs
    We assume each task requires to perform a binary classification.
    Parameters
    ----------
    gnn_out_feats : int
        Number of atom representation features after using GNN
    len_descriptors : int
        length of descriptors
    hyperbolic: str
        Riemannian Manifolds. Defalt: 'Poincare'
    rgcn_drop_out: float
        dropout rate for HRGCN layer
    n_tasks : int
        Number of prediction tasks
    classifier_hidden_feats : int
        Number of molecular graph features in hidden layers of the MLP Classifier
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    return_weight: bool
        Wether to return atom weight defalt=False
    """
    
    def __init__(self, gnn_rgcn_out_feats, ffn_hidden_feats, molecular_features_dim, ffn_dropout=0.25, classification=True, num_fc_layers=3, n_tasks=1, embed_molecular_features=False, device=None):
        super(BaseGNN, self).__init__()
        self.classification = classification
        self.rgcn_gnn_layers = nn.ModuleList()
        self.readout = WeightAndSum(gnn_rgcn_out_feats)
        
        if embed_molecular_features:
            self.molcular_features_embedding = self._make_molecular_features_embedding(molecular_features_dim)
            self.project_layers = self._make_fc_layers(num_fc_layers, ffn_dropout, gnn_rgcn_out_feats+100, ffn_hidden_feats)
        else:
            self.project_layers = self._make_fc_layers(num_fc_layers, ffn_dropout, gnn_rgcn_out_feats, ffn_hidden_feats)
        
        self.predict = self.output_layer(ffn_hidden_feats, n_tasks)
        
        self.embed_molecular_features = embed_molecular_features
        self.device = device
        
    def forward(self, g):
        # 将分子特征移动到正确的设备
        if hasattr(g, 'molecular_features'):
            # g.molecular_features = None
            g.molecular_features = g.molecular_features.to(self.device)
        else:
            g.molecular_features = None
        
        rgcn_node_feats = g.ndata.pop('node').float()
        rgcn_edge_feats = g.edata.pop('edge').long()
        smask_feats = g.ndata.pop('smask').unsqueeze(dim=1).float()
        molecular_features = g.molecular_features
        
        # Update atom features with GNNs
        for rgcn_gnn in self.rgcn_gnn_layers:
            rgcn_node_feats = rgcn_gnn(g, rgcn_node_feats, rgcn_edge_feats)
        # Compute molecule features from atom features and bond features
        graph_feats, weight = self.readout(g, rgcn_node_feats, smask_feats)
        if self.embed_molecular_features:
            molecular_features = self.molcular_features_embedding(molecular_features)
            graph_feats = torch.cat([graph_feats, molecular_features], dim=1)
        else:
            graph_feats = graph_feats

        h = self.project_layers(graph_feats)
        out = self.predict(h)

        del rgcn_node_feats, rgcn_edge_feats, smask_feats, molecular_features, graph_feats, h
        torch.cuda.empty_cache()
        
        # return out,  weight, (rgcn_node_feats,graph_feats, h)
        return out,  weight, None
    
    def _make_molecular_features_embedding(self, molecular_features_dim):
        feat_hidden_dim = min(molecular_features_dim, 100)
        return nn.Sequential(
            nn.Linear(molecular_features_dim, feat_hidden_dim),
            nn.PReLU(),
            nn.BatchNorm1d(feat_hidden_dim)
        )
    
    def _make_fc_layers(self, num_layers, dropout, in_feats, hidden_feats):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(in_feats if i == 0 else hidden_feats, hidden_feats))
            layers.append(nn.PReLU())
            layers.append(nn.BatchNorm1d(hidden_feats))
        return nn.Sequential(*layers)
    
    def output_layer(self, hidden_feats, out_feats):
        return nn.Sequential(
                nn.Linear(hidden_feats, out_feats)
                )


class RGCN(BaseGNN):
    """HRGCN based predictor for multitask prediction on molecular graphs
    We assume each task requires to perform a binary classification.
    Parameters
    ----------
    in_feats : int
        Number of input atom features
    Rgcn_hidden_feats : list of int
        rgcn_hidden_feats[i] gives the number of output atom features
        in the i+1-th HRGCN layer
    n_tasks : int
        Number of prediction tasks
    len_descriptors : int
        length of descriptors
    return_weight : bool
        Wether to return weight
    classifier_hidden_feats : int
        Number of molecular graph features in hidden layers of the MLP Classifier
    is_descriptor: bool
        Wether to use descriptor
    loop : bool
        Wether to use self loop
    gnn_drop_rate : float
        The probability for dropout of HRGCN layer. Default to be 0.5
    dropout : float
        The probability for dropout of MLP layer. Default to be 0.
    """
    
    def __init__(self, ffn_hidden_feats, rgcn_node_feats, rgcn_hidden_feats, molecular_features_dim, rgcn_drop_out=0.1, ffn_dropout=0.1,
                 classification=False,n_tasks=1, embed_molecular_features=False, device=None):
        super(RGCN, self).__init__(gnn_rgcn_out_feats=rgcn_hidden_feats[-1],
                                    ffn_hidden_feats=ffn_hidden_feats,
                                    ffn_dropout=ffn_dropout,
                                    classification=classification,
                                    n_tasks=n_tasks,
                                    molecular_features_dim=molecular_features_dim,
                                    embed_molecular_features=embed_molecular_features,
                                    device=device)
                                       
        for i in range(len(rgcn_hidden_feats)):
            rgcn_out_feats = rgcn_hidden_feats[i]
            self.rgcn_gnn_layers.append(RGCNLayer(rgcn_node_feats, rgcn_out_feats, loop=True,
                                                  rgcn_drop_out=rgcn_drop_out))
            rgcn_node_feats = rgcn_out_feats
        
    def load_my_state_dict(self, state_dict, logger):
        own_state = self.state_dict()  # 获取当前模型的状态字典
        for name, param in state_dict.items():
            if not name.startswith('rgcn_gnn_layers'):
                logger.info(f'SKIPPED: {name} - Not an RGCN layer parameter.')
                continue
            if name not in own_state:
                logger.info(f'NOT LOADED: {name} - Parameter not found in current model.')
                continue
            
            # 确保加载的权重和模型中的权重具有相同的形状
            if own_state[name].shape != param.shape:
                logger.warning(f'SKIPPED: {name} - Shape mismatch: '
                            f'model layer shape {own_state[name].shape}, '
                            f'checkpoint layer shape {param.shape}')
                continue

            try:
                # 如果 param 是 nn.parameter.Parameter，获取其数据
                if isinstance(param, nn.parameter.Parameter):
                    param = param.data
                own_state[name].copy_(param)  # 复制预训练权重到模型中
                logger.info(f'LOADED: {name} - Successfully loaded.')
            except Exception as e:
                logger.error(f'ERROR LOADING: {name} - {str(e)}')

        logger.info('Finished loading state_dict.')
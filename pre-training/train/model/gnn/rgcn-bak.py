
import torch.nn.functional as F
from dgl.nn.pytorch.conv import RelGraphConv
from torch import nn
import torch as th
from dgl.readout import sum_nodes
import torch

class SubstructureProcessor(nn.Module):
    def __init__(self, readout, feat_lin, out_lin):
        super(SubstructureProcessor, self).__init__()
        self.readout = readout
        self.feat_lin = feat_lin
        self.out_lin = out_lin
    
    def forward(self, rgcn_bg, rgcn_node_feats):
        """Compute substructure (motif) features from a batched DGL graph."""
        h_subs = []
        unique_motifs = rgcn_bg.motif_batch.unique()
        
        for motif in unique_motifs:
            if motif == 0:  # Skip motif 0 as it usually represents global nodes
                continue
            
            mask = (rgcn_bg.motif_batch == motif)
            sub_g = rgcn_bg.subgraph(mask)
            smask_feats = rgcn_bg.ndata['smask_full'].unsqueeze(dim=1).float()
            sub_smask_feats = smask_feats[mask]
            sub_node_feats = rgcn_node_feats[mask]
            
            graph_feats_sub, _ = self.readout(sub_g, sub_node_feats, sub_smask_feats)
            h_subs.append(graph_feats_sub)
        
        if h_subs:  # Ensure the list is not empty
            h_subs = torch.stack(h_subs, dim=0).squeeze(dim=1)
            feats_sub = self.feat_lin(h_subs)
            out_sub = self.out_lin(feats_sub)
        else:
            out_sub = None
        
        return out_sub
    
class WeightAndSum(nn.Module):
    """Compute importance weights for atoms and perform a weighted sum.

    Parameters
    ----------
    in_feats : int
        Input atom feature size
    """
    
    def __init__(self, in_feats):
        super(WeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )
    
    def forward(self, g, feats, smask):
        """Compute molecule representations out of atom representations

        Parameters
        ----------
        g : DGLGraph
            DGLGraph with batch size B for processing multiple molecules in parallel
        feats : FloatTensor of shape (N, self.in_feats)
            Representations for all atoms in the molecules
            * N is the total number of atoms in all molecules
        smask: substructure mask, atom node for 0, substructure node for 1.

        Returns
        -------
        FloatTensor of shape (B, self.in_feats)
            Representations for B molecules
        """
        with g.local_scope():
            g.ndata['h'] = feats
            weight = self.atom_weighting(g.ndata['h']) * smask
            g.ndata['w'] = weight
            h_g_sum = sum_nodes(g, 'h', 'w')
        return h_g_sum, weight


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
    hyperbolic: str
        Riemannian Manifolds. Defalt: 'Poincare'
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
    
    def __init__(self, gnn_rgcn_out_feats, ffn_hidden_feats, ffn_dropout=0.25, classification=False):
        super(BaseGNN, self).__init__()
        self.classification = classification
        self.rgcn_gnn_layers = nn.ModuleList()
        self.readout = WeightAndSum(gnn_rgcn_out_feats)
        self.feat_lin = nn.Linear(gnn_rgcn_out_feats, ffn_hidden_feats)
        self.out_lin = nn.Sequential(
            nn.Linear(ffn_hidden_feats, ffn_hidden_feats),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_hidden_feats, ffn_hidden_feats//2)
        )
        self.substructure_processor = SubstructureProcessor(self.readout, self.feat_lin, self.out_lin)
    
    def forward(self, rgcn_bg):
        """Multi-task prediction for a batch of molecules
        """
        rgcn_node_feats = rgcn_bg.ndata.pop('node').float()
        rgcn_edge_feats = rgcn_bg.edata.pop('edge').long()
        smask_feats = rgcn_bg.ndata.pop('smask').unsqueeze(dim=1).float()
        
        # Update atom features with GNNs
        for rgcn_gnn in self.rgcn_gnn_layers:
            rgcn_node_feats = rgcn_gnn(rgcn_bg, rgcn_node_feats, rgcn_edge_feats)
        # Compute molecule features from atom features and bond features
        graph_feats, weight = self.readout(rgcn_bg, rgcn_node_feats, smask_feats)
        feats_global = self.feat_lin(graph_feats)
        out_global = self.out_lin(feats_global)
        
        
        # motif_batch
        # graph_feats_sub,weight_sub = self.readout(rgcn_bg.motif_batch, rgcn_node_feats, smask_feats)[1:,:]
        # feats_sub = self.feat_lin(graph_feats_sub)
        # out_sub = self.out_lin(feats_sub)
        
        out_sub = self.substructure_processor(rgcn_bg, rgcn_node_feats)

        return graph_feats,out_global,out_sub
    
    # def fc_layer(self, dropout, in_feats, hidden_feats):
    #     return nn.Sequential(
    #             nn.Dropout(dropout),
    #             nn.Linear(in_feats, hidden_feats),
    #             nn.ReLU(),
    #             nn.BatchNorm1d(hidden_feats)
    #             )

    # def output_layer(self, hidden_feats, out_feats):
    #     return nn.Sequential(
    #             nn.Linear(hidden_feats, out_feats)
    #             )


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
    
    def __init__(self, ffn_hidden_feats, rgcn_node_feats, rgcn_hidden_feats, rgcn_drop_out=0.25, ffn_dropout=0.25,
                 classification=False):
        super(RGCN, self).__init__(gnn_rgcn_out_feats=rgcn_hidden_feats[-1],
                                       ffn_hidden_feats=ffn_hidden_feats,
                                       ffn_dropout=ffn_dropout,
                                       classification=classification,
                                       )
        for i in range(len(rgcn_hidden_feats)):
            rgcn_out_feats = rgcn_hidden_feats[i]
            self.rgcn_gnn_layers.append(RGCNLayer(rgcn_node_feats, rgcn_out_feats, loop=True,
                                                  rgcn_drop_out=rgcn_drop_out))
            rgcn_node_feats = rgcn_out_feats
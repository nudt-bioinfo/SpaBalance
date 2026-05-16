import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from math import sqrt

class Encoder_overall(Module):
      
    """\
    Overall encoder.

    Parameters
    ----------
    dim_in_feat_omics1 : int
        Dimension of input features for omics1.
    dim_in_feat_omics2 : int
        Dimension of input features for omics2. 
    dim_out_feat_omics1 : int
        Dimension of latent representation for omics1.
    dim_out_feat_omics2 : int
        Dimension of latent representation for omics2, which is the same as omics1.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    results: a dictionary including representations.

    """
     
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2, dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dropout = dropout
        self.act = act

        self.log_var1 = nn.Parameter(torch.zeros(1))
        self.log_var2 = nn.Parameter(torch.zeros(1))
        
        self.encoder_omics1 = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.encoder_omics2 = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)
        
        args = {
            'dim_h': 64,
            'embed_dropout_prob': 0.1,
            'num_layer': 3,
            'activation': 'relu',
            'use_ln': True
        }
        args = type('Args', (object,), args)()
        self.atten_omics1 =  MLP_Dropout(in_channels=64, args=args)
        self.atten_omics2 =  MLP_Dropout(in_channels=64, args=args)
        self.atten_cross = Multi_CrossAttention(self.dim_out_feat_omics1, self.dim_out_feat_omics2, 8)   
        
    def forward(self, features_omics1, features_omics2, features_omics1a, features_omics2a, adj_spatial_omics1, adj_feature_omics1, adj_spatial_omics2, adj_feature_omics2):
        
        # graph1
        hidden_spatial_omics1,emb_latent_spatial_omics1, ret_spatial_omics1, ret_spatial_omics1a = self.encoder_omics1(features_omics1, features_omics1a, adj_spatial_omics1)  
        hidden_spatial_omics2,emb_latent_spatial_omics2, ret_spatial_omics2, ret_spatial_omics2a = self.encoder_omics2(features_omics2, features_omics2a, adj_spatial_omics2)
        
        # graph2
        hidden_feature_omics1, emb_latent_feature_omics1, ret_feature_omics1, ret_feature_omics1a = self.encoder_omics1(features_omics1, features_omics1a, adj_feature_omics1)
        hidden_feature_omics2, emb_latent_feature_omics2, ret_feature_omics2, ret_feature_omics2a = self.encoder_omics2(features_omics2, features_omics2a, adj_feature_omics2)
        
        # within-modality attention aggregation layer
        emb_latent_omics1 = self.atten_omics1(emb_latent_spatial_omics1, emb_latent_feature_omics1)
        emb_latent_omics2 = self.atten_omics2(emb_latent_spatial_omics2, emb_latent_feature_omics2)
        
        # between-modality attention aggregation layer
        emb_latent_combined, alpha_omics_1_2 = self.atten_cross(emb_latent_omics1, emb_latent_omics2)
        
        # reverse the integrated representation back into the original expression space with modality-specific decoder
        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)
        
        # consistency encoding
        emb_latent_omics1_across_recon = self.decoder_omics2(emb_latent_omics1, adj_spatial_omics2)
        emb_hidden_omics1_across_recon, emb_latent_omics1_across_recon, ret_across_recon_omics1, ret_across_recon_omics1a = self.encoder_omics2(emb_latent_omics1_across_recon, features_omics1a, adj_spatial_omics2)
        emb_latent_omics2_across_recon = self.decoder_omics1(emb_latent_omics2, adj_spatial_omics1)
        emb_hidden_omics2_across_recon, emb_latent_omics2_across_recon, ret_across_recon_omics2, ret_across_recon_omics2a = self.encoder_omics1(emb_latent_omics2_across_recon, features_omics2a, adj_spatial_omics1)
        
        results = {'emb_latent_omics1':emb_latent_omics1,
                   'emb_latent_omics2':emb_latent_omics2,
                   'emb_latent_combined':emb_latent_combined,
                   'emb_recon_omics1':emb_recon_omics1,
                   'emb_recon_omics2':emb_recon_omics2,
                   'ret_spatial_omics1':ret_spatial_omics1,
                   'ret_spatial_omics2':ret_spatial_omics2,
                   'ret_spatial_omics1a':ret_spatial_omics1a,
                   'ret_spatial_omics2a':ret_spatial_omics2a,
                   'ret_feature_omics1':ret_feature_omics1,
                   'ret_feature_omics2':ret_feature_omics2,
                   'ret_feature_omics1a':ret_feature_omics1a,
                   'ret_feature_omics2a':ret_feature_omics2a,
                   'emb_latent_omics1_across_recon':emb_latent_omics1_across_recon,
                   'emb_latent_omics2_across_recon':emb_latent_omics2_across_recon,
                   'ret_across_recon_omics1':ret_across_recon_omics1,
                   'ret_across_recon_omics2':ret_across_recon_omics2,
                   'ret_across_recon_omics1a':ret_across_recon_omics1a,
                   'ret_across_recon_omics2a':ret_across_recon_omics2a,
                   'alpha':alpha_omics_1_2
                   }
        
        return results 

    def barlow_twins_loss(self, x, y, lam=0.005):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        N = x.size(0)
        cross_corr = (x.T @ y) / N  
        diag_loss = torch.sum((torch.diagonal(cross_corr) - 1) ** 2)  
        off_diag_loss = torch.sum(cross_corr ** 2) - diag_loss  
        return diag_loss + lam * off_diag_loss    

'''
---------------------
Encoder functions
author: Yahui Long https://github.com/JinmiaoChenLab/GraphST
AGPL-3.0 LICENSE
---------------------
'''
class Encoder(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 dropout=0.1):

        super().__init__()

        self.fc1 = nn.Linear(in_features, out_features)

        self.fc2 = nn.Linear(out_features, out_features)

        self.ln1 = nn.LayerNorm(out_features)

        self.ln2 = nn.LayerNorm(out_features)

        self.dropout = dropout

        self.disc = Discriminator(out_features)

        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

    def graph_conv(self, adj, x):

        if adj.is_sparse:
            return torch.spmm(adj, x)

        return torch.mm(adj, x)

    def forward(self, feat, feat_a, adj):

        z = self.fc1(feat)

        z = self.graph_conv(adj, z)

        z = self.ln1(z)

        z = F.relu(z)

        z = F.dropout(
            z,
            self.dropout,
            training=self.training
        )

        hidden_emb = z

        z = self.fc2(z)

        z = self.graph_conv(adj, z)

        z = self.ln2(z)

        emb = F.relu(z)

        ####################################

        z_a = self.fc1(feat_a)

        z_a = self.graph_conv(adj, z_a)

        z_a = self.ln1(z_a)

        emb_a = F.relu(z_a)

        ####################################

        g = self.read(emb, adj)

        g = self.sigm(g)

        g_a = self.read(emb_a, adj)

        g_a = self.sigm(g_a)

        ret = self.disc(g, emb, emb_a)

        ret_a = self.disc(g_a, emb_a, emb)

        return hidden_emb, emb, ret, ret_a     
    
'''
---------------------
Decoder functions
author: Yahui Long https://github.com/JinmiaoChenLab/SpatialGlue
AGPL-3.0 LICENSE
---------------------
'''

class Decoder(Module):
    
    """\
    Modality-specific GNN decoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features. 
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    Reconstructed representation.

    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        
        return x                  

class Multi_CrossAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads=4,
                 dropout=0.1):
        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.modality_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

        self.dropout = nn.Dropout(dropout)

    def reshape_heads(self, x):
        B = x.shape[0]

        x = x.view(B, self.num_heads, self.head_dim)
        x = x.transpose(0, 1)

        return x

    def forward(self, emb1, emb2):

        Q = self.reshape_heads(self.q_proj(emb1))
        K = self.reshape_heads(self.k_proj(emb2))
        V = self.reshape_heads(self.v_proj(emb2))

        attention_scores = torch.matmul(
            Q,
            K.transpose(-1, -2)
        ) / sqrt(self.head_dim)

        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_weights = self.dropout(attention_weights)

        attended = torch.matmul(attention_weights, V)

        attended = attended.transpose(0, 1).contiguous()
        attended = attended.view(
            emb1.shape[0],
            self.hidden_size
        )

        attended = self.out_proj(attended)

        gate_input = torch.cat([emb1, attended], dim=-1)

        modality_alpha = self.modality_gate(gate_input)

        modality_alpha = F.softmax(modality_alpha, dim=-1)

        fused = (
            modality_alpha[:, 0:1] * emb1 +
            modality_alpha[:, 1:2] * attended
        )

        return fused, modality_alpha
    
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
    
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1) 
    
class BatchDiscriminator(nn.Module):
    def __init__(self, input_dim, num_batches):
        super(BatchDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_batches) 
        )

    def forward(self, x):
        return self.fc(x)

class MLP_Dropout(torch.nn.Module):
    def __init__(self, in_channels: int, args):
        super(MLP_Dropout, self).__init__()
        out_channels = args.dim_h
        p_drop = args.embed_dropout_prob
        self.num_layer = args.num_layer
        self.args = args
        self.activation = ({
            'relu': F.relu, 
            'prelu': nn.PReLU(), 
            'rrelu': F.rrelu, 
            'elu': F.elu
        })[args.activation]
        self.factor = 2
        self.lins = nn.ModuleList()
        self.drops = nn.ModuleList()
        if args.use_ln:
            self.lns = nn.ModuleList()
        if self.num_layer >= 2:
            self.lins.append(nn.Linear(self.factor * in_channels, self.factor * out_channels))
            self.drops.append(nn.Dropout(p=p_drop))
            if args.use_ln:
                self.lns.append(nn.LayerNorm(self.factor * out_channels))
            for _ in range(1, self.num_layer - 1):
                self.drops.append(nn.Dropout(p=p_drop))
                self.lins.append(nn.Linear(self.factor * out_channels, self.factor * out_channels))
                if args.use_ln:
                    self.lns.append(nn.LayerNorm(self.factor * out_channels))
            self.drops.append(nn.Dropout(p=p_drop))
            self.lins.append(nn.Linear(self.factor * out_channels, out_channels))
            if args.use_ln:
                self.lns.append(nn.LayerNorm(out_channels))
        else:
            self.drops.append(nn.Dropout(p=p_drop))
            self.lins.append(nn.Linear(self.factor * in_channels, out_channels))
            if args.use_ln:
                self.lns.append(nn.LayerNorm(out_channels))

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor):
        """
        Forward pass combining two modalities (emb1 and emb2).
        
        Args:
            emb1 (torch.Tensor): Embedding from modality 1 with shape [batch_size, in_channels].
            emb2 (torch.Tensor): Embedding from modality 2 with shape [batch_size, in_channels].
            edge_index (torch.Tensor): Graph edge indices (not used in this implementation).
        
        Returns:
            torch.Tensor: Processed embeddings with shape [batch_size, out_channels].
        """
        # Combine two modalities by concatenating along the feature dimension
        x = torch.cat([emb1, emb2], dim=-1)  # Shape: [batch_size, 2 * in_channels]

        for i in range(self.num_layer):
            x = self.drops[i](x)
            x = self.lins[i](x)
            x = self.activation(x)
            if self.args.use_ln:
                x = self.lns[i](x)
        return x

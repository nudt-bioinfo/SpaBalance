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
    dim_in_feat_omics3 : int
        Dimension of input features for omics3. 
    dim_out_feat_omics1 : int
        Dimension of latent representation for omics1.
    dim_out_feat_omics2 : int
        Dimension of latent representation for omics2, which is the same as omics1.
    dim_out_feat_omics3 : int
        Dimension of latent representation for omics3, which is the same as omics1.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    results: a dictionary including representations and modality weights.

    """
     
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2, dim_in_feat_omics3, dim_out_feat_omics3, dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dim_in_feat_omics3 = dim_in_feat_omics3
        self.dim_out_feat_omics3 = dim_out_feat_omics3
        self.dropout = dropout
        self.act = act
        
        args = {
            'dim_h': 64,
            'embed_dropout_prob': 0.1,
            'num_layer': 3,
            'activation': 'relu',
            'use_ln': True
        }
        args = type('Args', (object,), args)()  
        self.encoder_omics1 = Encoder_sparse(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.encoder_omics2 = Encoder_sparse(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)
        self.encoder_omics3 = Encoder_sparse(self.dim_in_feat_omics3, self.dim_out_feat_omics3)
        self.decoder_omics3 = Decoder(self.dim_out_feat_omics3, self.dim_in_feat_omics3)

        self.atten_omics1 =  MLP_Dropout(in_channels=64, args=args)
        self.atten_omics2 =  MLP_Dropout(in_channels=64, args=args)
        self.atten_omics3 =  MLP_Dropout(in_channels=64, args=args)
        self.atten_cross = Multi_CrossAttention(self.dim_out_feat_omics1, self.dim_out_feat_omics2, 8) 

    def forward(self, features_omics1, features_omics2, features_omics3, features_omics1a, features_omics2a, features_omics3a, adj_spatial_omics1, adj_feature_omics1, adj_spatial_omics2, adj_feature_omics2, adj_spatial_omics3, adj_feature_omics3):
        
        # graph1
        hidden_spatial_omics1,emb_latent_spatial_omics1, ret_spatial_omics1, ret_spatial_omics1a = self.encoder_omics1(features_omics1, features_omics1a, adj_spatial_omics1)  
        hidden_spatial_omics2,emb_latent_spatial_omics2, ret_spatial_omics2, ret_spatial_omics2a = self.encoder_omics2(features_omics2, features_omics2a, adj_spatial_omics2)
        hidden_spatial_omics3,emb_latent_spatial_omics3, ret_spatial_omics3, ret_spatial_omics3a = self.encoder_omics3(features_omics3, features_omics3a, adj_spatial_omics3)
        
        # graph2
        hidden_feature_omics1, emb_latent_feature_omics1, ret_feature_omics1, ret_feature_omics1a = self.encoder_omics1(features_omics1, features_omics1a, adj_feature_omics1)
        hidden_feature_omics2, emb_latent_feature_omics2, ret_feature_omics2, ret_feature_omics2a = self.encoder_omics2(features_omics2, features_omics2a, adj_feature_omics2)
        hidden_feature_omics3, emb_latent_feature_omics3, ret_feature_omics3, ret_feature_omics3a = self.encoder_omics3(features_omics3, features_omics3a, adj_feature_omics3)
        
        # within-modality attention aggregation layer
        #emb_latent_omics1, alpha_omics1 = self.atten_omics1(emb_latent_spatial_omics1, emb_latent_feature_omics1)
        #emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spatial_omics2, emb_latent_feature_omics2)
        #emb_latent_omics3, alpha_omics3 = self.atten_omics3(emb_latent_spatial_omics3, emb_latent_feature_omics3)
        emb_latent_omics1 = self.atten_omics1(emb_latent_spatial_omics1, emb_latent_feature_omics1)
        emb_latent_omics2 = self.atten_omics2(emb_latent_spatial_omics2, emb_latent_feature_omics2)
        emb_latent_omics3 = self.atten_omics3(emb_latent_spatial_omics3, emb_latent_feature_omics3)
        
        # between-modality attention aggregation layer
        emb_latent_combined, alpha_omics_1_2_3 = self.atten_cross(emb_latent_omics1, emb_latent_omics2, emb_latent_omics3)
        
        # reverse the integrated representation back into the original expression space with modality-specific decoder
        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)
        emb_recon_omics3 = self.decoder_omics3(emb_latent_combined, adj_spatial_omics3)
        
        # consistency encoding
        emb_latent_omics1_omics2_across_recon = self.decoder_omics2(emb_latent_omics1, adj_spatial_omics2)
        emb_hidden_omics1_omics2_across_recon, emb_latent_omics1_omics2_across_recon, ret_across_recon_omics1_omics2, ret_across_recon_omics1a_omics2 = self.encoder_omics2(emb_latent_omics1_omics2_across_recon, features_omics1a, adj_spatial_omics2)
        emb_latent_omics1_omics3_across_recon = self.decoder_omics3(emb_latent_omics1, adj_spatial_omics3)
        emb_hidden_omics1_omics3_across_recon, emb_latent_omics1_omics3_across_recon, ret_across_recon_omics1_omics3, ret_across_recon_omics1a_omics3 = self.encoder_omics3(emb_latent_omics1_omics3_across_recon, features_omics1a, adj_spatial_omics3)
        emb_latent_omics2_omics1_across_recon = self.decoder_omics1(emb_latent_omics2, adj_spatial_omics1)
        emb_hidden_omics2_omics1_across_recon, emb_latent_omics2_omics1_across_recon, ret_across_recon_omics2_omics1, ret_across_recon_omics2a_omics1 = self.encoder_omics1(emb_latent_omics2_omics1_across_recon, features_omics2a, adj_spatial_omics1)
        emb_latent_omics2_omics3_across_recon = self.decoder_omics3(emb_latent_omics2, adj_spatial_omics3)
        emb_hidden_omics2_across_recon, emb_latent_omics2_omics3_across_recon, ret_across_recon_omics2_omics3, ret_across_recon_omics2a_omics3 = self.encoder_omics1(emb_latent_omics2_omics3_across_recon, features_omics2a, adj_spatial_omics1)
        emb_latent_omics3_omics1_across_recon = self.decoder_omics1(emb_latent_omics3, adj_spatial_omics1)
        emb_hidden_omics3_omics1_across_recon, emb_latent_omics3_omics1_across_recon, ret_across_recon_omics3_omics1, ret_across_recon_omics3a_omics1 = self.encoder_omics1(emb_latent_omics3_omics1_across_recon, features_omics3a, adj_spatial_omics1)
        emb_latent_omics3_omics2_across_recon = self.decoder_omics2(emb_latent_omics3, adj_spatial_omics2)
        emb_hidden_omics3_across_recon, emb_latent_omics3_omics2_across_recon, ret_across_recon_omics3_omics2, ret_across_recon_omics3a_omics2 = self.encoder_omics2(emb_latent_omics3_omics2_across_recon, features_omics3a, adj_spatial_omics2)

        
        results = {'emb_latent_omics1':emb_latent_omics1,
                   'emb_latent_omics2':emb_latent_omics2,
                   'emb_latent_omics3':emb_latent_omics3,
                   'emb_latent_combined':emb_latent_combined,
                   'emb_recon_omics1':emb_recon_omics1,
                   'emb_recon_omics2':emb_recon_omics2,
                   'emb_recon_omics3':emb_recon_omics3,
                   'ret_spatial_omics1':ret_spatial_omics1,
                   'ret_spatial_omics2':ret_spatial_omics2,
                   'ret_spatial_omics3':ret_spatial_omics3,
                   'ret_spatial_omics1a':ret_spatial_omics1a,
                   'ret_spatial_omics2a':ret_spatial_omics2a,
                   'ret_spatial_omics3a':ret_spatial_omics3a,
                   'ret_feature_omics1':ret_feature_omics1,
                   'ret_feature_omics2':ret_feature_omics2,
                   'ret_feature_omics3':ret_feature_omics3,
                   'ret_feature_omics1a':ret_feature_omics1a,
                   'ret_feature_omics2a':ret_feature_omics2a,
                   'ret_feature_omics3a':ret_feature_omics3a,
                   'emb_latent_omics1_omics2_across_recon':emb_latent_omics1_omics2_across_recon,
                   'emb_latent_omics1_omics3_across_recon':emb_latent_omics1_omics3_across_recon,
                   'emb_latent_omics2_omics1_across_recon':emb_latent_omics2_omics1_across_recon,
                   'emb_latent_omics2_omics3_across_recon':emb_latent_omics2_omics3_across_recon,
                   'emb_latent_omics3_omics1_across_recon':emb_latent_omics3_omics1_across_recon,
                   'emb_latent_omics3_omics2_across_recon':emb_latent_omics3_omics2_across_recon,
                   'ret_across_recon_omics1_omics2':ret_across_recon_omics1_omics2,
                   'ret_across_recon_omics1_omics3':ret_across_recon_omics1_omics3,
                   'ret_across_recon_omics2_omics1':ret_across_recon_omics2_omics1,
                   'ret_across_recon_omics2_omics3':ret_across_recon_omics2_omics3,
                   'ret_across_recon_omics3_omics1':ret_across_recon_omics3_omics1,
                   'ret_across_recon_omics3_omics2':ret_across_recon_omics3_omics2,
                   'ret_across_recon_omics1a_omics2':ret_across_recon_omics1a_omics2,
                   'ret_across_recon_omics1a_omics3':ret_across_recon_omics1a_omics3,
                   'ret_across_recon_omics2a_omics1':ret_across_recon_omics2a_omics1,
                   'ret_across_recon_omics2a_omics3':ret_across_recon_omics2a_omics3,
                   'ret_across_recon_omics3a_omics1':ret_across_recon_omics3a_omics1,
                   'ret_across_recon_omics3a_omics2':ret_across_recon_omics3a_omics2,
                   #'alpha_omics1':alpha_omics1,
                   #'alpha_omics2':alpha_omics2,
                   #'alpha_omics3':alpha_omics3,
                   'alpha':alpha_omics_1_2_3
                   }
        
        return results 

    def barlow_twins_loss(self, x, y, lam=0.005):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        N = x.size(0)
        cross_corr = (x.T @ y) / N  # 计算交叉相关矩阵
        diag_loss = torch.sum((torch.diagonal(cross_corr) - 1) ** 2)  # 对角项接近1
        off_diag_loss = torch.sum(cross_corr ** 2) - diag_loss  # 非对角项接近0
        return diag_loss + lam * off_diag_loss    

class Encoder(Module): 
    
    """\
    Modality-specific GNN encoder.

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
    Latent representation.

    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
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

class AttentionLayer(Module):
    
    """\
    Attention layer.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.     

    Returns
    -------
    Aggregated representations and modality weights.

    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)  
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined), self.alpha    

class Multi_CrossAttention(nn.Module):
    def __init__(self, hidden_size, all_head_size, head_num, num_modalities=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num
        self.num_modalities = num_modalities

        assert all_head_size % head_num == 0

        # Q, K, V 的线性映射层
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)

        # 每个模态的独立注意力头
        self.modality_attention_heads = nn.ModuleList([
            nn.Linear(1, self.h_size) for _ in range(num_modalities)
        ])
        
        # 归一化因子
        self.norm = sqrt(self.h_size)

    def forward(self, emb1, emb2, emb3):
        num_cells = emb1.size(0)

        # 融合特征生成 Q, K, V
        Q = self.linear_q(emb1).view(num_cells, self.num_heads, self.h_size).transpose(0, 1)
        # 分别生成 K 和 V
        K2 = self.linear_k(emb2).view(num_cells, self.num_heads, self.h_size).transpose(0, 1)
        K3 = self.linear_k(emb3).view(num_cells, self.num_heads, self.h_size).transpose(0, 1)
        K = K2 + K3  # 按权重加权相加或融合

        V2 = self.linear_v(emb2).view(num_cells, self.num_heads, self.h_size).transpose(0, 1)
        V3 = self.linear_v(emb3).view(num_cells, self.num_heads, self.h_size).transpose(0, 1)
        V = V2 + V3

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.norm  # [num_heads, 3484, 3484]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [num_heads, 3484, 3484]
    
        # 每个模态单独建模注意力权重
        modality_weights = []
        for head in self.modality_attention_heads:
            # 聚合多头的注意力输出
            aggregated_weights = attention_weights.mean(dim=0).mean(dim=-1)  # [3484, h_size]
            modality_weight = head(aggregated_weights.unsqueeze(-1))  # [3484, 1]
            modality_weights.append(modality_weight)
    
        # 拼接并归一化模态权重
        modality_weights = torch.cat(modality_weights, dim=-1)  # [3484, num_modalities]
        modality_weights = F.softmax(modality_weights, dim=-1)  # [3484, num_modalities]

        # 整合模态数据
        emb1_weighted = emb1 * modality_weights[:, 0:1]  # 对 emb1 加权, shape = [3484, hidden_size]
        emb2_weighted = emb2 * modality_weights[:, 1:2]  # 对 emb2 加权, shape = [3484, hidden_size]
        emb3_weighted = emb3 * modality_weights[:, 2:3]  # 对 emb3 加权, shape = [3484, hidden_size]
        integrated_embeddings = emb1_weighted + emb2_weighted + emb3_weighted # 模态数据整合, shape = [3484, hidden_size]

        return integrated_embeddings, modality_weights
    
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
    
class Encoder_sparse(Module):
    """
    Sparse version of Encoder
    """
    def __init__(self, in_features, out_features, dropout=0.0, act=F.relu):
        super(Encoder_sparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()
        
        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.spmm(adj, z)
        
        hidden_emb = z

        h = torch.mm(z, self.weight2)
        h = torch.spmm(adj, h)
        
        emb = self.act(z)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.spmm(adj, z_a)
        emb_a = self.act(z_a)
         
        #g = self.read(emb, self.graph_neigh)
        g = self.read(emb, adj)
        g = self.sigm(g)
        
        #g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.read(emb_a, adj)
        g_a =self.sigm(g_a)       
       
        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb)
        
        return hidden_emb, emb, ret, ret_a     
    
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
    
class MultiScaleFusion(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(MultiScaleFusion, self).__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        # Shared FC layer to reduce redundant computations
        self.shared_fc = nn.Linear(2, 2)
        self.output_fc = nn.Linear(feature_dim, output_dim)

    def forward(self, emb1, emb2):
        concatenated = torch.stack([emb1, emb2], dim=1)  # Shape: [batch_size, 2, feature_dim]
        gap_features = torch.mean(concatenated, dim=2)  # Shape: [batch_size, 2]

       # Calculate weights
        fc_output = self.shared_fc(gap_features)  # Shape: [batch_size, 2]
        combined_weights = F.softmax(fc_output, dim=1) * torch.sigmoid(fc_output)
        combined_weights = combined_weights.unsqueeze(2).expand(-1, -1, self.feature_dim)  # Shape: [batch_size, 2, feature_dim]

        weighted_features = concatenated * combined_weights  # Shape: [batch_size, 2, feature_dim]
        fused_feature = torch.sum(weighted_features, dim=1)  # Shape: [batch_size, feature_dim]
        output_feature = self.output_fc(fused_feature)  # Shape: [batch_size, output_dim]

        return output_feature
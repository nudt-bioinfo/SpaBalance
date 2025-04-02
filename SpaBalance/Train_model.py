import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from .model import Encoder_overall
from .preprocess import adjacent_matrix_preprocessing, permutation, add_contrastive_label
from .min_norm_solvers import MinNormSolver

'''
---------------------
The `Balance` class is modified based on `MMPareto`.
authors: Yake Wei, and Di Hu https://github.com/GeWu-Lab/MMPareto_ICML2024
AGPL-3.0 LICENSE
---------------------
'''
class Balance:
    def __init__(self, model, optimizer, device, initial_task_weights=None, initial_task_weights_both=None, initial_task_weights_omics=None, initial_task_weights_omics2=None, alpha=0.18, history_size=5, epochs=200, weight=30):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.alpha = alpha  # GradNorm hyperparameter, used to control the strength of weight updates
        self.epochs = epochs
        self.weight = weight 
        self.history_size = history_size

        # Initialize task weights
        if initial_task_weights is None:
            self.task_weights = {'both': 1.0, 'omics1': 1.0, 'omics2': 1.0}
        else:
            self.task_weights = initial_task_weights

        if initial_task_weights_both is None:
            self.task_weights_both= {'loss_corr_omics1': 3, 'loss_corr_omics2': 3, 'constractive_loss': 0.5}
        else:
            self.task_weights_both = initial_task_weights_both

        if initial_task_weights_omics is None:
            self.task_weights_omics = {'loss_recon': 10, 'loss_slf_1': 1, 'loss_slf_2': 1}
        else:
            self.task_weights_omics = initial_task_weights_omics

        if initial_task_weights_omics2 is None:
            self.task_weights_omics2 = {'loss_recon': 10, 'loss_slf_1': 1, 'loss_slf_2': 1}
        else:
            self.task_weights_omics2 = initial_task_weights_omics2

        # Initialize gradient queue
        self.history_grad_queues = {task: [] for task in self.task_weights.keys()}

    def compute_loss_omics(self, label_CSL, features, results, omics):
        loss = nn.BCEWithLogitsLoss()
        ret_f = results['ret_feature_' + omics]
        ret_fa = results['ret_feature_' + omics + 'a']
        loss_slf_1 = loss(ret_f, label_CSL)
        loss_slf_2 = loss(ret_fa, label_CSL)
        loss_recon = F.mse_loss(features, results['emb_recon_' + omics]) 

        total_loss = self.weight * loss_recon + loss_slf_1 + loss_slf_2

        return total_loss

    def compute_loss_shared(self, loss_corr1, loss_corr2, constractive_loss, epoch):
        """
        Calculate the total loss based on uncertainty dynamic weighting
        """
        # Move log_var1 and log_var2 to the same device as the input loss
        device = loss_corr1.device  # Get the device of the input tensor
        log_var1 = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
        log_var2 = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
        
        if epoch < self.epochs * 0.75:
            # Use simple weighting in the fixed weight stage
            loss_shared = loss_corr1 + loss_corr2 + 0.5 * constractive_loss
        else:
            # Dynamic weighting stage
            loss1 = (1 / (2 * torch.exp(log_var1))) * loss_corr1 + log_var1 / 2
            loss2 = (1 / (2 * torch.exp(log_var2))) * loss_corr2 + log_var2 / 2
            loss_shared = loss1 + loss2 + 0.5 * constractive_loss
        
        return loss_shared
    
    def compute_loss(self, results, constractive_loss, loss_omics1, loss_omics2, epoch):
        """Calculate the loss of all tasks"""
        loss_corr1 = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_omics1_across_recon'])
        loss_corr2 = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_omics2_across_recon'])
       
        # Calculate the loss of all tasks
        losses = {
            'both': self.compute_loss_shared(loss_corr1, loss_corr2, constractive_loss, epoch),
            'omics1':  loss_omics1,
            'omics2':  loss_omics2
        }
        return losses

    def calculate_gradients(self, losses):
        """Calculate the gradient of each task"""
        grads = {task: {} for task in losses.keys()}
        for task, loss in losses.items():
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)  # Retain the computation graph to calculate the gradients of multiple tasks
            grads[task]['concat'] = torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.grad is not None]).detach()
        return grads

    def update_task_weights_with_gradnorm_lr(self, losses, grads, omics=None):
        if omics is None: 
            self.task_weight = self.task_weights
        elif omics == 'omics1':
            self.task_weight = self.task_weights_omics
        elif omics == 'omics2':
            self.task_weight = self.task_weights_omics2
        else:
            self.task_weight = self.task_weights_both

        print(self.task_weight)
        """Improvement: Dynamically update weights, combining gradient adjustment history"""
        with torch.no_grad():
            grad_norms = {task: torch.norm(grad['concat'], p=2) for task, grad in grads.items()}
            weighted_losses = {task: self.task_weight[task] * losses[task] for task in losses.keys()}
            loss_avg = sum(weighted_losses.values()) / len(weighted_losses)

            # Dynamically adjust target gradients
            target_grad_norms = {
                task: grad_norms[task] * (losses[task] / loss_avg) ** self.alpha
                for task in losses.keys()
            }

            gradnorm_loss = sum(torch.abs(grad_norms[task] - target_grad_norms[task]) for task in losses.keys())

            # Dynamically adjust learning rate
            lr = self.optimizer.param_groups[0]['lr'] * gradnorm_loss.item()
            for task in self.task_weight.keys():
                self.task_weight[task] = max(self.task_weight[task] - lr, 1e-4)  # Ensure weights are non-negative

        return self.task_weight

    def update_task_weights_with_gradnorm_lr_hs(self, losses, grads):
        """Dynamically update weights, combining weighted historical gradient memory"""
        with torch.no_grad():
            # Current gradient norms
            grad_norms = {task: torch.norm(grad['concat'], p=2).item() for task, grad in grads.items()}

            # Update gradient queue
            for task in grad_norms.keys():
                self.history_grad_queues[task].append(grad_norms[task])
                if len(self.history_grad_queues[task]) > self.history_size:
                    self.history_grad_queues[task].pop(0)

            # Calculate weighted average gradient
            avg_grad_norms = {
                task: sum(self.history_grad_queues[task]) / len(self.history_grad_queues[task])
                for task in grad_norms.keys()
            }

            # Update target gradients
            weighted_losses = {task: self.task_weights[task] * losses[task] for task in losses.keys()}
            loss_avg = sum(weighted_losses.values()) / len(weighted_losses)

            target_grad_norms = {
                task: avg_grad_norms[task] * (losses[task] / loss_avg) ** self.alpha
                for task in losses.keys()
            }

            gradnorm_loss = sum(torch.abs(grad_norms[task] - target_grad_norms[task]) for task in losses.keys())

            # Dynamically adjust learning rate
            lr = self.optimizer.param_groups[0]['lr'] * gradnorm_loss.item()
            for task in self.task_weights.keys():
                self.task_weights[task] = max(self.task_weights[task] - lr, 1e-4)  # Ensure weights are non-negative

        return self.task_weights
    
    def compute_weights_with_similarity(self, grads):
        """Calculate weights based on gradient similarity"""
        tasks = list(grads.keys())
        weights = {task: 0.0 for task in tasks}

        # Find the maximum size
        max_size = max(grad['concat'].shape[0] for grad in grads.values())

        # Pad gradients to align
        for task in grads:
            grad_size = grads[task]['concat'].shape[0]
            if grad_size < max_size:
                grads[task]['concat'] = F.pad(grads[task]['concat'], (0, max_size - grad_size))

        # Calculate cosine similarity between tasks
        cos_omics1 = F.cosine_similarity(grads['both']['concat'], grads['omics1']['concat'], dim=0)
        cos_omics2 = F.cosine_similarity(grads['both']['concat'], grads['omics2']['concat'], dim=0)

        # Initialize weights
        omics1_k = [0, 0]
        omics2_k = [0, 0]

        # Handle weights for omics1
        if cos_omics1 > 0:  #  If the gradients are similar
            omics1_k[0] = 0.5  #  Weight equally
            omics1_k[1] = 0.5
        else:  # If the gradients conflict
            omics1_k, _ = MinNormSolver.find_min_norm_element(
                [list(grads[task].values()) for task in ['both', 'omics1']]
            )
            
        # Handle weights for omics2
        if cos_omics2 > 0:  
            omics2_k[0] = 0.5  
            omics2_k[1] = 0.5
        else:  
            omics2_k, _ = MinNormSolver.find_min_norm_element(
                [list(grads[task].values()) for task in ['both', 'omics2']]
            )

        # Combine weights
        weights['both'] = (omics1_k[0] + omics2_k[0]) / 2
        weights['omics1'] = omics1_k[1]
        weights['omics2'] = omics2_k[1]

        return weights

    
class Train_SpaBalance:
    def __init__(self, 
        data,
        datatype = 'SPOTS',
        device= torch.device('cpu'),
        random_seed = 2022,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=64
        ):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input, Our current model supports 'SPOTS', 'Stereo-CITE-seq', and 'Spatial-ATAC-RNA-seq'. We plan to extend our model for more data types in the future.  
            The default is 'SPOTS'.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 2022.    
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight decay to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 1500.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        weight_factors : list, optional
            Weight factors to balance the influcences of different omics data on model training.
    
        Returns
        -------
        The learned representation 'self.emb_combined'.

        '''
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.initial_task_weights_omics1 = None
        self.initial_task_weights_omics2 = None
        
        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2, dense=True)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)
        if 'label_CSL' not in self.adata_omics1.obsm.keys():    
           add_contrastive_label(self.adata_omics1)
        if 'label_CSL' not in self.adata_omics2.obsm.keys():    
           add_contrastive_label(self.adata_omics2)
        
        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        
        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
        
        if self.datatype == 'SPOTS':
           self.epochs = 600 
           self.weight = 120
           
        elif self.datatype == 'Stereo-CITE-seq':
           self.epochs = 1500 
           self.weight = 60
           
        elif self.datatype == '10x':
           self.epochs = 200
           self.weight = 30
            
        elif self.datatype == 'Spatial-epigenome-transcriptome': 
           self.epochs = 1600
           self.weight = 50

    def train(self):
        # 初始化模型和优化器
        self.training = True
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        self.model.train()

        # 初始化 Balance 实例
        pareto_solver = Balance(self.model, self.optimizer, self.device, epochs=self.epochs, weight=self.weight)
        self.label_CSL_omics1 = torch.FloatTensor(self.adata_omics1.obsm['label_CSL']).to(self.device)
        self.label_CSL_omics2 = torch.FloatTensor(self.adata_omics2.obsm['label_CSL']).to(self.device)

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            self.features_omics1a = permutation(self.features_omics1)
            self.features_omics2a = permutation(self.features_omics2)
            # 前向传播
            results = self.model(
                self.features_omics1, self.features_omics2,
                self.features_omics1a, self.features_omics2a,
                self.adj_spatial_omics1, self.adj_feature_omics1,
                self.adj_spatial_omics2, self.adj_feature_omics2,
            )

            # 计算对比损失
            constractive_loss = self.model.barlow_twins_loss(results['emb_latent_omics1'], results['emb_latent_omics2'])
            constractive_loss_reverse = self.model.barlow_twins_loss(results['emb_latent_omics2'], results['emb_latent_omics1'])
            constractive_loss = (constractive_loss + constractive_loss_reverse) / 2
            # 计算损失
            loss_omics1 = pareto_solver.compute_loss_omics(self.label_CSL_omics1, self.features_omics1, results, 'omics1')
            loss_omics2 = pareto_solver.compute_loss_omics(self.label_CSL_omics2, self.features_omics2, results, 'omics2')
            all_losses = pareto_solver.compute_loss(results, constractive_loss, loss_omics1, loss_omics2, epoch)
            # 计算总梯度
            grads = pareto_solver.calculate_gradients(all_losses)
            #计算权重（考虑任务相似性)
            weights = pareto_solver.compute_weights_with_similarity(grads) 
             #加权计算总损失
            total_loss = sum(weights[task] * all_losses[task] for task in all_losses.keys())
            # 更新模型参数
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss.item()}")

        print("Model training finished!\n")    

        # 推理阶段
        with torch.no_grad():
            self.training = False
            self.model.eval()
            results = self.model(
                self.features_omics1, self.features_omics2,
                self.features_omics1a, self.features_omics2a,
                self.adj_spatial_omics1, self.adj_feature_omics1,
                self.adj_spatial_omics2, self.adj_feature_omics2,
            )

        # 输出嵌入结果
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)  
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        output = {
            'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
            'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
            'SpaBalance': emb_combined.detach().cpu().numpy(),
            'alpha': results['alpha'].detach().cpu().numpy(),
        }

        return output

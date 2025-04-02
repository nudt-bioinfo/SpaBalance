import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from .model_3M import Encoder_overall
from .preprocess import adjacent_matrix_preprocessing, permutation, add_contrastive_label
from SpaBalance.min_norm_solvers import MinNormSolver

'''
---------------------
The `Balance` class is modified based on `MMPareto`.
authors: Yake Wei, and Di Hu https://github.com/GeWu-Lab/MMPareto_ICML2024
AGPL-3.0 LICENSE
---------------------
'''
class Balance:
    def __init__(self, model, optimizer, device, initial_task_weights=None, initial_task_weights_omics=None, initial_task_weights_omics2=None, initial_task_weights_omics3=None, alpha=0.18, history_size=5, epochs = 200):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.alpha = alpha  # GradNorm weight
        self.epochs = epochs 
        self.history_size = history_size

        # initialize task weights
        if initial_task_weights is None:
            self.task_weights = {'both': 1.0, 'omics1': 1.0, 'omics2': 1.0, 'omics3': 1.0 }
        else:
            self.task_weights = initial_task_weights

        if initial_task_weights_omics is None:
            self.task_weights_omics = {'loss_recon': 10, 'constrative_loss': 0.1, 'loss_slf_1': 1, 'loss_slf_2': 1, 'loss_corr':1}
        else:
            self.task_weights_omics = initial_task_weights_omics

        if initial_task_weights_omics2 is None:
            self.task_weights_omics2 = {'loss_recon': 10, 'constrative_loss': 0.1, 'loss_slf_1': 1, 'loss_slf_2': 1, 'loss_corr':1}
        else:
            self.task_weights_omics2 = initial_task_weights_omics2

        if initial_task_weights_omics3 is None:
            self.task_weights_omics3 = {'loss_recon': 10, 'constrative_loss': 0.1, 'loss_slf_1': 1, 'loss_slf_2': 1, 'loss_corr':1}
        else:
            self.task_weights_omics3 = initial_task_weights_omics2

        # initialize history queues
        self.history_grad_queues = {task: [] for task in self.task_weights.keys()}
    
    def loss_omics(self, label_CSL, features, results, constractive_loss1, constractive_loss2, omics):
        omicsa = 'omics2'
        omicsb = 'omics3'
        loss = nn.BCEWithLogitsLoss()
        ret_f = results['ret_feature_'+ omics]
        ret_fa = results['ret_feature_'+ omics + 'a']
        self.loss_slf_1 = loss(ret_f, label_CSL)
        self.loss_slf_2 = loss(ret_fa, label_CSL)
        if omics == 'omics2':
            omicsa = 'omics1'
        elif omics == 'omics3':
            omicsb = 'omics1'      
        loss_corr1 = F.mse_loss(results['emb_latent_'+omics], results['emb_latent_'+omics+'_'+omicsa+'_across_recon'])
        loss_corr2 = F.mse_loss(results['emb_latent_'+omics], results['emb_latent_'+omics+'_'+omicsb+'_across_recon'])
        loss_recon = F.mse_loss(features, results['emb_recon_'+omics])
        losses = {
            'loss_recon': loss_recon,
            'constrative_loss': constractive_loss1 + constractive_loss2,
            'loss_slf_1': self.loss_slf_1,
            'loss_slf_2': self.loss_slf_2,
            'loss_corr': loss_corr1 + loss_corr2,
        }
        return losses
    
    def compute_loss_omics(self, label_CSL, features, results, omics, epoch):
        loss = nn.BCEWithLogitsLoss()
        ret_f = results['ret_feature_' + omics]
        ret_fa = results['ret_feature_' + omics + 'a']
        loss_slf_1 = loss(ret_f, label_CSL)
        loss_slf_2 = loss(ret_fa, label_CSL)
        loss_recon = F.mse_loss(features, results['emb_recon_' + omics]) 
        total_loss = 10 * loss_recon + loss_slf_1 + loss_slf_2
        return total_loss

    def compute_loss_shared(self, loss_corr12, loss_corr13, loss_corr21, loss_corr23, loss_corr31, loss_corr32, constractive_loss, epoch):
        """
        calculate the shared loss based on the given losses
        """
        # move losses to the same device
        device = loss_corr12.device  # get the device of the loss
        log_var1 = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
        log_var2 = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
        log_var3 = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
        log_var4 = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
        log_var5 = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
        log_var6 = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
        
        if epoch < self.epochs * 0.75:
            # simple weighting
            loss_shared = loss_corr12 + loss_corr13 + loss_corr21 + loss_corr23 + loss_corr31 + loss_corr32 + 0.5 * constractive_loss
        else:
            # dynamic weighting based on uncertainty
            loss1 = (1 / (2 * torch.exp(log_var1))) * loss_corr12 + log_var1 / 2
            loss2 = (1 / (2 * torch.exp(log_var2))) * loss_corr13 + log_var2 / 2
            loss3 = (1 / (2 * torch.exp(log_var3))) * loss_corr21 + log_var3 / 2
            loss4 = (1 / (2 * torch.exp(log_var4))) * loss_corr23 + log_var4 / 2
            loss5 = (1 / (2 * torch.exp(log_var5))) * loss_corr31 + log_var5 / 2
            loss6 = (1 / (2 * torch.exp(log_var6))) * loss_corr32 + log_var6 / 2
            loss_shared = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + 0.5 * constractive_loss
        
        return loss_shared
    
    def compute_loss(self, results, constractive_loss, loss_omics1, loss_omics2, loss_omics3, epoch):
        """calculate the overall loss"""
        loss_corr12 = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_omics1_omics2_across_recon'])
        loss_corr13 = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_omics1_omics3_across_recon'])
        loss_corr21 = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_omics2_omics1_across_recon'])
        loss_corr23 = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_omics2_omics3_across_recon'])
        loss_corr31 = F.mse_loss(results['emb_latent_omics3'], results['emb_latent_omics3_omics1_across_recon'])
        loss_corr32 = F.mse_loss(results['emb_latent_omics3'], results['emb_latent_omics3_omics2_across_recon'])

        # calculate the loss for each task
        losses = {
            'both': self.compute_loss_shared(loss_corr12, loss_corr13, loss_corr21, loss_corr23, loss_corr31, loss_corr32, constractive_loss, epoch),
            'omics1':  loss_omics1,
            'omics2':  loss_omics2,
            'omics3':  loss_omics3
        }
        return losses

    def calculate_gradients(self, losses):
        """calculate the gradient of each task"""
        grads = {task: {} for task in losses.keys()}
        for task, loss in losses.items():
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)  # 保留计算图以计算多个任务的梯度
            grads[task]['concat'] = torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.grad is not None]).detach()
        return grads

    def update_task_weights_with_gradnorm(self, losses, grads):
        """update the task weights using grad norm"""
        with torch.no_grad():
            grad_norms = {task: torch.norm(grad['concat'], p=2) for task, grad in grads.items()}
            weighted_losses = {task: self.task_weights[task] * losses[task] for task in losses.keys()}
            loss_avg = sum(weighted_losses.values()) / len(weighted_losses)
            
            target_grad_norms = {
                task: grad_norms[task] * (losses[task] / loss_avg) ** self.alpha
                for task in losses.keys()
            }
            
            gradnorm_loss = sum(torch.abs(grad_norms[task] - target_grad_norms[task]) for task in losses.keys())
            self.task_weights = {
                task: max(self.task_weights[task] - self.optimizer.param_groups[0]['lr'] * gradnorm_loss.item(), 1e-4)
                for task in self.task_weights.keys()
            }
        return self.task_weights

    def update_task_weights_with_gradnorm_lr(self, losses, grads, omics=None):
        if omics is None:
            self.task_weight = self.task_weights
        else:
            if omics == 'omics1':
                self.task_weight = self.task_weights_omics
            elif omics == 'omics2':
                self.task_weight = self.task_weights_omics2
            elif omics == 'omics3':
                self.task_weight = self.task_weights_omics3
        print(self.task_weight)
        """dynamic learning with grad norm"""
        with torch.no_grad():
            grad_norms = {task: torch.norm(grad['concat'], p=2) for task, grad in grads.items()}
            weighted_losses = {task: self.task_weight[task] * losses[task] for task in losses.keys()}
            loss_avg = sum(weighted_losses.values()) / len(weighted_losses)

            target_grad_norms = {
                task: grad_norms[task] * (losses[task] / loss_avg) ** self.alpha
                for task in losses.keys()
            }

            gradnorm_loss = sum(torch.abs(grad_norms[task] - target_grad_norms[task]) for task in losses.keys())

            # dynamic learning rate
            lr = self.optimizer.param_groups[0]['lr'] * gradnorm_loss.item()
            for task in self.task_weight.keys():
                self.task_weight[task] = max(self.task_weight[task] - lr, 1e-4)  # 确保权重非负

        return self.task_weight

    def compute_weights_with_similarity(self, grads):
        """calcuate the similarity between two tasks and use it as a weight"""
        tasks = list(grads.keys())
        weights = {task: 0.0 for task in tasks}

        # find the largest gradient size
        max_size = max(grad['concat'].shape[0] for grad in grads.values())

        # pad the gradients to the largest size
        for task in grads:
            grad_size = grads[task]['concat'].shape[0]
            if grad_size < max_size:
                grads[task]['concat'] = F.pad(grads[task]['concat'], (0, max_size - grad_size))

        # calculate the cosine similarity between the gradients of both tasks
        cos_omics1 = F.cosine_similarity(grads['both']['concat'], grads['omics1']['concat'], dim=0)
        cos_omics2 = F.cosine_similarity(grads['both']['concat'], grads['omics2']['concat'], dim=0)
        cos_omics3 = F.cosine_similarity(grads['both']['concat'], grads['omics3']['concat'], dim=0)

        # initialize the weights
        omics1_k = [0, 0]
        omics2_k = [0, 0]
        omics3_k = [0, 0]

        # tackle omics1
        if cos_omics1 > 0:  # if gradients are similar
            omics1_k[0] = 0.5  # weight share
            omics1_k[1] = 0.5
        else:  # if gradients are conflicting
            omics1_k, _ = MinNormSolver.find_min_norm_element(
                [list(grads[task].values()) for task in ['both', 'omics1']]
            )

        # tackle omics2
        if cos_omics2 > 0:  
            omics2_k[0] = 0.5  
            omics2_k[1] = 0.5
        else:  
            omics2_k, _ = MinNormSolver.find_min_norm_element(
                [list(grads[task].values()) for task in ['both', 'omics2']]
            )

        # tackle omics3
        if cos_omics3 > 0: 
            omics3_k[0] = 0.5  
            omics3_k[1] = 0.5
        else: 
            omics3_k, _ = MinNormSolver.find_min_norm_element(
                [list(grads[task].values()) for task in ['both', 'omics3']]
            )

        # compute weights
        weights['both'] = (omics1_k[0] + omics2_k[0] + omics3_k[0]) / 2
        weights['omics1'] = omics1_k[1]
        weights['omics2'] = omics2_k[1]
        weights['omics3'] = omics3_k[1]

        print(weights)
        return weights

class Train_SpaBalance_3M:
    def __init__(self, 
        data,
        datatype = 'Triplet',
        device= torch.device('cpu'),
        random_seed = 2022,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=64,
        weight_factors = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input  
            The default is 'Triplet'. To date, real-worlk triplet modality data is still unavailable. We define default data type as 'Triplet' temporarily.
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
        self.weight_factors = weight_factors
        
        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adata_omics3 = self.data['adata_omics3']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2, self.adata_omics3)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_spatial_omics3 = self.adj['adj_spatial_omics3'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)
        self.adj_feature_omics3 = self.adj['adj_feature_omics3'].to(self.device)
        if 'label_CSL' not in self.adata_omics1.obsm.keys():
            add_contrastive_label(self.adata_omics1)
        if 'label_CSL' not in self.adata_omics2.obsm.keys():
            add_contrastive_label(self.adata_omics2)
        if 'label_CSL' not in self.adata_omics3.obsm.keys():
            add_contrastive_label(self.adata_omics3)
        
        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        self.features_omics3 = torch.FloatTensor(self.adata_omics3.obsm['feat'].copy()).to(self.device)
        
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        self.n_cell_omics3 = self.adata_omics3.n_obs
        
        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_input3 = self.features_omics3.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
        self.dim_output3 = self.dim_output
    
    def train(self):
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2, self.dim_input3, self.dim_output3).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        self.model.train()
        #initialize the Balance class
        pareto_solver = Balance(self.model, self.optimizer, self.device, epochs=self.epochs)
        self.label_CSL_omics1 = torch.FloatTensor(self.adata_omics1.obsm['label_CSL']).to(self.device)
        self.label_CSL_omics2 = torch.FloatTensor(self.adata_omics2.obsm['label_CSL']).to(self.device)
        self.label_CSL_omics3 = torch.FloatTensor(self.adata_omics3.obsm['label_CSL']).to(self.device)
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            self.features_omics1a = permutation(self.features_omics1)
            self.features_omics2a = permutation(self.features_omics2)
            self.features_omics3a = permutation(self.features_omics3)
            results = self.model(self.features_omics1, self.features_omics2, self.features_omics3,
                                 self.features_omics1a, self.features_omics2a, self.features_omics3a,
                                 self.adj_spatial_omics1, self.adj_feature_omics1, 
                                 self.adj_spatial_omics2, self.adj_feature_omics2, 
                                 self.adj_spatial_omics3, self.adj_feature_omics3)
            
            constractive_loss12 = self.model.barlow_twins_loss(results['emb_latent_omics1'], results['emb_latent_omics2'])
            constractive_loss12_reverse = self.model.barlow_twins_loss(results['emb_latent_omics2'], results['emb_latent_omics1'])
            constractive_loss12 = (constractive_loss12 + constractive_loss12_reverse) / 2
            constractive_loss13 = self.model.barlow_twins_loss(results['emb_latent_omics1'], results['emb_latent_omics3'])
            constractive_loss13_reverse = self.model.barlow_twins_loss(results['emb_latent_omics3'], results['emb_latent_omics1'])
            constractive_loss13 = (constractive_loss13 + constractive_loss13_reverse) / 2
            constractive_loss23 = self.model.barlow_twins_loss(results['emb_latent_omics2'], results['emb_latent_omics3'])
            constractive_loss23_reverse = self.model.barlow_twins_loss(results['emb_latent_omics3'], results['emb_latent_omics2'])
            constractive_loss23 = (constractive_loss23 + constractive_loss23_reverse) / 2
            constractive_loss = constractive_loss12 + constractive_loss13 + constractive_loss23
            loss_omics1 = pareto_solver.compute_loss_omics(self.label_CSL_omics1, self.features_omics1, results, 'omics1', epoch)
            loss_omics2 = pareto_solver.compute_loss_omics(self.label_CSL_omics2, self.features_omics2, results, 'omics2', epoch)
            loss_omics3 = pareto_solver.compute_loss_omics(self.label_CSL_omics3, self.features_omics3, results, 'omics3', epoch)
            #calculate the total loss
            all_losses = pareto_solver.compute_loss(results, constractive_loss, loss_omics1, loss_omics2, loss_omics3, epoch)
            #calculate the gradients
            grads = pareto_solver.calculate_gradients(all_losses)
            # calculate the weights based on the similarity of the gradients
            weights = pareto_solver.compute_weights_with_similarity(grads) 
            #calculate the total loss
            total_loss = sum(weights[task] * all_losses[task] for task in all_losses.keys())
            # update the model parameters
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        print("Model training finished!\n")    
    
        with torch.no_grad():
          self.model.eval()
          results = self.model(self.features_omics1, self.features_omics2, self.features_omics3,
                               self.features_omics1a, self.features_omics2a, self.features_omics3a,
                               self.adj_spatial_omics1, self.adj_feature_omics1, 
                               self.adj_spatial_omics2, self.adj_feature_omics2, 
                               self.adj_spatial_omics3, self.adj_feature_omics3)
 
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)  
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_omics3 = F.normalize(results['emb_latent_omics3'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        
        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'emb_latent_omics3': emb_omics3.detach().cpu().numpy(),
                  'SpaBalance': emb_combined.detach().cpu().numpy(),
                  'alpha': results['alpha'].detach().cpu().numpy()}
        
        return output
    
 
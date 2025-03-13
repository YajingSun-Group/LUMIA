import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .loss import NTXentLoss, WeightedNTXentLoss
import json
from .model.gnn import RGCN,GCN,GIN,GAT,MPNN
import os
import torch
import torch.nn.functional as F
from utils.pretrain_utils import snapshot,count_model_parameters
import sys

class Pretrainer(object):
    def __init__(self, dataset, config, logger):
        self.args = config
        self.device = config.device
        self.writer = SummaryWriter(log_dir=config.dump_path)
        self.dataset = dataset
        self.logger = logger
        self.nt_xent_criterion = NTXentLoss(self.device,self.args.temperature, self.args.use_cosine_similarity)
        self.weighted_nt_xent_criterion = WeightedNTXentLoss(self.device, self.args.temperature, self.args.use_cosine_similarity, self.args.lambda_1)

    def _build_model(self):
        if self.args.model == 'RGCN':
            model_config = json.load(open(self.args.model_config))
            model = RGCN(**model_config).to(self.device)
            self.logger.info("RGCN model built...")
        if self.args.model == 'GCN':
            model_config = json.load(open(self.args.model_config))
            model = GCN(**model_config).to(self.device)
            self.logger.info("GCN model built...")
        if self.args.model == 'GIN':
            model_config = json.load(open(self.args.model_config))
            model = GIN(**model_config).to(self.device)
            self.logger.info("GIN model built...")
        if self.args.model == 'GAT':
            model_config = json.load(open(self.args.model_config))
            model = GAT(**model_config).to(self.device)
            self.logger.info("GAT model built...")
        if self.args.model == 'MPNN':
            model_config = json.load(open(self.args.model_config))
            model = MPNN(**model_config).to(self.device)
            self.logger.info("MPNN model built...")
        
        if not os.path.exists(os.path.join(self.writer.log_dir, 'checkpoints')):
            os.makedirs(os.path.join(self.writer.log_dir, 'checkpoints'))
        # copy model_config to model_config.json
        with open(os.path.join(self.writer.log_dir, 'checkpoints', 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=4)
        
        # load pre-trained model
        try:
            checkpoints_folder = os.path.join(self.args.resume_from, 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            self.logger.info("Loaded pre-trained model with success.")
        except FileNotFoundError:
            self.logger.info("No pre-trained model found. Training from scratch.")
        
        self.logger.info(f"Model: {model}")
        
        return model
    
    
    def _save_config_file(self,model_checkpoints_folder,args):
        if not os.path.exists(model_checkpoints_folder):
            os.makedirs(model_checkpoints_folder)
            # save args as json
            with open(os.path.join(model_checkpoints_folder, 'config.json'), 'w') as f:
                # fix: Object of type Namespace is not JSON serializable
                args_dict = vars(args)
                json.dump(args_dict, f, indent=4)


    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss_global, valid_loss_sub = 0.0, 0.0
            counter = 0
            for bn, (g1, g2, mols, frag_mols) in enumerate(valid_loader):
                g1 = g1.to(self.device)
                g2 = g2.to(self.device)

                # get the representations and the projections
                __, z1_global, z1_sub = model(g1)  # [N,C]
                __, z2_global, z2_sub = model(g2)  # [N,C]

                # normalize projection feature vectors
                z1_global = F.normalize(z1_global, dim=1)
                z2_global = F.normalize(z2_global, dim=1)
                loss_global = self.weighted_nt_xent_criterion(z1_global, z2_global, mols)

                # normalize projection feature vectors
                z1_sub = F.normalize(z1_sub, dim=1)
                z2_sub = F.normalize(z2_sub, dim=1)
                loss_sub = self.nt_xent_criterion(z1_sub, z2_sub)

                valid_loss_global += loss_global.item()
                valid_loss_sub += loss_sub.item()

                counter += 1
            
            valid_loss_global /= counter
            valid_loss_sub /= counter

        model.train()
        return valid_loss_global, valid_loss_sub
    
    def train(self):
        
        # get data loaders
        train_loader, valid_loader = self.dataset.get_data_loaders()
        
        # build model
        model = self._build_model()
        
        # _,_,_ = count_model_parameters(model)
        # sys.exit(0)
        
        # check if model is on the correct device
        self.logger.info(f"Model Device: {next(model.parameters()).device}")
        
        
        # set optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                     weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs-9, eta_min=0, last_epoch=-1)
        
        # set model checkpoints folder
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        
        self._save_config_file(model_checkpoints_folder,self.args)
        
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        
        torch.cuda.empty_cache()
        
        for epoch_counter in range(self.args.epochs):
            # with tqdm.tqdm(train_loader) as tq_train:
            for bn, (g1, g2, mols, frag_mols) in enumerate(train_loader):
                
                torch.cuda.empty_cache()

                optimizer.zero_grad()
                
                # check if graph is on the correct device
                g1 = g1.to(self.device)
                g2 = g2.to(self.device)
                
                _, z1_global, z1_sub = model(g1)  # [N,C]
                _, z2_global, z2_sub = model(g2)  # [N,C]
                
                # normalize projection feature vectors
                z1_global = F.normalize(z1_global, dim=1)
                z2_global = F.normalize(z2_global, dim=1)
                loss_global = self.weighted_nt_xent_criterion(z1_global, z2_global, mols)
                
                # normalize projection feature vectors
                z1_sub = F.normalize(z1_sub, dim=1)
                z2_sub = F.normalize(z2_sub, dim=1)
                assert z1_sub.size(0) == z2_sub.size(0)
                
                loss_sub = self.nt_xent_criterion(z1_sub, z2_sub)
                
                loss = loss_global + self.args.lambda_2 * loss_sub
                
                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss_global', loss_global, global_step=n_iter)
                    self.writer.add_scalar('loss_sub', loss_sub, global_step=n_iter)
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    self.logger.info(f"Epoch: {epoch_counter}, Batch: {bn}, Loss: {loss.item()}, Loss_global: {loss_global.item()}, Loss_sub: {loss_sub.item()}")
                    
                loss.backward()
                optimizer.step()
                n_iter += 1
                
            # validate the model if requested
            if epoch_counter % self.args.eval_every_n_epochs == 0:
                valid_loss_global, valid_loss_sub = self._validate(model, valid_loader)
                valid_loss = valid_loss_global + 0.5 * valid_loss_sub
                print(epoch_counter, bn, valid_loss_global, valid_loss_sub, valid_loss, '(validation)')
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
            
                self.writer.add_scalar('valid_loss_global', valid_loss_global, global_step=valid_n_iter)
                self.writer.add_scalar('valid_loss_sub', valid_loss_sub, global_step=valid_n_iter)
                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
                

            if (epoch_counter+1) % 1 == 0:
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter))))
                
            if epoch_counter >= self.args.warmup-1:
                scheduler.step()
        
            # save the global model
            if n_iter % 1000 == 0:
                snapshot(model, n_iter, model_checkpoints_folder, 'model')
    
                      
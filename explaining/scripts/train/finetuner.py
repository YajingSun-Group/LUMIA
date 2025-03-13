import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import json

from tqdm import tqdm
from .model.gnn import RGCN
import os
import torch
import torch.nn.functional as F
from .meter import Meter, format_scores, save_scores, Normalizer, organize_results
from .earlystop import EarlyStopping

class Finetuner(object):
    def __init__(self, dataset, config, logger):
        self.args = config
        self.device = config.device
        self.writer = SummaryWriter(log_dir=config.dump_path)
        self.dataset = dataset
        self.logger = logger
        self.task_name = self.args.task_name
        
        if self.args.task in ['classification','multi-task-classification']:
            self.criteria = torch.nn.BCEWithLogitsLoss()
        elif self.args.task in ['regression','multi-task-regression']:
            # if self.args.task in ['qm7', 'qm8']:
            self.criteria = torch.nn.MSELoss()
        
        if self.args.normalize:
            self.normalize = Normalizer(self.dataset.data[self.args.task_list].values)
            # self.normalize = Normalizer(self.dataset.data.query('split == "train"')[self.args.task_list].values)
        else:
            self.normalize = None
            
                
    def build_model(self):
        if self.args.model == 'RGCN':
            model_config = self.args.model_config
            if self.args.task in ['regression','multi-task-regression']:
                model_config['classification'] = False
            elif self.args.task in ['classification','multi-task-classification']:
                model_config['classification'] = True
            model_config['n_tasks'] = len(self.args.task_list)
            model_config['embed_molecular_features'] = self.args.embed_molecular_features
            model_config['molecular_features_dim'] = self.args.molecular_features_dim
            model_config['device'] = self.device
            model = RGCN(**model_config).to(self.device)
            self.logger.info("RGCN model built...")
        
        # load pretrained model weights
        if self.args.resume_from is not None:
            model.load_state_dict(torch.load(os.path.join(self.args.resume_from, 'checkpoints', 'model.pth')))
            self.logger.info("Loaded ckpt model with success.")

        else:
            if self.args.fine_tune_from is not None and not self.args.train_from_scratch:
                checkpoints_folder = os.path.join(self.args.fine_tune_from, 'checkpoints')
                state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
                model.load_my_state_dict(state_dict,self.logger)
                self.logger.info("Loaded pre-trained model with success.")
            elif self.args.train_from_scratch:
                self.logger.info("Training from scratch.")
            else:
                raise ValueError('fine_tune_from is None or not exists')
                
        
        self.logger.info(f"Model: {model}")
        
        self.model = model
        
    
    
    def _save_config_file(self,model_checkpoints_folder,args):
        if not os.path.exists(model_checkpoints_folder):
            os.makedirs(model_checkpoints_folder)
            # save args as json
            with open(os.path.join(model_checkpoints_folder, 'config.json'), 'w') as f:
                # fix: Object of type Namespace is not JSON serializable
                args_dict = vars(args)
                json.dump(args_dict, f, indent=4)

    
    def _run_a_train_epoch(self, train_loader, optimizer):
        self.model.train()
        train_meter = Meter(self.args.task_list, self.normalize)
        total_loss = 0
        n_mol = 0
        
        
        if not self.args.disable_tqdm:
        # use tqdm for progress bar      
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=False, disable=self.args.disable_tqdm)
        else:
            progress_bar = enumerate(train_loader)
      
        for bn, batch_data in progress_bar:
            smiles, bg, labels = batch_data
            bg = bg.to(self.device)
            if self.normalize is not None:
                labels = self.normalize.norm(labels)
            labels = labels.float().to(self.device)
            
            preds, weight, _ = self.model(bg)
            loss = (self.criteria(preds, labels)).mean()
            
            optimizer.zero_grad()
            loss.backward()
            
            total_loss = total_loss + loss * len(smiles)
            n_mol = n_mol + len(smiles)
            optimizer.step()
            
            progress_bar.set_postfix({
            "loss": loss.item(),  # now loss (per example)
            "avg_loss": total_loss / n_mol # running avg loss
                })
            
            train_meter.update(preds, labels,smiles)
            del bg, labels, preds, weight, loss
            torch.cuda.empty_cache()
            
        train_score = train_meter.compute_metric(self.args.task)
        average_loss = total_loss / n_mol
        return train_score, average_loss
            
    def run_an_eval_epoch(self,valid_loader, results_save_path=None):
        # validation steps
        self.model.eval()
        valid_meter = Meter(self.args.task_list, self.normalize)
        total_loss = 0
        n_mol = 0
        
        if not self.args.disable_tqdm:
            progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Validation", leave=False, disable=self.args.disable_tqdm)
        else:
            progress_bar = enumerate(valid_loader)
        
        with torch.no_grad():
            for bn, batch_data in progress_bar:
                smiles, bg, labels = batch_data
                bg = bg.to(self.device)
                if self.normalize is not None:
                    labels = self.normalize.norm(labels)
                labels = labels.float().to(self.device)
                
                preds, weight, _ = self.model(bg)
                loss = (self.criteria(preds, labels)).mean()
                
                total_loss = total_loss + loss * len(smiles)
                n_mol = n_mol + len(smiles)
                valid_meter.update(preds, labels, smiles)
                
                progress_bar.set_postfix({
                "loss": loss.item(),
                "avg_loss": total_loss / n_mol
                    })
                
                del bg, labels, preds, weight, loss
                torch.cuda.empty_cache()
        
        average_loss = total_loss / n_mol
        valid_score = valid_meter.compute_metric(self.args.task)
        
        if results_save_path is not None:
            y_true, y_pred = valid_meter.return_pred_true(save_to_csv=True, csv_path=results_save_path)

        return valid_score, average_loss
    
    def train(self):
        
        # get data loaders
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()
        
        # build model
        self.build_model()
        
        # set optimizer and scheduler
        layer_list = []
        for name, param in self.model.named_parameters():
            if 'project_layers' in name or 'predict' in name or 'readout' in name:
                self.logger.info(f"Fine-tuning layer: {name}")
                layer_list.append(name)
        
        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, self.model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, self.model.named_parameters()))))

        if self.args.train_from_scratch:    
            optimizer = torch.optim.Adam(
                    [{'params': params, 'lr': self.args.lr}, 
                 {'params': base_params, 'lr': self.args.lr * 1}
                ],
                weight_decay=self.args.weight_decay
            )
        
        else:
            optimizer = torch.optim.Adam(
                    [{'params': params, 'lr': self.args.lr}, 
                 {'params': base_params, 'lr': self.args.lr * 0.4}
                ],
                weight_decay=self.args.weight_decay
            )
        
        
        # set model checkpoints folder
        task_name_folder = os.path.join(self.writer.log_dir, self.task_name)
        model_checkpoints_folder = os.path.join(self.writer.log_dir,self.task_name,'checkpoints')
        
        self._save_config_file(model_checkpoints_folder,self.args)
        

        # set early stopping
        stopper = EarlyStopping(patience=self.args.patience, 
                                mode=self.args.early_stopping_mode, 
                                filename=os.path.join(model_checkpoints_folder, 'early_stop.pth'),
                                logger = self.logger,
                                )
        
        torch.cuda.empty_cache()
        for epoch_counter in range(self.args.epochs):
            # with tqdm.tqdm(train_loader) as tq_train:
            train_scores, train_loss = self._run_a_train_epoch(train_loader, optimizer)            
            valid_scores, valid_loss = self.run_an_eval_epoch(valid_loader)
            test_scores, test_loss = self.run_an_eval_epoch(test_loader)
            
            self.logger.info(f"Epoch: {epoch_counter}, "
                            f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Test Loss: {test_loss:.4f}, "
                            f"Train Scores: [{format_scores(train_scores)}], "
                            f"Valid Scores: [{format_scores(valid_scores)}], "
                            f"Test Scores: [{format_scores(test_scores)}]")
            
            self.writer.add_scalar('train_loss', train_loss, global_step=epoch_counter)
            self.writer.add_scalar('valid_loss', valid_loss, global_step=epoch_counter)
            self.writer.add_scalar('test_loss', test_loss, global_step=epoch_counter)
            
            early_stop = stopper.step(test_scores[self.args.early_stopping_metric], self.model)
            
            if early_stop:
                self.logger.info("Early stopping triggered.")
                break
        
        # load the best model
        stopper.load_checkpoint(self.model)
        train_scores, train_loss = self.run_an_eval_epoch(train_loader, results_save_path=os.path.join(task_name_folder, 'prediction_train_results.csv'))
        valid_scores, valid_loss = self.run_an_eval_epoch(valid_loader, results_save_path=os.path.join(task_name_folder, 'prediction_valid_results.csv'))
        test_scores, test_loss = self.run_an_eval_epoch(test_loader, results_save_path=os.path.join(task_name_folder, 'prediction_test_results.csv'))
        
        # print the final results
        self.logger.info('*'*50)
        self.logger.info(f"Train Scores: [{format_scores(train_scores)}], "
                        f"Valid Scores: [{format_scores(valid_scores)}], "
                        f"Test Scores: [{format_scores(test_scores)}]")
        self.logger.info('*'*50)
        
        # save scores
        save_scores(self.args, train_scores, valid_scores, test_scores, os.path.join(task_name_folder, 'scores.csv'))

        # organize results
        organize_results(task_name_folder, self.args.task_list)

    def project(self,data_loader,tag='train'):
        assert self.model is not None, "Model is not loaded."
        self.model.eval()
        # results = []
        node_feats_all = []
        graph_feats_all = []
        h_all = []
        smiles_all = []
        self.logger.info(f"Projecting {tag} data...")
        with torch.no_grad():
            for bn, batch_data in enumerate(data_loader):
                smiles, bg, labels = batch_data
                bg = bg.to(self.device)
                labels = labels.float().to(self.device)
                
                preds, weight, project_feats = self.model(bg)
                rgcn_node_feats, graph_feats, h = project_feats
                
                # results.append([smiles, preds.cpu().numpy(), labels.cpu().numpy()])
                node_feats_all.append(rgcn_node_feats.cpu().numpy())
                graph_feats_all.append(graph_feats.cpu().numpy())
                h_all.append(h.cpu().numpy())
                smiles_all.extend(smiles)
        self.logger.info(f"Projecting {tag} data finished.")
        
        # save results
        # results_save_path = os.path.join(self.writer.log_dir, f'prediction_results_{tag}.csv')
        graph_feats_save_path = os.path.join(self.writer.log_dir, f'graph_feats_{tag}.npy')
        node_feats_save_path = os.path.join(self.writer.log_dir, f'node_feats_{tag}.npy')
        h_save_path = os.path.join(self.writer.log_dir, f'h_{tag}.npy')
        
        np.save(graph_feats_save_path, np.concatenate(graph_feats_all, axis=0))
        np.save(node_feats_save_path, np.concatenate(node_feats_all, axis=0))
        np.save(h_save_path, np.concatenate(h_all, axis=0))
        # pd.DataFrame(results, columns=['smiles', 'preds', 'labels']).to_csv(results_save_path)
        np.save(os.path.join(self.writer.log_dir, f'smiles_{tag}.npy'), np.array(smiles_all))
        
    def predict(self,data_loader,tag='test'):
        assert self.model is not None, "Model is not loaded."
        self.model.eval()
        results = []
        with torch.no_grad():
            for bn, batch_data in enumerate(data_loader):
                smiles, bg, labels = batch_data
                bg = bg.to(self.device)
                labels = labels.float().to(self.device)
                
                preds, weight, project_feats = self.model(bg)
                results.append([smiles, preds.cpu().numpy(), labels.cpu().numpy()])
                
        results_save_path = os.path.join(self.writer.log_dir, f'prediction_results_{tag}.csv')
        pd.DataFrame(results, columns=['smiles', 'preds', 'labels']).to_csv(results_save_path)
        
        return results

                      
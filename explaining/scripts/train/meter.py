import os
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score
import pandas as pd

class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""
    
    def __init__(self,task_list,normalizer=None):
        self.y_pred = []
        self.y_true = []
        self.smiles = []  # 用于记录SMILES字符串
        # self.losses = []  # 用于记录所有的loss
        self.task_list = task_list
        self.normalizer = normalizer
        
    def update(self, y_pred, y_true, smiles=None):
        """Update for the result of an iteration
        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        smiles : list of str, optional
            List of SMILES strings corresponding to the batch.
        loss : float32 tensor, optional
            Loss value to record for this iteration.
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        if smiles is not None:
            self.smiles.extend(smiles)  # 更新SMILES
        # if loss is not None:
        #     self.losses.append(loss.detach().cpu().item())  # 记录loss
    
    def accuracy_score(self):
        """Compute accuracy score for each task.
        Returns
        -------
        float
            Accuracy score for all tasks
        """
        y_pred = torch.cat(self.y_pred, dim=0)
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.numpy()
        y_pred_label = np.array([pro2label(x) for x in y_pred])
        y_true = torch.cat(self.y_true, dim=0).numpy()
        scores = round(accuracy_score(y_true, y_pred_label), 4)
        return scores
    
    def auroc(self):
        """Compute Area Under the Receiver Operating Characteristic Curve (AUROC).
        Returns
        -------
        float
            AUROC score for all tasks
        """
        y_pred = torch.cat(self.y_pred, dim=0).numpy()
        y_true = torch.cat(self.y_true, dim=0).numpy()
        
        # 计算整体 AUROC
        scores = round(roc_auc_score(y_true, y_pred), 4)
        return scores
    
    def r2(self):
        """Compute R2 score.
        Returns
        -------
        float
            R2 score
        """
        y_pred = torch.cat(self.y_pred, dim=0).numpy()
        y_true = torch.cat(self.y_true, dim=0).numpy()
        scores = round(r2_score(y_true, y_pred), 4)
        return scores
    
    def mae(self):
        """Compute Mean Absolute Error (MAE).
        Returns
        -------
        float
            MAE score
        """
        y_pred = torch.cat(self.y_pred, dim=0).numpy()
        y_true = torch.cat(self.y_true, dim=0).numpy()
        scores = round(mean_absolute_error(y_true, y_pred), 4)
        return scores
    
    def rmse(self):
        """Compute Root Mean Squared Error (RMSE).
        Returns
        -------
        float
            RMSE score
        """
        y_pred = torch.cat(self.y_pred, dim=0).numpy()
        y_true = torch.cat(self.y_true, dim=0).numpy()
        scores = round(np.sqrt(mean_squared_error(y_true, y_pred)), 4)
        return scores
    
    def return_pred_true(self, save_to_csv=False, csv_path='predictions.csv'):
        """Return predictions and true values.
        Optionally save them to a CSV file.
        Parameters
        ----------
        save_to_csv : bool, optional
            Whether to save the predictions and true values to a CSV file.
        csv_path : str, optional
            The path to save the CSV file.
        Returns
        -------
        tuple of tensors
            Predictions and true values.
        """
        y_pred = torch.cat(self.y_pred, dim=0).numpy()
        y_true = torch.cat(self.y_true, dim=0).numpy()
        if self.normalizer is not None:
            y_pred = self.normalizer.denorm(y_pred)
            y_true = self.normalizer.denorm(y_true)

        if save_to_csv:
            # 创建一个字典来存储所有列
            data_dict = {'smiles': self.smiles}
            
            # 为每个输出维度添加真实值和预测值列
            for i in range(y_true.shape[1]):
                data_dict[f'label_{self.task_list[i]}'] = y_true[:, i]
                data_dict[f'pred_{self.task_list[i]}'] = y_pred[:, i]
            
            df = pd.DataFrame(data_dict)
            df.to_csv(csv_path, index=False)
        
        return y_true, y_pred
        
    def compute_metric(self, task_type):
        """Compute metrics for either regression or classification tasks.
        Parameters
        ----------
        task_type : str
            Either 'regression' or 'classification'. Determines which metrics to compute.
        Returns
        -------
        dict
            A dictionary with metric names as keys and their corresponding values.
        """
        assert task_type in ['regression', 'classification','multi-task-classification','multi-task-regression'], \
            'Expect task_type to be either "regression" or "classification", got {}'.format(task_type)

        metrics = {}


        if task_type in ['classification', 'multi-task-classification']:
            metrics['accuracy'] = self.accuracy_score()
            metrics['auroc'] = self.auroc()

        elif task_type in ['regression', 'multi-task-regression']:
            metrics['r2'] = self.r2()
            metrics['mae'] = self.mae()
            metrics['rmse'] = self.rmse()

        return metrics
    
    # def get_losses(self):
    #     """Return the recorded losses.
    #     Returns
    #     -------
    #     list of float
    #         List of all recorded losses.
    #     """
    #     return self.losses
    
def format_scores(scores):
    """格式化字典中的分数，按照 'key: value' 的形式输出"""
    return ', '.join([f"{key}: {value:.4f}" for key, value in scores.items()])
  
def save_scores(args,train_scores,valid_score, test_scores, path):
    
    result_pd = pd.DataFrame()
    if args.task in ['classification', 'multi-task-classification']:
        result_pd['index'] = ['accuracy', 'auroc']

    else:
        result_pd['index'] = ['r2', 'mae', 'rmse']

    
    stop_train_list = []
    stop_val_list = []
    stop_test_list = []
    
    for key in train_scores.keys():
        stop_train_list.append(train_scores[key])
        stop_val_list.append(valid_score[key])
        stop_test_list.append(test_scores[key])    
    
    result_pd['train'] = stop_train_list
    result_pd['val'] = stop_val_list
    result_pd['test'] = stop_test_list
    
    result_pd.to_csv(path, index=False)
    
    
def pro2label(x):
    if x < 0.5:
        return 0
    else:
        return 1
    
    

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        # 确保输入是 torch.Tensor 类型
        if isinstance(tensor, np.ndarray):
            tensor = torch.FloatTensor(tensor)
        elif isinstance(tensor, pd.DataFrame):
            tensor = torch.FloatTensor(tensor.values)
        elif isinstance(tensor, pd.Series):
            tensor = torch.FloatTensor(tensor.values)
        self.mean = torch.mean(tensor, dim=0)
        self.std = torch.std(tensor, dim=0)
        
        # avoid divide by zero
        self.std[self.std < 1e-12] = 1.0

    def norm(self, tensor):
        if isinstance(tensor, np.ndarray):
            tensor = torch.FloatTensor(tensor)
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        if isinstance(normed_tensor, np.ndarray):
            normed_tensor = torch.FloatTensor(normed_tensor)
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_my_state_dict(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)



def organize_results(results_path, task_list):
    train_predictions = pd.read_csv(os.path.join(results_path, 'prediction_train_results.csv'))
    valid_predictions = pd.read_csv(os.path.join(results_path, 'prediction_valid_results.csv'))
    test_predictions = pd.read_csv(os.path.join(results_path, 'prediction_test_results.csv'))
    
    # 创建一个字典来存储所有结果
    results_dict = {
        'property': task_list,  # 使用task_list作为index
        'train_mae': [],
        'valid_mae': [],
        'test_mae': [],
        'train_rmse': [],
        'valid_rmse': [],
        'test_rmse': [],
        'train_r2': [],
        'valid_r2': [],
        'test_r2': []
    }
    
    for task in task_list:
        # calculate the mae, rmse, r2 for each task
        train_mae = mean_absolute_error(train_predictions[f'label_{task}'], train_predictions[f'pred_{task}'])
        valid_mae = mean_absolute_error(valid_predictions[f'label_{task}'], valid_predictions[f'pred_{task}'])
        test_mae = mean_absolute_error(test_predictions[f'label_{task}'], test_predictions[f'pred_{task}'])
        
        train_rmse = np.sqrt(mean_squared_error(train_predictions[f'label_{task}'], train_predictions[f'pred_{task}']))
        valid_rmse = np.sqrt(mean_squared_error(valid_predictions[f'label_{task}'], valid_predictions[f'pred_{task}']))
        test_rmse = np.sqrt(mean_squared_error(test_predictions[f'label_{task}'], test_predictions[f'pred_{task}']))
        
        train_r2 = r2_score(train_predictions[f'label_{task}'], train_predictions[f'pred_{task}'])
        valid_r2 = r2_score(valid_predictions[f'label_{task}'], valid_predictions[f'pred_{task}'])
        test_r2 = r2_score(test_predictions[f'label_{task}'], test_predictions[f'pred_{task}'])
        
        # 将结果添加到对应的列表中
        results_dict['train_mae'].append(train_mae)
        results_dict['valid_mae'].append(valid_mae)
        results_dict['test_mae'].append(test_mae)
        results_dict['train_rmse'].append(train_rmse)
        results_dict['valid_rmse'].append(valid_rmse)
        results_dict['test_rmse'].append(test_rmse)
        results_dict['train_r2'].append(train_r2)
        results_dict['valid_r2'].append(valid_r2)
        results_dict['test_r2'].append(test_r2)
    
    # 创建DataFrame并设置index
    results_pd = pd.DataFrame(results_dict)
    results_pd = results_pd.set_index('property')
    
    # 保存结果
    results_pd.to_csv(os.path.join(results_path, 'results_summary.csv'))
    
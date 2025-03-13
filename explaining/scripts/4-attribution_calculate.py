import pandas as pd
import numpy as np
import os

for task_name in ['delta_est','s0t1','s0s1','hl']:
    for sub_type in ['brics']:
        for CV_NUM in range(0,5):
            attribution_result = pd.DataFrame()
            print('{} {}'.format(task_name, sub_type))

            result_sub = pd.read_csv('../prediction/summary/{}_{}_cv_{}_prediction_summary.csv'.format(task_name, sub_type, CV_NUM))
            result_mol = pd.read_csv('../prediction/summary/{}_{}_cv_{}_prediction_summary.csv'.format(task_name, 'mol', CV_NUM))

            # 使用merge来替代手动查找
            merged_result = pd.merge(result_sub, result_mol[['smiles', 'pred_mean', 'pred_std']], on='smiles', how='left',
                                    suffixes=('_sub', '_mol'))

            attribution_result['smiles'] = merged_result['smiles']
            attribution_result['label'] = merged_result['label']
            attribution_result['sub_name'] = merged_result['sub_name']
            attribution_result['split'] = merged_result['split']
            attribution_result['sub_pred_mean'] = merged_result['pred_mean_sub']
            attribution_result['sub_pred_std'] = merged_result['pred_std_sub']
            attribution_result['mol_pred_mean'] = merged_result['pred_mean_mol']
            attribution_result['mol_pred_std'] = merged_result['pred_std_mol']

            attribution_result['attribution'] = attribution_result['mol_pred_mean'] - attribution_result['sub_pred_mean']
            attribution_result['attribution_normalized'] = (np.exp(attribution_result['attribution'].values) - np.exp(
                -attribution_result['attribution'].values)) / (np.exp(attribution_result['attribution'].values) + np.exp(
                -attribution_result['attribution'].values))

            dirs = '../prediction/attribution/'
            if not os.path.exists(dirs):
                os.makedirs(dirs)

            attribution_result.to_csv('{}/{}_{}_cv_{}_attribution_summary.csv'.format(dirs,task_name, sub_type, CV_NUM), index=False)

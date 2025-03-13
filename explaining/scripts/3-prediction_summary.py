import pandas as pd
import os
# task_list = ['ESOL', 'Mutagenicity', 'hERG']
task_list = ['delta_est','s0t1','s0s1','hl']
# sub_type_list = ['mol','brics']
sub_type_list = ['mol', 'brics']
# sub_type_list = ['mol', 'fg', 'murcko', 'brics', 'brics_emerge', 'murcko_emerge']

# sub_type_list = ['mol', 'fg', 'murcko', 'brics', 'brics_emerge', 'murcko_emerge']
# task_list = ['Mutagenicity_data_hERG_task']
# sub_type_list = ['mol']

for task_name in task_list:
    for sub_type in sub_type_list:
        # try:
        # 将训练集，验证集，测试集数据合并
        for CV_NUM in range(0,5):
            result_summary = pd.DataFrame()
            for i in range(0,1):
                seed = i + 1
                result_train = pd.read_csv('../prediction/{}/{}_{}_{}_cv_{}_train_prediction.csv'.format(sub_type, task_name, sub_type, seed, CV_NUM))
                result_val = pd.read_csv('../prediction/{}/{}_{}_{}_cv_{}_val_prediction.csv'.format(sub_type, task_name, sub_type, seed, CV_NUM))
                result_test = pd.read_csv('../prediction/{}/{}_{}_{}_cv_{}_test_prediction.csv'.format(sub_type, task_name, sub_type, seed, CV_NUM))
                group_list = ['train' for x in range(len(result_train))] + ['valid' for x in range(len(result_val))] + ['test' for x in range(len(result_test))]
                result = pd.concat([result_train, result_val, result_test], axis=0)
                
                # mol是模型最初预测的时候给的结果，batch是会随机乱序的，所以需要重新排序
                result['split'] = group_list
                if sub_type == 'mol':
                    result.sort_values(by='smiles', inplace=True)
                # 合并五个随机种子结果，并统计方差和均值
                if seed == 1:
                    result_summary['smiles'] = result['smiles']
                    result_summary['label'] = result['label']
                    result_summary['sub_name'] = result['sub_name']
                    result_summary['split'] = result['split']
                    result_summary['pred_{}'.format(seed)] = result['pred'].tolist()
                if seed > 1:
                    result_summary['pred_{}'.format(seed)] = result['pred'].tolist()
            pred_columnms = ['pred_{}'.format(i+1) for i in range(0,1)]
            data_pred = result_summary[pred_columnms]
            result_summary['pred_mean'] = data_pred.mean(axis=1)
            result_summary['pred_std'] = data_pred.std(axis=1)
            dirs = '../prediction/summary/'
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            result_summary.to_csv('../prediction/summary/{}_{}_cv_{}_prediction_summary.csv'.format(task_name, sub_type, CV_NUM), index=False)
            print('{} {} sum succeed.'.format(task_name, sub_type))

        # except:
            # print('{} {} sum failed.'.format(task_name, sub_type))
import os
import shutil
import pandas as pd


## INDO
import os
import shutil
import pandas as pd


# 1. 先生成group data
five_fold_path = '/home/qianzhang/MyProject/LUMIA/dataset/finetune/ocelot/5_fold_cross_validation'
graph_data_path = '../data/graph_data'
prediction_path = '/home/qianzhang/MyProject/LUMIA/fine-tuning/dumped/1213-finetune-ocelot-no-embed'
mol_path = '../prediction/mol'
ATTR = 'hl'
for i in range(5):
    fold_path = os.path.join(five_fold_path, f'ocelot_clean_5fold_{i}.csv')
    # 读取目标数据
    target_data = pd.read_csv(fold_path)
    
    # 读取需要更新的group文件
    hr_group_for_mol = pd.read_csv(os.path.join(graph_data_path, f'{ATTR}_group.csv'))
    # hr_group_for_fg = pd.read_csv(os.path.join(graph_data_path, f'{ATTR}_group_for_fg.csv'))
    hr_group_for_brics = pd.read_csv(os.path.join(graph_data_path, f'{ATTR}_group_for_brics.csv'))
    # hr_group_for_skelton = pd.read_csv(os.path.join(graph_data_path, f'{ATTR}_group_for_skelton.csv'))
    # hr_group_for_side_chain = pd.read_csv(os.path.join(graph_data_path, f'{ATTR}_group_for_side_chain.csv'))
    
    # 创建smiles到split的映射
    target_data_unique = target_data.drop_duplicates(subset='smiles')
    mapping = dict(zip(target_data_unique['smiles'], target_data_unique['split']))
    
    # 更新split列
    hr_group_for_mol['split'] = hr_group_for_mol['smiles'].map(mapping)
    # hr_group_for_fg['split'] = hr_group_for_fg['smiles'].map(mapping)
    hr_group_for_brics['split'] = hr_group_for_brics['smiles'].map(mapping)
    # hr_group_for_skelton['split'] = hr_group_for_skelton['smiles'].map(mapping)
    # hr_group_for_side_chain['split'] = hr_group_for_side_chain['smiles'].map(mapping)
        
    # 保存更新后的文件
    hr_group_for_mol.to_csv(os.path.join(graph_data_path, f'{ATTR}_group_for_mol_{i}.csv'), index=False)
    # hr_group_for_fg.to_csv(os.path.join(graph_data_path, f'{ATTR}_group_for_fg_{i}.csv'), index=False)
    hr_group_for_brics.to_csv(os.path.join(graph_data_path, f'{ATTR}_group_for_brics_{i}.csv'), index=False)
    # hr_group_for_skelton.to_csv(os.path.join(graph_data_path, f'{ATTR}_group_for_skelton_{i}.csv'), index=False)
    # hr_group_for_side_chain.to_csv(os.path.join(graph_data_path, f'{ATTR}_group_for_side_chain_{i}.csv'), index=False)
    
    # 2. 复制 prediction文件
    hr_path = os.path.join(prediction_path, f'cross_validation_{i}/{ATTR}')
    
    prediction_test_file = os.path.join(hr_path, 'prediction_test_results.csv')
    prediction_train_file = os.path.join(hr_path, 'prediction_train_results.csv')
    prediction_val_file = os.path.join(hr_path, 'prediction_valid_results.csv')
    
    # 复制到/home/qianzhang/MyProject/LUMIA/explaining/prediction/mol
    # shutil.copy(prediction_test_file, os.path.join(mol_path, 'hr_mol_1_cv_{}_test_prediction.csv'.format(i)))
    test_result = pd.read_csv(prediction_test_file)
    train_result = pd.read_csv(prediction_train_file) 
    valid_result = pd.read_csv(prediction_val_file)

    # 添加sub_name列
    test_result['sub_name'] = 'noname'
    train_result['sub_name'] = 'noname'
    valid_result['sub_name'] = 'noname'

    # 重命名列
    test_result = test_result.rename(columns={f'label_{ATTR}': 'label', f'pred_{ATTR}': 'pred'})
    train_result = train_result.rename(columns={f'label_{ATTR}': 'label', f'pred_{ATTR}': 'pred'})
    valid_result = valid_result.rename(columns={f'label_{ATTR}': 'label', f'pred_{ATTR}': 'pred'})

    # 确保目标目录存在
    os.makedirs(mol_path, exist_ok=True)

    # 保存文件
    test_result.to_csv(os.path.join(mol_path, f'{ATTR}_mol_1_cv_{i}_test_prediction.csv'), index=False)
    train_result.to_csv(os.path.join(mol_path, f'{ATTR}_mol_1_cv_{i}_train_prediction.csv'), index=False)
    valid_result.to_csv(os.path.join(mol_path, f'{ATTR}_mol_1_cv_{i}_val_prediction.csv'), index=False)



# # 1. 先生成group data
# five_fold_path = '/home/qianzhang/MyProject/LUMIA/dataset/finetune/ocelot/5_fold_cross_validation'
# graph_data_path = '../data/graph_data'
# prediction_path = '/home/qianzhang/MyProject/LUMIA/fine-tuning/dumped/1213-finetune-ocelot-no-embed'
# mol_path = '../prediction/mol'
# ATTR = 'hr'
# for i in range(5):
#     fold_path = os.path.join(five_fold_path, f'ocelot_clean_5fold_{i}.csv')
#     # 读取目标数据
#     target_data = pd.read_csv(fold_path)
    
#     # 读取需要更新的group文件
#     hr_group_for_mol = pd.read_csv(os.path.join(graph_data_path, f'{ATTR}_group_for_mol.csv'))
#     hr_group_for_fg = pd.read_csv(os.path.join(graph_data_path, f'{ATTR}_group_for_fg.csv'))
#     hr_group_for_brics = pd.read_csv(os.path.join(graph_data_path, f'{ATTR}_group_for_brics.csv'))
    
#     # 创建smiles到split的映射
#     target_data_unique = target_data.drop_duplicates(subset='smiles')
#     mapping = dict(zip(target_data_unique['smiles'], target_data_unique['split']))
    
#     # 更新split列
#     hr_group_for_mol['split'] = hr_group_for_mol['smiles'].map(mapping)
#     hr_group_for_fg['split'] = hr_group_for_fg['smiles'].map(mapping)
#     hr_group_for_brics['split'] = hr_group_for_brics['smiles'].map(mapping)
    
#     # 保存更新后的文件
#     hr_group_for_mol.to_csv(os.path.join(graph_data_path, f'{ATTR}_group_for_mol_{i}.csv'), index=False)
#     hr_group_for_fg.to_csv(os.path.join(graph_data_path, f'{ATTR}_group_for_fg_{i}.csv'), index=False)
#     hr_group_for_brics.to_csv(os.path.join(graph_data_path, f'{ATTR}_group_for_brics_{i}.csv'), index=False)
    
    
#     # 2. 复制 prediction文件
#     hr_path = os.path.join(prediction_path, f'cross_validation_{i}/hr')
    
#     prediction_test_file = os.path.join(hr_path, 'prediction_test_results.csv')
#     prediction_train_file = os.path.join(hr_path, 'prediction_train_results.csv')
#     prediction_val_file = os.path.join(hr_path, 'prediction_valid_results.csv')
    
#     # 复制到/home/qianzhang/MyProject/LUMIA/explaining/prediction/mol
#     # shutil.copy(prediction_test_file, os.path.join(mol_path, 'hr_mol_1_cv_{}_test_prediction.csv'.format(i)))
#     test_result = pd.read_csv(prediction_test_file)
#     train_result = pd.read_csv(prediction_train_file) 
#     valid_result = pd.read_csv(prediction_val_file)

#     # 添加sub_name列
#     test_result['sub_name'] = 'noname'
#     train_result['sub_name'] = 'noname'
#     valid_result['sub_name'] = 'noname'

#     # 重命名列
#     test_result = test_result.rename(columns={'label_hr': 'label', 'pred_hr': 'pred'})
#     train_result = train_result.rename(columns={'label_hr': 'label', 'pred_hr': 'pred'})
#     valid_result = valid_result.rename(columns={'label_hr': 'label', 'pred_hr': 'pred'})

#     # 确保目标目录存在
#     os.makedirs(mol_path, exist_ok=True)

#     # 保存文件
#     test_result.to_csv(os.path.join(mol_path, f'{ATTR}_mol_1_cv_{i}_test_prediction.csv'), index=False)
#     train_result.to_csv(os.path.join(mol_path, f'{ATTR}_mol_1_cv_{i}_train_prediction.csv'), index=False)
#     valid_result.to_csv(os.path.join(mol_path, f'{ATTR}_mol_1_cv_{i}_val_prediction.csv'), index=False)

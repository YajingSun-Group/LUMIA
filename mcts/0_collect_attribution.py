import os
import pandas as pd
import numpy as np
from rdkit import Chem
import pickle
import tqdm
from multiprocessing import Pool, cpu_count

def process_molecule(smiles):
    """处理单个分子的函数"""
    mol = Chem.MolFromSmiles(smiles)
    n_atom = mol.GetNumAtoms()
    attribution = data[data['smiles'] == smiles].attribution.tolist()
    
    return {
        'smiles': smiles,
        'atom_contributions': {i: attribution[i] for i in range(n_atom)}
    }
def find_best_cv(smiles, task_name='delta_est'):
    """找到给定SMILES在所有CV中MAE最小的那个CV"""
    mae_per_cv = []
    for cv in range(5):
        data_cv = pd.read_csv(f'../explaining/lumia/prediction/attribution/{task_name}_brics_cv_{cv}_attribution_summary.csv')
        mol = Chem.MolFromSmiles(smiles)
        smiles_canon = Chem.MolToSmiles(mol)
        mol_data = data_cv[data_cv['smiles'] == smiles_canon]
        
        if not mol_data.empty:
            mae = abs(mol_data.mol_pred_mean.values[0] - mol_data.label.values[0])
            mae_per_cv.append(mae)
        else:
            mae_per_cv.append(float('inf'))
    
    return np.argmin(mae_per_cv)


def find_best_cv_wrapper(args):
    smiles, attr = args
    return find_best_cv(smiles, attr)
    
if __name__ == '__main__':
    
    ATTR = 'hr'
    
    for CV_NUM in range(5):
        data = pd.read_csv('../explaining/lumia/prediction/attribution/{}_brics_cv_{}_attribution_summary.csv'.format(ATTR, CV_NUM))
        smiles_list = data['smiles'].unique()

        # 创建进程池，使用CPU核心数量的进程
        n_cores = cpu_count()
        print(f"使用 {n_cores} 个CPU核心进行并行计算")
        
        # 使用进程池进行并行计算
        with Pool(n_cores) as pool:
            attribution_dict_list = list(tqdm.tqdm(
                pool.imap(process_molecule, smiles_list),
                total=len(smiles_list),
                desc=f'Processing CV_{CV_NUM}'
            ))

        # 保存为pkl文件
        os.makedirs(f'./{ATTR}', exist_ok=True)
        with open(f'./{ATTR}/attribution_cv_{CV_NUM}.pkl', 'wb') as f:
            pickle.dump(attribution_dict_list, f)
    


   # 为每个SMILES找到最佳的CV
    print("查找每个SMILES的最佳CV...")
    with Pool(n_cores) as pool:
        best_cv_results = list(tqdm.tqdm(
        pool.imap(find_best_cv_wrapper, [(smiles, ATTR) for smiles in smiles_list]),
        total=len(smiles_list),
        desc='Finding best CV'
        ))
    best_cv_dict = dict(zip(smiles_list, best_cv_results))

    # 定义收集attribution的函数
    def collect_best_attribution(smiles):
        best_cv = best_cv_dict[smiles]
        data = pd.read_csv(f'../explaining/lumia/prediction/attribution/{ATTR}_brics_cv_{best_cv}_attribution_summary.csv')
        mol = Chem.MolFromSmiles(smiles)
        n_atom = mol.GetNumAtoms()
        attribution = data[data['smiles'] == smiles].attribution.tolist()
        
        return {
            'smiles': smiles,
            'best_cv': best_cv,
            'atom_contributions': {i: attribution[i] for i in range(n_atom)}
        }

    # 并行收集最佳attribution
    print("收集最佳attribution...")
    with Pool(n_cores) as pool:
        final_attribution_list = list(tqdm.tqdm(
            pool.imap(collect_best_attribution, smiles_list),
            total=len(smiles_list),
            desc='Collecting best attribution'
        ))
    
    # 创建保存目录
    
    # 保存最终结果
    output_file = f'./{ATTR}/best_attribution.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(final_attribution_list, f)
    
    print(f"最佳attribution结果已保存到: {output_file}")

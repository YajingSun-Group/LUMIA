import json
import numpy as np
import build_data
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from train.model.gnn import RGCN
from train.meter import Meter, format_scores, save_scores, Normalizer, organize_results


from maskgnn import collate_molgraphs, EarlyStopping, run_a_train_epoch, \
    run_an_eval_epoch, set_random_seed, pos_weight
import pickle as pkl
import time
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# fix parameters of model
def SMEG_explain_for_substructure(task_name, hyperparameter, rgcn_hidden_feats=[64, 64, 64], ffn_hidden_feats=128,
               lr=0.0003, classification=True, sub_type='fg'):
    args = {}
    args['device'] = "cuda"
    args['node_data_field'] = 'node'
    args['edge_data_field'] = 'edge'
    args['substructure_mask'] = 'smask'
    args['classification'] = classification
    # model parameter
    args['num_epochs'] = 1500
    args['patience'] = 30
    args['batch_size'] = 1024
    args['mode'] = 'higher'

    args['lr'] = lr
    args['loop'] = True
    # task name (model name)
    args['task_name'] = task_name  # change
    args['data_name'] = task_name  # change
    args['bin_path'] = '../data/graph_data/{}_for_{}.bin'.format(args['data_name'], sub_type)
    args['group_path'] = '../data/graph_data/{}_group_for_{}_{}.csv'.format(args['data_name'], sub_type, CV_NUM)
    args['smask_path'] = '../data/graph_data/{}_smask_for_{}.npy'.format(args['data_name'], sub_type)
    args['seed'] = 0
    
    
    if hyperparameter['normalize']:
        dataset_task_data = pd.read_csv(hyperparameter['dataset_path'])[task_name]
        normalize = Normalizer(dataset_task_data)
        
        if task_name == 'indo':
            normalize.load_my_state_dict('./normalizer_hr.pth')
        
    
    else:
        normalize = None
    
    
    
    print('***************************************************************************************************')
    print('{}, seed {}, substructure type {}'.format(args['task_name'], args['seed']+1, sub_type))
    print('***************************************************************************************************')
    train_set, val_set, test_set, task_number = build_data.load_graph_from_csv_bin_for_splited(
        bin_path=args['bin_path'],
        group_path=args['group_path'],
        smask_path=args['smask_path'],
        classification=False,
        random_shuffle=False,
        normalize=normalize,
        additional_features_path=hyperparameter['additional_features']
    )
    
    print("Molecule graph is loaded!")
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              collate_fn=collate_molgraphs)
    
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs)
    
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)
    if args['classification']:
        pos_weight_np = pos_weight(train_set)
        loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='none',
                                                    pos_weight=pos_weight_np.to(args['device']))
    else:
        loss_criterion = torch.nn.MSELoss(reduction='none')
    
    model = RGCN(**hyperparameter['model_config']).to(hyperparameter['device'])
    checkpoint = torch.load(
        os.path.join(hyperparameter['resume_from'], 'checkpoints', 'early_stop.pth'),
        map_location=hyperparameter['device']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print('model is loaded!')
    

    
    model.to(args['device'])
    pred_name = '{}_{}_{}'.format(args['task_name'], sub_type, args['seed'] + 1)
    
    save_dir = '../prediction/{}'.format(sub_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    stop_test_list, _ = run_an_eval_epoch(args, model, test_loader, loss_criterion,
                                          out_path='../prediction/{}/{}_cv_{}_test'.format(sub_type, pred_name, CV_NUM),normalize=normalize,embed_molecular_features=hyperparameter['model_config']['embed_molecular_features'])
    stop_train_list, _ = run_an_eval_epoch(args, model, train_loader, loss_criterion,
                                           out_path='../prediction/{}/{}_cv_{}_train'.format(sub_type, pred_name, CV_NUM),normalize=normalize,embed_molecular_features=hyperparameter['model_config']['embed_molecular_features'])
    stop_val_list, _ = run_an_eval_epoch(args, model, val_loader, loss_criterion,
                                         out_path='../prediction/{}/{}_cv_{}_val'.format(sub_type, pred_name, CV_NUM),normalize=normalize,embed_molecular_features=hyperparameter['model_config']['embed_molecular_features'])
    print('Mask prediction is generated!')



CV_NUM=4

for task in ['delta_est','s0t1','s0s1', 'hl']:
    # for sub_type in ['fg', 'murcko', 'brics', 'brics_emerge', 'murcko_emerge']:
    for sub_type in ['brics']:
        print('task: {}, sub_type: {}'.format(task, sub_type))
    # for sub_type in ['fg']:
    # for sub_type in ['murcko', 'brics', 'brics_emerge', 'murcko_emerge']:
        # load
        if task == 'hr':
            resume_path = '/home/qianzhang/MyProject/LUMIA/fine-tuning/dumped/1213-finetune-ocelot-no-embed/cross_validation_{}/hr'.format(CV_NUM)
            hyperparameter = json.load(open(os.path.join(resume_path, 'checkpoints', 'config.json')))
            hyperparameter['resume_from'] = resume_path
            hyperparameter['dataset_path'] = '../data/origin_data/hr.csv'
            if hyperparameter['model_config']['embed_molecular_features'] == True:
                hyperparameter['additional_features'] = '/home/qianzhang/MyProject/LUMIA/dataset/finetune/ocelot/ocelot_descriptors_processed.csv'
            else:
                hyperparameter['additional_features'] = None
        elif task == 'indo':
            resume_path = '/home/qianzhang/MyProject/LUMIA/fine-tuning/dumped/1213-finetune-ocelot-no-embed/cross_validation_{}/hr'.format(CV_NUM)
            hyperparameter = json.load(open(os.path.join(resume_path, 'checkpoints', 'config.json')))
            hyperparameter['resume_from'] = resume_path
            hyperparameter['dataset_path'] = '../data/origin_data/indo.csv'
            hyperparameter['additional_features'] = None
        elif task == 'delta_est':
            resume_path = '/home/qianzhang/MyProject/LUMIA/fine-tuning/dumped/1213-finetune-ocelot-no-embed/cross_validation_{}/delta_est'.format(CV_NUM)
            hyperparameter = json.load(open(os.path.join(resume_path, 'checkpoints', 'config.json')))
            hyperparameter['resume_from'] = resume_path
            hyperparameter['dataset_path'] = '../data/origin_data/delta_est.csv'
            hyperparameter['additional_features'] = None
        elif task == 's0t1':
            resume_path = '/home/qianzhang/MyProject/LUMIA/fine-tuning/dumped/1213-finetune-ocelot-no-embed/cross_validation_{}/s0t1'.format(CV_NUM)
            hyperparameter = json.load(open(os.path.join(resume_path, 'checkpoints', 'config.json')))
            hyperparameter['resume_from'] = resume_path
            hyperparameter['dataset_path'] = '../data/origin_data/s0t1.csv'
            hyperparameter['additional_features'] = None
        elif task == 's0s1':
            resume_path = '/home/qianzhang/MyProject/LUMIA/fine-tuning/dumped/1213-finetune-ocelot-no-embed/cross_validation_{}/s0s1'.format(CV_NUM)
            hyperparameter = json.load(open(os.path.join(resume_path, 'checkpoints', 'config.json')))
            hyperparameter['resume_from'] = resume_path
            hyperparameter['dataset_path'] = '../data/origin_data/s0s1.csv'
            hyperparameter['additional_features'] = None
        elif task == 'hl':
            resume_path = '/home/qianzhang/MyProject/LUMIA/fine-tuning/dumped/1213-finetune-ocelot-no-embed/cross_validation_{}/hl'.format(CV_NUM)
            hyperparameter = json.load(open(os.path.join(resume_path, 'checkpoints', 'config.json')))
            hyperparameter['resume_from'] = resume_path
            hyperparameter['dataset_path'] = '../data/origin_data/hl.csv'
            hyperparameter['additional_features'] = None
        else:
            resume_path ='/home/qianzhang/MyProject/LUMIA/fine-tuning/dumped/1209-finetune-qm9-sub/RE-seed-0-bs128-early-stopping-r2-resplit-lr0.0001/RE'
            hyperparameter = json.load(open(os.path.join(resume_path, 'checkpoints', 'config.json')))
            hyperparameter['resume_from'] = resume_path
            hyperparameter['dataset_path'] = '/home/qianzhang/MyProject/LUMIA/explaining/data/origin_data/RE.csv'
            hyperparameter['additional_features'] = None
            hyperparameter['device'] = 'cuda:0'
        
        
        # RE
        # resume_path ='/home/qianzhang/MyProject/LUMIA/fine-tuning/dumped/1209-finetune-qm9-sub/RE-seed-0-bs128-early-stopping-r2-resplit-lr0.0001/RE'
        # hyperparameter = json.load(open(os.path.join(resume_path, 'checkpoints', 'config.json')))
        # hyperparameter['resume_from'] = resume_path
        # hyperparameter['dataset_path'] = '/home/qianzhang/MyProject/LUMIA/explaining/data/origin_data/RE.csv'
        # hyperparameter['additional_features'] = None
        # hyperparameter['device'] = 'cuda:0'
        
        
        
        SMEG_explain_for_substructure(task_name=task,
                                      hyperparameter = hyperparameter,
                                      rgcn_hidden_feats=hyperparameter['model_config']['rgcn_hidden_feats'],
                                      ffn_hidden_feats=hyperparameter['model_config']['ffn_hidden_feats'],
                                      lr=hyperparameter['lr'], classification=False, sub_type=sub_type)

        print('Task Done!')



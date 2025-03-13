import pandas as pd
from utils import parse_predict_args,set_random_seed, get_featurizers, SMILESToBigraph, MoleculeDataset,read_dataset,count_model_parameters
from train import Finetuner
import warnings
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
import sys


def main():
    args,logger = parse_predict_args()
    args.task_name = 'regression'
    if args.additional_features:
        additional_features = pd.read_csv(args.additional_features)
    else:
        additional_features = None
    # data
    assert args.data_path is not None, "Data path is not provided." # for normalize
    assert args.resume_from is not None, "Resume from path is not provided."
    assert args.predict_data_path is not None, "Predict data path is not provided."
    
    args.normalize_path = './normalizer_hr.pth'
    
    data = pd.read_csv(args.data_path)
    logger.info(f"{args.data_path} data loaded, shape: {data.shape}")
    
    target_data = pd.read_csv(args.predict_data_path)
    logger.info(f"{args.predict_data_path} data loaded, shape: {target_data.shape}")
    
    # set random seed
    set_random_seed(args.seed)
    
    # set dataset and dataloader
    atom_featurizer, bond_featurizer = get_featurizers(args)
    smiles_to_g = SMILESToBigraph(node_featurizer=atom_featurizer,
                                  edge_featurizer=bond_featurizer,
                                  )
    
    dataset = MoleculeDataset(data=target_data,
                              additional_features=additional_features,
                              smiles_to_graph=smiles_to_g,
                              args=args)
    
    data_loader = dataset.get_full_loader()
    
    finetuner = Finetuner(dataset=dataset, config=args, logger=logger,
                          is_predict=True,normalize_path=args.normalize_path)
    finetuner.build_model()
    
    # _,_,_ = count_model_parameters(finetuner.model)
    # sys.exit(0)
    
    # project
    finetuner.predict(data_loader,tag='full_dataset')
    logger.info("Prediction finished!")
    
if __name__ == '__main__':
    main()
    
    
    
    
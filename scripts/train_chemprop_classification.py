import torch.multiprocessing as mp
import asyncio
import torch
import time
import json
import pandas as pd
from rdkit import Chem
import torch
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger
from lightning import pytorch as pl
from chemprop import data, featurizers, models, nn
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, precision_recall_curve, auc, mean_squared_error, mean_absolute_error, r2_score
from collections import defaultdict
import argparse
import os

import pickle

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = [None] * len(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.cache[idx] is None:
            self.cache[idx] = self.dataset[idx]
        return self.cache[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Sample argument parser for processing input.")
    
    # Integer field for cluster

    parser.add_argument('--num_runs', type=int, required=True, help='An integer field representing the cluster value.')

    parser.add_argument('--train_file', type=str, required=True, help='Path to train file.')

    parser.add_argument('--val_file', type=str, required=True, help='Path to val file.')

    parser.add_argument('--test_file', type=str, required=True, help='Path to test file.')

    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory.')
    
    # String field for output filename
    #parser.add_argument('--outputfilename', type=str, required=True, help='A string representing the output file name.')

    parser.add_argument('--outdir', type=str)
    
    # Boolean field for mean_replace

    parser.add_argument('--results_path', type=str, required=False, help='Path to save results.')

    parser.add_argument('--experiment_name', type=str, required=False, help='Name of experiment.')

    parser.add_argument('--max_epochs', type=int, required=False, default=20, help='Number of epochs to train for.')

    parser.add_argument('--num_workers', type=int, required=False, default=1, help='Number of workers to use for data loading.')

    parser.add_argument('--weight_column_name', type=str, required=False, default=None, help='Name of column to use as weight for training data.')

    parser.add_argument('--random_seed', type = int, required = False, default = None, help = 'random seed to set')




    

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    
    

    train_df = pd.read_csv(os.path.join(args.data_dir, args.train_file))
    val_df = pd.read_csv(os.path.join(args.data_dir, args.val_file))
    test_df = pd.read_csv(os.path.join(args.data_dir, args.test_file))



    for i in range(args.num_runs):
        if args.random_seed is not None:
            torch.manual_seed(args.random_seed)
            np.random.seed(args.random_seed)
        #start_time = time.time()
        num_workers = args.num_workers
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        if args.weight_column_name != None:
            train_data = [data.MoleculeDatapoint.from_smi(smi, y, weight = w) for smi, y, w in zip(train_df.loc[:, 'compound_smiles'].values, train_df.loc[:, ['classification_label']].values, train_df.loc[:, [args.weight_column_name]].values)]
        else:
            train_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(train_df.loc[:, 'compound_smiles'].values, train_df.loc[:, ['classification_label']].values)]
        
        
        val_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(val_df.loc[:, 'compound_smiles'].values, val_df.loc[:, ['classification_label']].values)]
        test_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(test_df.loc[:, 'compound_smiles'].values, test_df.loc[:, ['classification_label']].values)]


        
        use_cached_data = True

        if use_cached_data:
            original_train_dset = data.MoleculeDataset(train_data, featurizer)

            train_dset = CachedDataset(original_train_dset)

            original_val_dset = data.MoleculeDataset(val_data, featurizer)

            val_dset = CachedDataset(original_val_dset)

            original_test_dset = data.MoleculeDataset(test_data, featurizer)


            test_dset = CachedDataset(original_test_dset)
        else:
            train_dset = data.MoleculeDataset(train_data, featurizer)


            val_dset = data.MoleculeDataset(val_data, featurizer)


            test_dset = data.MoleculeDataset(test_data, featurizer)





        pin_memory = True
        
        train_loader = data.build_dataloader(train_dset, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False, pin_memory=pin_memory)
        test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False, pin_memory=pin_memory)
    
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
       
        
        mp = nn.BondMessagePassing()
        agg = nn.MeanAggregation()
    

        ffn = nn.BinaryClassificationFFN()
        
        batch_norm = True
        
        metric_list = None
        
        mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)
    
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
            enable_progress_bar=True,
            accelerator="auto",
            devices=1,
            max_epochs=args.max_epochs, # number of epochs to train for
        )

        if trainer.local_rank == 0:
            pickle.dump(vars(args), open("{}.args".format(args.results_path), "wb"))
    
        trainer.fit(mpnn, train_loader, val_loader)

        end_time = time.time()
        #print(f"Training time: {end_time - start_time}")
    
        results = trainer.test(mpnn, test_loader)
    
    
        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=True,
                accelerator="auto",
                devices=1
            )
            test_preds = trainer.predict(mpnn, test_loader)
        test_preds = np.concatenate(test_preds, axis =0 )
        test_df['classification_preds'] = test_preds

        train_base = os.path.splitext(os.path.basename(args.train_file))[0]
        test_base = os.path.splitext(os.path.basename(args.test_file))[0]
        outfile_name = f"{train_base}_{test_base}_run_{i}"

        if args.weight_column_name != None:
            outfile_name = f"{args.weight_column_name}_weighted_{outfile_name}"

        if args.random_seed is not None:
            outfile_name = f"{outfile_name}_seed_{args.random_seed}"

        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        test_df.to_csv(os.path.join(args.outdir, f"{outfile_name}.csv"), index=False)
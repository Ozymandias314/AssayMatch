import torch.multiprocessing as mp
import torch
import time
import pandas as pd
import numpy as np
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from chemprop import data, featurizers, models, nn
import argparse
import os


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
    
    parser.add_argument('--train_file', type=str, required=True, help='Path to total train file.')

    parser.add_argument('--total_file', type=str, required=True, help='Path to total file.')

    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory.')

    parser.add_argument('--outdir', type=str)

    parser.add_argument('--results_path', type=str, required=False, help='Path to save results.')

    parser.add_argument('--experiment_name', type=str, required=False, help='Name of experiment.')

    parser.add_argument('--max_epochs', type=int, required=False, default=20, help='Number of epochs to train for.')

    parser.add_argument('--num_workers', type=int, required=False, default=1, help='Number of workers to use for data loading.')

    parser.add_argument('--weight_column_name', type=str, required=False, default=None, help='Name of column to use as weight for training data.')

    parser.add_argument('--index', type=int, required=True, help='Index of the experiment, what binary vector to use')

    parser.add_argument('--binary_vector_dir', type=str, required=True, help='Directory containing the binary vectors')

    parser.add_argument('--save_checkpoints', type = bool, required = False, default = False, help = 'whether to save checkpoints')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    total_train_df = pd.read_csv(os.path.join(args.data_dir, args.train_file))
    total_df = pd.read_csv(os.path.join(args.data_dir, args.total_file))

    #load a precomputed binary vector to subset the training data
    binary_vector = np.load(os.path.join(args.binary_vector_dir, f"{args.index}.npy"))

    n_rows = len(total_train_df)

    if len(binary_vector) < n_rows:
        raise ValueError(
            f"Binary vector only has {len(binary_vector)} entries "
            f"but train_df has {n_rows} rows."
        )
    binary_vector = binary_vector[:n_rows]
    
    train_df = total_train_df[binary_vector.astype(bool)]

    start_time = time.time()
    num_workers = args.num_workers
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(train_df.loc[:, 'compound_smiles'].values, train_df.loc[:, ['classification_label']].values)]
    
    total_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(total_df.loc[:, 'compound_smiles'].values, total_df.loc[:, ['classification_label']].values)]

    train_dset = data.MoleculeDataset(train_data, featurizer)

    total_dset = data.MoleculeDataset(total_data, featurizer)

    pin_memory = True
    
    train_loader = data.build_dataloader(train_dset, num_workers=num_workers, pin_memory=pin_memory)

    total_loader = data.build_dataloader(total_dset, num_workers=num_workers, shuffle=False, pin_memory=pin_memory)
    
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()

    ffn = nn.BinaryClassificationFFN()
    
    batch_norm = True
    
    metric_list = None
    
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=args.max_epochs,
    )

    trainer.fit(mpnn, train_loader)

    trainer.save_checkpoint(f"{args.outdir}/model_{args.index}.ckpt")

    end_time = time.time()
    print(f"Training time: {end_time - start_time}")

    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="auto",
            devices=1
        )
        test_preds = trainer.predict(mpnn, total_loader)
    test_preds = np.concatenate(test_preds, axis =0)
   
    np.save(os.path.join(args.outdir, f'{args.index}_results.npy'), test_preds)
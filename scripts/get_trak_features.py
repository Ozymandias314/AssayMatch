import torch
import warnings
from chemprop import data, featurizers, models, nn
from trak import TRAKer
from trak import JacRevGradientComputer
from lightning.fabric.utilities.data import AttributeDict
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import os
import shutil
import time
from dataclasses import InitVar, dataclass, field
from typing import Optional, NamedTuple, Sequence, Iterable
from chemprop.data.samplers import ClassBalanceSampler, SeededSampler
from torch.utils.data import DataLoader, Sampler
from torch import Tensor
from chemprop.data.molgraph import MolGraph

#override chemprop data classes to make compatible with TRAK
@dataclass(repr=False, eq=False, slots=True)
class BatchMolGraph:
    """A :class:`BatchMolGraph` represents a batch of individual :class:`MolGraph`\s.

    It has all the attributes of a ``MolGraph`` with the addition of the ``batch`` attribute. This
    class is intended for use with data loading, so it uses :obj:`~torch.Tensor`\s to store data
    """

    mgs: InitVar[Sequence[MolGraph]]
    """A list of individual :class:`MolGraph`\s to be batched together"""
    V: Tensor = field(init=False)
    """the atom feature matrix"""
    E: Tensor = field(init=False)
    """the bond feature matrix"""
    edge_index: Tensor = field(init=False)
    """an tensor of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: Tensor = field(init=False)
    """A tensor of shape ``E`` that maps from an edge index to the index of the source of the
    reverse edge in the ``edge_index`` attribute."""
    batch: Tensor = field(init=False)
    """the index of the parent :class:`MolGraph` in the batched graph"""

    __size: int = field(init=False)

    def __post_init__(self, mgs: Sequence[MolGraph]):
        self.__size = len(mgs)

        Vs = []
        Es = []
        edge_indexes = []
        rev_edge_indexes = []
        batch_indexes = []

        num_nodes = 0
        num_edges = 0
        for i, mg in enumerate(mgs):
            Vs.append(mg.V)
            Es.append(mg.E)
            edge_indexes.append(mg.edge_index + num_nodes)
            rev_edge_indexes.append(mg.rev_edge_index + num_edges)
            batch_indexes.append([i] * len(mg.V))

            num_nodes += mg.V.shape[0]
            num_edges += mg.edge_index.shape[1]

        self.V = torch.from_numpy(np.concatenate(Vs)).float()
        self.E = torch.from_numpy(np.concatenate(Es)).float()
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes)).long()
        self.rev_edge_index = torch.from_numpy(np.concatenate(rev_edge_indexes)).long()
        self.batch = torch.tensor(np.concatenate(batch_indexes)).long()

    def __len__(self) -> int:
        """the number of individual :class:`MolGraph`\s in this batch"""
        return self.__size

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)
        return self

    def cuda(self):
        return self.to('cuda')
    
class TrainingBatch(NamedTuple):
    bmg: Optional[BatchMolGraph] 
    V_d: Optional[torch.Tensor]
    X_d: Optional[torch.Tensor]
    Y: Optional[torch.Tensor]
    w: torch.Tensor
    lt_mask: Optional[torch.Tensor]
    gt_mask: Optional[torch.Tensor]

    def to(self, device: torch.device):
        # Now self.bmg.cuda() will return the bmg object itself

        moved_bmg = self.bmg.to(device) if self.bmg is not None else None
        return TrainingBatch(
            bmg=moved_bmg, # Assign the returned object
            V_d=self.V_d.to(device) if self.V_d is not None else None,
            X_d=self.X_d.to(device) if self.X_d is not None else None,
            Y=self.Y.to(device)     if self.Y is not None     else None,
            w=self.w.to(device),
            lt_mask=self.lt_mask.to(device) if self.lt_mask is not None else None,
            gt_mask=self.gt_mask.to(device) if self.gt_mask is not None else None,
        )

def collate_batch(batch) -> TrainingBatch:
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    return TrainingBatch(
        BatchMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )


def build_dataloader(
    dataset,
    batch_size: int = 64,
    num_workers: int = 0,
    class_balance: bool = False,
    seed: int | None = None,
    shuffle: bool = True,
    epsilon: Optional[float] = 0.1,
    n_sample: Optional[int] = None,
    drop_last: Optional[bool] = None,
    **kwargs,
):
    """Return a :obj:`~torch.utils.data.DataLoader` for :class:`MolGraphDataset`\s

    Parameters
    ----------
    dataset : MoleculeDataset | ReactionDataset | MulticomponentDataset
        The dataset containing the molecules or reactions to load.
    batch_size : int, default=64
        the batch size to load.
    num_workers : int, default=0
        the number of workers used to build batches.
    class_balance : bool, default=False
        Whether to perform class balancing (i.e., use an equal number of positive and negative
        molecules). Class balance is only available for single task classification datasets. Set
        shuffle to True in order to get a random subset of the larger class.
    seed : int, default=None
        the random seed to use for shuffling (only used when `shuffle` is `True`).
    shuffle : bool, default=False
        whether to shuffle the data during sampling.
    """
    if drop_last is None:
        if len(dataset) % batch_size == 1:
            warnings.warn(
                f"Dropping last batch of size 1 to avoid issues with batch normalization \
    (dataset size = {len(dataset)}, batch_size = {batch_size})"
            )
            drop_last = True
        else:
            drop_last = False

    if class_balance:
        sampler = ClassBalanceSampler(dataset.Y, seed, shuffle)
    elif shuffle and seed is not None:
        sampler = SeededSampler(len(dataset), seed)
    else:
        sampler = None

    collate_fn = collate_batch

    return DataLoader(
        dataset,
        batch_size,
        sampler is None and shuffle,
        sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        **kwargs,
    )
    
def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Compute TRAK scores for a set of checkpoints.")
    parser.add_argument('--train_file', type=str, required=True, help='Path to train file.')

    parser.add_argument('--test_file', type=str, required=True, help='Path to test file.')

    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory.')

    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing the model checkpoints (e.g., model_0.ckpt, model_1.ckpt, ...).')
    parser.add_argument('--num_checkpoints', type=int, required=True,
                        help='Number of checkpoints to load from the checkpoint_dir.')

    # TRAK arguments
    parser.add_argument('--proj_dim', type=int, default=512,
                        help='Dimension for the TRAK projection.')

    # Output arguments
    parser.add_argument('--outdir', type=str, required=True,
                        help='Path to save the computed TRAK scores (e.g., /path/to/scores.npy).')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for computations.')
    parser.add_argument('--results_path', type=str, required=False, help='Path to save results.')

    parser.add_argument('--experiment_name', type=str, required=False, help='Name of experiment.')
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    """Main execution function."""


    # Ensure the output directory exists
    output_dir = os.path.dirname(args.outdir)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Construct checkpoint file paths
    ckpt_files = [os.path.join(args.checkpoint_dir, f'model_{i}.ckpt') for i in range(args.num_checkpoints)]
    print(f"Attempting to load {args.num_checkpoints} checkpoints from {args.checkpoint_dir}...")

    # Add AttributeDict to safe globals for loading Lightning checkpoints
    torch.serialization.add_safe_globals([AttributeDict])

    # Load checkpoints
    try:
        ckpts_loaded = [torch.load(ckpt, map_location="cpu", weights_only=False) for ckpt in tqdm(ckpt_files, desc="Loading checkpoints")]
        # Extract state_dict (assuming standard PyTorch Lightning structure)
        ckpts = [ckpt['state_dict'] for ckpt in ckpts_loaded]
        print(f"Successfully loaded {len(ckpts)} checkpoints.")
    except FileNotFoundError as e:
        print(f"Error loading checkpoints: {e}")
        print("Please ensure --checkpoint_dir and --num_checkpoints are correct and all checkpoint files exist.")
    except KeyError as e:
        print(f"Error accessing 'state_dict' in checkpoint: {e}")
        print("Checkpoint structure might be different than expected (PyTorch Lightning standard).")
    except Exception as e:
        print(f"An unexpected error occurred during checkpoint loading: {e}")



    # Define Model Architecture (must match the architecture used for training the checkpoints)
    # Assuming Binary Classification based on original script
    print("Defining model architecture (MPNN for Binary Classification)...")
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    ffn = nn.BinaryClassificationFFN() # Use BinaryClassificationFFN for classification
    batch_norm = True
    metric_list = None # Metrics aren't needed for TRAK computation
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list).to(args.device).eval() # Put model on device and set to eval mode

    # Load Data
    print("Loading and processing data...")
    try:
        train_df = pd.read_csv(os.path.join(args.data_dir, args.train_file))
        test_df = pd.read_csv(os.path.join(args.data_dir, args.test_file))
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    # Ensure target column is treated correctly for classification (often integer or float 0/1)
    # Using [['classification_label']] ensures it remains a 2D array-like structure for Chemprop
    train_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(train_df['compound_smiles'].values, train_df[['classification_label']].values)]
    test_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(test_df['compound_smiles'].values, test_df[['classification_label']].values)]

    train_dset = data.MoleculeDataset(train_data, featurizer)
    test_dset = data.MoleculeDataset(test_data, featurizer)

    # Create DataLoaders
    pin_memory = (args.device == 'cuda') # Use pin_memory only if using GPU
    train_loader = build_dataloader(train_dset,
                                        num_workers=args.num_workers,
                                        shuffle=False, # Important: DO NOT shuffle train loader for TRAK
                                        pin_memory=pin_memory)
    test_loader = build_dataloader(test_dset,
                                       num_workers=args.num_workers,
                                       shuffle=False,
                                       pin_memory=pin_memory)
    
    print(f"Created train loader with {len(train_loader.dataset)} samples and test loader with {len(test_loader.dataset)} samples.")
    exp_name = f"{os.path.basename(args.test_file)}_{args.num_checkpoints}_checkpoints_{args.proj_dim}_proj_dim"
    # Initialize TRAKer
    print("Initializing TRAKer...")
    try:
        traker = TRAKer(model=mpnn,
                        task='chemprop_classification', # Specify task type
                        train_set_size=len(train_loader.dataset),
                        gradient_computer=JacRevGradientComputer, # Using JacRev as in original
                        proj_dim=args.proj_dim,
                        save_dir=f'{args.outdir}/{exp_name}_results',
                        device=args.device)
    except Exception as e:
        print(f"Error initializing TRAKer: {e}")

    # Featurize Training Data
    print(f"Featurizing training data using {len(ckpts)} checkpoints...")
    for model_id, ckpt in enumerate(tqdm(ckpts, desc="Featurizing Checkpoints")):
        traker.load_checkpoint(ckpt, model_id=model_id)
        for batch in tqdm(train_loader, desc=f"Featurizing Model {model_id}", leave=False):
            # Move batch to the correct device
            traker.featurize(batch=batch.to('cuda'), num_samples=len(batch.bmg)) # Use batch graph length

    try:
        traker.finalize_features()
        print("Finalized TRAK features for training data.")
    except Exception as e:
        print(f"Error finalizing features: {e}")



    # Score Test Data
    print(f"Scoring test data using {len(ckpts)} checkpoints...")
    # Experiment name can be derived or kept simple
    

    for model_id, ckpt in enumerate(tqdm(ckpts, desc="Scoring Checkpoints")):
        traker.start_scoring_checkpoint(exp_name=exp_name,
                                        checkpoint=ckpt,
                                        model_id=model_id,
                                        num_targets=len(test_loader.dataset))
        for batch in tqdm(test_loader, desc=f"Scoring Model {model_id}", leave=False):
            # Move batch to the correct device
            batch = batch.to(args.device)
            # Ensure targets are float if needed
            if hasattr(batch, 'y') and batch.y is not None:
                batch.y = batch.y.float()
            traker.score(batch=batch, num_samples=len(batch.bmg)) # Use batch graph length

    try:
        scores = traker.finalize_scores(exp_name=exp_name)
        print(f"Finalized TRAK scores. Score matrix shape: {scores.shape}") # type: ignore
    except Exception as e:
        print(f"Error finalizing scores: {e}")

    # Save Scores
  
    try:
        np.save(os.path.join(args.outdir, f"{exp_name}.npy"), scores)  # type: ignore
        print("Scores saved successfully.")


    except Exception as e:
        print(f"Error saving scores to {os.path.join(args.outdir, args.output_file)}: {e}")
        

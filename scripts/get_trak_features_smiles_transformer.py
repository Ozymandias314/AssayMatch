import torch
from chemprop import data, featurizers, models, nn
from trak import TRAKer
from trak import JacRevGradientComputerForSmilesTransformer
from lightning.fabric.utilities.data import AttributeDict
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import os
import shutil
import time
import sys
sys.path.append('/data/rbg/users/vincentf/data_uncertainty_take_2/scripts')
from transformer_mlp import TorchMLPClassifier
from torch.utils.data import TensorDataset, DataLoader
sys.path.append("/data/rbg/users/vincentf/data_uncertainty_take_2/smiles-transformer/smiles_transformer")
from pretrain_trfm import TrfmSeq2seq
from utils import split
PAD_INDEX = 0
UNK_INDEX = 1
EOS_INDEX = 2
SOS_INDEX = 3
MASK_INDEX = 4
from build_vocab import WordVocab
VOCAB = WordVocab.load_vocab('/data/rbg/users/vincentf/data_uncertainty_take_2/smiles-transformer/experiments/vocab.pkl')
def get_inputs(sm):
    seq_len = 220
    sm = sm.split()
    if len(sm)>218:
        print('SMILES is too long ({:d})'.format(len(sm)))
        sm = sm[:109]+sm[-109:]
    ids = [VOCAB.stoi.get(token, UNK_INDEX) for token in sm]
    ids = [SOS_INDEX] + ids + [EOS_INDEX]
    seg = [1]*len(ids)
    padding = [PAD_INDEX]*(seq_len - len(ids))
    ids.extend(padding), seg.extend(padding)
    return ids, seg


def get_array(smiles):
    x_id, x_seg = [], []
    for sm in smiles:
        a,b = get_inputs(sm)
        x_id.append(a)
        x_seg.append(b)
    return torch.tensor(x_id), torch.tensor(x_seg)


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
    print(f"Attempting to load {args.num_checkpoints} checkpoints from {args.checkpoint_dir}...")
    ckpts = [torch.load(os.path.join(args.checkpoint_dir, f'model_{i}.pth'), map_location = torch.device('cpu'), weights_only = False) for i in range(args.num_checkpoints)]
    print(f"Successfully loaded {len(ckpts)} checkpoints.")


    print("Defining model architecture...")

    trfm = TrfmSeq2seq(len(VOCAB), 256, len(VOCAB), 4)
    trfm.load_state_dict(torch.load('/data/rbg/users/vincentf/data_uncertainty_take_2/smiles-transformer/trfm_12_23000.pkl', map_location = torch.device('cpu')))
    trfm.to('cuda')
    trfm.eval()
    clf_for_trak = TorchMLPClassifier(
                encoder=trfm,
                hidden_layer_sizes=100,  # or [100, 50] for multiple layers
                max_iter=1000,
                verbose=True,
                batch_size=100,
                learning_rate=0.001,
                encoder_on_cpu = False)
    clf_for_trak._build_mlp(1024, 2)
    clf_for_trak.eval()

    grad_wrt_params = []

    # Add all encoder parameters (embed, pe, transformer encoder, out layer)
    for name, _ in clf_for_trak.encoder.named_parameters():
        if not name.startswith('trfm.decoder'):
            grad_wrt_params.append(f'encoder.{name}')

    # Add all MLP parameters
    for name, _ in clf_for_trak.mlp.named_parameters():
        grad_wrt_params.append(f'mlp.{name}')

    print("Loading data...")
    try:
        train_df = pd.read_csv(os.path.join(args.data_dir, args.train_file))
        test_df = pd.read_csv(os.path.join(args.data_dir, args.test_file))
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
    
    print("Featurizing data...")

    train_x_split = [split(sm) for sm in train_df['compound_smiles'].values]
    train_xid, train_xseg = get_array(train_x_split)

    test_x_split = [split(sm) for sm in test_df['compound_smiles'].values]
    test_xid, test_xseg = get_array(test_x_split)

    train_loader = DataLoader(TensorDataset(train_xid, torch.tensor(train_df['classification_label'].values)), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_xid, torch.tensor(test_df['classification_label'].values)), batch_size=32, shuffle=False)

    print(f"Created train loader with {len(train_loader.dataset)} samples and test loader with {len(test_loader.dataset)} samples.")
    exp_name = f"{os.path.basename(args.test_file)}_{args.num_checkpoints}_checkpoints_{args.proj_dim}_proj_dim"
    # Initialize TRAKer
    print("Initializing TRAKer...")
    try:
        traker = TRAKer(model=clf_for_trak,
                        task='smilestransformer_classification', # Specify task type
                        train_set_size=len(train_loader.dataset),
                        gradient_computer=JacRevGradientComputerForSmilesTransformer, # Using JacRev as in original
                        proj_dim=args.proj_dim,
                        save_dir=f'{args.outdir}/{exp_name}_results',
                        device=args.device,
                        grad_wrt=grad_wrt_params)
    except Exception as e:
        print(f"Error initializing TRAKer: {e}")
    
        # Featurize Training Data
    print(f"Featurizing training data using {len(ckpts)} checkpoints...")
    for model_id, ckpt in enumerate(tqdm(ckpts, desc="Featurizing Checkpoints")):
        traker.load_checkpoint(ckpt, model_id=model_id)
        for batch in tqdm(train_loader, desc=f"Featurizing Model {model_id}", leave=False):
            # Move batch to the correct device

            traker.featurize(batch=[batch[0].to('cuda'), batch[1].to('cuda')], num_samples=len(batch[0]))

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
            batch = [batch[0].to('cuda'), batch[1].to('cuda')]
            # Ensure targets are float if needed
            traker.score(batch=batch, num_samples=len(batch[0]))

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


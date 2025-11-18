import sys
sys.path.append("smiles-transformer/smiles_transformer")
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
import torch.multiprocessing as mp
import torch
import pandas as pd
import torch
import numpy as np
import argparse
import os

from transformer_mlp import TorchMLPClassifier

VOCAB = WordVocab.load_vocab('smiles-transformer/smiles_transformer/vocab.pkl')

PAD_INDEX = 0
UNK_INDEX = 1
EOS_INDEX = 2
SOS_INDEX = 3
MASK_INDEX = 4
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
    parser = argparse.ArgumentParser(description="Sample argument parser for processing input.")

    parser.add_argument('--train_file', type=str, required=True, help='Path to total train file.')

    parser.add_argument('--total_file', type=str, required=True, help='Path to total file.')

    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory.')

    parser.add_argument('--outdir', type=str)

    parser.add_argument('--results_path', type=str, required=False, help='Path to save results.')

    parser.add_argument('--experiment_name', type=str, required=False, help='Name of experiment.')

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

    trfm = TrfmSeq2seq(len(VOCAB), 256, len(VOCAB), 4)
    trfm.load_state_dict(torch.load('smiles-transformer/trfm_12_23000.pkl', map_location = torch.device('cuda')))
    trfm.to('cuda')
    trfm.eval()

    clf = TorchMLPClassifier(
        encoder=trfm,
        hidden_layer_sizes=100,  # or [100, 50] for multiple layers
        max_iter=1000,
        verbose=True,
        batch_size=200,
        learning_rate=0.001,
        encoder_on_cpu=False
    )

    train_x_split = [split(sm) for sm in train_df['compound_smiles'].values]
    train_xid, train_xseg = get_array(train_x_split)

    total_x_split = [split(sm) for sm in total_df['compound_smiles'].values]
    total_xid, total_xseg = get_array(total_x_split)

    clf.fit(train_xid, train_df['classification_label'])

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    clf.save_weights(f"{args.outdir}/model_{args.index}.pth")

    y_score = clf.predict_proba(total_xid)[:, 1]

    y_score = np.array(y_score)
    
    np.save(os.path.join(args.outdir, f'{args.index}_results.npy'), y_score)
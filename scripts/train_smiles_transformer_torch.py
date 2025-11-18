import sys
sys.path.append("../smiles-transformer/smiles_transformer")
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
import torch
import pandas as pd
import torch
import argparse
import os

from transformer_mlp import TorchMLPClassifier

VOCAB = WordVocab.load_vocab('../smiles-transformer/smiles_transformer/vocab.pkl')

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
    
    parser.add_argument('--num_runs', type=int, required=True, help='An integer field representing the cluster value.')

    parser.add_argument('--train_file', type=str, required=True, help='Path to train file.')

    parser.add_argument('--test_file', type=str, required=True, help='Path to test file.')

    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory.')
    
    parser.add_argument('--outdir', type=str)
    
    parser.add_argument('--results_path', type=str, required=False, help='Path to save results.')

    parser.add_argument('--experiment_name', type=str, required=False, help='Name of experiment.')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    train_df = pd.read_csv(os.path.join(args.data_dir, args.train_file))
    test_df = pd.read_csv(os.path.join(args.data_dir, args.test_file))

    for i in range(args.num_runs):
        trfm = TrfmSeq2seq(len(VOCAB), 256, len(VOCAB), 4)
        trfm.load_state_dict(torch.load('/data/rbg/users/vincentf/data_uncertainty_take_2/smiles-transformer/trfm_12_23000.pkl', map_location = torch.device('cuda')))
        trfm.eval()
        trfm.to('cuda')
        train_x_split = [split(sm) for sm in train_df['compound_smiles'].values]
        train_xid, train_xseg = get_array(train_x_split)

        test_x_split = [split(sm) for sm in test_df['compound_smiles'].values]
        test_xid, test_xseg = get_array(test_x_split)

        clf = TorchMLPClassifier(
            encoder=trfm,
            hidden_layer_sizes=100,  
            max_iter=1000,
            verbose=True,
            batch_size=200,
            learning_rate=0.001,
            encoder_on_cpu=False
        )

        clf.fit(train_xid, train_df['classification_label'])
        y_score = clf.predict_proba(test_xid)[:, 1]

        test_df['classification_preds'] = y_score

        train_base = os.path.splitext(os.path.basename(args.train_file))[0]
        test_base = os.path.splitext(os.path.basename(args.test_file))[0]
        outfile_name = f"{train_base}_{test_base}_run_{i}"

        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        test_df.to_csv(os.path.join(args.outdir, f"{outfile_name}.csv"), index=False)
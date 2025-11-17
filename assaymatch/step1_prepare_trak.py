import json
import numpy as np
import random 
import os
from collections import defaultdict
import argparse
from tqdm import tqdm
import itertools
import shutil
import pandas as pd

parser = argparse.ArgumentParser(description="Generate Data")
parser.add_argument(
    "--base_path",
    "-b",
    type=str,
    required=True,
    help="base path for this experiment run",
)



if __name__ == "__main__":
    args = parser.parse_args()

    BASE_PATH = args.base_path

    maximum_index = max([int(name.split('_')[-1]) for name in os.listdir(BASE_PATH) if name.startswith('run_')], default=0)

    os.makedirs(os.path.join(BASE_PATH, f'run_{maximum_index + 1}'), exist_ok=True)

    with open('/data/rbg/users/vincentf/data_uncertainty_take_2/data/finetuning_data/chembl_id_to_idx_list.json', 'r') as f:
        chembl_id_to_idx_list = json.load(f)
    with open('/data/rbg/users/vincentf/data_uncertainty_take_2/data/finetuning_data/idx_to_description.json', 'r') as f:
        idx_to_description = json.load(f)
    with open('/data/rbg/users/vincentf/data_uncertainty_take_2/data/finetuning_data/description_to_idx.json', 'r') as f:
        description_to_idx = json.load(f)

    CHEMBL_ID_LIST = ['CHEMBL340', 'CHEMBL240', 'CHEMBL247','CHEMBL220', 'CHEMBL279', 'CHEMBL203'] 

    for chembl_id in CHEMBL_ID_LIST:
        os.mkdir(os.path.join(BASE_PATH, f'run_{maximum_index + 1}', chembl_id))
        os.mkdir(os.path.join(BASE_PATH, f'run_{maximum_index + 1}', chembl_id, 'chemprop'))
        os.mkdir(os.path.join(BASE_PATH, f'run_{maximum_index + 1}', chembl_id, 'st'))

        df = pd.read_csv(f'/data/rbg/users/vincentf/data_uncertainty_take_2/data/finalized_data/{chembl_id}_total.csv')

        train_description_idxs = random.sample(chembl_id_to_idx_list[chembl_id], int(0.9*len(chembl_id_to_idx_list[chembl_id])))

        train_descriptions = set([idx_to_description[str(i)] for i in train_description_idxs])

        train_df = df[df['assay_description'].isin(train_descriptions)]
        test_df = df[~df['assay_description'].isin(train_descriptions)]

        df.to_csv(os.path.join(BASE_PATH, f'run_{maximum_index + 1}', chembl_id, 'total.csv'))
        train_df.to_csv(os.path.join(BASE_PATH, f'run_{maximum_index + 1}', chembl_id, 'train.csv'))
        test_df.to_csv(os.path.join(BASE_PATH, f'run_{maximum_index + 1}', chembl_id, 'test.csv'))



    config = {
        "script": "train_chemprop_for_datamodels_classification",
        "cartesian_hyperparams": {
            "max_epochs": [
                20
            ],
            "num_workers": [
                2
            ],
            "train_file": [
                "train.csv"
            ],
            "total_file": [
                "train.csv"
            ],
            "binary_vector_dir": [
                "/data/rbg/users/vincentf/data_uncertainty_take_2/data/random_vectors"
            ],
            "index": [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                87,
                88,
                89,
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                99
            ]
        },
        "paired_hyperparams": {
            "outdir": [os.path.join(BASE_PATH, f'run_{maximum_index+1}', chembl_id, 'chemprop') for chembl_id in CHEMBL_ID_LIST],
            "data_dir": [os.path.join(BASE_PATH, f'run_{maximum_index+1}', chembl_id) for chembl_id in CHEMBL_ID_LIST],
        },
        "available_gpus": [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
        ]
    }

    with open(os.path.join(BASE_PATH, f'run_{maximum_index + 1}', 'trak_training_config.json'), 'w') as f:
        json.dump(config, f, indent = 4)

    trak_config = {
        "script": "get_trak_features",
        "cartesian_hyperparams": {
            "num_checkpoints": [
                100
            ],
            "train_file": [
                "train.csv"
            ],
            "test_file": [
                "train.csv"
            ]
        },
        "paired_hyperparams": {
            "outdir": [os.path.join(BASE_PATH, f'run_{maximum_index+1}', chembl_id, 'chemprop') for chembl_id in CHEMBL_ID_LIST],
            "checkpoint_dir": [os.path.join(BASE_PATH, f'run_{maximum_index+1}', chembl_id, 'chemprop') for chembl_id in CHEMBL_ID_LIST],
            "data_dir": [os.path.join(BASE_PATH, f'run_{maximum_index+1}', chembl_id) for chembl_id in CHEMBL_ID_LIST]
        },
        "available_gpus": [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7"
        ]
    }

    with open(os.path.join(BASE_PATH, f'run_{maximum_index + 1}', 'trak_featurization_config.json'), 'w') as f:
        json.dump(trak_config, f, indent = 4)

    config_st = {
        "script": "train_smiles_transformer_for_trak_classification",
        "cartesian_hyperparams": {
            "train_file": [
                "train.csv"
            ],
            "total_file": [
                "train.csv"
            ],
            "binary_vector_dir": [
                "/data/rbg/users/vincentf/data_uncertainty_take_2/data/random_vectors"
            ],
            "index": [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                87,
                88,
                89,
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                99
            ]
        },
        "paired_hyperparams": {
            "outdir": [os.path.join(BASE_PATH, f'run_{maximum_index+1}', chembl_id, 'st') for chembl_id in CHEMBL_ID_LIST],
            "data_dir": [os.path.join(BASE_PATH, f'run_{maximum_index+1}', chembl_id) for chembl_id in CHEMBL_ID_LIST],
        },
        "available_gpus": [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
        ]
    }

    with open(os.path.join(BASE_PATH, f'run_{maximum_index + 1}', 'trak_training_config_st.json'), 'w') as f:
        json.dump(config_st, f, indent = 4)

    trak_config_st = {
        "script": "get_trak_features_smiles_transformer",
        "cartesian_hyperparams": {
            "num_checkpoints": [
                100
            ],
            "train_file": [
                "train.csv"
            ],
            "test_file": [
                "train.csv"
            ]
        },
        "paired_hyperparams": {
            "outdir": [os.path.join(BASE_PATH, f'run_{maximum_index+1}', chembl_id, 'st') for chembl_id in CHEMBL_ID_LIST],
            "checkpoint_dir": [os.path.join(BASE_PATH, f'run_{maximum_index+1}', chembl_id, 'st') for chembl_id in CHEMBL_ID_LIST],
            "data_dir": [os.path.join(BASE_PATH, f'run_{maximum_index+1}', chembl_id) for chembl_id in CHEMBL_ID_LIST]
        },
        "available_gpus": [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7"
        ]
    }

    with open(os.path.join(BASE_PATH, f'run_{maximum_index + 1}', 'trak_featurization_config_st.json'), 'w') as f:
        json.dump(trak_config_st, f, indent = 4)
    
import json
import numpy as np
import random 
import os
import argparse
from tqdm import tqdm
import itertools
import time
import pandas as pd
from typing import Dict, List

def compute_true_ranking_per_group(
    df: pd.DataFrame,
    trak_scores: np.ndarray,
    global_idx_to_description: Dict[str, str],
    relevant_global_indices: List[int]
) -> Dict[int, List[int]]:
    """
    Computes a 'true' ranking for a specific group of items (e.g., a single chembl_id).

    This function operates on a single group's data and correctly maps local
    DataFrame/array indices to the global indices required for the final output.

    Args:
        df (pd.DataFrame): DataFrame for the specific group, indexed from 0.
        trak_scores (np.ndarray): The score matrix corresponding to the group's DataFrame.
        global_idx_to_description (Dict[str, str]): The complete, global mapping from
                                                    any index (as a string) to its description.
        relevant_global_indices (List[int]): The list of global integer indices that
                                             define this specific group.

    Returns:
        A dictionary where each key is a global index from the current group, and the
        value is a list of all global indices within that same group, sorted by score.
    """
    print("Starting group-specific ranking computation...")
    start_time = time.time()

    # --- Step 1: Create Mappings Between Global and Local Indices ---
    # The `df` and `trak_scores` for this group use local indices (0, 1, 2...).
    # We need to map them to and from the `relevant_global_indices`.
    local_to_global_map = {i: g_idx for i, g_idx in enumerate(relevant_global_indices)}
    
    # Create a local description map using local indices (0, 1, ...) as keys.
    # JSON keys are strings, so we must cast the global index to a string for the lookup.
    local_idx_to_description = {
        local_idx: global_idx_to_description[str(global_idx)]
        for local_idx, global_idx in local_to_global_map.items()
    }

    # --- Step 2: Pre-compute Index Lookups by Description (using local indices) ---
    # This groups by description to find all local indices that share that description.
    description_to_local_indices = df.groupby('assay_description').groups

    # The list of all local indices we need to generate rankings for.
    all_local_indices = list(local_idx_to_description.keys())
    
    true_ranking_dict = {}
    loop_start_time = time.time()

    # --- Step 3: Optimized Comparison Loop Within the Group ---
    # Iterate through each item in the current group to generate its ranking.
    for _, anchor_local_idx in enumerate(all_local_indices):
        anchor_desc = local_idx_to_description[anchor_local_idx]
        
        # Get all local indices for data points sharing the anchor's description.
        anchor_indices_in_df = description_to_local_indices.get(anchor_desc)
        
        if anchor_indices_in_df is None or len(anchor_indices_in_df) == 0:
            continue # This item's description isn't in the DataFrame.

        scores_for_anchor = []
        # Compare the anchor item against every other item *in the same group*.
        for target_local_idx in all_local_indices:

            
            target_desc = local_idx_to_description[target_local_idx]
            target_indices_in_df = description_to_local_indices.get(target_desc)
            
            if target_indices_in_df is None or len(target_indices_in_df) == 0:
                continue 
            score = 0.0
            if target_indices_in_df is not None and len(target_indices_in_df) > 0:
                # Core calculation uses local indices to slice the `trak_scores` matrix.
                #score = np.mean(np.abs(trak_scores[np.ix_(target_indices_in_df, anchor_indices_in_df)]))
                score = np.mean(trak_scores[np.ix_(target_indices_in_df, anchor_indices_in_df)])
            
            # Associate the score with the target's GLOBAL index for the final ranked list.
            target_global_idx = local_to_global_map[target_local_idx]
            scores_for_anchor.append((score, target_global_idx))

        # Sort by score (descending) and global index (descending, for tie-breaking).
        scores_for_anchor.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        # Extract the sorted GLOBAL indices.
        sorted_global_indices = [g_idx for score, g_idx in scores_for_anchor]
        
        # The final dictionary is keyed by the anchor's GLOBAL index.
        anchor_global_idx = local_to_global_map[anchor_local_idx]
        true_ranking_dict[anchor_global_idx] = sorted_global_indices


    print(f"Total group computation complete in {time.time() - start_time:.2f}s.")
    return true_ranking_dict

parser = argparse.ArgumentParser(description="Generate Data")
parser.add_argument(
    "--base_path",
    "-b",
    type=str,
    required=True,
    help="base path for this experiment run",
)

CHEMBL_ID_LIST = ['CHEMBL340', 'CHEMBL240', 'CHEMBL247', 'CHEMBL220', 'CHEMBL279', 'CHEMBL203'] 

def sample_train_test_triplets(idx_to_ranking, chembl_id_to_idx_list, top_6_list, test_size = 0.1):
    #first split out train indices and test indices. then sample triplets as follows: given a triplet (a, i, j) i is from the first half of the list and j is from the second half
    # if a, i, j contain all train points, add it to train_triplets, if it contains exactly one test point, add it to the test_triplets, otherwise discard 

    train_indices = []
    for chembl_id, indices in chembl_id_to_idx_list.items():

        if chembl_id in top_6_list:
            train_indices.extend(indices)
        else:
            # Randomly sample indices for training
            num_train_indices = int(len(indices) * (1 - test_size))
            sampled_train_indices = random.sample(indices, num_train_indices)
            train_indices.extend(sampled_train_indices)
    train_indices = set(train_indices)

    train_triplets = []
    test_triplets = []

    for idx, ranking in tqdm(idx_to_ranking.items()):
        num_triplets = min(40, max(1, int(len(ranking)*(len(ranking) - 1)/2 * 0.005)))
        midpoint = len(ranking) // 2
        first_half = ranking[:midpoint]
        second_half = ranking[midpoint:]

        # Generate all combinations of two indices from the ranking
        combinations = list(itertools.product(first_half, second_half))
        # Randomly sample the specified number of triplets
        sampled_combinations = random.sample(combinations, num_triplets)
        # Create triplets with the current id
        for x, y in sampled_combinations:
            if x != idx and y != idx:
                if x in train_indices and y in train_indices and idx in train_indices:
                    train_triplets.append((idx, x, y))
                if sum([i in train_indices for i in [idx, x, y]]) == 2:
                    test_triplets.append((idx, x, y))
    
    return train_triplets, test_triplets, train_indices

if __name__ == "__main__":

    args = parser.parse_args()

    with open('data/chembl_id_to_idx_list.json', 'r') as f:
        chembl_id_to_idx_list = json.load(f)
    with open('data/idx_to_description.json', 'r') as f:
        idx_to_description = json.load(f)
    with open('data/description_to_idx.json', 'r') as f:
        description_to_idx = json.load(f)


    with open('data/true_rankings_no_top_6.json') as f:
        true_rankings = json.load(f)
    true_rankings = {int(k): v for k, v in true_rankings.items()}

    #extend true_rankings here
    BASE_PATH = args.base_path
    for chembl_id in CHEMBL_ID_LIST:
        global_indices = chembl_id_to_idx_list[chembl_id]

        df = pd.read_csv(os.path.join(BASE_PATH, chembl_id, 'train.csv'))
        trak_scores = np.load(os.path.join(BASE_PATH, chembl_id, 'chemprop', 'train.csv_100_checkpoints_512_proj_dim.npy'))

        df = df.reset_index(drop = True)
        ranking_for_group = compute_true_ranking_per_group(
            df,
            trak_scores,
            idx_to_description,      # Pass the full, global description map.
            global_indices           # Pass the list of global indices for this specific group.
        )

        # Add the results for this group to the main dictionary.
        true_rankings.update(ranking_for_group)


    train_triplets, test_triplets, train_indices = sample_train_test_triplets(true_rankings, chembl_id_to_idx_list, CHEMBL_ID_LIST)
    test_triplets = random.sample(test_triplets, 20000)
    train_triplets = np.array(train_triplets)
    test_triplets = np.array(test_triplets)

    np.save(os.path.join(BASE_PATH, 'train_triplets_chemprop.npy'), train_triplets)
    np.save(os.path.join(BASE_PATH, 'val_triplets_chemprop.npy'), test_triplets)
    with open(os.path.join(BASE_PATH, 'train_indices_chemprop.json'), 'w') as f:
        json.dump(list(train_indices), f)
    print(f"Saved new run data in {BASE_PATH}")

    config = {
        "script": "finetune_embeddings_predefined",
        "cartesian_hyperparams": {
            "embedding_file": [
                "data/description_embeddings.npy"
            ],
            "train_triplets_file": [
                os.path.join(BASE_PATH,  'train_triplets_chemprop.npy')
            ],
            "validation_triplets_file": [
                os.path.join(BASE_PATH,  'val_triplets_chemprop.npy')
            ],
            "output_dir": [
                BASE_PATH
            ],
            "projection_dim": [
                768
            ],
            "hidden_dim_ratio": [
                0.8
            ],
            "batch_size": [
                512
            ],
            "learning_rate": [
                1e-4
            ],
            "margin": [
                0.1
            ],
            "model_type": [
                "chemprop"
            ]
        },
        "available_gpus": [
            "0"
        ]
    }

    with open(os.path.join(BASE_PATH, 'finetune_config_chemprop.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    with open('data/true_rankings_no_top_6_st.json') as f:
        true_rankings = json.load(f)
    true_rankings = {int(k): v for k, v in true_rankings.items()}

    BASE_PATH = args.base_path
    for chembl_id in CHEMBL_ID_LIST:
        global_indices = chembl_id_to_idx_list[chembl_id]

        df = pd.read_csv(os.path.join(BASE_PATH, chembl_id, 'train.csv'))
        trak_scores = np.load(os.path.join(BASE_PATH, chembl_id, 'st', 'train.csv_100_checkpoints_512_proj_dim.npy'))

        df = df.reset_index(drop = True)
        ranking_for_group = compute_true_ranking_per_group(
            df,
            trak_scores,
            idx_to_description,      # Pass the full, global description map.
            global_indices           # Pass the list of global indices for this specific group.
        )

        # Add the results for this group to the main dictionary.
        true_rankings.update(ranking_for_group)


    train_triplets, test_triplets, train_indices = sample_train_test_triplets(true_rankings, chembl_id_to_idx_list, CHEMBL_ID_LIST)
    test_triplets = random.sample(test_triplets, 20000)
    train_triplets = np.array(train_triplets)
    test_triplets = np.array(test_triplets)


    np.save(os.path.join(BASE_PATH, 'train_triplets_st.npy'), train_triplets)
    np.save(os.path.join(BASE_PATH, 'val_triplets_st.npy'), test_triplets)
    with open(os.path.join(BASE_PATH, 'train_indices_st.json'), 'w') as f:
        json.dump(list(train_indices), f)
    print(f"Saved new run data in {BASE_PATH}")

    config_st = {
        "script": "finetune_embeddings_predefined",
        "cartesian_hyperparams": {
            "embedding_file": [
                "data/description_embeddings.npy"
            ],
            "train_triplets_file": [
                os.path.join(BASE_PATH,  'train_triplets_st.npy')
            ],
            "validation_triplets_file": [
                os.path.join(BASE_PATH,  'val_triplets_st.npy')
            ],
            "output_dir": [
                BASE_PATH
            ],
            "projection_dim": [
                768
            ],
            "hidden_dim_ratio": [
                0.8
            ],
            "batch_size": [
                512
            ],
            "learning_rate": [
                1e-4
            ],
            "margin": [
                0.1
            ],
            "model_type": [
                "st"
            ]
        },
        "available_gpus": [
            "0"
        ]
    }

    with open(os.path.join(BASE_PATH, 'finetune_config_st.json'), 'w') as f:
        json.dump(config_st, f, indent=4)
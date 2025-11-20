import time
import pandas as pd
import numpy as np
from typing import Dict, List
import json
from tqdm import tqdm

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
    local_to_global_map = {i: g_idx for i, g_idx in enumerate(relevant_global_indices)}
    
    # Create a local description map using local indices (0, 1, ...) as keys.
    # JSON keys are strings, so we must cast the global index to a string for the lookup.
    local_idx_to_description = {
        local_idx: global_idx_to_description[str(global_idx)]
        for local_idx, global_idx in local_to_global_map.items()
    }

    # --- Step 2: Pre-compute Index Lookups by Description (using local indices) ---
    description_to_local_indices = df.groupby('assay_description').groups

    # The list of all local indices we need to generate rankings for.
    all_local_indices = list(local_idx_to_description.keys())
    
    true_ranking_dict = {}
    loop_start_time = time.time()

    # --- Step 3: Optimized Comparison Loop Within the Group ---
    for i, anchor_local_idx in enumerate(all_local_indices):
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
            
            score = 0.0
            if target_indices_in_df is not None and len(target_indices_in_df) > 0:
                score = np.mean(trak_scores[np.ix_(target_indices_in_df, anchor_indices_in_df)])
            
            target_global_idx = local_to_global_map[target_local_idx]
            scores_for_anchor.append((score, target_global_idx))

        # Sort by score (descending) and global index (descending, for tie-breaking).
        scores_for_anchor.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        # Extract the sorted GLOBAL indices.
        sorted_global_indices = [g_idx for score, g_idx in scores_for_anchor]
        
        # The final dictionary is keyed by the anchor's GLOBAL index.
        anchor_global_idx = local_to_global_map[anchor_local_idx]
        true_ranking_dict[anchor_global_idx] = sorted_global_indices

        if (i + 1) % 100 == 0 or (i + 1) == len(all_local_indices):
            elapsed = time.time() - loop_start_time
            print(f"Processed {i + 1}/{len(all_local_indices)} items... ({elapsed:.2f}s elapsed)")

    print(f"Total group computation complete in {time.time() - start_time:.2f}s.")
    return true_ranking_dict

if __name__ == '__main__':
    with open('data/chembl_id_to_idx_list.json', 'r') as f:
        chembl_id_to_idx_list = json.load(f)
    with open('data/idx_to_description.json', 'r') as f:
        idx_to_description = json.load(f)

    final_true_ranking_dict = {}

    for chembl_id, global_indices in tqdm(chembl_id_to_idx_list.items()):
        print(f"--- Processing chembl_id: {chembl_id} ---")
        
        # Load the DataFrame and scores for the current group.
        df = pd.read_csv(f'data/{chembl_id}/total.csv')
        trak_scores = np.load(f'data/{chembl_id}/total.csv_100_checkpoints_512_proj_dim.npy')

        df = df.reset_index(drop=True)

        ranking_for_group = compute_true_ranking_per_group(
            df,
            trak_scores,
            idx_to_description,      
            global_indices          
        )

        final_true_ranking_dict.update(ranking_for_group)

    print("\nAll processing complete.")

    with open('true_rankings.json', 'w') as f:
        json.dump(final_true_ranking_dict, f, indent=4)
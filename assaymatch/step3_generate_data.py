import sys, os
import torch
import argparse
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import random

scripts_path = os.path.abspath(os.path.join(os.getcwd(), "scripts"))
sys.path.append(scripts_path)

from finetune_embeddings_predefined import TripletFineTuner

CHEMBL_ID_LIST = ['CHEMBL340', 'CHEMBL240', 'CHEMBL247', 'CHEMBL220', 'CHEMBL279', 'CHEMBL203'] 

def project_embeddings(ckpt_path: str, emb_array: np.ndarray) -> np.ndarray:
    """
    Load the trained MLP from checkpoint and project emb_array (l×768)
    to l×projection_dim, forcing everything on CPU.
    """
    # 1) force map_location to CPU so weights never hit GPU
    cpu = torch.device("cpu")
    model = TripletFineTuner.load_from_checkpoint(
        ckpt_path,
        map_location=cpu,
        embedding_file="data/description_embeddings.npy",
        projection_dim=768
    )
    # 2) ensure model is on CPU
    model.to(cpu).eval()

    # 3) convert input to a CPU tensor
    x = torch.from_numpy(emb_array).float().to(cpu)

    # 4) forward‐pass on CPU
    with torch.no_grad():
        proj = model.mlp(x)

    # 5) back to NumPy
    return proj.cpu().numpy()


parser = argparse.ArgumentParser(description="Generate Data")
parser.add_argument(
    "--base_path",
    "-b",
    type=str,
    required=True,
    help="base path for this experiment run",
)




def create_train_test_sets_with_multiple_strategies(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_sizes: list,
    assay_similarities_list: list, # Changed from assay_similarities (dict) to list of dicts
    num_assays_per_size: int = 10,
    min_test_size: int = 5,
    max_test_size: int = 100,
    random_seed: int = 42
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    test_desc_counts = test_df['assay_description'].value_counts(dropna=False).to_dict()
    
    valid_test_descriptions = []
    for desc, count in test_desc_counts.items():
        if min_test_size <= count <= max_test_size:
            # If assay_similarities_list is not empty, desc must be in at least one sim_dict
            if assay_similarities_list:
                if any(desc in sim_dict for sim_dict in assay_similarities_list):
                    valid_test_descriptions.append(desc)
            else: # If list is empty, no similarity check needed for validity
                valid_test_descriptions.append(desc)
    train_desc_counts = train_df['assay_description'].value_counts(dropna=False).to_dict()
    all_train_descriptions_global = list(train_desc_counts.keys()) # All available train descriptions
    
    results = {}

    num_to_sample = min(num_assays_per_size, len(valid_test_descriptions))
    if num_to_sample == 0:
        return

    chosen_test_descs_for_loop = random.sample(valid_test_descriptions, num_to_sample)
    random_test_df = test_df[test_df['assay_description'].isin(chosen_test_descs_for_loop)]


    for train_size_target, percentile in tqdm(train_sizes, desc="Processing train_sizes"):
        results[percentile] = {'random_train_sets':[], 'similarity_based': []}

        results[percentile]['random_test_set'] = random_test_df

        for _ in range(num_to_sample):
            shuffled_train_descriptions = all_train_descriptions_global.copy()
            random.shuffle(shuffled_train_descriptions)
            
            selected_random_train_descs = []
            current_random_train_row_count = 0
            for desc_cand in shuffled_train_descriptions:
                if current_random_train_row_count < train_size_target:
                    selected_random_train_descs.append(desc_cand)
                    current_random_train_row_count += train_desc_counts.get(desc_cand, 0)
                else:
                    break
            if not selected_random_train_descs:
                random_train_sample_df = pd.DataFrame(columns=train_df.columns)
            else:
                # Handle NaN descriptions correctly
                conditions = []
                non_na_selected = [d for d in selected_random_train_descs if pd.notna(d)]
                if non_na_selected:
                    conditions.append(train_df['assay_description'].isin(non_na_selected))
                if any(pd.isna(d) for d in selected_random_train_descs):
                    conditions.append(train_df['assay_description'].isna())
                
                if conditions:
                    combined_condition = conditions[0]
                    for cond in conditions[1:]: combined_condition |= cond
                    random_train_sample_df = train_df[combined_condition].copy()
                else:
                    random_train_sample_df = pd.DataFrame(columns=train_df.columns)
            results[percentile]['random_train_sets'].append(random_train_sample_df)
        
        for test_desc_current in chosen_test_descs_for_loop:
            generated_train_sets = []            
            if pd.isna(test_desc_current):
                current_test_df = test_df[test_df['assay_description'].isna()].copy()
            else:
                current_test_df = test_df[test_df['assay_description'] == test_desc_current].copy()
            for sim_dict in assay_similarities_list:
                current_similarity_map = sim_dict.get(test_desc_current, {})
                
                train_candidates_with_scores = []
                for train_d_cand in all_train_descriptions_global:
                    score = current_similarity_map.get(train_d_cand, -float('inf'))
                    train_candidates_with_scores.append((train_d_cand, score))
                
                train_candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                selected_sim_train_descs = []
                current_sim_train_row_count = 0
                for desc_cand, _ in train_candidates_with_scores:
                    if current_sim_train_row_count < train_size_target:
                        selected_sim_train_descs.append(desc_cand)
                        current_sim_train_row_count += train_desc_counts.get(desc_cand, 0)
                    else:
                        break
                
                # Construct the similarity-based training DataFrame
                if not selected_sim_train_descs:
                    sim_train_sample_df = pd.DataFrame(columns=train_df.columns)
                else:
                    # Handle NaN descriptions correctly
                    conditions = []
                    non_na_selected = [d for d in selected_sim_train_descs if pd.notna(d)]
                    if non_na_selected:
                        conditions.append(train_df['assay_description'].isin(non_na_selected))
                    if any(pd.isna(d) for d in selected_sim_train_descs):
                        conditions.append(train_df['assay_description'].isna())

                    if conditions:
                        combined_condition = conditions[0]
                        for cond in conditions[1:]: combined_condition |= cond
                        sim_train_sample_df = train_df[combined_condition].copy()
                    else:
                        sim_train_sample_df = pd.DataFrame(columns=train_df.columns)
                generated_train_sets.append(sim_train_sample_df)

            final_tuple_for_result = tuple(generated_train_sets + [current_test_df])
            results[percentile]["similarity_based"].append(final_tuple_for_result)
        
    return results

            

        #do the random subsets first. generate num


def generate_data(chembl_id, chembl_id_to_idx_list, idx_to_description, description_to_idx, description_embeddings, projected_description_embeddings, base_path, args, model_type, bao_dict):
    train_df = pd.read_csv(os.path.join(base_path, chembl_id, 'train.csv'))
    test_df = pd.read_csv(os.path.join(base_path, chembl_id, 'test.csv'))
    df = pd.concat([train_df, test_df])
    unique_test_descriptions = set(test_df['assay_description'].values)
    unique_train_descriptions = set(train_df['assay_description'].values)


    train_sizes = [(int(len(train_df) * i/10), i*10) for i in range(1, 10, 1)]

    assay_similarities = {}

    for test_description in set(test_df['assay_description'].values):
        test_idx = int(description_to_idx[test_description])
        test_embedding = description_embeddings[test_idx, :]

        assay_similarities[test_description] = {}

        for train_description in set(train_df['assay_description'].values):
            train_idx = int(description_to_idx[train_description])
            train_embedding = description_embeddings[train_idx, :]

            similarity = np.dot(test_embedding, train_embedding) / (np.linalg.norm(test_embedding) * np.linalg.norm(train_embedding))
            assay_similarities[test_description][train_description] = similarity

    assay_similarities_projected = {}

    for test_description in set(test_df['assay_description'].values):
        test_idx = int(description_to_idx[test_description])
        test_embedding = projected_description_embeddings[test_idx, :]

        assay_similarities_projected[test_description] = {}

        for train_description in set(train_df['assay_description'].values):
            train_idx = int(description_to_idx[train_description])
            train_embedding = projected_description_embeddings[train_idx, :]

            similarity = np.dot(test_embedding, train_embedding) / (np.linalg.norm(test_embedding) * np.linalg.norm(train_embedding))
            assay_similarities_projected[test_description][train_description] = similarity

    # --- Create Datasets using all 4 Strategies ---
    print("Creating final train/test sets...")
    result = create_train_test_sets_with_multiple_strategies(
        train_df, test_df, train_sizes,
        [assay_similarities, assay_similarities_projected],
        num_assays_per_size=10,
    )

    # --- Save Datasets and Create Config ---
    print("Saving datasets and generating configuration file...")
    base_save_path_prefix = f"{base_path}/{chembl_id}/{model_type}/datasets"
    os.makedirs(base_save_path_prefix, exist_ok=True)
    os.makedirs(f'{base_path}/{chembl_id}/{model_type}/results', exist_ok=True)
    
    script_name = "train_chemprop_classification" if model_type == 'chemprop' else "train_smiles_transformer_torch"
    config_dict = {
        "script": script_name,
        "cartesian_hyperparams": {
            "num_runs": [1], "outdir": [f'{base_path}/{chembl_id}/{model_type}/results'],
            "max_epochs": [20], "data_dir": [base_save_path_prefix],
            "num_workers": [2], "val_file": ["test_10_0.csv"]
        },
        "available_gpus": ["0", "1", "2", "3", "4", "5", "6", "7"],
        "paired_hyperparams": {"train_file": [], "test_file": []}
    }

    #if model_type == 'st' pop the keys max_epochs, num_workers, and val_files from the cartesian hyperparams
    if model_type == 'st':
        config_dict['cartesian_hyperparams'].pop('max_epochs')
        config_dict['cartesian_hyperparams'].pop('num_workers')
        config_dict['cartesian_hyperparams'].pop('val_file')

    for size, size_dict in result.items():
        size_dict['random_test_set'].to_csv(f'{base_save_path_prefix}/random_test_{size}.csv', index=False)
        
        for i, sample_tuple in enumerate(size_dict['similarity_based']):
            try:
                embedding_train_df, projected_train_df, test_df_sample = sample_tuple
            except ValueError:
                print(f"ERROR: Unpacking sample tuple for size {size}, index {i}. Expected 2 DataFrames.")
                continue
            
            try:
                # Save similarity-based training sets
                embedding_train_df.to_csv(f'{base_save_path_prefix}/embedding_train_{size}_{i}.csv', index=False)
                projected_train_df.to_csv(f'{base_save_path_prefix}/assaymatch_train_{size}_{i}.csv', index=False)

                
                # Save the common test set for this group
                test_df_sample.to_csv(f'{base_save_path_prefix}/test_{size}_{i}.csv', index=False)

                # Update config dictionary

                to_extend = [f'embedding_train_{size}_{i}.csv', f'assaymatch_train_{size}_{i}.csv']


                config_dict['paired_hyperparams']['train_file'].extend(to_extend)
                config_dict['paired_hyperparams']['test_file'].extend([f'test_{size}_{i}.csv'] * len(to_extend))

                if size == 10:
                    baos = {bao_dict[str(assay_id)] for assay_id in test_df_sample['assay_id'] if str(assay_id) in bao_dict}

                    bao_train_subdf = train_df[train_df['assay_id'].apply(lambda x: bao_dict.get(str(x)) in baos if str(x) in bao_dict else False)]

                    bao_train_subdf.to_csv(f'{base_save_path_prefix}/bao_train_{size}_{i}.csv', index=False)
                    config_dict['paired_hyperparams']['train_file'].append(f'bao_train_{size}_{i}.csv')
                    config_dict['paired_hyperparams']['test_file'].append(f'test_{size}_{i}.csv')

            except AttributeError as e:
                print(f"ERROR: Could not save CSVs for size {size}, index {i}: {e}")
                continue
        
        for i, train_df_random in enumerate(size_dict['random_train_sets']):
            train_df_random.to_csv(f'{base_save_path_prefix}/random_train_{size}_{i}.csv', index=False)
            config_dict['paired_hyperparams']['train_file'].append(f'random_train_{size}_{i}.csv')
            config_dict['paired_hyperparams']['test_file'].append(f'random_test_{size}.csv')
    
    train_df.to_csv(f'{base_save_path_prefix}/train.csv', index=False)
    test_df.to_csv(f'{base_save_path_prefix}/test.csv', index = False)


    config_dict['paired_hyperparams']['train_file'].append(f'train.csv')
    config_dict['paired_hyperparams']['test_file'].append(f'test.csv')

    
    
    with open(f'{base_path}/{chembl_id}_config_{model_type}.json', 'w') as f:
        json.dump(config_dict, f, indent=4)

    print(f"--- Finished data generation for {chembl_id} ---")

# =================================================================
if __name__ == "__main__":
    args = parser.parse_args()
    chembl_ids = CHEMBL_ID_LIST

    with open('data/chembl_id_to_idx_list.json', 'r') as f:
        chembl_id_to_idx_list = json.load(f)
    with open('data/idx_to_description.json', 'r') as f:
        idx_to_description = json.load(f)
    with open('data/description_to_idx.json', 'r') as f:
        description_to_idx = json.load(f)
    
    with open('data/chembl_id_to_ic50_assay_bao_dicts.json', 'r') as f:
        chembl_id_to_ic50_assay_bao_dicts = json.load(f)

    # train_pairs_positive = np.load(f'{args.base_path}/assay_split_positive_pairs_train.npy').flatten()
    # train_pairs_negative = np.load(f'{args.base_path}/assay_split_negative_pairs_train.npy').flatten()

    # with open(f'{args.base_path}/train_indices.json', 'r') as f:
    #     train_indices = json.load(f)

    description_embeddings = np.load('data/description_embeddings.npy')


    #find only 

    projected_description_embeddings_chemprop = project_embeddings(f'{args.base_path}/{[f for f in os.listdir(args.base_path) if f.endswith("chemprop.ckpt")][0]}', description_embeddings)
    for chembl_id in chembl_ids:
        generate_data(chembl_id, chembl_id_to_idx_list, idx_to_description, description_to_idx, description_embeddings, projected_description_embeddings_chemprop, args.base_path, args, 'chemprop', chembl_id_to_ic50_assay_bao_dicts[chembl_id])

    projected_description_embeddings_st = project_embeddings(f'{args.base_path}/{[f for f in os.listdir(args.base_path) if f.endswith("st.ckpt")][0]}', description_embeddings)
    for chembl_id in chembl_ids:
        generate_data(chembl_id, chembl_id_to_idx_list, idx_to_description, description_to_idx, description_embeddings, projected_description_embeddings_chemprop, args.base_path, args, 'st', chembl_id_to_ic50_assay_bao_dicts[chembl_id])
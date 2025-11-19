import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # make sure scipy is installed
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import re
import itertools
import collections
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
from sklearn.metrics import roc_auc_score

def plot_auroc_vs_size(ax, data, smooth_sigma=0, legend=True, title = None):
    plot_data = data.copy()

    bao_auroc = plot_data.pop('bao', {}).get('10', None)

    if not plot_data:
        print("No data to plot.")
        return

    sizes = sorted(int(s) for s in next(iter(plot_data.values())).keys())

    for method, results in plot_data.items():
        # Extract MAE values in order of sizes, defaulting if a size is missing
        aurocs = [results.get(str(size), np.nan) for size in sizes]

        y_values = np.array(aurocs, dtype=float)
        
        if smooth_sigma > 0:
            # Apply 1D Gaussian smoothing, ignoring NaNs
            y_values = gaussian_filter1d(y_values, sigma=smooth_sigma)

        ax.plot(sizes, y_values, marker='o', linestyle='-', label=method)
    
    if bao_auroc is not None:
        ax.axhline(y=bao_auroc, color='red', linestyle='--', label='BAO selection baseline')

    ax.set_xlabel('Percentage of Full Train Set')
    ax.set_ylabel('AUROC')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('AUROC vs. Training Set Size for Different Methods')
    ax.grid(True, linestyle='--', linewidth=0.5)

    if legend:
        ax.legend(title='Selection Strategy')

# --- Constants ---
CHEMBL_ID_LIST = ['CHEMBL340', 'CHEMBL240', 'CHEMBL247', 'CHEMBL220', 'CHEMBL279', 'CHEMBL203']  
GROUPED_TAGS = {'embedding', 'assaymatch', 'bao'}
LABEL_COL = 'classification_label'
PREDS_COL = 'classification_preds'
REQUIRED_COLS = {LABEL_COL, PREDS_COL}

def _group_filepaths(base_paths: List[str], chembl_ids: List[str], subpaths: List[str], model_type: str) -> Dict[str, Any]:
    """
    Scans directories to find and group result filepaths by category, correctly
    separating runs from different chembl_id directories.
    """
    filename_re = re.compile(
        r"^(embedding|assaymatch|random|bao)_train_.*?_test_(\d+).*?(run_\d+)\.csv$"
    )
    # Temporary structure now includes the directory path to keep runs separate
    temp_grouped = collections.defaultdict( # key: dirpath
        lambda: collections.defaultdict( # key: category
            lambda: collections.defaultdict( # key: size
                lambda: collections.defaultdict(list) # key: run_id
            )
        )
    )
    full_files = []


    for b_path, c_id, s_path in itertools.product(base_paths, chembl_ids, subpaths):
        dirpath = Path(b_path) / c_id  / model_type / s_path
        if not dirpath.is_dir():
            print(f"Warning: Directory not found '{dirpath}'. Skipping.")
            continue

        # Isolate 'full' files, which are not grouped by run or size
        for entry in dirpath.glob('*train_random_test*.csv'):
            if entry.is_file():
                full_files.append(entry)

        # Group all other files
        for entry in dirpath.iterdir():
            if not entry.is_file() or 'train_test_run' in entry.name or "train_random_test" in entry.name:
                continue
            match = filename_re.match(entry.name)
            if match:
                category, size_str, run_id = match.groups()
                size = int(size_str)
                temp_grouped[dirpath][category][size][run_id].append(entry)

    # --- Aggregate results from the detailed temp_grouped structure ---
    final_files = collections.defaultdict(dict)
    final_files['full'] = full_files
    
    # This structure will hold the properly separated run file lists
    # Format: {category: {size: [[run1_files], [run2_files], ...]}}
    aggregated_runs = collections.defaultdict(lambda: collections.defaultdict(list))

    for dirpath, categories in temp_grouped.items():
        for category, sizes in categories.items():
            for size, runs in sizes.items():
                # Each value in runs.values() is a list of files for a unique
                # combination of dirpath and run_id.
                for run_files in runs.values():
                    aggregated_runs[category][size].append(run_files)

    # Now, format the aggregated runs into the final structure
    for category, sizes in aggregated_runs.items():
        if category in GROUPED_TAGS:
            final_files[category] = sizes
        elif category == 'random':
            # For 'random', flatten the run lists into a single list per size
            final_files[category] = collections.defaultdict(list)
            for size, run_groups in sizes.items():
                for file_list in run_groups:
                    final_files[category][size].extend(file_list)
            
    return final_files



def _read_csv_results(filepath: Path) -> Optional[Tuple[List, List]]:
    """Reads a CSV file, validates its columns, and returns labels and predictions."""
    try:
        df = pd.read_csv(filepath)
        if df.empty or not REQUIRED_COLS.issubset(df.columns):
            print(f"Warning: Missing required columns or empty file in '{filepath}'. Skipping.")
            return None
        return df[LABEL_COL].tolist(), df[PREDS_COL].tolist()
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Warning: Cannot read '{filepath}' ({e}). Skipping.")
        return None


def _calculate_auc(labels: List, preds: List, context: str) -> Optional[float]:
    """Calculates ROC AUC score, handling cases with a single class."""
    if len(set(labels)) < 2:
        print(f"Info: Only one class present for {context}. AUC is undefined.")
        return None
    try:
        return roc_auc_score(labels, preds)
    except ValueError as e:
        print(f"Warning: Could not compute AUC for {context}: {e}")
        return None

def finalize_classification_results(base_paths: List[str], chembl_ids: List[str], paths: List[str], model_type: str) -> Dict[str, Any]:
    """
    Finalizes classification results by calculating average AUROC scores.
    """
    if not isinstance(base_paths, list): base_paths = [base_paths]
    if not isinstance(chembl_ids, list): chembl_ids = [chembl_ids]
    if not isinstance(paths, list): paths = [paths]

    grouped_files = _group_filepaths(base_paths, chembl_ids, paths, model_type)
    final_aucs = {}
    final_aucs_before_mean = {}

    for tag, data in grouped_files.items():
        if tag in GROUPED_TAGS:
            # --- 1. Logic for fully grouped tags ('embedding', 'assaymatch', 'bao') ---
            final_aucs[tag] = {}
            final_aucs_before_mean[tag] = {}
            for size, run_groups in data.items():
                run_groups = [sorted(group) for group in run_groups]
                run_groups.sort(key=lambda g: g[0])
                aucs_for_size = []
                for file_group in run_groups: # A group of files for one run
                    all_labels, all_preds = [], []
                    for fp in file_group:
                        results = _read_csv_results(fp)
                        if results:
                            all_labels.extend(results[0])
                            all_preds.extend(results[1])
                    if all_labels:
                        auc = _calculate_auc(all_labels, all_preds, f"{tag}/{size}")
                        if auc is not None: aucs_for_size.append(auc)
                if aucs_for_size:
                    final_aucs[tag][str(size)] = statistics.mean(aucs_for_size)
                    final_aucs_before_mean[tag][str(size)] = aucs_for_size
        
        elif tag == 'random':
            # --- 2. Logic for 'random' (group by size, but AUC per file) ---
            final_aucs[tag] = {}
            final_aucs_before_mean[tag] = {}
            for size, filepaths in data.items(): 
                filepaths.sort()
                aucs_for_size = []
                for fp in filepaths:
                    results = _read_csv_results(fp)
                    if results:
                        # Calculate AUC for each individual file
                        auc = _calculate_auc(results[0], results[1], f"{tag}/{size}/{fp.name}")
                        if auc is not None: aucs_for_size.append(auc)
                if aucs_for_size:
                    final_aucs_before_mean[tag][str(size)] = aucs_for_size
                    final_aucs[tag][str(size)] = statistics.mean(aucs_for_size)

        else: # This will handle 'full'
            # --- 3. Logic for non-grouped tags ('full') ---
            aucs_for_tag = []
            data.sort()
            for filepath in data: # data is a flat list of filepaths
                results = _read_csv_results(filepath)
                if results:
                    auc = _calculate_auc(results[0], results[1], f"{tag}/{filepath.name}")
                    if auc is not None: aucs_for_tag.append(auc)
            if aucs_for_tag:
                final_aucs[tag] = statistics.mean(aucs_for_tag)
                final_aucs_before_mean[tag] = aucs_for_tag
                
    return final_aucs, final_aucs_before_mean


if __name__ == '__main__':
    results, _= finalize_classification_results([f'benchmark/run_{i}' for i in range(1, 16)],['CHEMBL340', 'CHEMBL240', 'CHEMBL203', 'CHEMBL220', 'CHEMBL279', 'CHEMBL247'], 'results', 'chemprop')
    results['random']['100'] = results['full']
    results['embedding']['100'] = results['full']
    results['assaymatch']['100'] = results['full']
    results.pop('full')
    fig, ax = plt.subplots(figsize = (10, 6), dpi=100)

    plot_auroc_vs_size(ax, results)
    plt.savefig('benchmark/chemprop_learning_curve.png')

    full_results_chemprop = {}
    for chembl_id in ['CHEMBL340', 'CHEMBL240', 'CHEMBL203', 'CHEMBL220', 'CHEMBL279', 'CHEMBL247']: 
        _, results = finalize_classification_results([f'benchmark/run_{i}' for i in range(1, 16)], chembl_id, 'results', 'chemprop')
        full_results_chemprop[chembl_id] = results 
    with open('benchmark/chemprop_full_results.json', 'w') as f:
        json.dump(full_results_chemprop, f, indent=4)

    results, _= finalize_classification_results([f'benchmark/run_{i}' for i in range(1, 16)],['CHEMBL340', 'CHEMBL240', 'CHEMBL203', 'CHEMBL220', 'CHEMBL279', 'CHEMBL247'], 'results', 'st')
    results['random']['100'] = results['full']
    results['embedding']['100'] = results['full']
    results['assaymatch']['100'] = results['full']
    results.pop('full')
    fig, ax = plt.subplots(figsize = (10, 6), dpi=100)

    plot_auroc_vs_size(ax, results)
    plt.savefig('benchmark/st_learning_curve.png')

    full_results_st = {}
    for chembl_id in ['CHEMBL340', 'CHEMBL240', 'CHEMBL203', 'CHEMBL220', 'CHEMBL279', 'CHEMBL247']: 
        _, results = finalize_classification_results([f'benchmark/run_{i}' for i in range(1, 16)], chembl_id, 'results', 'st')
        full_results_chemprop[chembl_id] = results 
    with open('benchmark/st_full_results.json', 'w') as f:
        json.dump(full_results_chemprop, f, indent=4)

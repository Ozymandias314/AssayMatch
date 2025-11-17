import argparse

import pandas as pd
import sqlite3 
import json
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
import numpy as np

db_path = '/data/rbg/users/vincentf/data_uncertainty/chembl_34/chembl_34/chembl_34_sqlite/chembl_34.db'

conn = sqlite3.connect(db_path, timeout = 10)
cur = conn.cursor()


parser = argparse.ArgumentParser(description="Make a dataset.")
parser.add_argument(
    "--target_chembl_id",
    "-t",
    type=str,
    required=True,
    help="target ChEMBL ID",
)
parser.add_argument(
    "--output_dir",
    "-o",
    type=str,
    required=True,
    help="output directory",
)
parser.add_argument(
    "--no_mutants",
    action="store_true",
    default=False,
    help="don't include mutant targets",
)
parser.add_argument(
    "--deduplicate",
    action="store_true",
    default=False,
    help="deduplicate the dataset",
)
parser.add_argument(
     "--assay_list_path",
    type=str,
    default=None,
    help="path to a json containing an iterable of assay IDs to include",
)
parser.add_argument('--results_path', type=str, required=False, help='Path to save results.')

parser.add_argument('--experiment_name', type=str, required=False, help='Name of experiment.')
def get_assay_activities(a_id):
    assay_activities = cur.execute(f'''
                                SELECT 
                                    molregno,
                                    relation,
                                    value,
                                    units,
                                    standard_type,
                                    compound_records.compound_name,
                                    compound_structures.canonical_smiles
                                FROM 
                                    activities 
                                JOIN
                                    compound_records USING (molregno)
                                JOIN
                                    compound_structures USING (molregno)
                                WHERE
                                    assay_id = '{a_id}'
                                AND 
                                    standard_type = 'IC50'
                                GROUP BY
                                    molregno,
                                    relation,
                                    value,
                                    units,
                                    standard_type
                                ''').fetchall()
    
    return [{"molregno": molregno, "relation": relation, "value": value, "units": units, "standard_type": standard_type, "compound_name": compound_name, "compound_smiles": smiles}  for molregno, relation, value, units, standard_type, compound_name, smiles in assay_activities]

def get_all_target_info(t_chembl_id):
        target_to_assays = cur.execute(f'''
                                SELECT 
                                    tid,
                                    target_dictionary.pref_name,
                                    target_dictionary.chembl_id,
                                    assay_id,
                                    assays.description,
                                    docs.doi,
                                    docs.journal,
                                    docs.pubmed_id,
                                    docs.abstract,
                                    COUNT(DISTINCT molregno) as cnt,
                                    COUNT(activity_id)
                                FROM 
                                    activities 
                                JOIN 
                                    assays USING (assay_id)
                                JOIN 
                                    target_dictionary USING (tid)
                                JOIN 
                                    docs ON (assays.doc_id = docs.doc_id)
                                WHERE 
                                    target_dictionary.chembl_id = '{t_chembl_id}'
                                AND 
                                    standard_type = 'IC50'
                                GROUP BY
                                    tid,
                                    target_dictionary.pref_name,
                                    target_dictionary.chembl_id,
                                    assay_id
                                ORDER BY
                                    cnt DESC
                                ''').fetchall()
        toreturn = []
        for tid, t_pref_name, t_chembl_id, a_id, description, doi, journal, pmid, abstract, m_cnt, a_cnt in target_to_assays:
            to_add = {
                "assay": a_id,
                "target": t_pref_name,
                "target_id": tid,
                "num_activities": a_cnt,
                "assay_description": description,}
            if doi is not None:
                to_add["document"] = {"doi": doi, "pmid": pmid, "journal": journal, "abstract": abstract}
            to_add["activities"] = get_assay_activities(a_id)
            toreturn.append(to_add)
        return toreturn

units_with_conversion = [
    "10'-6 mol/L",
    "mM",
    "M",
    "nM",
    "nmol/L",
    "pM",
    "mg/ml",
    "ug ml-1",
    "10'-1 ug/ml",
    "10'-10M",
    "10'-1microM",
    "10'-2 ug/ml",
    "10'-2microM",
    "10'-3 ug/ml",
    "10'-5g/ml",
    "10'-6g/ml",
    "10'-7M",
    "10^-8M",
    "10'1 uM",
    "10'2 uM",
    "10'3 uM",
    "10'1 ug/ml",
    "10'2 ug/ml",
    "uM l-1",
    "umol/L",
    "10'-2M",
    "10'-8M",
    "10^-4microM",
    "10'3pM",
    "10^-3microM",
    "10'-3microM",
    "10'-9M",
    "10'-4microM",
    "10'6pM",
    "10'5pM",
    "10'-12M",
    "10'-4M",
    "10'2pM",
    "10'-6M",
    "10'-3M",
    "10'-5M",
    "10^-5M",
    "10^-7M",
    "10^-9M",
    "microM",
    "10'-7g/ml",
    "um",
    "g/ml",
    "10'4nM",
    "10'3nM",
    "10'-9mol/L",
    "nM l-1",
]


if __name__ == "__main__":
    args = parser.parse_args()

    all_assays = get_all_target_info(args.target_chembl_id)
    with open(f"{args.output_dir}/{args.target_chembl_id}_all_ic50_activities.json", "w") as f:
        json.dump(all_assays, f, indent=4)

    if args.assay_list_path is not None:
        with open(args.assay_list_path, "r") as f:
            assay_list = json.load(f)
        assay_list = [int(a) for a in assay_list]

        data = []
        for a in all_assays:
            if a['assay'] in assay_list:
                data.append(a)
    else:
        data = all_assays
    
    assay_id = []
    relation = []
    value = []
    unit = []
    compound_smiles = []
    compound_name = []
    doi = []
    assay_description = []
    year = []
    volume = []
    issue = []
    for assay in data:
        for activity in assay['activities']:
            assay_id.append(assay['assay'])
            relation.append(activity['relation'])
            value.append(activity['value'])
            unit.append(activity['units'])
            assay_description.append(assay['assay_description'])
            compound_smiles.append(activity['compound_smiles'])
            compound_name.append(activity['compound_name'])
            if 'document' in assay:
                if 'doi' in assay['document']:
                    doi.append(assay['document']['doi'])
                else:
                    doi.append(pd.NA)
                if 'year' in assay['document'] and assay['document']['year'] is not None:
                    year.append(assay['document']['year'])
                else:
                    year.append(pd.NA)
                if 'volume' in assay['document'] and assay['document']['volume'] is not None:
                    volume.append(assay['document']['volume'])
                else:
                    volume.append(pd.NA)
                if 'issue' in assay['document'] and assay['document']['issue'] is not None:
                    issue.append(assay['document']['issue'])
                else:
                    issue.append(pd.NA)
            else:
                doi.append(pd.NA)
                year.append(pd.NA)
                volume.append(pd.NA)
                issue.append(pd.NA)
    for i in range(len(value)):
        if value[i] is not None:
            
            if unit[i] == '10\'-6 mol/L':
                unit[i] = 'uM'
            if unit[i] == 'um':
                unit[i] = 'uM'
    
            elif unit[i] == 'ng/ml':
                unit[i] = 'uM'
            elif unit[i] == 'microM':
                unit[i] = 'uM'
            elif unit[i] == "10'4nM":
                value[i] == value[i] * 10
                unit[i] = 'uM'
            elif unit[i] == "10'3nM":
                unit[i] = 'uM'
            
            elif unit[i] == 'mM':
                value[i] = value[i] * 1000  # Convert mM to uM
                unit[i] = 'uM'
            
            elif unit[i] == 'M':
                value[i] = value[i] * 1000000  # Convert M to uM
                unit[i] = 'uM'
            
            elif unit[i] == 'nM':
                value[i] = value[i] / 1000  # Convert nM to uM
                unit[i] = 'uM'
            elif unit[i] == "10'-9mol/L":
                value[i] = value[i] / 1000  # Convert nM to uM
                unit[i] = 'uM'
            
            elif unit[i] == 'nmol/L':
                value[i] = value[i] / 1000  # Convert nmol/L to uM
                unit[i] = 'uM'
            elif unit[i] == 'nM l-1':
                value[i] = value[i] / 1000  # Convert nmol/L to uM
                unit[i] = 'uM'
            elif unit[i] == 'pM':
                value[i] = value[i] / 1000000  # Convert pM to uM
                unit[i] = 'uM'
            
            elif unit[i] == 'mg/ml':
                mol_wt = ExactMolWt(Chem.MolFromSmiles(compound_smiles[i]))
                value[i] = value[i] * 1000000 / mol_wt  # Convert mg/ml to uM
                unit[i] = 'uM'
            
            elif unit[i] == 'ug ml-1':
                mol_wt = ExactMolWt(Chem.MolFromSmiles(compound_smiles[i]))
                value[i] = value[i] * 1000 / mol_wt  # Convert ug/ml to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-1 ug/ml":
                mol_wt = ExactMolWt(Chem.MolFromSmiles(compound_smiles[i]))
                value[i] = value[i] * 0.1 * 1000 / mol_wt  # Convert 10^-1 ug/ml to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-10M":
                value[i] = value[i] * 1e-10 * 1000000  # Convert 10^-10 M to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-1microM":
                value[i] = value[i] * 0.1  # Convert 10^-1 microM to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-2 ug/ml":
                mol_wt = ExactMolWt(Chem.MolFromSmiles(compound_smiles[i]))
                value[i] = value[i] * 0.01 * 1000 / mol_wt  # Convert 10^-2 ug/ml to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-2microM":
                value[i] = value[i] * 0.01  # Convert 10^-2 microM to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-3 ug/ml":
                mol_wt = ExactMolWt(Chem.MolFromSmiles(compound_smiles[i]))
                value[i] = value[i] * 0.001 * 1000 / mol_wt  # Convert 10^-3 ug/ml to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-5g/ml":
                mol_wt = ExactMolWt(Chem.MolFromSmiles(compound_smiles[i]))
                value[i] = value[i] * 1e-5 * 1000000 / mol_wt  # Convert 10^-5 g/ml to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-6g/ml":
                mol_wt = ExactMolWt(Chem.MolFromSmiles(compound_smiles[i]))
                value[i] = value[i] * 1e-6 * 1000000 / mol_wt  # Convert 10^-6 g/ml to uM
                unit[i] = 'uM'
            elif unit[i] == "10'-7g/ml":
                mol_wt = ExactMolWt(Chem.MolFromSmiles(compound_smiles[i]))
                value[i] = value[i] * 1e-7 * 1000000 / mol_wt  # Convert 10^-6 g/ml to uM
                unit[i] = 'uM'
            elif unit[i] == "g/ml":
                mol_wt = ExactMolWt(Chem.MolFromSmiles(compound_smiles[i]))
                value[i] = value[i] * 1e-7 * 1000000 / mol_wt  # Convert 10^-6 g/ml to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-7M":
                value[i] = value[i] * 1e-7 * 1000000  # Convert 10^-7 M to uM
                unit[i] = 'uM'
            
            elif unit[i] == '10^-8M':
                value[i] = value[i] * 1e-8 * 1000000  # Convert 10^-8 M to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'1 uM":
                value[i] = value[i] * 10  # Convert 10^1 uM to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'2 uM":
                value[i] = value[i] * 100  # Convert 10^2 uM to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'3 uM":
                value[i] = value[i] * 1000  # Convert 10^3 uM to uM
                unit[i] = 'uM'
            elif unit[i] == "10'1 ug/ml":
                mol_wt = ExactMolWt(Chem.MolFromSmiles(compound_smiles[i]))
                value[i] = value[i] * 10 * 1000 / mol_wt  # Convert 10^1 ug/ml to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'2 ug/ml":
                mol_wt = ExactMolWt(Chem.MolFromSmiles(compound_smiles[i]))
                value[i] = value[i] * 100 * 1000 / mol_wt  # Convert 10^2 ug/ml to uM
                unit[i] = 'uM'
            
            elif unit[i] == 'uM l-1':
                unit[i] = 'uM'  # Just a synonym for uM, no conversion needed
            
            elif unit[i] == 'umol/L':
                unit[i] = 'uM'  # Also a synonym for uM, no conversion needed
            elif unit[i] == "10'-2M":
                value[i] = value[i] * 1e-2 * 1e6  # Convert 10^-2 M to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-8M":
                value[i] = value[i] * 1e-8 * 1e6  # Convert 10^-8 M to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10^-4microM":
                value[i] = value[i] * 1e-4  # Convert 10^-4 microM to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'3pM":
                value[i] = value[i] * 1e3 * 1e-6  # Convert 10^3 pM to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10^-3microM":
                value[i] = value[i] * 1e-3  # Convert 10^-3 microM to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-3microM":
                value[i] = value[i] * 1e-3  # Convert 10^-3 microM to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-9M":
                value[i] = value[i] * 1e-9 * 1e6  # Convert 10^-9 M to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-4microM":
                value[i] = value[i] * 1e-4  # Convert 10^-4 microM to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'6pM":
                value[i] = value[i] * 1e6 * 1e-6  # Convert 10^6 pM to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'5pM":
                value[i] = value[i] * 1e5 * 1e-6  # Convert 10^5 pM to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-12M":
                value[i] = value[i] * 1e-12 * 1e6  # Convert 10^-12 M to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-4M":
                value[i] = value[i] * 1e-4 * 1e6  # Convert 10^-4 M to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'2pM":
                value[i] = value[i] * 1e2 * 1e-6  # Convert 10^2 pM to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-6M":
                value[i] = value[i] * 1e-6 * 1e6  # Convert 10^-6 M to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-3M":
                value[i] = value[i] * 1e-3 * 1e6  # Convert 10^-3 M to uM
                unit[i] = 'uM'
            
            elif unit[i] == "10'-5M":
                value[i] = value[i] * 1e-5 * 1e6  # Convert 10^-5 M to uM
                unit[i] = 'uM'
            elif unit[i] == "10^-5M":
                value[i] = value[i] * 1e-5 * 1e6  # Convert 10^-5 M to uM
                unit[i] = 'uM'
            elif unit[i] == "10^-7M":
                value[i] = value[i] * 1e-7 * 1e6  # Convert 10^-5 M to uM
                unit[i] = 'uM'
            elif unit[i] == "10^-9M":
                value[i] = value[i] * 1e-9 * 1e6  # Convert 10^-5 M to uM
                unit[i] = 'uM'
    df = pd.DataFrame({'assay_id': assay_id, 'doi': doi, 'year': year, 'volume': volume, 'issue': issue, 'assay_description': assay_description, 'compound_smiles': compound_smiles, 'compound_name': compound_name, 'value': value, 'unit': unit, 'relation': relation})
    
    df_equality = df.loc[
        (df['relation'] == '=') &
        df['value'].notna() &
        (df['value'] != 0) &
        (df['unit'] == 'uM')
    ].copy()            # <‑‑ add copy()

    clean_df = df_equality
    
    clean_df.loc[:, 'log_value'] = np.log10(clean_df['value'])
    clean_df.loc[:, 'classification_label'] = (clean_df['log_value'] > 0).astype(int)

    clean_df['scaffold_smiles'] = [Chem.MolToSmiles(GetScaffoldForMol(Chem.MolFromSmiles(smi))) for smi in clean_df['compound_smiles']]
    
    clean_df.to_csv(f'{args.output_dir}/{args.target_chembl_id}_ic50_equality_data.csv')

    #mutants = set(df[df['assay_description'].str.contains('mutant', case=False, na=False)]['assay_id'])

    if args.no_mutants:
        mutants = set(clean_df[clean_df['assay_description'].str.contains('mutant', case=False, na=False)]['assay_id'])
        clean_df_no_mutants = clean_df[~clean_df['assay_id'].isin(mutants)]

        clean_df_no_mutants.to_csv(f'{args.output_dir}/{args.target_chembl_id}_ic50_equality_data_no_mutants.csv')

        clean_df = clean_df_no_mutants
    
    if args.deduplicate:
        df = clean_df 
        smiles_list = df['compound_smiles'].tolist()
        smiles_with_duplicates = [smiles for smiles in smiles_list if smiles_list.count(smiles) > 1]
        # Create a combined column for sorting, handling NaN values
        df["year_volume_issue"] = (
            df["year"].fillna(-1).astype(int).astype(str).str.zfill(4)
            + "."
            + df["volume"].fillna(-1).astype(int).astype(str).str.zfill(2)
            + "."
            + df["issue"].fillna(-1).astype(int).astype(str).str.zfill(2)
        )

        # Define a function to process each group
        def process_group(group):
            # Sort by year.volume.issue, placing NaN values last
            group = group.sort_values("year_volume_issue", na_position="last")
            to_keep = []
            seen = []
            
            for _, row in group.iterrows():
                value = row["value"]
                # Check if value is within 1% of any already retained value
                if not any(abs(value - x) / x <= 0.01 for x in seen):
                    to_keep.append(row)
                    seen.append(value)
            
            return pd.DataFrame(to_keep)

        # Apply the function to each group
        result = df.groupby("compound_smiles", group_keys=False).apply(process_group)

        # Drop helper column for clarity
        result = result.drop(columns=["year_volume_issue"])

        result = result.drop(columns=['Unnamed: 0'], errors="ignore")

        if args.no_mutants:
            result.to_csv(f'{args.output_dir}/{args.target_chembl_id}_ic50_equality_data_no_mutants_deduplicated.csv', index=False)
        else:
            result.to_csv(f'{args.output_dir}/{args.target_chembl_id}_ic50_equality_data_deduplicated.csv', index=False)
        
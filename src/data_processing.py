"""
Data loading, cleaning, deduplication, and SMILES standardization
for ChEMBL-style bioactivity data (e.g. DPP-4).
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import SaltRemover
from molvs import standardize_smiles

CHEMBL_COLUMNS = [
    'Molecule ChEMBL ID', 'Smiles', 'Standard Type',
    'Standard Value', 'Standard Units', 'Document Year', 'Target Name'
]


def load_and_clean_data(file_path, sep=';'):
    """Load a raw ChEMBL export, drop incomplete rows, and keep nM records."""
    df = pd.read_csv(file_path, sep=sep)
    df = df[df['Smiles'].notna()]
    df = df[df['Standard Value'].notna()]
    df = df[df['Standard Units'].notna()]
    df = df[CHEMBL_COLUMNS]
    df = df[df['Standard Units'].str.contains('nM', na=False)]
    print(f"Loaded {df.shape[0]} compounds with nM-unit records")
    return df


def deduplicate_by_mean(df, id_col='Molecule ChEMBL ID', value_col='Standard Value'):
    """
    Collapse repeated measurements per compound to a single row by taking the
    mean of Standard Value across all records sharing the same ID, keeping
    the lowest resulting value per compound (mirrors the notebook's
    groupby -> transform("mean") -> sort -> drop_duplicates workflow).
    """
    df = df.copy()
    df['new_value'] = df.groupby(id_col)[value_col].transform('mean')
    df = df.sort_values('new_value', ascending=True)
    df = df.drop_duplicates(id_col, keep='first')
    print(f"Deduplicated to {df.shape[0]} unique compounds")
    return df


def standardize_and_strip_salts(smiles_list, salt_defn="[Cl,Br]"):
    """Standardize SMILES with MolVS and strip simple salts with RDKit."""
    standardized = [standardize_smiles(smi) for smi in smiles_list]
    remover = SaltRemover.SaltRemover(defnData=salt_defn)
    mols = [Chem.MolFromSmiles(smi) for smi in standardized]
    stripped = [remover.StripMol(mol) if mol is not None else None for mol in mols]
    clean_smiles = [Chem.MolToSmiles(mol) if mol is not None else None for mol in stripped]
    return clean_smiles


def label_by_threshold(df, value_col='Standard Value', threshold=10000):
    """Binary-label compounds: active (<= threshold nM) vs inactive."""
    df = df.copy()
    df['label'] = (df[value_col] <= threshold).astype(int)
    return df


if __name__ == "__main__":
    print("Data processing module ready.")

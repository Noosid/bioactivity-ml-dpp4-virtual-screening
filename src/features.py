"""
Feature generation: Morgan fingerprints, Murcko scaffolds,
and Tanimoto similarity utilities.
"""
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity


def smiles_to_morgan_fp(smiles_list, radius=2, n_bits=1024):
    """Convert a list of SMILES strings to Morgan (ECFP-like) fingerprints."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp))
    return np.array(fps)


def smiles_to_morgan_bitvects(smiles_list, radius=2, n_bits=1024):
    """Same as smiles_to_morgan_fp but returns raw RDKit BitVects, needed
    for Tanimoto similarity calculations (which operate on BitVects, not
    plain numpy arrays)."""
    bitvects = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            bitvects.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    return bitvects


def get_murcko_scaffolds(smiles_list):
    """Return the Murcko (ring-system) scaffold SMILES for each input molecule."""
    scaffolds = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.append(Chem.MolToSmiles(scaffold_mol))
        else:
            scaffolds.append(None)
    return scaffolds


def cluster_by_scaffold(df, scaffold_col='Murcko', min_cluster_size=2):
    """
    Split a dataframe into train/test sets based on Murcko scaffold groups.
    Scaffolds observed >= min_cluster_size times go to train (so at least
    one structurally-related analog is available for learning), all
    singleton scaffolds go to test. This is a simple, self-contained
    stand-in for the external clustering step in the original notebook.
    """
    counts = df[scaffold_col].value_counts()
    train_scaffolds = counts[counts >= min_cluster_size].index
    train_df = df[df[scaffold_col].isin(train_scaffolds)].reset_index(drop=True)
    test_df = df[~df[scaffold_col].isin(train_scaffolds)].reset_index(drop=True)
    print(f"Scaffold split -> train: {train_df.shape[0]}, test: {test_df.shape[0]}")
    return train_df, test_df


def pairwise_tanimoto(train_fps, train_ids, test_fps, test_ids):
    """
    Compute pairwise Tanimoto similarity between every training and test
    fingerprint. Returns a long-format DataFrame with columns:
    Xtrain, Xtest, TC (Tanimoto coefficient).
    """
    rows = []
    for i, fp1 in enumerate(train_fps):
        for j, fp2 in enumerate(test_fps):
            tc = round(TanimotoSimilarity(fp1, fp2), 2)
            rows.append((train_ids[i], test_ids[j], tc))
    tc_df = pd.DataFrame(rows, columns=['Xtrain', 'Xtest', 'TC'])
    return tc_df.sort_values('TC', ascending=False)


if __name__ == "__main__":
    print("Features module ready.")

"""
Tanimoto similarity analysis.

Computes pairwise Tanimoto similarity between the scaffold-split train and
test sets to check how structurally distinct the test set is from
training data (a key sanity check for scaffold-based splitting).
"""
from src.data_processing import load_and_clean_data, deduplicate_by_mean, label_by_threshold
from src.features import get_murcko_scaffolds, cluster_by_scaffold, smiles_to_morgan_bitvects, pairwise_tanimoto

# 1. Load, clean, deduplicate, label, scaffold-split
df = load_and_clean_data("data/raw/dpp4.csv")
df = deduplicate_by_mean(df)
df = label_by_threshold(df, threshold=10000)
df['Murcko'] = get_murcko_scaffolds(df['Smiles'].tolist())
train_df, test_df = cluster_by_scaffold(df, scaffold_col='Murcko', min_cluster_size=2)

# 2. Fingerprint each split
train_fps = smiles_to_morgan_bitvects(train_df['Smiles'].tolist())
test_fps = smiles_to_morgan_bitvects(test_df['Smiles'].tolist())
train_ids = train_df['Molecule ChEMBL ID'].tolist()
test_ids = test_df['Molecule ChEMBL ID'].tolist()

# 3. Pairwise Tanimoto similarity, sorted descending (highest overlap first)
tc_combined = pairwise_tanimoto(train_fps, train_ids, test_fps, test_ids)

print(tc_combined.head(20))
tc_combined.to_csv("results/tanimoto_train_test_similarity.csv", index=False)
print(f"Saved {tc_combined.shape[0]} pairwise similarity scores.")

"""
Murcko scaffold-based split pipeline.

Reduces chemical-series bias (vs. a random split) by grouping compounds
by ring scaffold and ensuring structurally related analogs cluster
together rather than leaking across train/test.
"""
from src.data_processing import load_and_clean_data, deduplicate_by_mean, label_by_threshold
from src.features import smiles_to_morgan_fp, get_murcko_scaffolds, cluster_by_scaffold
from src.modeling import train_random_forest, evaluate_model, cross_validate_roc_auc, save_model

# 1. Load, clean, deduplicate, label
df = load_and_clean_data("data/raw/dpp4.csv")
df = deduplicate_by_mean(df)
df = label_by_threshold(df, threshold=10000)

# 2. Compute Murcko scaffolds and split by scaffold cluster
df['Murcko'] = get_murcko_scaffolds(df['Smiles'].tolist())
train_df, test_df = cluster_by_scaffold(df, scaffold_col='Murcko', min_cluster_size=2)

# 3. Featurize each split independently
X_train = smiles_to_morgan_fp(train_df['Smiles'].tolist())
y_train = train_df['label'].values
X_test = smiles_to_morgan_fp(test_df['Smiles'].tolist())
y_test = test_df['label'].values

print("Train label distribution:\n", train_df['label'].value_counts())
print("Test label distribution:\n", test_df['label'].value_counts())

# 4. Train and evaluate
rf_model, rf_grid = train_random_forest(X_train, y_train)
evaluate_model(rf_model, X_test, y_test, label="Random Forest (scaffold split)")
cross_validate_roc_auc(rf_model, X_train, y_train)

# 5. Persist
save_model(rf_model, "models/rf_dpp4_scaffold_split.pkl")

print("Scaffold split training completed!")

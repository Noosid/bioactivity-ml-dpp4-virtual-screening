"""
SMOTE-balanced pipeline.

Uses the scaffold-based train/test split (to avoid leakage), then applies
SMOTE oversampling to the training set only, before training a Random
Forest classifier. Useful when active/inactive classes are imbalanced.
"""
from src.data_processing import load_and_clean_data, deduplicate_by_mean, label_by_threshold
from src.features import smiles_to_morgan_fp, get_murcko_scaffolds, cluster_by_scaffold
from src.modeling import train_with_smote, evaluate_model, save_model

# 1. Load, clean, deduplicate, label
df = load_and_clean_data("data/raw/dpp4.csv")
df = deduplicate_by_mean(df)
df = label_by_threshold(df, threshold=10000)

# 2. Scaffold split (kept identical to the non-SMOTE scaffold pipeline
#    so the two approaches are directly comparable)
df['Murcko'] = get_murcko_scaffolds(df['Smiles'].tolist())
train_df, test_df = cluster_by_scaffold(df, scaffold_col='Murcko', min_cluster_size=2)

X_train = smiles_to_morgan_fp(train_df['Smiles'].tolist())
y_train = train_df['label'].values
X_test = smiles_to_morgan_fp(test_df['Smiles'].tolist())
y_test = test_df['label'].values

print("Train label distribution before SMOTE:\n", train_df['label'].value_counts())

# 3. SMOTE + Random Forest
rf_model, rf_grid = train_with_smote(X_train, y_train)
evaluate_model(rf_model, X_test, y_test, label="Random Forest (SMOTE-balanced, scaffold split)")

# 4. Persist
save_model(rf_model, "models/rf_dpp4_smote.pkl")

print("SMOTE-balanced training completed!")

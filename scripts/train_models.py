"""
Baseline pipeline: random train/test split.

Loads cleaned DPP-4 data, deduplicates by mean potency, labels compounds
active/inactive, generates Morgan fingerprints, and trains + compares
Random Forest, SVM, Naive Bayes, and KNN classifiers on a random split.
"""
from sklearn.model_selection import train_test_split

from src.data_processing import load_and_clean_data, deduplicate_by_mean, label_by_threshold
from src.features import smiles_to_morgan_fp
from src.modeling import (
    train_random_forest,
    train_baseline_classifiers,
    evaluate_model,
    cross_validate_roc_auc,
    save_model,
)

# 1. Load, clean, and deduplicate
df = load_and_clean_data("data/raw/dpp4.csv")
df = deduplicate_by_mean(df)
df = label_by_threshold(df, threshold=10000)

# 2. Featurize
X = smiles_to_morgan_fp(df['Smiles'].tolist())
y = df['label'].values

# 3. Random split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train Random Forest (primary model) + baselines
rf_model, rf_grid = train_random_forest(X_train, y_train)
evaluate_model(rf_model, X_test, y_test, label="Random Forest (random split)")
cross_validate_roc_auc(rf_model, X_train, y_train)

baselines = train_baseline_classifiers(X_train, y_train)
for name, clf in baselines.items():
    evaluate_model(clf, X_test, y_test, label=name)

# 5. Persist the primary model
save_model(rf_model, "models/rf_dpp4_random_split.pkl")

print("Random split training completed!")

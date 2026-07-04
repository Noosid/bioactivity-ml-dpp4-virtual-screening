## Bioactivity Predictions and Virtual Screening Using Machine Learning

Machine learning pipeline for predicting compound bioactivity (DPP-4 inhibitors) and performing virtual screening.

**Key Components:**
- Data preprocessing from ChEMBL (cleaning, deduplication by mean potency, SMILES standardization, salt stripping)
- Murcko scaffold generation and scaffold-based clustering (chemical bias reduction)
- Morgan fingerprints
- Tanimoto similarity analysis between train/test splits
- Random Forest, SVM, Naive Bayes, and KNN classifiers
- Random split, scaffold split, and SMOTE-balanced modeling pipelines
- Virtual screening on external libraries (e.g. ZINC, FDA-approved drugs)

**Project layout:**
```
src/
  data_processing.py       # loading, cleaning, deduplication, standardization
  features.py               # Morgan fingerprints, Murcko scaffolds, Tanimoto similarity
  modeling.py                # classifiers, cross-validation, SMOTE, evaluation
scripts/
  train_models.py                # baseline: random train/test split
  train_scaffold_split.py        # Murcko scaffold-based train/test split
  train_smote_model.py           # SMOTE-balanced scaffold-split modeling
  tanimoto_analysis.py           # train/test structural similarity check
  screen_library.py              # apply a trained model to an external library
```

**Technologies:** RDKit, scikit-learn, imbalanced-learn, pandas, numpy

**Data setup:**
This repo ships architecture and code only — no ChEMBL/ZINC-derived data files are bundled, since those are subject to their own source terms. To run the pipelines, place the following in `data/raw/`:
```
data/raw/
  dpp4.csv    # ChEMBL bioactivity export used for TRAINING (columns: Molecule ChEMBL ID, Smiles,
              # Standard Type, Standard Value, Standard Units, Document Year, Target Name)
  fda.csv     # ZINC-curated FDA-approved drug set used ONLY for screening a trained model
              # (columns: zinc_id, smiles) -- never used for training or labeling
```
Trained models are written to `models/`, and screening/analysis outputs (Tanimoto tables, repurposing hits) are written to `results/`.

**Methodology note on scaffold clustering:**
The original workflow clustered Murcko scaffolds using an external tool (`mayachemtools`' `RDKitClusterMolecules.py`), which groups scaffolds by structural similarity above a threshold. The `cluster_by_scaffold` function in `src/features.py` is a lightweight, dependency-free stand-in: it groups compounds by **exact** Murcko scaffold SMILES and assigns any scaffold occurring fewer than `min_cluster_size` times to the test set. This captures the same intent (keep structurally related analogs together, avoid train/test leakage) but is **not** a drop-in equivalent of the original similarity-based clustering algorithm -- results from the two approaches will differ, and this distinction should be noted if referencing the original clustering methodology in the article.

**Note:** Code adapted from a research workflow. Original snippets credited to Uresearcher.com (copyright protected).

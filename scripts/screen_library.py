"""
Virtual screening / drug repurposing.

The model here is always trained on DPP-4 ChEMBL bioactivity data
(see train_models.py / train_scaffold_split.py / train_smote_model.py).
This script never trains on the FDA set -- it only applies an already
trained model to it.

Screening library: FDA-approved drugs, sourced from ZINC (fda.csv),
with columns 'zinc_id' and 'smiles'. Used purely for prospective
drug-repurposing screening, not for training or labeling.
"""
import pandas as pd

from src.features import smiles_to_morgan_fp
from src.modeling import load_model

MODEL_PATH = "models/rf_dpp4_scaffold_split.pkl"   # trained on DPP-4 data only
SCREENING_LIBRARY_PATH = "data/raw/fda.csv"          # FDA-approved drugs (ZINC), columns: zinc_id, smiles
PROBABILITY_THRESHOLD = 0.8
OUTPUT_PATH = "results/fda_repurposing_hits.csv"

# 1. Load trained model (trained exclusively on DPP-4 data)
model = load_model(MODEL_PATH)

# 2. Load FDA/ZINC screening library and featurize
library = pd.read_csv(SCREENING_LIBRARY_PATH)
library = library[["zinc_id", "smiles"]]
fps = smiles_to_morgan_fp(library['smiles'].tolist())

# 3. Predict class + probability of DPP-4 activity for each FDA drug
predictions = model.predict(fps)
probabilities = model.predict_proba(fps)[:, 1]

library['predicted'] = predictions
library['prob1'] = probabilities

# 4. Filter to high-confidence repurposing candidates
hits = library[library['prob1'] >= PROBABILITY_THRESHOLD]
hits.to_csv(OUTPUT_PATH, index=False)

print(f"Screened {library.shape[0]} FDA-approved drugs against the DPP-4 model.")
print(f"Found {len(hits)} potential repurposing candidates "
      f"(probability >= {PROBABILITY_THRESHOLD}), saved to {OUTPUT_PATH}.")

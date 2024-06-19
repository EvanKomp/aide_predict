# tests/test_zero_shot_models.py
'''
* Author: Evan Komp
* Created: 6/18/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

This file exists to test wrapped zero shot models against literature values.

Tests here:
- ESM zero shot as a DMS predictor for small ProteinGym assay: ENVZ_ECOLI_Ghose_2023
   Expected Spearman about 0.2
'''
import os
import pytest

import pandas as pd
from scipy.stats import spearmanr

import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

def test_esm_zero_shot():
    # this model requires no MSA
    from aide_predict.bespoke_models.esm import ESMPredictorWrapper

    model = ESMPredictorWrapper(
        model_checkpoint="esm2_t6_8M_UR50D",
        marginal_method="masked_marginal",
        positions=None,
        device=DEVICE,
        pool=True,
        wt="LADDRTLLMAGVSHDLRTPLTRIRLATEMMSEQDGYLAESINKDIEECNAIIEQFIDYLR",
        metadata_folder='./tmp/esm',
    )

    assay_data = pd.read_csv(
        os.path.join('tests', 'data', 'ENVZ_ECOLI_Ghose_2023.csv'))
    sequences = assay_data['mutated_sequence'].tolist()
    scores = assay_data['DMS_score'].tolist()

    model.fit(sequences) # does nothing
    predictions = model.predict(sequences)
    spearman = spearmanr(scores, predictions)[0]
    print(f"ESM Spearman: {spearman}")
    assert abs(spearman - 0.2) < 0.05

if __name__ == "__main__":
    test_esm_zero_shot()

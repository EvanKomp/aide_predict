# tests/test_bespoke_models/test_predictors/test_esm2_likelihood.py
'''
* Author: Evan Komp
* Created: 6/26/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

This file exists to test wrapped zero shot models against literature values.

Tests here:
- VESPA zero shot as a DMS predictor for small ProteinGym assay: ENVZ_ECOLI_Ghose_2023
   Expected Spearman about 0.135
'''
import os
import pytest

import pandas as pd
from scipy.stats import spearmanr

from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence

import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

@pytest.mark.optional
def test_vespa_zero_shot():
    # this model requires no MSAs

    from aide_predict.bespoke_models.predictors.vespa import VESPAWrapper

    wt_sequence = "LADDRTLLMAGVSHDLRTPLTRIRLATEMMSEQDGYLAESINKDIEECNAIIEQFIDYLR"
    wt = ProteinSequence(wt_sequence, id="ENVZ_ECOLI")

    model = VESPAWrapper(
        wt=wt,
        light=True,  # Using VESPAl
        metadata_folder='./tmp/vespa',
    )

    assay_data = pd.read_csv(
        os.path.join('tests', 'data', 'ENVZ_ECOLI_Ghose_2023.csv'))
    sequences = ProteinSequences.from_list(assay_data['mutated_sequence'].tolist())
    scores = assay_data['DMS_score'].tolist()

    model.fit(sequences)  # This should initialize the VESPA predictor
    print('VESPA model fitted!')
    predictions = model.predict(sequences)
    spearman = spearmanr(scores, predictions)[0]
    print(f"VESPA Spearman: {spearman}")
    assert abs(spearman - 0.135) < 0.03  

    # Repeat for non-light (full VESPA) model
    model = VESPAWrapper(
        wt=wt,
        light=False,
        metadata_folder='./tmp/vespa',
    )
    model.fit(sequences)
    predictions = model.predict(sequences)
    spearman = spearmanr(scores, predictions)[0]
    print(f"Full VESPA Spearman: {spearman}")
    assert abs(spearman - 0.135) < 0.03

if __name__ == "__main__":
    test_vespa_zero_shot()
# tests/test_not_base_models/test_eve.py
'''
* Author: Evan Komp
* Created: 10/28/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os
import sys
import pytest
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from aide_predict.utils.data_structures import ProteinSequences, ProteinSequencesOnFile, ProteinSequence
from aide_predict.bespoke_models.predictors.eve import EVEWrapper

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

@pytest.mark.skipif(
    os.environ.get('EVE_CONDA_ENV') is None or os.environ.get('EVE_REPO') is None,
    reason="EVE environment variables not set"
)
def test_eve_zero_shot():
    """
    Test EVE's performance on the ENVZ_ECOLI benchmark dataset from ProteinGym.
    
    This test:
    1. Uses a small MSA and minimal training steps for quick testing
    2. Verifies the correlation is in the expected range
    3. Checks basic model functionality
    """
    # Load the benchmark data
    assay_data = pd.read_csv(
        os.path.join('tests', 'data', 'ENVZ_ECOLI_Ghose_2023.csv'))
    sequences = ProteinSequences.from_list(assay_data['mutated_sequence'].tolist())
    scores = assay_data['DMS_score'].tolist()

    # Define wild type sequence
    wt_sequence = "LADDRTLLMAGVSHDLRTPLTRIRLATEMMSEQDGYLAESINKDIEECNAIIEQFIDYLR"
    wt = ProteinSequence(wt_sequence, id='ENVZ_ECOLI/1-60')

    # Load MSA
    msa_file = os.path.join('tests', 'data', 'ENVZ_ECOLI_extreme_filtered.a2m')
    msa = ProteinSequencesOnFile.from_fasta(msa_file)

    # Initialize EVE with minimal training parameters for testing
    model = EVEWrapper(
        metadata_folder='./tmp/eve',
        wt=wt,
        # Reduce training time for testing
        # take default values
        training_steps=30000
    )

    print('Fitting EVE model...')
    model.fit(msa)
    print('EVE model fitted!')

    # ensure briefly that the model is capable of handling multiple mutations
    test_sequences = ProteinSequences([
        wt.upper()._mutate(10, 'C')._mutate(11, 'A'),
        wt.upper()._mutate(10, 'C')._mutate(11, 'C')
    ])
    _ = model.predict(test_sequences)
        
    print('Making predictions...')
    predictions = model.predict(sequences)
    spearman = spearmanr(scores, predictions, nan_policy='omit')[0]
    print(f"EVE Spearman correlation: {spearman}")

    # The correlation should be in a reasonable range
    # Note: This is a minimal model, so we expect lower performance than the full model
    assert not np.isnan(spearman), "Correlation should not be NaN"
    assert spearman > -1 and spearman < 1, "Correlation should be between -1 and 1"
    assert abs(spearman - 0.03) < 0.18, "Correlation should be in expected range"

if __name__ == '__main__':
    test_eve_zero_shot()
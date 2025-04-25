# tests/test_not_base_models/test_ssemb.py
'''
* Author: Evan Komp
* Created: 4/2/2025
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
from aide_predict.utils.data_structures.structures import ProteinStructure
from aide_predict.bespoke_models.predictors.ssemb import SSEmbWrapper

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

@pytest.mark.skipif(
    os.environ.get('SSEMB_CONDA_ENV') is None or os.environ.get('SSEMB_REPO') is None,
    reason="SSEmb environment variables not set"
)
def test_ssemb_zero_shot():
    """
    Test SSEmb's performance on the ENVZ_ECOLI benchmark dataset from ProteinGym.
    
    This test:
    1. Uses a small MSA and protein structure for quick testing
    2. Verifies the correlation is in the expected range
    3. Checks basic model functionality
    """
    # Load the benchmark data
    assay_data = pd.read_csv(
        os.path.join('tests', 'data', 'ENVZ_ECOLI_Ghose_2023.csv'))
    sequences = ProteinSequences.from_list(assay_data['mutated_sequence'].tolist())
    scores = assay_data['DMS_score'].tolist()

    # Define wild type sequence with structure
    pdb_path = os.path.join('tests', 'data', 'ENVZ_ECOLI.pdb')
    structure = ProteinStructure(pdb_file=pdb_path)

    # Load MSA
    msa_file = os.path.join('tests', 'data', 'ENVZ_ECOLI_extreme_filtered.a2m')
    wt = ProteinSequence.from_fasta(msa_file).upper()
    wt.msa = wt.msa.upper()
    wt.structure = structure

    # Initialize SSEmb model
    model = SSEmbWrapper(
        metadata_folder='./tmp/ssemb',
        wt=wt,
        gpu_id=0  # Use first GPU if available, otherwise CPU will be used
    )

    print('Fitting SSEmb model...')
    model.fit()
    print('SSEmb model fitted!')

    # ensure briefly that the model is capable of handling multiple mutations
    test_sequences = ProteinSequences([
        wt.upper()._mutate(10, 'C')._mutate(11, 'A'),
        wt.upper()._mutate(10, 'C')._mutate(11, 'C')
    ])
    multi_predictions = model.predict(test_sequences)
    print(f"Multiple mutation test predictions: {multi_predictions}")
    assert len(multi_predictions) == 2, "Should have predictions for both test sequences"
        
    print('Making predictions...')
    predictions = model.predict(sequences)
    spearman = spearmanr(scores, predictions, nan_policy='omit')[0]
    print(f"SSEmb Spearman correlation: {spearman}")

    # The correlation should be in a reasonable range
    assert not np.isnan(spearman), "Correlation should not be NaN"
    assert spearman > -1 and spearman < 1, "Correlation should be between -1 and 1"
    # Adjust expected correlation range based on SSEmb performance on this dataset
    assert abs(spearman - 0.05) < 0.2, "Correlation should be in expected range"

if __name__ == '__main__':
    test_ssemb_zero_shot()
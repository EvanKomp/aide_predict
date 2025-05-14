# tests/test_bespoke_models/test_predictors/test_evmutation.py
'''
* Author: Evan Komp
* Created: 7/12/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os
import pytest
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from aide_predict.utils.data_structures import ProteinSequences, ProteinSequencesOnFile, ProteinSequence
from aide_predict.bespoke_models.predictors.evmutation import EVMutationWrapper

def test_evcouplings_zero_shot():
    # Load the data
    assay_data = pd.read_csv(
        os.path.join('tests', 'data', 'ENVZ_ECOLI_Ghose_2023.csv'))
    sequences = ProteinSequences.from_list(assay_data['mutated_sequence'].tolist())
    scores = assay_data['DMS_score'].tolist()

    # Create a small MSA for testing (in practice, you'd use a real MSA)
    msa_file = os.path.join('tests', 'data', 'ENVZ_ECOLI_extreme_filtered.a2m')
    wt = ProteinSequence.from_fasta(msa_file)

    # Test with standard protocol
    model = EVMutationWrapper(
        metadata_folder='./tmp/evcouplings',
        wt=wt,
        protocol="standard",
        theta=0.8,
        iterations=100,
    )

    model.fit()
    print('EVCouplings model fitted!')
    predictions = model.predict(sequences)
    spearman = spearmanr(scores, predictions)[0]
    print(f"EVCouplings Spearman (standard): {spearman}")
    assert abs(spearman - 0.1) < 0.05 

if __name__ == "__main__":
    test_evcouplings_zero_shot()
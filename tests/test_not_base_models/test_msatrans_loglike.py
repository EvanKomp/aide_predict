# tests/test_not_base_models/test_msatrans_loglike.py
'''
* Author: Evan Komp
* Created: 7/8/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os
import pytest

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from aide_predict.utils.data_structures import ProteinSequencesOnFile, ProteinSequences

import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


@pytest.mark.optional
def test_msa_transformer_zero_shot():
    from aide_predict.bespoke_models.predictors.msa_transformer import MSATransformerLikelihoodWrapper

    # Load the MSA
    msa_file = os.path.join('tests', 'data', 'ENVZ_ECOLI_extreme_filtered.a2m')
    msa = ProteinSequencesOnFile.from_fasta(msa_file)

    # Load the assay data
    assay_data = pd.read_csv(os.path.join('tests', 'data', 'ENVZ_ECOLI_Ghose_2023.csv'))
    sequences = ProteinSequences.from_list(assay_data['mutated_sequence'].tolist())
    scores = assay_data['DMS_score'].tolist()

    wt_sequence = "LADDRTLLMAGVSHDLRTPLTRIRLATEMMSEQDGYLAESINKDIEECNAIIEQFIDYLR"

    # Test wt marginal method
    model = MSATransformerLikelihoodWrapper(
        marginal_method="wildtype_marginal",
        positions=None,
        device=DEVICE,
        pool=True,
        wt=wt_sequence,
        batch_size=128,
        metadata_folder='./tmp/msa_transformer',
    )

    model.fit(msa)
    print('wt marginal method fitted')
    predictions = model.predict(sequences)
    spearman = spearmanr(scores, predictions)[0]
    print(f"MSA Transformer (masked marginal) Spearman: {spearman}")
    assert abs(spearman - 0.2) < 0.1  # Adjust the expected correlation as needed

    # Test masked marginal method
    model = MSATransformerLikelihoodWrapper(
        marginal_method="masked_marginal",
        positions=None,
        device=DEVICE,
        pool=True,
        wt=wt_sequence,
        batch_size=128,
        metadata_folder='./tmp/msa_transformer',
    )

    model.fit(msa)
    print('masked marginal method fitted')
    predictions = model.predict(sequences)
    spearman = spearmanr(scores, predictions)[0]
    print(f"MSA Transformer (wildtype marginal) Spearman: {spearman}")
    assert abs(spearman - 0.2) < 0.1  # Adjust the expected correlation as needed

    # Test mutant marginal method
    # THIS ONE TAKES A LONG TIME, 1k calls
    model = MSATransformerLikelihoodWrapper(
        marginal_method="mutant_marginal",
        positions=None,
        device=DEVICE,
        pool=True,
        wt=wt_sequence,
        batch_size=128,
        metadata_folder='./tmp/msa_transformer',
    )

    model.fit(msa)
    print('mutant marginal method fitted')
    predictions = model.predict(sequences)
    spearman = spearmanr(scores, predictions)[0]
    print(f"MSA Transformer (wildtype marginal) Spearman: {spearman}")
    assert abs(spearman - 0.2) < 0.1  # Adjust the expected correlation as needed

    # Test with specific positions and no pooling
    model = MSATransformerLikelihoodWrapper(
        marginal_method="wildtype_marginal",
        positions=[8, 9, 10],
        device=DEVICE,
        pool=False,
        wt=wt_sequence,
        batch_size=128,
        metadata_folder='./tmp/msa_transformer',
    )

    model.fit(msa)
    print('mutant marginal model fitted')
    predictions = model.predict(sequences)
    assert len(predictions) == len(sequences)
    assert predictions.shape[1] == 3

if __name__ == "__main__":
    test_msa_transformer_zero_shot()
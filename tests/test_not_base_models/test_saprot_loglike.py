# tests/test_not_base_models/test_saprot_loglike.py
'''
* Author: Evan Komp
* Created: 7/16/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os
import pytest

import pandas as pd
from scipy.stats import spearmanr

from aide_predict.utils.data_structures import ProteinSequences, ProteinStructure, ProteinSequence

import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

@pytest.mark.optional
def test_saprot_zero_shot():
    from aide_predict.bespoke_models.predictors.saprot import SaProtLikelihoodWrapper

    # Load the structure
    structure = ProteinStructure('tests/data/ENVZ_ECOLI.pdb')

    wt_sequence = "LADDRTLLMAGVSHDLRTPLTRIRLATEMMSEQDGYLAESINKDIEECNAIIEQFIDYLR"
    wt = ProteinSequences([ProteinSequence(wt_sequence, structure=structure)])[0]

    assay_data = pd.read_csv(os.path.join('tests', 'data', 'ENVZ_ECOLI_Ghose_2023.csv'))
    sequences = ProteinSequences.from_list(assay_data['mutated_sequence'].tolist())
    scores = assay_data['DMS_score'].tolist()

    # Repeat for wildtype marginal
    model = SaProtLikelihoodWrapper(
        model_checkpoint="westlake-repl/SaProt_650M_AF2",
        marginal_method="wildtype_marginal",
        positions=None,
        device=DEVICE,
        pool=True,
        wt=wt,
        metadata_folder='./tmp/saprot',
        foldseek_path='foldseek'
    )
    model.fit(sequences)  # does nothing
    predictions = model.predict(sequences)
    spearman = spearmanr(scores, predictions)[0]
    print(f"SaProt Spearman (wildtype_marginal): {spearman}")
    assert abs(spearman - 0.15) < 0.05  # Adjust this threshold as needed

    model = SaProtLikelihoodWrapper(
        model_checkpoint="westlake-repl/SaProt_650M_AF2",
        marginal_method="masked_marginal",
        positions=None,
        device=DEVICE,
        pool=True,
        wt=wt,
        metadata_folder='./tmp/saprot',
        foldseek_path='foldseek'  # Adjust this path if necessary
    )

    model.fit(sequences)  # does nothing
    print('SaProt model fitted!')
    predictions = model.predict(sequences)
    spearman = spearmanr(scores, predictions)[0]
    print(f"SaProt Spearman (masked_marginal): {spearman}")
    assert abs(spearman - 0.15) < 0.05  # Adjust this threshold as needed

    # Run with positions specified and get position-specific scores
    model = SaProtLikelihoodWrapper(
        model_checkpoint="westlake-repl/SaProt_650M_AF2",
        marginal_method="wildtype_marginal",
        positions=[8, 9, 10],
        device=DEVICE,
        pool=False,
        wt=wt,
        metadata_folder='./tmp/saprot',
        foldseek_path='foldseek'
    )
    model.fit(sequences)  # does nothing
    predictions = model.predict(sequences)
    assert len(predictions) == len(sequences)
    assert len(predictions[0]) == 3

    # Repeat for mutant marginal
    model = SaProtLikelihoodWrapper(
        model_checkpoint="westlake-repl/SaProt_650M_AF2",
        marginal_method="mutant_marginal",
        positions=None,
        device=DEVICE,
        pool=True,
        wt=wt,
        metadata_folder='./tmp/saprot',
        foldseek_path='foldseek'
    )
    model.fit(sequences)  # does nothing
    predictions = model.predict(sequences)
    spearman = spearmanr(scores, predictions)[0]
    print(f"SaProt Spearman (mutant_marginal): {spearman}")
    assert abs(spearman - 0.15) < 0.05  # Adjust this threshold as needed

if __name__ == "__main__":
    test_saprot_zero_shot()
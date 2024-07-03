# tests/test_bespoke_models/test_predictors/test_hmm.py
'''
* Author: Evan Komp
* Created: 6/26/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os
import pytest
import pandas as pd

from aide_predict.bespoke_models.predictors.hmm import HMMWrapper
from aide_predict.utils.data_structures import ProteinSequences

from sklearn.metrics import roc_auc_score

# def test_hmm():
#     """Score HMM 17 on p740 data, should get 0.52 auroc"""

#     sequences = ProteinSequences.from_fasta(os.path.join('tests', 'data', 'hmm-17.fa'))

#     test_data = pd.read_csv(os.path.join('tests', 'data', 'p740_labels.csv'))
#     labels = test_data['target'].values > 1e-5 # testing for nonzero activity
#     seq_dict = test_data.set_index('id')['sequence'].to_dict()
#     test_sequences = ProteinSequences.from_dict(seq_dict)

#     model = HMMWrapper(metadata_folder='./tmp/hmm', threshold=0.0)
#     model.fit(sequences)
#     predictions = model.predict(test_sequences)

#     auroc = roc_auc_score(labels, predictions)
#     assert abs(0.52 - auroc) < 0.03

# if __name__ == "__main__":
#     test_hmm()
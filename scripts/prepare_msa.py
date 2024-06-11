# scripts/prepare_msa.py
'''
* Author: Evan Komp
* Created: 5/9/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology

Cleanup and compute MSA weights from MSA built on WT protein.
This is handled by the MSAProcessing class, Adapted from the Mark's lab (see aide_predict/utils/msa.py for references)

Note that we can also add additional weights manually via IDs. 

'''
import os
import pickle
import dvc.api

from aide_predict.utils.msa import MSAProcessing, MSAProcessingArgs, convert_sto_a2m

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filemode='w', filename='./logs/prepare_msa.log')

PARAMS = dvc.api.params_show()['msaprocessing']

def main():
    # first prepare directories
    EXECDIR = os.getcwd()
    if not os.path.exists(os.path.join(EXECDIR, 'data', 'msa')):
        os.makedirs(os.path.join(EXECDIR, 'data', 'msa'))

    # prepare the arguments
    args = MSAProcessingArgs(
        theta=PARAMS['theta'],
        use_weights=PARAMS['use_weights'],
        preprocess_MSA=PARAMS['preprocess'],
        threshold_focus_cols_frac_gaps=PARAMS['threshold_focus_cols_frac_gaps'],
        threshold_sequence_frac_gaps=PARAMS['threshold_sequence_frac_gaps'],
        remove_sequences_with_indeterminate_AA_in_focus_cols=PARAMS['remove_sequences_with_indeterminate_AA_in_focus_cols'],
    )
    msa = MSAProcessing(args)

    # get the wild type sequence ID
    with open(os.path.join(EXECDIR, 'data', 'wt.fasta'), 'r') as f:
        lines = f.readlines()
        wt_id = lines[0].split()[1].strip()
    logger.info(f'wt_id: {wt_id}')

    # process the MSA
    msa.process(
        MSA_location=os.path.join(EXECDIR, 'data', 'jackhmmer', 'jackhmmer.a2m'),
        weights_location=os.path.join(EXECDIR, 'data', 'msa', 'raw_weights.npy'),
        focus_seq_id=wt_id,
        additional_weights=None,
        new_a2m_location=os.path.join(EXECDIR, 'data', 'msa', 'msa_clean.a2m'),
    )

    # pickle the MSAProcessing object for loading by downstream processes
    with open(os.path.join(EXECDIR, 'data', 'msa', 'msa_obj.pkl'), 'wb') as f:
        pickle.dump(msa, f)
    

if __name__ == '__main__':
    main()
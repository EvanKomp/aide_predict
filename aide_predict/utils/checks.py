# aide_predict/utils/checks.py
'''
* Author: Evan Komp
* Created: 6/13/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Common checks to ensure that different pipeline components are compatable.
'''
import os
import dvc.api

def check_dvc_params():
    """Ensures the user provided the necessary data and did not ask for
    incompatible steps.

    Returns pre-run metrics about the pipeline.


    Checks done:
    - If the user asks for a model that requires an MSA, ensure the MSA step is on.
    - If the user is using specifying specific positions to score, ensure that any
      sequences to be evaluated are fixed length and all models are capable of position specific scoring.
      Currently incompatable with supervised models.
    - If the user gives either training or test data, and any have different legnths, ensure that
      models are capable of handling variable length sequences.
    - If the user asks for jackhmmer search, ensure that wt.fasta was provided
    - If the user asks for supervised and or/msa mode that adds training sequences, ensure that the training sequences are provided.
    - If the user asks for CV, ensure training data is provided.
    """
    params = dvc.api.params_show()
    if not 'data' in os.listdir() and 'dvc.yaml' in os.listdir():
        raise FileNotFoundError("This directory does not appear to be the aide_predict dvc pipeline. `check_dvc_params` must be run from the root of the dvc pipeline.")


def check_input_sequences_against_model_capability():
    # we need to catch if the user passed non fixed length sequences and the model is not capable of that
    # or if it is capable of not fixed length AS WELL AS position specific scoring, how can both be recitifed?
    # If the user passes an alignment and positions, that might work.     
    
    pass
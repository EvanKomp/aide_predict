# aide_predict/bespoke_models/predictors/evmutation.py
'''
* Author: Evan Komp
* Created: 7/12/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Wrapper around EVmutation model from the EVCouplings repositorty:
https://github.com/debbiemarkslab/EVcouplings/tree/develop

Hopf T. A., Green A. G., Schubert B., et al. The EVcouplings Python framework for coevolutionary sequence analysis. Bioinformatics 35, 1582–1584 (2019)
'''
import os
import warnings
import shutil
from typing import List, Union, Optional
import numpy as np
import pandas as pd
from evcouplings.couplings.model import CouplingsModel
from evcouplings.couplings import protocol
from evcouplings.align.protocol import describe_frequencies, Alignment, parse_header
from aide_predict.bespoke_models.base import ProteinModelWrapper, RequiresMSAMixin, CanRegressMixin, RequiresWTToFunctionMixin, RequiresFixedLengthMixin, MessageBool
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence

from tqdm import tqdm

# check for plmc
if shutil.which("plmc") is None:
    AVAILABLE = MessageBool(False, "The 'plmc' executable is required for EVCouplings.")
else:
    AVAILABLE = MessageBool(True, "EVMutation is available.")
    plmc = shutil.which("plmc")


class EVMutationWrapper(
    RequiresWTToFunctionMixin, 
    RequiresFixedLengthMixin,
    RequiresMSAMixin,
    CanRegressMixin,
    ProteinModelWrapper):
    """
    A wrapper for EVCouplings that implements the ProteinModelWrapper interface.
    """
    _available = AVAILABLE

    def __init__(self, metadata_folder: str, 
                 wt: Optional[Union[str, ProteinSequence]] = None,
                 protocol: str = "standard",
                 theta: float = 0.8,
                 iterations: int = 100,
                 lambda_h: float = 1e-2,
                 lambda_J: float = 1e-2,
                 lambda_group: float = None,
                 min_sequence_distance: int = 6,
                 cpu: int = 1):
        """
        Initialize the EVCouplingsWrapper.

        Args:
            metadata_folder (str): Folder to store metadata and intermediate files.
            wt (Optional[Union[str, ProteinSequence]]): Wild-type sequence.
            protocol (str): EVCouplings protocol to use ("standard", "complex", or "mean_field").
            theta (float): Sequence clustering threshold.
            iterations (int): Number of iterations for inference.
            lambda_h (float): Regularization strength on fields.
            lambda_J (float): Regularization strength on couplings.
            lambda_group (float): Group regularization strength.
            cpu (int): Number of CPUs to use.
        """
        super().__init__(metadata_folder=metadata_folder, wt=wt)
        self.protocol = protocol
        self.theta = theta
        self.iterations = iterations
        self.lambda_h = lambda_h
        self.lambda_J = lambda_J
        self.lambda_group = lambda_group
        self.cpu = cpu
        self.min_sequence_distance = min_sequence_distance

        if not self.wt.id:
            raise ValueError("The wild-type sequence must have an ID, of expected format 'ID/START-END'")
        else:
            id_, start_region, end_region = parse_header(self.wt.id)
            self._start = start_region
            self._end = end_region

    def check_metadata(self) -> None:
        # ensure metadata folder is empty, otherwise delete whats inside
        if os.path.exists(self.metadata_folder):
            shutil.rmtree(self.metadata_folder)

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'EVCouplingsWrapper':
        """
        Fit the EVCouplings model to the given MSA.

        Args:
            X (ProteinSequences): The input MSA.
            y (Optional[np.ndarray]): Ignored. Present for API consistency.

        Returns:
            EVCouplingsWrapper: The fitted model.
        """
        if not X.width == len(self.wt):
            raise ValueError("The sequences in the MSA must all have the same length as the wild-type sequence")
        if not str(X[0]).upper() == str(self.wt).upper():
            raise ValueError("The wild-type sequence must be present in the MSA as the first sequence.")
        if not self.wt.id == X[0].id:
            raise ValueError("The wild-type sequence must have the same ID as the first sequence in the MSA.")

        # Prepare the alignment file
        alignment_file = os.path.join(self.metadata_folder, "alignment.a2m")
        X.to_fasta(alignment_file)

        # need to define frequency file, expected by EVCouplings
        with open(alignment_file, "r") as f:
            ali = Alignment.from_file(f, format='fasta')
            freqs = describe_frequencies(ali, self._start, 0)
            freqs.to_csv(os.path.join(self.metadata_folder, "frequencies.csv"))

        # Prepare the configuration for EVCouplings
        config = {
            "alignment_file": alignment_file,
            "prefix": os.path.join(self.metadata_folder, "evc_"),
            "theta": self.theta,
            "iterations": self.iterations,
            "lambda_h": self.lambda_h,
            "lambda_J": self.lambda_J,
            "lambda_group": self.lambda_group,
            "cpu": None,
            "protocol": self.protocol,
            "focus_mode": False,
            "focus_sequence": self.wt.id,
            "segments": None,
            "min_sequence_distance": self.min_sequence_distance,
            "frequencies_file": os.path.join(self.metadata_folder, "frequencies.csv"),
            "alphabet": None,
            "ignore_gaps": True,
            "scale_clusters": None,
            "plmc": plmc,
            "reuse_ecs": True,
            "lambda_J_times_Lq": True,
            "scoring_model": "logistic_regression"
        }

        # Run EVCouplings
        outcfg = protocol.run(**config)

        # Load the resulting model
        self.model_ = CouplingsModel(outcfg["model_file"])

        return self

    def _transform(self, X: ProteinSequences) -> np.ndarray:
        """
        Predict the effect of mutations using the fitted EVCouplings model.

        Args:
            X (ProteinSequences): The input sequences to predict.

        Returns:
            np.ndarray: An array of predicted effects for each input sequence.
        """
        predictions = []
        for seq in tqdm(X):
            mutations = self.wt.get_mutations(seq)
            evc_mutations = [(int(m[1:-1]), m[0], m[-1]) for m in mutations]
            # remove mutations that evc does not haves scores for
            new_mutations = []
            for i, (subs_pos, subs_from, subs_to) in enumerate(evc_mutations):
                if subs_pos not in self.model_.index_map or subs_to not in self.model_.alphabet_map:
                    warnings.warn(f"Mutation {subs_from}{subs_pos}{subs_to} is not supported by EVCouplings statistics and will be ignored for scoring.")
                    continue
                new_mutations.append((subs_pos, subs_from, subs_to))
                
            delta_h, _, _ = self.model_.delta_hamiltonian(new_mutations)
            predictions.append(delta_h)
        return np.array(predictions).reshape(-1, 1)

    def _partial_fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> None:
        """
        EVCouplings does not support partial fitting.
        """
        raise NotImplementedError("EVCouplings does not support partial fitting.")

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.

        Args:
            input_features (Optional[List[str]]): Ignored. Present for API consistency.

        Returns:
            List[str]: A list containing a single feature name.
        """
        return ["EVCouplings_delta_hamiltonian"]
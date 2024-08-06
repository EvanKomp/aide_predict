# aide_predict/bespoke_models/predictors/vespa.py
'''
* Author: Evan Komp
* Created: 8/1/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Wrapper of VESPA:
Marquet, C. et al. Embeddings from protein language models predict conservation and variant effects. Hum Genet 141, 1629–1647 (2022).

This model embeds the sequences with a PLM, then uses the embeddings for a pretrained logistic regression model for conservation. These
are input into a model to predict single mutation effects.
'''
import os
from typing import Optional, Union, List
import numpy as np
import pandas as pd
import subprocess

from aide_predict.utils.data_structures import ProteinSequence, ProteinSequences
from aide_predict.bespoke_models.base import ProteinModelWrapper, CanRegressMixin, RequiresWTDuringInferenceMixin, RequiresWTToFunctionMixin
from aide_predict.utils.common import MessageBool

try:
    import vespa.predict.config
    from vespa.predict.vespa import VespaPred
    from vespa.predict import utils
    import torch
    AVAILABLE = MessageBool(True, "VESPA is available.")
except ImportError:
    AVAILABLE = MessageBool(False, "VESPA is not available. Please install it with requirements-vespa.txt.")

import logging
logger = logging.getLogger(__name__)


class VESPAWrapper(CanRegressMixin, RequiresWTDuringInferenceMixin, RequiresWTToFunctionMixin, ProteinModelWrapper):
    """
    A wrapper class for the VESPA (Variant Effect Score Prediction using Attention) model.

    This class provides an interface to use VESPA within the AIDE framework,
    allowing for prediction of variant effects on protein sequences.

    Attributes:
        light (bool): If True, uses the lighter VESPAl model. If False, uses the full VESPA model.
    """

    _available = AVAILABLE

    def __init__(self, metadata_folder: Optional[str] = None, 
                 wt: Optional[Union[str, ProteinSequence]] = None, 
                 light: bool = True) -> None:
        """
        Initialize the VESPAWrapper.

        Args:
            metadata_folder (Optional[str]): Folder to store metadata.
            wt (Optional[Union[str, ProteinSequence]]): Wild-type protein sequence.
            light (bool): If True, use the lighter VESPAl model. If False, use the full VESPA model.
        """
        super().__init__(metadata_folder=metadata_folder, wt=wt)
        self.light = light

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'VESPAWrapper':
        """
        Fit the VESPA model. This method is a placeholder as VESPA doesn't require fitting.

        Args:
            X (ProteinSequences): The input protein sequences.
            y (Optional[np.ndarray]): Ignored. Present for API consistency.

        Returns:
            VESPAWrapper: The fitted model (self).
        """
        self.fitted_ = True
        return self

    def _transform(self, X: ProteinSequences) -> np.ndarray:
        """
        Transform the input sequences using the VESPA model.

        This method checks that each input sequence is a single point mutation from the wild type,
        writes the mutations to a file, runs the VESPA model, and processes the results.

        Args:
            X (ProteinSequences): The input protein sequences to transform.

        Returns:
            np.ndarray: The log-transformed VESPA scores for each input sequence.

        Raises:
            ValueError: If any input sequence is not a single point mutation from the wild type,
                        or if VESPA fails to return predictions.
        """
        # Check that each sequence is maximum a single point mutation from the wild type
        for seq in X:
            mutations = self.wt.get_mutations(seq)
            if len(mutations) != 1:
                raise ValueError(f"Sequence {seq} is not a single point mutation from the wild type sequence {self.wt}")

        mutation_file = os.path.join(self.metadata_folder, "mutations.txt")
        wt_fasta_file = os.path.join(self.metadata_folder, "wt.fasta")

        # Write mutations to file
        with open(mutation_file, 'w') as f:
            for seq in X:
                mutation = self.wt.get_mutations(seq)[0]
                fromAA, pos, toAA = mutation[0], mutation[1:-1], mutation[-1]
                f.write(f"{self.wt.id}_{fromAA}{int(pos)-1}{toAA}\n")

        # Create a temporary file for the wild type sequence
        ProteinSequences([self.wt]).to_fasta(wt_fasta_file)

        # Run the model
        cmd = ["vespa", os.path.basename(wt_fasta_file), "-m", os.path.basename(mutation_file), "--prott5_weights_cache", '.']
        if self.light:
            cmd.append("--vespal")
        else:
            cmd.append("--vespa")
        logger.info(f"Running command: {cmd}")
        stdout, stderr = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            cwd=self.metadata_folder
        ).communicate()
        if stderr:
            logger.error(f"VESPA gave: {stderr.decode()}")
        
        results = []
        outpath = os.path.join(self.metadata_folder, "vespa_run_directory", "output", "0.csv")
        if not os.path.exists(outpath):
            raise ValueError("VESPA did not return predictions, check logs.")
        column = "VESPA" if not self.light else "VESPAl"
        df = pd.read_csv(outpath, sep=';').set_index('Mutant')
        for seq in X:
            mutation = self.wt.get_mutations(seq)[0]
            fromAA, pos, toAA = mutation[0], mutation[1:-1], mutation[-1]
            result = df.loc[f"{fromAA}{int(pos)-1}{toAA}"][column]
            results.append(result)

        return np.log(1-np.array(results).reshape(-1, 1))

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get the names of the output features.

        Args:
            input_features (Optional[List[str]]): Ignored. Present for API consistency.

        Returns:
            List[str]: A list containing the name of the output feature.
        """
        return ["VESPA_score"]


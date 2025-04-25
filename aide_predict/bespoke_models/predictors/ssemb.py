# aide_predict/bespoke_models/predictors/ssemb.py
'''
* Author: Evan Komp
* Created: 4/2/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Wrapper for SSEmb:
Blaabjerg, L.M., Jonsson, N., Boomsma, W. et al. SSEmb: A joint embedding of protein sequence and structure enables robust variant effect predictions. 
    Nat Commun 15, 9646 (2024). https://doi.org/10.1038/s41467-024-53982-z
'''
import os
import sys
import json
import subprocess
from typing import Optional, Union, Dict, Any, List
import warnings
import numpy as np
import pandas as pd

from aide_predict.bespoke_models.base import (
    ProteinModelWrapper, 
    RequiresWTToFunctionMixin,
    RequiresStructureMixin,
    RequiresWTMSAMixin,
    CanRegressMixin
)
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence, ProteinStructure
from aide_predict.utils.common import MessageBool
import aide_predict

import logging
logger = logging.getLogger(__name__)

# Check environment setup
SSEMB_ENV = os.environ.get('SSEMB_CONDA_ENV')
SSEMB_REPO = os.environ.get('SSEMB_REPO')

SSEMB_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(aide_predict.__file__)),
    "external_calls",
    "ssemb",
    "_ssemb_score.py"
)

if SSEMB_ENV is None or SSEMB_REPO is None:
    AVAILABLE = MessageBool(False, "SSEmb requires SSEMB_CONDA_ENV and SSEMB_REPO environment variables to be set")
elif not os.path.exists(SSEMB_REPO):
    AVAILABLE = MessageBool(False, f"SSEmb repository directory not found: {SSEMB_REPO}")
elif not os.path.exists(os.path.join(SSEMB_REPO, "weights")):
    AVAILABLE = MessageBool(False, f"SSEmb weights directory not found: {os.path.join(SSEMB_REPO, 'weights')}")
else:
    AVAILABLE = MessageBool(True, "SSEmb model is available")


class SSEmbWrapper(RequiresWTToFunctionMixin, RequiresStructureMixin, RequiresWTMSAMixin, CanRegressMixin, ProteinModelWrapper):
    """
    Wrapper for SSEmb model to predict variant effects on protein stability.
    
    SSEmb combines protein structure and sequence information using a graph neural network
    and MSA Transformer to predict the effects of mutations. This wrapper provides an
    interface to run SSEmb within the AIDE framework.
    
    The SSEmb model requires:
    1. A multiple sequence alignment (MSA) for the protein family
    2. A structure for the wild-type protein
    3. A wild-type sequence reference
    
    The model predicts a score for each variant, where higher scores indicate better predicted stability/function.
    
    Attributes:
        _available (MessageBool): Indicates whether SSEmb is available based on environment setup.
        gpu_id (int): GPU device ID to use for model inference.
        msa_ (ProteinSequences): The multiple sequence alignment used for training.
        fitted_ (bool): Whether the model has been fitted.
    """

    _available = AVAILABLE

    def __init__(
        self,
        metadata_folder: str = None,
        wt: Optional[Union[str, ProteinSequence]] = None,
        gpu_id: int = 0
    ):
        """
        Initialize the SSEmb wrapper.

        Args:
            metadata_folder (str, optional): Folder to store metadata and intermediate files.
            wt (Union[str, ProteinSequence], optional): Wild-type protein sequence.
            gpu_id (int, optional): GPU device ID to use. Defaults to 0.
        """
        super().__init__(metadata_folder=metadata_folder, wt=wt)
        self.gpu_id = gpu_id
        
        # Create necessary directories in metadata folder
        self._io_dir = os.path.join(self.metadata_folder, "ssemb_io")
        self._run_dir = os.path.join(self.metadata_folder, "ssemb_run")
        
        os.makedirs(self._io_dir, exist_ok=True)
        os.makedirs(self._run_dir, exist_ok=True)

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'SSEmbWrapper':
        """
        Fit the SSEmb model to the MSA.

        Args:
            X (ProteinSequences): Input MSA for the protein family.
            y (Optional[np.ndarray]): Ignored. Present for API consistency.

        Returns:
            SSEmbWrapper: The fitted wrapper.

        Raises:
            ValueError: If the first sequence in the MSA doesn't match the wild-type sequence.
        """
        X = self.wt.msa
        
        # we can check that MSA and WT are the same sequence
        if not str(self.wt) == str(X[0]):
            raise ValueError('WT sequence and first MSA sequence expected to be the same')
        
        # Store the MSA for prediction
        self.msa_ = X
        self.fitted_ = True
        
        # Save MSA to file for later use
        msa_path = os.path.join(self._io_dir, "msa.a3m")
        self.msa_.to_fasta(msa_path)
        
        # Save reference to structure
        if isinstance(self.wt.structure, str):
            self._structure_path = os.path.abspath(self.wt.structure)
        elif isinstance(self.wt.structure, ProteinStructure):
            self._structure_path = os.path.abspath(self.wt.structure.pdb_file)
        else:
            raise ValueError(f'Unknown structure data type: {type(self.wt.structure)}')
        
        return self

    def _transform(self, X: ProteinSequences) -> np.ndarray:
        """
        Predict variant effects using SSEmb.

        Args:
            X (ProteinSequences): Input protein variants to score.

        Returns:
            np.ndarray: Array of SSEmb scores for the variants.

        Raises:
            ValueError: If any variant differs from the wild-type by more than one mutation.
            RuntimeError: If SSEmb execution fails.
        """
        # Paths for files
        pdb_path = os.path.join(self._io_dir, "wt.pdb")
        msa_path = os.path.join(self._io_dir, "msa.a3m")
        variants_path = os.path.join(self._io_dir, "variants.txt")
        multi_scores_path = os.path.join(self._run_dir, "ssemb_multi_scores.csv")
        
        # Create symlink to structure if it doesn't exist
        if not os.path.exists(pdb_path):
            if os.path.islink(pdb_path):
                os.unlink(pdb_path)
            os.symlink(self._structure_path, pdb_path)
        
        # Create variants file for multi-mutations
        self._create_variants_file(X, variants_path)
        
        # Run SSEmb
        cmd = [
            "conda", "run",
            "-n", SSEMB_ENV,
            "--no-capture-output",
            "python", SSEMB_SCRIPT,
            "--pdb", pdb_path,
            "--msa", msa_path,
            "--run", self._run_dir,
            "--weights", os.path.join(SSEMB_REPO, "weights"),
            "--variants", variants_path,
            "--gpu-id", str(self.gpu_id)
        ]
        
        logger.info(f"Running SSEmb with command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.debug(f"SSEmb stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"SSEmb stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"SSEmb execution failed: {e.stderr}")
        
        # Check if results file exists
        if not os.path.exists(multi_scores_path):
            raise RuntimeError(f"SSEmb failed to produce multi-mutation scores at {multi_scores_path}")
        
        # Read scores
        scores_df = pd.read_csv(multi_scores_path)
        
        # Extract scores for input variants
        variant_scores = []
        for seq in X:
            mutations = self.wt.get_mutations(seq)
            mutation_string = ";".join(mutations) if mutations else f"{self.wt[0]}{1}{self.wt[0]}"
            
            # Find matching row in scores DataFrame
            match = scores_df[scores_df["variant"] == mutation_string]
            if len(match) == 0:
                raise ValueError(f"Variant with mutations {mutation_string} not found in SSEmb results")
            
            variant_scores.append(match["score_ml"].values[0])
        
        return np.array(variant_scores).reshape(-1, 1)
        

    def _create_variants_file(self, X: ProteinSequences, output_path: str) -> None:
        """
        Create a variants file for SSEmb from the input sequences.
        
        Args:
            X (ProteinSequences): Input protein variants.
            output_path (str): Path to save the variants file.
            
        Raises:
            ValueError: If any variant cannot be properly represented in the SSEmb format.
        """
        with open(output_path, 'w') as f:
            for seq in X:
                mutations = self.wt.get_mutations(seq)
                if not mutations:  # If sequence is identical to wild type
                    # Add a placeholder "mutation" that changes an amino acid to itself
                    # This allows SSEmb to score the wild-type sequence
                    f.write(f"{self.wt[0]}{1}{self.wt[0]}\n")
                else:
                    f.write(';'.join(mutations) + '\n')
        
        logger.debug(f"Created variants file at {output_path}")

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.
        
        Args:
            input_features (Optional[List[str]]): Input feature names (not used).
            
        Returns:
            List[str]: A list containing the name of the output feature.
        """
        return ["SSEmb_score"]
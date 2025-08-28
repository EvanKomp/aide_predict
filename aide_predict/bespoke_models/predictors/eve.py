# aide_predict/bespoke_models/predictors/eve.py
'''
* Author: Evan Komp
* Created: 10/28/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Wrapper for EVE (Evolutionary Variational Autoencoder) model.
Please see original paper and implementation:
https://github.com/OATML/EVE
'''

import os
import sys
import json
import subprocess
import tempfile
from typing import Optional, Union, Dict, Any, List
import warnings
import numpy as np
import pandas as pd
from hashlib import sha256

from aide_predict.bespoke_models.base import (
    ProteinModelWrapper, 
    RequiresWTToFunctionMixin,
    RequiresFixedLengthMixin, 
    RequiresWTMSAMixin,
    RequiresMSAForFitMixin,
    AcceptsLowerCaseMixin,
    RequiresWTDuringInferenceMixin,
    CanRegressMixin
)
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.utils.common import MessageBool
import aide_predict

import logging
logger = logging.getLogger(__name__)

# Check environment setup
EVE_ENV = os.environ.get('EVE_CONDA_ENV')
EVE_REPO = os.environ.get('EVE_REPO')

EVE_SCRIPT_1 = os.path.join(
    os.path.dirname(os.path.dirname(aide_predict.__file__)),
    "external_calls",
    "eve",
    "_train_VAE_one.py"
)
EVE_SCRIPT_2 = os.path.join(
    os.path.dirname(os.path.dirname(aide_predict.__file__)),
    "external_calls",
    "eve",
    "_compute_evol_indices_one.py"
)

if EVE_ENV is None or EVE_REPO is None:
    AVAILABLE = MessageBool(False, "EVE requires EVE_CONDA_ENV and EVE_REPO environment variables to be set")
else:
    AVAILABLE = MessageBool(True, "EVE model is available")

class EVEWrapper(RequiresWTToFunctionMixin, RequiresFixedLengthMixin, RequiresWTDuringInferenceMixin, RequiresMSAForFitMixin, RequiresWTMSAMixin, AcceptsLowerCaseMixin, CanRegressMixin, ProteinModelWrapper):
    """
    Wrapper for EVE (Evolutionary Variational Autoencoder) model.
    
    This wrapper provides an interface to train and use EVE models within the AIDE framework.
    EVE is run in a separate conda environment specified by EVE_CONDA_ENV environment variable.
    The EVE repository location must be specified in EVE_REPO environment variable.

    NOTE: SHould this refit on sequences?

    Attributes:
        _available (MessageBool): Indicates whether EVE is available based on environment setup.
    """

    _available = AVAILABLE

    def __init__(
        self,
        metadata_folder: str = None,
        wt: Optional[Union[str, ProteinSequence]] = None,
        # MSA processing parameters
        theta: float = 0.2,
        
        # Encoder parameters
        encoder_hidden_layers: List[int] = [2000, 1000, 300],
        encoder_z_dim: int = 50,
        encoder_convolve_input: bool = False,
        encoder_convolution_input_depth: int = 40,
        encoder_nonlinear_activation: str = "relu",
        encoder_dropout_proba: float = 0.0,
        
        # Decoder parameters
        decoder_hidden_layers: List[int] = [300, 1000, 2000],
        decoder_z_dim: int = 50,
        decoder_bayesian: bool = True,
        decoder_first_nonlinearity: str = "relu",
        decoder_last_nonlinearity: str = "relu",
        decoder_dropout_proba: float = 0.1,
        decoder_convolve_output: bool = True,
        decoder_convolution_output_depth: int = 40,
        decoder_temperature_scaler: bool = True,
        decoder_sparsity: bool = False,
        decoder_num_tiles_sparsity: int = 0,
        decoder_logit_sparsity_p: float = 0.0,
        
        # Training parameters
        training_steps: int = 400000,
        learning_rate: float = 1e-4,
        training_batch_size: int = 256,
        annealing_warm_up: int = 0,
        kl_latent_scale: float = 1.0,
        kl_global_params_scale: float = 1.0,
        l2_regularization: float = 0.0,
        use_lr_scheduler: bool = False,
        use_validation_set: bool = False,
        validation_set_pct: float = 0.2,
        validation_freq: int = 1000,
        log_training_info: bool = True,
        log_training_freq: int = 1000,
        save_model_freq: int = 500000,
        
        # Inference parameters
        inference_batch_size: int = 256,
        num_samples: int = 10,
    ):
        """
        Initialize the EVE wrapper with all configurable parameters exposed.

        Args:
            metadata_folder (str): Folder to store intermediate files and model artifacts.
            wt (Optional[Union[str, ProteinSequence]]): Wild-type sequence.
            
            # MSA Processing
            theta (float): Parameter for MSA sequence reweighting.
            
            # Encoder Parameters
            encoder_hidden_layers (List[int]): Sizes of hidden layers in encoder.
            encoder_z_dim (int): Dimensionality of latent space.
            encoder_convolve_input (bool): Whether to apply convolution to input.
            encoder_convolution_input_depth (int): Depth of input convolution.
            encoder_nonlinear_activation (str): Activation function for encoder.
            encoder_dropout_proba (float): Dropout probability in encoder.
            
            # Decoder Parameters
            decoder_hidden_layers (List[int]): Sizes of hidden layers in decoder.
            decoder_z_dim (int): Dimensionality of latent space (should match encoder).
            decoder_bayesian (bool): Whether to use Bayesian decoder.
            decoder_first_nonlinearity (str): Activation for first layer.
            decoder_last_nonlinearity (str): Activation for last layer.
            decoder_dropout_proba (float): Dropout probability in decoder.
            decoder_convolve_output (bool): Whether to apply convolution to output.
            decoder_convolution_output_depth (int): Depth of output convolution.
            decoder_temperature_scaler (bool): Whether to use temperature scaling.
            decoder_sparsity (bool): Whether to enforce sparsity.
            decoder_num_tiles_sparsity (int): Number of tiles for sparsity.
            decoder_logit_sparsity_p (float): Sparsity parameter.
            
            # Training Parameters
            training_steps (int): Number of training steps.
            learning_rate (float): Learning rate for optimization.
            training_batch_size (int): Batch size during training.
            annealing_warm_up (int): Steps for KL annealing warmup.
            kl_latent_scale (float): Scale for latent KL term.
            kl_global_params_scale (float): Scale for global parameters KL term.
            l2_regularization (float): L2 regularization strength.
            use_lr_scheduler (bool): Whether to use learning rate scheduler.
            use_validation_set (bool): Whether to use validation set.
            validation_set_pct (float): Percentage of data for validation.
            validation_freq (int): Frequency of validation.
            log_training_info (bool): Whether to log training information.
            log_training_freq (int): Frequency of logging.
            save_model_freq (int): Frequency of model saving.
            
            # Inference Parameters
            inference_batch_size (int): Batch size for computing evolutionary indices.
            num_samples (int): Number of samples for approximating delta ELBO.
        """
        super().__init__(metadata_folder=metadata_folder, wt=wt)
        
        # Store all parameters as attributes
        # MSA processing
        self.theta = theta
        
        # Encoder parameters
        self.encoder_hidden_layers = encoder_hidden_layers
        self.encoder_z_dim = encoder_z_dim
        self.encoder_convolve_input = encoder_convolve_input
        self.encoder_convolution_input_depth = encoder_convolution_input_depth
        self.encoder_nonlinear_activation = encoder_nonlinear_activation
        self.encoder_dropout_proba = encoder_dropout_proba
        
        # Decoder parameters
        self.decoder_hidden_layers = decoder_hidden_layers
        self.decoder_z_dim = decoder_z_dim
        self.decoder_bayesian = decoder_bayesian
        self.decoder_first_nonlinearity = decoder_first_nonlinearity
        self.decoder_last_nonlinearity = decoder_last_nonlinearity
        self.decoder_dropout_proba = decoder_dropout_proba
        self.decoder_convolve_output = decoder_convolve_output
        self.decoder_convolution_output_depth = decoder_convolution_output_depth
        self.decoder_temperature_scaler = decoder_temperature_scaler
        self.decoder_sparsity = decoder_sparsity
        self.decoder_num_tiles_sparsity = decoder_num_tiles_sparsity
        self.decoder_logit_sparsity_p = decoder_logit_sparsity_p
        
        # Training parameters
        self.training_steps = training_steps
        self.learning_rate = learning_rate
        self.training_batch_size = training_batch_size
        self.annealing_warm_up = annealing_warm_up
        self.kl_latent_scale = kl_latent_scale
        self.kl_global_params_scale = kl_global_params_scale
        self.l2_regularization = l2_regularization
        self.use_lr_scheduler = use_lr_scheduler
        self.use_validation_set = use_validation_set
        self.validation_set_pct = validation_set_pct
        self.validation_freq = validation_freq
        self.log_training_info = log_training_info
        self.log_training_freq = log_training_freq
        self.save_model_freq = save_model_freq
        
        # Inference parameters
        self.inference_batch_size = inference_batch_size
        self.num_samples = num_samples

    def _get_parameter_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert individual parameters to EVE's expected parameter dictionary structure."""
        return {
            "encoder_parameters": {
                "hidden_layers_sizes": self.encoder_hidden_layers,
                "z_dim": self.encoder_z_dim,
                "convolve_input": self.encoder_convolve_input,
                "convolution_input_depth": self.encoder_convolution_input_depth,
                "nonlinear_activation": self.encoder_nonlinear_activation,
                "dropout_proba": self.encoder_dropout_proba
            },
            "decoder_parameters": {
                "hidden_layers_sizes": self.decoder_hidden_layers,
                "z_dim": self.decoder_z_dim,
                "bayesian_decoder": self.decoder_bayesian,
                "first_hidden_nonlinearity": self.decoder_first_nonlinearity,
                "last_hidden_nonlinearity": self.decoder_last_nonlinearity,
                "dropout_proba": self.decoder_dropout_proba,
                "convolve_output": self.decoder_convolve_output,
                "convolution_output_depth": self.decoder_convolution_output_depth,
                "include_temperature_scaler": self.decoder_temperature_scaler,
                "include_sparsity": self.decoder_sparsity,
                "num_tiles_sparsity": self.decoder_num_tiles_sparsity,
                "logit_sparsity_p": self.decoder_logit_sparsity_p
            },
            "training_parameters": {
                "num_training_steps": self.training_steps,
                "learning_rate": self.learning_rate,
                "batch_size": self.training_batch_size,
                "annealing_warm_up": self.annealing_warm_up,
                "kl_latent_scale": self.kl_latent_scale,
                "kl_global_params_scale": self.kl_global_params_scale,
                "l2_regularization": self.l2_regularization,
                "use_lr_scheduler": self.use_lr_scheduler,
                "use_validation_set": self.use_validation_set,
                "validation_set_pct": self.validation_set_pct,
                "validation_freq": self.validation_freq,
                "log_training_info": self.log_training_info,
                "log_training_freq": self.log_training_freq,
                "save_model_params_freq": self.save_model_freq
            }
        }

    def _save_parameters(self) -> str:
        """Save model parameters to a JSON file in the metadata folder."""
        params = self._get_parameter_dict()
        params_file = os.path.join(self.metadata_folder, "model_parameters.json")
        with open(params_file, "w") as f:
            json.dump(params, f, indent=4)
        return params_file

    def _run_eve_script(self, script_path: str, args: List[str]) -> None:
        """
        Run an EVE script in the EVE conda environment.

        Args:
            script_path (str): path to the EVE script to run.
            args (List[str]): Command line arguments for the script.

        Raises:
            RuntimeError: If the script execution fails.
        """
        cmd = [
            "conda", "run",
            "-n", EVE_ENV,
            "--cwd", EVE_REPO,
            "python", "-u",  # Add -u flag to force unbuffered output
            script_path,
            *args
        ]
        logger.info(f"Running EVE script: {' '.join(cmd)}")
        
        # Use subprocess.run instead of Popen for simpler handling
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},  # Force Python unbuffered output
            bufsize=0  # Completely unbuffered
        )

        # Print output
        if process.stdout:
            print(process.stdout)
        if process.stderr:
            print(process.stderr, file=sys.stderr)

        if process.returncode != 0:
            raise RuntimeError(f"EVE script failed with exit code {process.returncode}")


    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'EVEWrapper':
        """
        Fit the EVE model to the input MSA.

        Args:
            X (ProteinSequences): Input MSA.
            y (Optional[np.ndarray]): Ignored. Present for API consistency.

        Returns:
            EVEWrapper: The fitted model.

        Raises:
            RuntimeError: If EVE training fails.
        """
        X = self.wt.msa

        # Set up folders within metadata directory
        subfolders = ['weights', 'checkpoints', 'logs']
        for folder in subfolders:
            os.makedirs(os.path.join(self.metadata_folder, folder), exist_ok=True)

        # Save MSA to temporary file
        msa_file = os.path.join(self.metadata_folder, "msa.fasta")
        X.to_fasta(msa_file)

        # Save model parameters
        params_file = self._save_parameters()

        # Run EVE training script
        model_name = f"eve_model_{sha256(X[0].id.encode()).hexdigest()[:8]}"
        args = [
            "--msa_file", msa_file,
            "--model_name", model_name,
            "--model_parameters", params_file,
            "--theta_reweighting", str(self.theta),
            "--weights_folder", os.path.join(self.metadata_folder, "weights"),
            "--checkpoint_folder", os.path.join(self.metadata_folder, "checkpoints"),
            "--logs_folder", os.path.join(self.metadata_folder, "logs")
        ]
        
        self._run_eve_script(EVE_SCRIPT_1, args)
        
        # Store the model name and final checkpoint location
        self.model_name_ = model_name
        self.checkpoint_ = os.path.join(self.metadata_folder, "checkpoints", f"{model_name}_final")
        self.fitted_ = True
        
        return self

    def _transform(self, X: ProteinSequences) -> np.ndarray:
        """
        Transform sequences using EVE to get evolutionary indices.

        Args:
            X (ProteinSequences): Input sequences to score.

        Returns:
            np.ndarray: Array of evolutionary indices for input sequences.

        Raises:
            RuntimeError: If computation of evolutionary indices fails.
        """
        # Identify wild-type sequences in the input
        wt_indices = []
        non_wt_indices = []
        non_wt_sequences = []
        
        for i, seq in enumerate(X):
            if str(seq).upper() == str(self.wt).upper():
                wt_indices.append(i)
            else:
                non_wt_indices.append(i)
                non_wt_sequences.append(seq)
        
        # If all sequences are wild-type, return zeros
        if len(non_wt_sequences) == 0:
            return np.zeros((len(X), 1))
        
        # Create mutations file only for non-wild-type sequences
        non_wt_X = ProteinSequences(non_wt_sequences)
        mutations = []
        for seq in non_wt_X:
            mut_list = self.wt.get_mutations(seq)
            mutations.append(':'.join(mut_list))
            if mutations[-1] == '':
                print(mut_list)
                raise ValueError(f"Sequence {seq.id} is equal to WT but should have been filtered out for eve call")
            

        # Save mutations to file
        mutations_df = pd.DataFrame({"mutations": mutations})
        mutations_file = os.path.join(self.metadata_folder, "mutations.csv")
        mutations_df.to_csv(mutations_file, index=False)

        # Save MSA to file (required for EVE)
        msa_file = os.path.join(self.metadata_folder, "msa.fasta")
        
        # Save parameters
        params_file = self._save_parameters()

        # Set up results folder
        results_folder = os.path.join(self.metadata_folder, "results")
        os.makedirs(results_folder, exist_ok=True)

        # Run EVE compute script
        args = [
            "--msa_file", msa_file,
            "--model_name", self.model_name_,
            "--model_parameters", params_file,
            "--checkpoint", self.checkpoint_,
            "--theta_reweighting", str(self.theta),
            "--weights_folder", os.path.join(self.metadata_folder, "weights"),
            "--computation_mode", "input_mutations_list",
            "--mutations_file", mutations_file,
            "--output_folder", results_folder,
            "--num_samples", str(self.num_samples),
            "--batch_size", str(self.inference_batch_size)
        ]

        self._run_eve_script(EVE_SCRIPT_2, args)

        # Read and process results
        results_file = os.path.join(
            results_folder,
            f"{self.model_name_}_{self.num_samples}_samples.csv"
        )
        
        if not os.path.exists(results_file):
            raise RuntimeError(f"EVE did not generate results file: {results_file}")
            
        results = pd.read_csv(results_file).drop_duplicates()
        
        # Create a mapping from mutation lists to scores
        result_dict = {}
        for _, row in results.iterrows():
            mutation_set = frozenset(row['mutations'].split(':'))
            result_dict[mutation_set] = row['evol_indices']
        
        # Reconstruct scores in original order
        final_scores = np.zeros((len(X), 1))
        
        # For wild-type sequences, score is 0 (reference point)
        # For non-wild-type sequences, look up score in results
        for original_idx, seq_idx in enumerate(non_wt_indices):
            mutation_set = frozenset(self.wt.get_mutations(X[seq_idx]))
            if mutation_set in result_dict:
                # EVE scores are better when more negative, so we negate them
                final_scores[seq_idx] = -result_dict[mutation_set]  
            else:
                logger.warning(f"No score found for sequence at index {seq_idx} with mutations {mutation_set}")
                final_scores[seq_idx] = np.nan
        
        return final_scores

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names."""
        return ["EVE_score"]
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.
        
        Args:
            deep (bool): If True, will return the parameters for this estimator and
                        contained subobjects that are estimators.

        Returns:
            Dict[str, Any]: Parameter names mapped to their values.
        """
        params = super().get_params(deep=deep)
        
        # Remove private attributes
        params = {k: v for k, v in params.items() if not k.startswith('_')}
        
        return params

    def set_params(self, **params: Any) -> 'EVEWrapper':
        """
        Set the parameters of this estimator.
        
        Args:
            **params: Estimator parameters.

        Returns:
            EVEWrapper: Return self to enable chaining.
        """
        return super().set_params(**params)
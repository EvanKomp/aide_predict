# aide_predict/bespoke_models/predictors/pretrained_transformers.py
'''
* Author: Evan Komp
* Created: 7/11/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Base class for log likelihood based transformer models. Supports wildtype, mutant, and masked marginal methods.

See: 
Meier, J. et al. Language models enable zero-shot prediction of the effects of mutations on protein function. Preprint at https://doi.org/10.1101/2021.07.09.450648 (2021).

'''
from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Any, Callable
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
import os

from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.bespoke_models.base import ProteinModelWrapper, PositionSpecificMixin, RequiresWTDuringInferenceMixin, CanRegressMixin, ExpectsNoFitMixin, RequiresFixedLengthMixin, CacheMixin

class MarginalMethod(Enum):
    MASKED = "masked_marginal"
    WILDTYPE = "wildtype_marginal"
    MUTANT = "mutant_marginal"

class ModelDeviceManager:
    _instances = {}
    
    @classmethod
    def get_instance(cls, model_instance: Any, device: str):
        if model_instance not in cls._instances:
            cls._instances[model_instance] = cls(model_instance, device)
        return cls._instances[model_instance]

    def __init__(self, model_instance: Any, device: str = 'cpu'):
        self.model_instance = model_instance
        self.device = device
        self.keep_on_device = os.environ.get('KEEP_MODEL_ON_DEVICE', '').lower() in ('true', '1', 'yes')
        self.model_loaded = False

    @contextmanager
    def model_on_device(self, load_func: Callable[[], None], cleanup_func: Callable[[], None]):
        try:
            # Load the model only if it's not already loaded
            if not self.model_loaded:
                load_func()
                self.model_loaded = True
            yield
        finally:
            # Clean up only if not set to keep on device
            if not self.keep_on_device:
                cleanup_func()
                self.model_loaded = False
                self._reclaim_memory()

    def _reclaim_memory(self):
        if self.device.startswith('cuda'):
            import torch
            torch.cuda.empty_cache()
        elif self.device.startswith('mps'):
            import torch
            torch.mps.empty_cache()

@contextmanager
def model_device_context(model_instance: Any, load_func: Callable[[], None], cleanup_func: Callable[[], None], device: str = 'cpu'):
    """Context manager used to load and clean up a model on a specific device.
    
    This ensures model weights are not sitting on the GPU when not being accessed,
    unless the KEEP_MODEL_ON_DEVICE environment variable is set to True.
    If set to True, the model is loaded only once and kept on the device across multiple calls.
    """
    manager = ModelDeviceManager.get_instance(model_instance, device)
    with manager.model_on_device(load_func, cleanup_func):
        yield

class LikelihoodTransformerBase(PositionSpecificMixin, RequiresFixedLengthMixin, ExpectsNoFitMixin, CanRegressMixin, RequiresWTDuringInferenceMixin, CacheMixin, ProteinModelWrapper, ABC):
    """
    Base class for likelihood transformer models.

    This abstract base class provides a framework for implementing likelihood transformer models
    that can compute various types of marginal likelihoods for protein sequences.

    Attributes:
        marginal_method (MarginalMethod): The method used to compute marginal likelihoods.
        batch_size (int): The number of sequences to process in each batch.
        device (str): The device to use for computations ('cpu' or 'cuda').
    """

    def __init__(self, metadata_folder: str=None, 
                 marginal_method: MarginalMethod = MarginalMethod.WILDTYPE.value,
                 positions: Optional[List[int]] = None, 
                 flatten: bool = False,
                 pool: bool = True,
                 batch_size: int = 2,
                 device: str = 'cpu',
                 wt: Optional[Union[str, ProteinSequence]] = None,
                 **kwargs
                 ):
        """
        Initialize the LikelihoodTransformerBase.

        Args:
            metadata_folder (str): Folder to store metadata.
            marginal_method (MarginalMethod): Method to compute marginal likelihoods.
            positions (Optional[List[int]]): Specific positions to consider.
            flatten (bool): Whether to flatten the output.
            pool (bool): Whether to pool the likelihoods across positions.
            batch_size (int): Number of sequences to process in each batch.
            device (str): Device to use for computations ('cpu' or 'cuda').
            wt (Optional[Union[str, ProteinSequence]]): Wild type sequence.
        """
        super().__init__(metadata_folder=metadata_folder, wt=wt, positions=positions, pool=pool, flatten=flatten, **kwargs)
        self.marginal_method = marginal_method
        self.batch_size = batch_size
        self.device = device

    def _transform(self, X: ProteinSequences) -> np.ndarray:
        """
        Transform the input sequences using the specified marginal method.

        Args:
            X (ProteinSequences): Input protein sequences.

        Returns:
            np.ndarray: Computed log likelihoods.

        Raises:
            ValueError: If an unknown marginal method is specified.
        """
        with model_device_context(self, self._load_model, self._cleanup_model, self.device):
            if self.marginal_method == MarginalMethod.MUTANT.value:
                log_likelihoods = self._compute_mutant_marginal(X)
            elif self.marginal_method == MarginalMethod.WILDTYPE.value:
                log_likelihoods = self._compute_wildtype_marginal(X)
            elif self.marginal_method == MarginalMethod.MASKED.value:
                log_likelihoods = self._compute_masked_marginal(X)
            else:
                raise ValueError(f"Unknown marginal method: {self.marginal_method}")
            
            return self._post_process_likelihoods(log_likelihoods, X)
    
    def _validate_marginals(self, marginals: Union[np.ndarray, List[np.ndarray]], sequences: ProteinSequences) -> None:
        """
        Validate the computed marginal likelihoods.
        
        Args:
            marginals (Union[np.ndarray, List[np.ndarray]]): Computed marginal likelihoods.
            sequences (ProteinSequences): Input protein sequences.

        Raises:
            ValueError: If the marginals do not meet the expected criteria.
        """
        if isinstance(marginals, list):
            if len(marginals) != len(sequences):
                raise ValueError(f"Expected {len(sequences)} marginal likelihood arrays, but got {len(marginals)}")
            for i, (m, seq) in enumerate(zip(marginals, sequences)):
                self._validate_single_marginal(m, seq, f"Sequence {i}")
        else:
            self._validate_single_marginal(marginals, sequences[0], "Single sequence")

    def _validate_single_marginal(self, marginal: np.ndarray, sequence: ProteinSequence, identifier: str) -> None:
        """
        Validate a single marginal likelihood array.
        
        Args:
            marginal (np.ndarray): Marginal likelihood array to validate.
            sequence (ProteinSequence): The corresponding input sequence.
            identifier (str): Identifier for error messages.
        
        Raises:
            ValueError: If the marginal likelihood array does not meet the expected criteria.
        """
        if marginal.shape[1] != len(sequence):
            raise ValueError(f"{identifier}: Expected marginal likelihoods of shape (1, {len(sequence)}), "
                             f"but got shape {marginal.shape}")
        
        if np.any(np.isinf(marginal)):
            raise ValueError(f"{identifier}: Marginal likelihoods contain infinite values")

    def _validate_log_probs(self, log_probs: Union[np.ndarray, List[np.ndarray]], sequences: ProteinSequences) -> None:
        """
        Validate the log probabilities computed by child classes.

        Args:
            log_probs (Union[np.ndarray, List[np.ndarray]]): Log probabilities to validate.
            sequences (ProteinSequences): The input sequences used to compute the log probabilities.

        Raises:
            ValueError: If the log probabilities do not meet the expected criteria.
        """
        if isinstance(log_probs, list):
            if len(log_probs) != len(sequences):
                raise ValueError(f"Expected {len(sequences)} log probability arrays, but got {len(log_probs)}")
            for i, (lp, seq) in enumerate(zip(log_probs, sequences)):
                self._validate_single_log_prob(lp, seq, f"Sequence {i}")
        else:
            self._validate_single_log_prob(log_probs, sequences[0], "Single sequence")

    def _validate_single_log_prob(self, log_prob: np.ndarray, sequence: ProteinSequence, identifier: str) -> None:
        """
        Validate a single log probability array.

        Args:
            log_prob (np.ndarray): Log probability array to validate.
            sequence (ProteinSequence): The corresponding input sequence.
            identifier (str): Identifier for error messages.

        Raises:
            ValueError: If the log probability array does not meet the expected criteria.
        """
        if log_prob.shape[0] != len(sequence):
            raise ValueError(f"{identifier}: Expected log probabilities of shape ({len(sequence)}, vocab_size), "
                             f"but got shape {log_prob.shape}")
        
        if not np.allclose(np.exp(log_prob).sum(axis=-1), 1.0, atol=1e-5):
            raise ValueError(f"{identifier}: Log probabilities do not sum to 1.0 after exponentiation")
        
        if np.any(np.isnan(log_prob)):
            raise ValueError(f"{identifier}: Log probabilities contain NaN values")
        
        if np.any(np.isinf(log_prob)):
            raise ValueError(f"{identifier}: Log probabilities contain infinite values")

    @abstractmethod
    def _compute_log_likelihoods(self, X: ProteinSequences, mask_positions: Optional[List[List[int]]] = None) -> List[np.ndarray]:
        """
        Compute log likelihoods of each vocab unit at each position in input sequences.
        
        Args:
            X (ProteinSequences): Input protein sequences.
            mask_positions (Optional[List[List[int]]]): Positions to mask in the sequences.

        Returns:
            List[np.ndarray]: List of log likelihood arrays for each sequence.
            Each is expected to have shape (len(sequence), vocab_size).
        """
        pass

    @abstractmethod
    def _index_log_probs(self, log_probs: np.ndarray, sequences: ProteinSequences) -> np.ndarray:
        """
        Index log probabilities by the observed amino acids in the sequences.

        Args:
            log_probs (np.ndarray): Log probabilities to index. These are each of shape (seq_len, vocab_size).
            sequences (ProteinSequences): Sequences to use for indexing.

        Returns:
            np.ndarray: Indexed log probabilities. These should be of shape (1, seq_len).
        """
        pass

    @abstractmethod
    def _load_model(self) -> None:
        """
        Load and the model and other objects into memory on device such that they can be accessed in
        `_compute_log_likelihoods` and `_index_log_probs`.

        This is used in a context manager to make sure the model is only on device when needed.
        """
        pass

    @abstractmethod
    def _cleanup_model(self) -> None:
        """
        Clean up the model and other objects loaded into memory in `_load_model`.

        This is used in a context manager to make sure the model is only on device when needed.
        """
        pass

    def _compute_log_likelihoods_batches(self, X: ProteinSequences, mask_positions: List[int] = []) -> List[np.ndarray]:
        """
        Compute log likelihoods for the input sequences in batches.

        Args:
            X (ProteinSequences): Input protein sequences.
            mask_positions (List[int]): Positions to mask in the sequences.

        Returns:
            List[np.ndarray]: List of log likelihood arrays for each sequence.
        """
        log_likelihoods = []
        index = 0
        total = len(X)
        bar = tqdm(total=total, desc="Computing log likelihoods", unit="sequence")
        for batch in X.iter_batches(self.batch_size):
            log_probs = self._compute_log_likelihoods(batch, mask_positions[index:index + len(batch)])
            self._validate_log_probs(log_probs, batch)
            log_likelihoods.extend(log_probs)
            index += len(batch)
            bar.update(len(batch))

        return log_likelihoods

    def _compute_mutant_marginal(self, X: ProteinSequences) -> List[np.ndarray]:
        """
        Compute mutant marginal log likelihoods.

        Args:
            X (ProteinSequences): Input protein sequences.

        Returns:
            List[np.ndarray]: List of mutant marginal log likelihoods for each sequence.
        """
        log_probs = self._compute_log_likelihoods_batches(X)
        self._validate_log_probs(log_probs, X)
        
        log_likelihoods = [self._index_log_probs(l, ProteinSequences([X[i]])) for i, l in enumerate(log_probs)]
        self._validate_marginals(log_likelihoods, X)
        
        if self.wt is not None:
            if not X.aligned or X.width != len(self.wt):
                pass  # Compare to WT in post-processing
            else:
                wt_likelihoods = [self._index_log_probs(l, ProteinSequences([self.wt])) for l in log_probs]
                self._validate_marginals(wt_likelihoods, ProteinSequences([self.wt] * len(X)))
                
                log_likelihoods = [ll - wl for ll, wl in zip(log_likelihoods, wt_likelihoods)]
                for i, (seq, ll) in enumerate(zip(X, log_likelihoods)):
                    mutation_positions = np.array(seq.mutated_positions(self.wt))
                    mutation_mask = np.isin(np.arange(len(seq)), mutation_positions).reshape(1,-1)
                    ll[~mutation_mask] = np.nan
        
        return log_likelihoods

    def _compute_wildtype_marginal(self, X: ProteinSequences) -> List[np.ndarray]:
        """
        Compute wildtype marginal log likelihoods.

        Args:
            X (ProteinSequences): Input protein sequences.

        Returns:
            List[np.ndarray]: List of wildtype marginal log likelihoods for each sequence.

        Raises:
            ValueError: If wildtype sequence is not provided or sequences are not aligned.
        """
        if self.wt is None:
            raise ValueError("Wildtype sequence must be provided for wildtype marginal likelihoods.")
        
        if not X.aligned and X.width != len(self.wt):
            raise ValueError("Wildtype marginal likelihoods require aligned sequences with the same length as the wildtype.")
        
        wt_log_probs = self._compute_log_likelihoods_batches(ProteinSequences([self.wt]))[0]
        self._validate_log_probs(wt_log_probs, ProteinSequences([self.wt]))
        
        log_likelihoods = [self._index_log_probs(wt_log_probs, ProteinSequences([seq])) for seq in X]
        self._validate_marginals(log_likelihoods, X)
        
        wt_likelihoods = self._index_log_probs(wt_log_probs, ProteinSequences([self.wt]))
        self._validate_marginals(wt_likelihoods, ProteinSequences([self.wt]))
        
        log_likelihoods = [ll - wt_likelihoods for ll in log_likelihoods]
        
        for i, (seq, ll) in enumerate(zip(X, log_likelihoods)):
            mutation_positions = np.array(seq.mutated_positions(self.wt))
            mutation_mask = np.isin(np.arange(len(seq)), mutation_positions).reshape(1,-1)
            ll[~mutation_mask] = np.nan
        
        return log_likelihoods

    def _compute_masked_marginal(self, X: ProteinSequences) -> List[np.ndarray]:
        """
        Compute masked marginal log likelihoods.

        Args:
            X (ProteinSequences): Input protein sequences.

        Returns:
            List[np.ndarray]: List of masked marginal log likelihoods for each sequence.

        Raises:
            ValueError: If wildtype sequence is not provided, sequences are not aligned, or no mask positions are found.
        """
        if self.wt is None:
            raise ValueError("Wildtype sequence must be provided for masked marginal likelihoods.")
        
        if not X.aligned and X.width != len(self.wt):
            raise ValueError("Masked marginal likelihoods require aligned sequences with the same length as the wildtype.")
        
        mask_positions = self.positions if self.positions is not None else X.mutated_positions
        if len(mask_positions) == 0:
            raise ValueError("No mask positions found for masked marginal likelihoods.")
        log_likelihoods = np.vstack([np.full(len(seq), np.nan) for seq in X])

        all_mask_positions = [[pos] for pos in mask_positions]
        wt_sequences = ProteinSequences([self.wt] * len(mask_positions))

        all_masked_log_probs = self._compute_log_likelihoods_batches(wt_sequences, all_mask_positions)
        self._validate_log_probs(all_masked_log_probs, wt_sequences)

        log_likelihoods_matrix = np.full((len(self.wt), all_masked_log_probs[0].shape[1]), np.nan)
        for i, pos in enumerate(mask_positions):
            masked_log_probs = all_masked_log_probs[i]
            log_likelihoods_matrix[pos] = masked_log_probs[pos]

        wt_marginal = self._index_log_probs(log_likelihoods_matrix, ProteinSequences([self.wt]))
        self._validate_marginals(wt_marginal, ProteinSequences([self.wt]))
        mutant_marginals = self._index_log_probs(log_likelihoods_matrix, X)
        self._validate_marginals(mutant_marginals, X)
        log_likelihoods = mutant_marginals - wt_marginal

        for i, (seq, ll) in enumerate(zip(X, log_likelihoods)):
            mutation_positions = np.array(seq.mutated_positions(self.wt))
            mutation_mask = np.isin(np.arange(len(seq)), mutation_positions)
            ll[~mutation_mask] = np.nan

        return [l.reshape(1, -1) for l in log_likelihoods]

    def _post_process_likelihoods(self, log_likelihoods: List[np.ndarray], X: ProteinSequences) -> np.ndarray:
        """
        Post-process the computed log likelihoods.

        This method handles position filtering, pooling, and the edge case for mutant marginal
        with variable length sequences.

        Args:
            log_likelihoods (List[np.ndarray]): Computed log likelihoods.
            X (ProteinSequences): Input protein sequences.

        Returns:
            np.ndarray: Post-processed log likelihoods.
        """
        if self.positions is not None:
            log_likelihoods = np.vstack([ll[:, self.positions] for ll in log_likelihoods])
        
        if self.pool:
            # if all values are nana, that means it is equal to wildtype and nanmean will fail, return
            # zero for that case
            new_log_likelihoods = []
            for ll in log_likelihoods:
                if np.all(np.isnan(ll)):
                    new_log_likelihoods.append(0.0)
                else:
                    new_log_likelihoods.append(np.nanmean(ll))
            log_likelihoods = np.array(new_log_likelihoods)
            
            # Handle the edge case for mutant marginal with variable length sequences
            if self.marginal_method == MarginalMethod.MUTANT.value and self.wt is not None and not X.aligned and X.width != len(self.wt):
                # Compute WT log likelihood
                wt_log_probs = self._compute_log_likelihoods_batches(ProteinSequences([self.wt]))[0]
                wt_likelihood = self._index_log_probs(wt_log_probs, ProteinSequences([self.wt]))
                wt_pooled = np.nanmean(wt_likelihood)
                
                # Subtract WT pooled likelihood from each pooled mutant likelihood
                log_likelihoods = log_likelihoods - wt_pooled
        else:
            log_likelihoods = np.vstack(log_likelihoods)
    
        return log_likelihoods.reshape(-1, 1) if log_likelihoods.ndim == 1 else log_likelihoods

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.

        Args:
            input_features (Optional[List[str]]): Input feature names (not used in this method).

        Returns:
            List[str]: Output feature names.

        Raises:
            ValueError: If the model hasn't been fitted or if feature names can't be generated.
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Model has not been fitted yet. Call fit() before using this method.")
        
        if self.pool:
            return [f"{self.__class__.__name__}_log_likelihood"]
        elif self.flatten:
            positions = self.positions
            if positions is None:
                raise ValueError("Cannot return feature names for flattened output without positions. unknown number of features")
            return [f"{self.__class__.__name__}_pos{p}_log_likelihood" for p in positions]
        else:
            raise ValueError("Cannot return feature names for non-flattened, non-pooled output.")
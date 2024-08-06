# aide_predict/bespoke_models/embedders/ohe.py
'''
* Author: Evan Komp
* Created: 7/5/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Two classes: OneHotProteinEmbedding for fixed length sequences and OneHotAlignmentEmbedding which
will dynamically align sequences to reference alignment before encoding.

'''
import warnings
from typing import List, Union, Optional
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from aide_predict.bespoke_models.base import ProteinModelWrapper, RequiresMSAMixin, PositionSpecificMixin, CanHandleAlignedSequencesMixin, RequiresFixedLengthMixin
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.utils.constants import AA_SINGLE, GAP_CHARACTERS

class OneHotProteinEmbedding(PositionSpecificMixin, RequiresFixedLengthMixin, ProteinModelWrapper):
    """
    A protein sequence embedder that performs one-hot encoding with position-specific capabilities.
    
    This class wraps sklearn's OneHotEncoder to provide one-hot encoding
    specifically for protein sequences. It expects fixed-length sequences
    without gaps and uses a 20 amino acid vocabulary. It also allows for
    position-specific encoding.

    Attributes:
        vocab (List[str]): The vocabulary of amino acids used for encoding.
        encoder (OneHotEncoder): The underlying sklearn OneHotEncoder.
        positions (Optional[List[int]]): Specific positions to encode. If None, all positions are encoded.
        pool (bool): Ignored
        flatten (bool): Whether to flatten the output array.
        seq_length (Optional[int]): The length of the sequences, determined during fitting.
    """

    def __init__(self, metadata_folder: str=None, wt: Optional[Union[str, ProteinSequence]] = None,
                 positions: Optional[List[int]] = None, flatten: bool = True, pool: bool = False):
        """
        Initialize the OneHotProteinEmbedding.

        Args:
            metadata_folder (str): The folder where metadata is stored.
            wt (Optional[Union[str, ProteinSequence]]): The wild type sequence, if any.
            positions (Optional[List[int]]): Specific positions to encode. If None, all positions are encoded.
            flatten (bool): Whether to flatten the output array.

        Notes: WT is set to None to avoid normalization. For an embedder this is effectively a feature scaler which you
        should do manually if you want
        """
        super().__init__(metadata_folder=metadata_folder, wt=None, positions=positions, pool=False, flatten=flatten)
        self._vocab = list(AA_SINGLE)

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'OneHotProteinEmbedding':
        """
        Fit the one-hot encoder to the protein sequences.

        Args:
            X (ProteinSequences): The input protein sequences.
            y (Optional[np.ndarray]): Ignored. Present for API consistency.

        Returns:
            OneHotProteinEmbedding: The fitted embedder.

        Raises:
            ValueError: If the input sequences are not of fixed length or contain gaps.
        """
        if not X.fixed_length:
            raise ValueError("Input sequences must be of fixed length.")
        
        if X.has_gaps:
            raise ValueError("Input sequences must not contain gaps.")
        
        self.seq_length_ = X.width
        self.positions_len_ = len(self.positions) if self.positions is not None else self.seq_length_
        
        # Initialize the encoder with the correct number of features
        self.encoder_ = OneHotEncoder(categories=[self._vocab] * self.positions_len_, 
                                     sparse_output=False, handle_unknown='ignore')
        
        # Prepare the sequences for fitting
        sequences_array = X.as_array()
        if self.positions is not None:
            sequences_array = sequences_array[:, self.positions]

        self.encoder_.fit(sequences_array)
        
        return self

    def _transform(self, X: ProteinSequences) -> np.ndarray:
        """
        Transform the protein sequences into one-hot encoded vectors.

        Args:
            X (ProteinSequences): The input protein sequences.

        Returns:
            np.ndarray: The one-hot encoded sequences.

        Raises:
            ValueError: If the input sequences are not of fixed length or contain gaps.
        """
        if not X.fixed_length:
            raise ValueError("Input sequences must be of fixed length.")
        
        if X.has_gaps:
            raise ValueError("Input sequences must not contain gaps.")
        
        if X.width != self.seq_length_:
            raise ValueError(f"Input sequences must have length {self.seq_length_}")
        
        # Prepare the sequences for encoding
        sequences_array = X.as_array()
        if self.positions is not None:
            sequences_array = sequences_array[:, self.positions]
        
        # Encode the sequences
        encoded = self.encoder_.transform(sequences_array)
        
        # Reshape the encoded sequences to (n_samples, seq_length, n_features)
        encoded = encoded.reshape(len(X), self.positions_len_, -1)
        
        return encoded
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.

        Args:
            input_features (Optional[List[str]]): Ignored. Present for API consistency.

        Returns:
            List[str]: Output feature names.
        """
        if not hasattr(self, 'encoder_'):
            raise ValueError("Encoder has not been fitted yet. Call fit() before using this method.")
        
        positions = self.positions if self.positions is not None else range(self.seq_length_)
        if self.flatten:
            return [f"pos{i}_{aa}" for i in positions for aa in self._vocab]
        else:
            return [f"pos{i}" for i in positions]
        

class OneHotAlignedEmbedding(PositionSpecificMixin, RequiresMSAMixin, CanHandleAlignedSequencesMixin, ProteinModelWrapper):
    """
    A protein sequence embedder that performs one-hot encoding for aligned sequences.
    
    This class allows for variable-length sequences and requires an MSA for fitting.
    It creates an encoding on the alignment including gaps. At prediction time,
    it can handle both aligned and unaligned sequences.

    Attributes:
        vocab (List[str]): The vocabulary of amino acids and gap characters used for encoding.
        encoder (OneHotEncoder): The underlying sklearn OneHotEncoder.
        positions (Optional[List[int]]): Specific positions to encode. If None, all positions are encoded.
        pool (bool): Whether to pool the encoded vectors across positions.
        flatten (bool): Whether to flatten the output array.
        alignment_width (int): The width of the original alignment.
        original_alignment (ProteinSequences): The original alignment used for fitting.
    """

    def __init__(self, metadata_folder: str, wt: Optional[Union[str, ProteinSequence]] = None,
                 positions: Optional[List[int]] = None, flatten: bool = True, pool: bool = False):
        """
        Initialize the OneHotAlignedEmbedding.

        Args:
            metadata_folder (str): The folder where metadata is stored.
            wt (Optional[Union[str, ProteinSequence]]): The wild type sequence, if any.
            positions (Optional[List[int]]): Specific positions to encode. If None, all positions are encoded.
            flatten (bool): Whether to flatten the output array.
            pool (bool): Ignored

        Notes: WT is set to None to avoid normalization. For an embedder this is effectively a feature scaler which you
        should do manually if you want
        """
        super().__init__(metadata_folder=metadata_folder, wt=None, positions=positions, pool=False, flatten=flatten)
        self._vocab = list(AA_SINGLE.union(GAP_CHARACTERS))

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'OneHotAlignedEmbedding':
        """
        Fit the one-hot encoder to the aligned protein sequences.

        Args:
            X (ProteinSequences): The input aligned protein sequences.
            y (Optional[np.ndarray]): Ignored. Present for API consistency.

        Returns:
            OneHotAlignedEmbedding: The fitted embedder.

        Raises:
            ValueError: If the input sequences are not aligned.
        """
        if not X.aligned:
            raise ValueError("Input sequences must be aligned for fitting.")
        
        self.alignment_width_ = X.width
        self.original_alignment_ = X
        self.positions_len_ = len(self.positions) if self.positions is not None else self.alignment_width_
        
        # Initialize the encoder with the correct number of features
        self.encoder_ = OneHotEncoder(categories=[self._vocab] * self.positions_len_, 
                                      sparse_output=False, handle_unknown='ignore')
        
        # Prepare the sequences for fitting
        sequences_array = X.as_array()
        if self.positions is not None:
            sequences_array = sequences_array[:, self.positions]

        self.encoder_.fit(sequences_array)
        
        return self
    
    def _transform(self, X: ProteinSequences) -> np.ndarray:
        """
        Transform the protein sequences into one-hot encoded vectors.

        Args:
            X (ProteinSequences): The input protein sequences.

        Returns:
            np.ndarray: The one-hot encoded sequences.

        Raises:
            ValueError: If the input sequences are aligned but not of the same width as the original alignment.
        """
        if X.aligned and not X.fixed_length:
            if X.width != self.alignment_width_:
                raise ValueError(f"Aligned input sequences must have width {self.alignment_width_}")
            warnings.warn("Input sequences are already aligned. Using them as-is for encoding.")
            sequences_to_encode = X
        else:
            warnings.warn("Input sequences are not aligned. Aligning them to the original alignment.")
            sequences_to_encode = X.align_to(self.original_alignment_, return_only_new=True, realign=False)
        
        # Prepare the sequences for encoding
        sequences_array = sequences_to_encode.as_array()
        if self.positions is not None:
            sequences_array = sequences_array[:, self.positions]
        
        # Encode the sequences
        encoded = self.encoder_.transform(sequences_array)
        
        # Reshape the encoded sequences to (n_samples, seq_length, n_features)
        encoded = encoded.reshape(len(X), self.positions_len_, -1)
        
        return encoded
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.

        Args:
            input_features (Optional[List[str]]): Ignored. Present for API consistency.

        Returns:
            List[str]: Output feature names.
        """
        if not hasattr(self, 'encoder_'):
            raise ValueError("Encoder has not been fitted yet. Call fit() before using this method.")
        
        positions = self.positions if self.positions is not None else range(self.alignment_width_)
        if self.flatten:
            return [f"pos{i}_{aa}" for i in positions for aa in self._vocab]
        else:
            return [f"pos{i}" for i in positions]
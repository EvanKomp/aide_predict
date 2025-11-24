# aide_predict/bespoke_models/embedders/aa_properties.py
'''
* Author: Evan Komp
* Created: 11/24/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Simple amino acid property embedder for testing position-specific functionality.
'''
import numpy as np
from typing import List, Union, Optional

from aide_predict.bespoke_models.base import (
    ProteinModelWrapper, 
    PositionSpecificMixin,
    CanHandleAlignedSequencesMixin,
    ExpectsNoFitMixin
)
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.utils.common import MessageBool

import logging
logger = logging.getLogger(__name__)

AVAILABLE = MessageBool(True, "AAPropertiesEmbedding is always available")


# Simple physicochemical properties for the 20 standard amino acids
AA_PROPERTIES = {
    'A': [1.8, 0.0, 0.0],   # Alanine: hydrophobicity, charge, size
    'C': [2.5, 0.0, 0.0],   # Cysteine
    'D': [-3.5, -1.0, 0.0], # Aspartic acid
    'E': [-3.5, -1.0, 0.5], # Glutamic acid
    'F': [2.8, 0.0, 1.0],   # Phenylalanine
    'G': [-0.4, 0.0, -1.0], # Glycine
    'H': [-3.2, 0.5, 0.5],  # Histidine
    'I': [4.5, 0.0, 0.5],   # Isoleucine
    'K': [-3.9, 1.0, 0.5],  # Lysine
    'L': [3.8, 0.0, 0.5],   # Leucine
    'M': [1.9, 0.0, 0.5],   # Methionine
    'N': [-3.5, 0.0, 0.0],  # Asparagine
    'P': [-1.6, 0.0, 0.0],  # Proline
    'Q': [-3.5, 0.0, 0.5],  # Glutamine
    'R': [-4.5, 1.0, 1.0],  # Arginine
    'S': [-0.8, 0.0, -0.5], # Serine
    'T': [-0.7, 0.0, 0.0],  # Threonine
    'V': [4.2, 0.0, 0.0],   # Valine
    'W': [-0.9, 0.0, 1.5],  # Tryptophan
    'Y': [-1.3, 0.0, 1.0],  # Tyrosine
}


class AAPropertiesEmbedding(
    ExpectsNoFitMixin,
    PositionSpecificMixin, 
    CanHandleAlignedSequencesMixin,
    ProteinModelWrapper
):
    """
    A simple amino acid property embedder for testing position-specific functionality.
    
    This embedder converts each amino acid to a 3-dimensional vector based on:
    - Hydrophobicity (Kyte-Doolittle scale approximation)
    - Charge (at physiological pH)
    - Size (relative volume)
    
    This is a simple, fast embedder that can handle aligned sequences with gaps
    and is useful for testing the PositionSpecificMixin functionality.
    
    Attributes:
        positions (Optional[List[int]]): Specific positions to encode. If None, all positions are encoded.
        pool (bool): Whether to pool the encoded vectors across positions.
        flatten (bool): Whether to flatten the output array.
        handle_aligned (bool): Whether to handle aligned sequences with gaps.
        gap_fill_value (float): Value to use for gap positions.
    """
    
    _available = AVAILABLE

    def __init__(
        self,
        metadata_folder: str = None,
        positions: Optional[List[int]] = None,
        flatten: bool = False,
        pool: bool = False,
        handle_aligned: bool = True,
        gap_fill_value: float = 0.0,
        wt: Optional[Union[str, ProteinSequence]] = None,
        **kwargs
    ):
        """
        Initialize the AAPropertiesEmbedding.

        Args:
            metadata_folder (str): The folder where metadata is stored.
            positions (Optional[List[int]]): Specific positions to encode. If None, all positions are encoded.
            flatten (bool): Whether to flatten the output array.
            pool (bool): Whether to pool the encoded vectors across positions.
            handle_aligned (bool): Whether to handle aligned sequences with gaps.
            gap_fill_value (float): Value to use for gap positions.
            wt (Optional[Union[str, ProteinSequence]]): The wild type sequence, if any.
        """
        super().__init__(
            metadata_folder=metadata_folder,
            wt=wt,
            positions=positions,
            pool=pool,
            flatten=flatten,
            handle_aligned=handle_aligned,
            gap_fill_value=gap_fill_value,
            **kwargs
        )
        self.embedding_dim_ = 3  # 3 properties per amino acid

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'AAPropertiesEmbedding':
        """
        Fit the embedder (no actual fitting needed as properties are predefined).

        Args:
            X (ProteinSequences): The input protein sequences.
            y (Optional[np.ndarray]): Ignored. Present for API consistency.

        Returns:
            AAPropertiesEmbedding: The fitted embedder.
        """
        self.fitted_ = True
        return self

    def _transform(self, X: ProteinSequences) -> List[np.ndarray]:
        """
        Transform the protein sequences into amino acid property embeddings.

        Args:
            X (ProteinSequences): The input protein sequences.

        Returns:
            List[np.ndarray]: The amino acid property embeddings for the sequences.
        """
        all_embeddings = []
        
        for seq in X:
            seq_str = str(seq).upper()
            seq_len = len(seq_str)
            
            # Create embedding matrix: (seq_len, 3)
            embedding = np.zeros((1, seq_len, 3), dtype=np.float32)
            
            for i, aa in enumerate(seq_str):
                if aa in AA_PROPERTIES:
                    embedding[0, i, :] = AA_PROPERTIES[aa]
                else:
                    # Unknown amino acid - use zeros
                    logger.warning(f"Unknown amino acid '{aa}' in sequence {seq.id}, using zeros")
                    embedding[0, i, :] = [0.0, 0.0, 0.0]
            
            all_embeddings.append(embedding)
        
        # Return as list - PositionSpecificMixin will handle position selection, pooling, and alignment remapping
        return all_embeddings

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.

        Args:
            input_features (Optional[List[str]]): Ignored. Present for API consistency.

        Returns:
            List[str]: Output feature names.
        """
        if not hasattr(self, 'fitted_'):
            raise ValueError("Model has not been fitted yet. Call fit() before using this method.")
        
        positions = self.positions
        property_names = ['hydrophobicity', 'charge', 'size']
        
        if self.pool:
            return [f"AAProps_{prop}" for prop in property_names]
        elif self.flatten:
            if positions is None:
                raise ValueError("Cannot return feature names for flattened embeddings without specifying positions")
            return [f"pos{p}_{prop}" for p in positions for prop in property_names]
        else:
            raise ValueError("Cannot return feature names for non-flattened non-pooled embeddings.")

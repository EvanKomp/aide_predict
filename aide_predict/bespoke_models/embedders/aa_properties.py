# aide_predict/bespoke_models/embedders/aa_properties.py
'''
* Author: Evan Komp
* Created: 11/24/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Amino acid property embedder using AAindex database for physicochemical properties.
'''
import numpy as np
from typing import List, Union, Optional, Dict, Tuple
from aaindex import aaindex1

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

# Default AAindex indices to use for each property
DEFAULT_AAINDEX_PROPERTIES = [
    ('KYTJ820101', 'hydrophobicity'),      # Kyte-Doolittle hydropathy index
    ('KLEP840101', 'charge'),              # Net charge
    ('FASG760104', 'pKa_amino_terminus'),  # pKa N-terminus
    ('FASG760105', 'pKa_carboxyl_terminus'), # pKa C-terminus
    ('GRAR740103', 'volume'),              # Volume
    ('GRAR740102', 'polarity'),            # Polarity
    ('FASG760101', 'molecular_weight'),    # Molecular weight
    ('ZIMJ680104', 'isoelectric_point'),   # Isoelectric point
    ('ZIMJ680102', 'bulkiness'),           # Bulkiness (related to aromaticity/structure)
]


def _check_aaindex_available() -> MessageBool:
    """Check if AAindex is available and can be accessed."""
    try:
        from aaindex import aaindex1
        # Try to access a known index to verify it works
        _ = aaindex1['KYTJ820101']
        return MessageBool(True, "AAindex is available")
    except Exception as e:
        return MessageBool(False, f"AAindex initialization failed: {str(e)}")


AVAILABLE = _check_aaindex_available()


def _build_aa_property_lookup(
    aaindex_ids: List[Tuple[str, str]],
    include_aromatic: bool = True
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Build a lookup table for amino acid properties from AAindex.
    
    Args:
        aaindex_ids: List of tuples (aaindex_id, property_name)
        include_aromatic: Whether to include a boolean aromatic property
        
    Returns:
        Tuple of (lookup_dict, property_names) where:
            - lookup_dict maps amino acid letter to property vector
            - property_names is the list of property names in order
    """
    # Standard amino acids
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aromatic_aas = {'F', 'W', 'Y'}  # Phenylalanine, Tryptophan, Tyrosine
    
    property_names = [name for _, name in aaindex_ids]
    if include_aromatic:
        property_names.append('aromatic')
    
    # Build lookup table
    lookup = {}
    for aa in amino_acids:
        properties = []
        for aaindex_id, _ in aaindex_ids:
            try:
                record = aaindex1[aaindex_id]
                value = record.values.get(aa)
                if value is None:
                    logger.warning(f"No value for {aa} in {aaindex_id}, using 0.0")
                    value = 0.0
                properties.append(float(value))
            except Exception as e:
                logger.warning(f"Error getting {aaindex_id} for {aa}: {e}, using 0.0")
                properties.append(0.0)
        
        # Add aromatic boolean as 1.0 or 0.0
        if include_aromatic:
            properties.append(1.0 if aa in aromatic_aas else 0.0)
        
        lookup[aa] = np.array(properties, dtype=np.float32)
    
    return lookup, property_names


class AAPropertiesEmbedding(
    ExpectsNoFitMixin,
    PositionSpecificMixin, 
    CanHandleAlignedSequencesMixin,
    ProteinModelWrapper
):
    """
    An amino acid property embedder using AAindex database.
    
    This embedder converts each amino acid to a vector of physicochemical properties
    from the AAindex database. By default, it uses:
    - Hydrophobicity (Kyte-Doolittle scale)
    - Net charge
    - pKa of amino terminus
    - pKa of carboxyl terminus
    - Volume
    - Polarity
    - Molecular weight
    - Isoelectric point
    - Bulkiness
    - Aromatic (boolean: 1.0 for F/W/Y, 0.0 otherwise)
    
    Custom properties can be specified by providing a list of (aaindex_id, property_name) tuples.
    
    Attributes:
        positions (Optional[List[int]]): Specific positions to encode. If None, all positions are encoded.
        pool (bool): Whether to pool the encoded vectors across positions.
        flatten (bool): Whether to flatten the output array.
        handle_aligned (bool): Whether to handle aligned sequences with gaps.
        gap_fill_value (float): Value to use for gap positions.
        aaindex_properties (List[Tuple[str, str]]): List of (aaindex_id, property_name) tuples.
        include_aromatic (bool): Whether to include aromatic boolean property.
        aa_property_lookup_ (Dict[str, np.ndarray]): Lookup table for amino acid properties.
        property_names_ (List[str]): Names of the properties in order.
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
        aaindex_properties: Optional[List[Tuple[str, str]]] = None,
        include_aromatic: bool = True,
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
            aaindex_properties (Optional[List[Tuple[str, str]]]): List of (aaindex_id, property_name) tuples.
                If None, uses DEFAULT_AAINDEX_PROPERTIES.
            include_aromatic (bool): Whether to include a boolean aromatic property (F, W, Y = 1.0, others = 0.0).
        """
        if aaindex_properties is None:
            aaindex_properties = DEFAULT_AAINDEX_PROPERTIES
        self.aaindex_properties = aaindex_properties
        self.include_aromatic = include_aromatic
        
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

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'AAPropertiesEmbedding':
        """
        Fit the embedder by building the amino acid property lookup table.

        Args:
            X (ProteinSequences): The input protein sequences.
            y (Optional[np.ndarray]): Ignored. Present for API consistency.

        Returns:
            AAPropertiesEmbedding: The fitted embedder.
        """
        # Build lookup table from AAindex
        self.aa_property_lookup_, self.property_names_ = _build_aa_property_lookup(
            self.aaindex_properties,
            include_aromatic=self.include_aromatic
        )
        self.embedding_dim_ = len(self.property_names_)
        
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
            
            # Create embedding matrix: (1, seq_len, n_properties)
            embedding = np.zeros((1, seq_len, self.embedding_dim_), dtype=np.float32)
            
            for i, aa in enumerate(seq_str):
                if aa in self.aa_property_lookup_:
                    embedding[0, i, :] = self.aa_property_lookup_[aa]
                else:
                    # Unknown amino acid (including gaps) - use zeros or gap_fill_value
                    if aa == '-':
                        embedding[0, i, :] = self.gap_fill_value
                    else:
                        logger.warning(f"Unknown amino acid '{aa}' in sequence {seq.id}, using zeros")
                        embedding[0, i, :] = 0.0
            
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
        if not hasattr(self, 'fitted_') or not self.fitted_:
            raise ValueError("Model has not been fitted yet. Call fit() before using this method.")
        
        positions = self.positions
        
        if self.pool:
            return [f"AAProps_{prop}" for prop in self.property_names_]
        elif self.flatten:
            if positions is None:
                raise ValueError("Cannot return feature names for flattened embeddings without specifying positions")
            return [f"pos{p}_{prop}" for p in positions for prop in self.property_names_]
        else:
            raise ValueError("Cannot return feature names for non-flattened non-pooled embeddings.")

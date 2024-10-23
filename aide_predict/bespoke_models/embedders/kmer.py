# aide_predict/bespoke_models/embedders/kmer.py
'''
* Author: Evan Komp
* Created: 8/9/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import numpy as np
from typing import List, Union, Optional
from collections import defaultdict

from aide_predict.bespoke_models.base import ProteinModelWrapper, CanHandleAlignedSequencesMixin
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.utils.common import MessageBool
AVAILABLE = MessageBool(True, "Available")

class KmerEmbedding(CanHandleAlignedSequencesMixin, ProteinModelWrapper):
    """
    A fast K-mer embedding class for protein sequences.

    This class generates K-mer embeddings for protein sequences, handling both
    aligned and unaligned sequences efficiently.

    Attributes:
        k (int): The size of the K-mers.
        normalize (bool): Whether to normalize the K-mer counts.
    """
    _available=AVAILABLE
    def __init__(self, metadata_folder: str = None, 
                 k: int = 3, 
                 normalize: bool = True,
                 wt: ProteinSequence = None):
        """
        Initialize the KmerEmbedding.

        Args:
            metadata_folder (str): Folder to store metadata.
            k (int): The size of the K-mers.
            normalize (bool): Whether to normalize the K-mer counts.
        """
        super().__init__(metadata_folder=metadata_folder, wt=None)
        if k < 1:
            raise ValueError("K must be a positive integer.")
        if not isinstance(k, int):
            raise ValueError("K must be an integer.")

        self.k = k
        self.normalize = normalize
        self._kmer_to_index = {}

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'KmerEmbedding':
        """
        Fit the K-mer embedding model.

        This method creates a mapping of K-mers to indices.

        Args:
            X (ProteinSequences): The input protein sequences.
            y (Optional[np.ndarray]): Ignored. Present for API consistency.

        Returns:
            KmerEmbedding: The fitted model.
        """
        if len(X) == 0:
            raise ValueError("Cannot fit KmerEmbedding with no sequences.")
        unique_kmers = set()
        for seq in X:
            seq_str = str(seq).upper().replace('-', '')  # Remove gaps
            if len(seq_str) < self.k:
                raise ValueError(f"Sequence {seq.id} is too short for K={self.k}.")
            unique_kmers.update(seq_str[i:i+self.k] for i in range(len(seq_str) - self.k + 1))
        
        self._kmer_to_index = {kmer: i for i, kmer in enumerate(sorted(unique_kmers))}
        self.n_features_ = len(self._kmer_to_index)
        self.fitted_ = True
        return self

    def _transform(self, X: ProteinSequences) -> np.ndarray:
        """
        Transform the protein sequences into K-mer embeddings.

        Args:
            X (ProteinSequences): The input protein sequences.

        Returns:
            np.ndarray: The K-mer embeddings for the sequences.
        """
        embeddings = np.zeros((len(X), self.n_features_), dtype=np.float32)

        for i, seq in enumerate(X):
            seq_str = str(seq).upper().replace('-', '')  # Remove gaps
            if len(seq_str) < self.k:
                raise ValueError(f"Sequence {seq.id} is too short for K={self.k}.")
            kmer_counts = defaultdict(int)
            for j in range(len(seq_str) - self.k + 1):
                kmer = seq_str[j:j+self.k]
                if kmer in self._kmer_to_index:
                    kmer_counts[self._kmer_to_index[kmer]] += 1
            
            for idx, count in kmer_counts.items():
                embeddings[i, idx] = count

        if self.normalize:
            row_sums = embeddings.sum(axis=1)
            embeddings = embeddings / row_sums[:, np.newaxis]

        return embeddings

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.

        Args:
            input_features (Optional[List[str]]): Ignored. Present for API consistency.

        Returns:
            List[str]: Output feature names.
        """
        return [f"kmer_{kmer}" for kmer in self._kmer_to_index.keys()]
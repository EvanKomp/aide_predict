# aide_predict/utils/msa.py
'''
* MSAProcessing class Refactored from Frazer et al. 
@article{Frazer2021DiseaseVP,
  title={Disease variant prediction with deep generative models of evolutionary data.},
  author={Jonathan Frazer and Pascal Notin and Mafalda Dias and Aidan Gomez and Joseph K Min and Kelly P. Brock and Yarin Gal and Debora S. Marks},
  journal={Nature},
  year={2021}
}

* Author: Evan Komp
* Created: 5/8/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology

Peocessing of MSAs for preparation of input data for the zero-shot model.
Note that The MSAProcessing class IS A REFACTORING  of the MSA processing class from The marks Lab https://github.com/OATML-Markslab/EVE/blob/master/utils/data_utils.py
Credit is given to them for the original implementation and the methodology of sequence weighting.
Here, we make it more pythonic and readbale, as well as an order of magnitude speed up.

In addition to refactoring, we add some additional functionality:
- A focus seq need not be present, in which case all columns are considered focus columns and contribute to weight computation
- One Hot encoding is reworked to use sklearn's OneHotEncoder instead of a loop of loops, with about an order of magnitude speedup
- Weight computation leverages numpy array indexing instead of a loop, and if torch is available
  and advanced hardware is present, GPU is used.
      Tested on 10000 protein sequences sequences of length 55:
        - original: 8.9 seconds
        - cpu array operations: 1.2 seconds (7.4x speedup)
        - gpu array operations: 0.2 seconds (44.5x speedup)
- other minor speedups with array operations
'''
import numpy as np
from typing import Optional
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.utils.constants import AA_SINGLE, GAP_CHARACTERS
from aide_predict import OneHotAlignedEmbedding
import copy

import logging
logger = logging.getLogger(__name__)


class MSAProcessing:
    def __init__(self, 
                 theta: float = 0.2,
                 use_weights: bool = True,
                 preprocess_msa: bool = True,
                 threshold_sequence_frac_gaps: float = 0.5,
                 threshold_focus_cols_frac_gaps: float = 0.3,
                 remove_sequences_with_indeterminate_aa_in_focus_cols: bool = True,
                 weight_computation_batch_size: int = 10000):
        """
        Initialize the MSAProcessing class.

        Args:
            theta (float): Sequence weighting hyperparameter.
            use_weights (bool): Whether to compute and use sequence weights.
            preprocess_msa (bool): Whether to preprocess the MSA.
            threshold_sequence_frac_gaps (float): Threshold for removing sequences with too many gaps.
            threshold_focus_cols_frac_gaps (float): Threshold for determining focus columns.
            remove_sequences_with_indeterminate_aa_in_focus_cols (bool): Whether to remove sequences with indeterminate AAs in focus columns.
            weight_computation_batch_size (int): Batch size for weight computation.
        """
        self.theta = theta
        self.use_weights = use_weights
        self.preprocess_msa = preprocess_msa
        self.threshold_sequence_frac_gaps = threshold_sequence_frac_gaps
        self.threshold_focus_cols_frac_gaps = threshold_focus_cols_frac_gaps
        self.remove_sequences_with_indeterminate_aa_in_focus_cols = remove_sequences_with_indeterminate_aa_in_focus_cols
        self.weight_computation_batch_size = weight_computation_batch_size
        
        logger.debug(f"MSAProcessing initialized with parameters: theta={theta}, use_weights={use_weights}, "
                     f"preprocess_msa={preprocess_msa}, threshold_sequence_frac_gaps={threshold_sequence_frac_gaps}, "
                     f"threshold_focus_cols_frac_gaps={threshold_focus_cols_frac_gaps}, "
                     f"remove_sequences_with_indeterminate_aa_in_focus_cols={remove_sequences_with_indeterminate_aa_in_focus_cols}, "
                     f"weight_computation_batch_size={weight_computation_batch_size}")

    def process(self, msa: ProteinSequences, focus_seq_id: Optional[str] = None) -> ProteinSequences:
        """
        Process the input MSA.

        Args:
            msa (ProteinSequences): The input multiple sequence alignment.
            focus_seq_id (Optional[str]): The ID of the focus sequence. If None, no focus sequence is used.

        Returns:
            ProteinSequences: The processed MSA with computed weights.
        """
        logger.info(f"Starting MSA processing with {len(msa)} sequences")
        
        if focus_seq_id is None:
            logger.debug("No focus_seq_id provided, processing without focus")
            self.focus_seq = None
        else:
            logger.debug(f"Using provided focus_seq_id: {focus_seq_id}")
            self.focus_seq = msa[focus_seq_id]
            logger.debug(f"Focus sequence length: {len(self.focus_seq)}")
        
        if self.preprocess_msa:
            logger.info("Preprocessing MSA")
            msa = self._preprocess_msa(msa)
        else:
            logger.info("Skipping MSA preprocessing")
        
        if self.focus_seq:
            self.focus_cols = self._get_focus_columns(msa[focus_seq_id])
            logger.debug(f"Number of focus columns: {np.sum(self.focus_cols)}")
        else:
            self.focus_cols = np.ones(msa.width, dtype=bool)
            logger.debug("No focus sequence, all columns are considered focus columns")
        
        if self.use_weights:
            logger.info("Computing sequence weights")
            weights = self._compute_weights(msa)
            msa.weights = weights
            logger.debug(f"Weight statistics: min={np.min(weights):.4f}, max={np.max(weights):.4f}, "
                         f"mean={np.mean(weights):.4f}, std={np.std(weights):.4f}")
        else:
            logger.info("Skipping weight computation")
        
        logger.info("MSA processing completed")
        return msa

    def _preprocess_msa(self, msa: ProteinSequences) -> ProteinSequences:
        """
        Preprocess the MSA by removing sequences with too many gaps and identifying focus columns.

        Args:
            msa (ProteinSequences): The input MSA.

        Returns:
            ProteinSequences: The preprocessed MSA.
        """
        logger.debug(f"Starting MSA preprocessing with {len(msa)} sequences")
        
        msa_array = msa.as_array()
        ids = np.array([seq.id for seq in msa])
        
        if self.focus_seq:
            focus_seq_array = np.array(list(str(self.focus_seq)))
            non_gap_cols = focus_seq_array != '-'
            msa_array = msa_array[:, non_gap_cols]
            logger.debug(f"Number of non-gap columns in focus sequence: {np.sum(non_gap_cols)}")
        
        logger.debug(f"MSA shape: {msa_array.shape}")
        
        seq_gaps_frac = np.mean(msa_array == '-', axis=1)
        seq_below_threshold = seq_gaps_frac <= self.threshold_sequence_frac_gaps
        msa_array = msa_array[seq_below_threshold]
        ids = ids[seq_below_threshold]
        logger.debug(f"Sequences removed due to high gap fraction: {np.sum(~seq_below_threshold)}")
        
        if self.focus_seq:
            columns_gaps_frac = np.mean(msa_array == '-', axis=0)
            focus_cols = columns_gaps_frac <= self.threshold_focus_cols_frac_gaps
            logger.debug(f"Number of focus columns: {np.sum(focus_cols)}")
            msa_array[:, ~focus_cols] = np.char.lower(msa_array[:, ~focus_cols])
        else:
            focus_cols = np.ones(msa_array.shape[1], dtype=bool)
            logger.debug("No focus sequence, all columns are considered focus columns")
        
        if self.remove_sequences_with_indeterminate_aa_in_focus_cols:
            AA_SINGLE_LOWER = [aa.lower() for aa in AA_SINGLE]
            valid_aa = set(AA_SINGLE).union(GAP_CHARACTERS).union(AA_SINGLE_LOWER)
            valid_sequences = np.all(np.isin(msa_array[:, focus_cols], list(valid_aa)), axis=1)
            msa_array = msa_array[valid_sequences]
            ids = ids[valid_sequences]
            logger.debug(f"Sequences removed due to indeterminate AAs: {np.sum(~valid_sequences)}")
        
        processed_msa = ProteinSequences([
            ProteinSequence(''.join(seq), id=ids[i])
            for i, seq in enumerate(msa_array)
        ])
        
        logger.info(f"Preprocessed MSA: {len(processed_msa)} sequences, {msa_array.shape[1]} columns")
        return processed_msa

    def _get_focus_columns(self, focus_seq: ProteinSequence) -> np.ndarray:
        """
        Identify focus columns in the focus sequence.

        Args:
            focus_seq (ProteinSequence): The focus sequence.

        Returns:
            np.ndarray: Boolean array indicating focus columns.
        """
        focus_seq_array = np.array(list(str(focus_seq)))
        is_uppercase = np.char.isupper(focus_seq_array)
        is_not_gap = focus_seq_array != '-'
        focus_cols = is_uppercase & is_not_gap
        logger.debug(f"Focus columns identified: {np.sum(focus_cols)} out of {len(focus_cols)}")
        return focus_cols

    def _compute_weights(self, msa: ProteinSequences) -> np.ndarray:
        """
        Compute sequence weights for the MSA.

        Args:
            msa (ProteinSequences): The input MSA.

        Returns:
            np.ndarray: Array of sequence weights.
        """
        logger.debug("Starting weight computation")
        
        ohe = OneHotAlignedEmbedding(metadata_folder=None, flatten=False)
        ohe.fit(msa)
        one_hot = ohe.transform(msa)
        logger.debug(f"One-hot encoding shape: {one_hot.shape}")
        
        one_hot = one_hot.reshape(len(msa), -1)
        logger.debug(f"Reshaped one-hot encoding: {one_hot.shape}")
        
        weights = []
        for i in range(0, len(one_hot), self.weight_computation_batch_size):
            batch = one_hot[i:i+self.weight_computation_batch_size]
            batch_weights = self._compute_weight_batch(batch, self.theta)
            weights.append(batch_weights)
            logger.debug(f"Computed weights for batch {i//self.weight_computation_batch_size + 1}: "
                         f"min={np.min(batch_weights):.4f}, max={np.max(batch_weights):.4f}")
        
        weights = np.concatenate(weights)
        neff = np.sum(weights)
        logger.info(f"Computed weights: Neff (effective sequence count) = {neff:.2f}")
        logger.debug(f"Final weight statistics: min={np.min(weights):.4f}, max={np.max(weights):.4f}, "
                     f"mean={np.mean(weights):.4f}, std={np.std(weights):.4f}")
        
        return weights

    @staticmethod
    def _compute_weight_batch(sequences: np.ndarray, theta: float) -> np.ndarray:
        """
        Compute weights for a batch of sequences.

        Args:
            sequences (np.ndarray): Batch of sequences in one-hot encoding.
            theta (float): Sequence weighting hyperparameter.

        Returns:
            np.ndarray: Array of sequence weights for the batch.
        """
        logger.debug(f"Computing weights for batch of shape {sequences.shape}")
        
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
            logger.debug(f"Using device: {device}")
            
            sequences = torch.from_numpy(sequences.astype(np.float32)).to(device)
            seq_dot_seq = torch.einsum('ij,ij->i', sequences, sequences)
            list_seq_dot_seq = sequences @ sequences.T
            denom = (list_seq_dot_seq / seq_dot_seq[:, None]).cpu().numpy()
        except ImportError:
            logger.debug("PyTorch not available, using NumPy for computations")
            seq_dot_seq = np.einsum('ij,ij->i', sequences, sequences)
            list_seq_dot_seq = sequences @ sequences.T
            denom = list_seq_dot_seq / seq_dot_seq[:, None]
        
        num_similar = np.sum(denom >= 1 - theta, axis=1)
        weights = 1.0 / num_similar
        logger.debug(f"Computed batch weights: min={np.min(weights):.4f}, max={np.max(weights):.4f}, "
                     f"mean={np.mean(weights):.4f}, std={np.std(weights):.4f}")
        return weights
    
    def get_most_populated_chunk(self, msa: ProteinSequences, chunk_size: int) -> ProteinSequences:
        """
        Get the most populated chunk of contiguous columns from the MSA.

        Args:
            msa (ProteinSequences): The input MSA.
            chunk_size (int): The size of the chunk.
            
        Returns:
            ProteinSequences: The chunk of contiguous columns.
        """

        msa_array = msa.as_array()
        # count non gap characters
        non_gap = ~np.isin(msa_array, GAP_CHARACTERS)
        non_gap_counts = np.sum(non_gap, axis=0)

        # check for the most populated chunk
        max_populated = 0
        max_start = 0
        for i in range(len(non_gap_counts) - chunk_size):
            populated = np.sum(non_gap_counts[i:i+chunk_size])
            if populated > max_populated:
                max_populated = populated
                max_start = i

        chunk = msa_array[:, max_start:max_start+chunk_size]
        sequences = []
        for i in range(len(msa)):
            sequences.append(ProteinSequence(''.join(chunk[i]), id=msa[i].id, structure=msa[i].structure))
        return ProteinSequences(sequences)
    

    def compute_conservation(self, msa, normalize=True):
        """
        Compute the conservation score for each column in the MSA.
        
        This method calculates the entropy-based conservation for each position in the alignment,
        with an option to normalize values between 0 (variable) and 1 (conserved).
        
        Parameters
        ----------
        msa : ProteinSequences
            The multiple sequence alignment to analyze.
        normalize : bool, optional (default=True)
            Whether to normalize entropy scores to range from 0 (variable) to 1 (conserved).
            
        Returns
        -------
        numpy.ndarray
            Vector of length L with conservation scores for each column.
        
        Notes
        -----
        - If sequence weights are available in the MSA, they will be used to calculate
        weighted frequencies for more accurate conservation measurement.
        - Conservation is calculated using the Shannon entropy of the amino acid distribution
        at each position, with an option to normalize to the [0,1] range.
        """
        import numpy as np
        from collections import Counter
        
        logger.debug(f"Computing conservation scores for MSA with {len(msa)} sequences, {msa.width} positions")
        
        # Get alignment as array for easier manipulation
        msa_array = msa.as_array()
        
        # Determine if we have sequence weights to use
        if hasattr(msa, 'weights') and msa.weights is not None:
            weights = msa.weights
            logger.debug("Using pre-computed sequence weights for conservation calculation")
        else:
            # Use uniform weights if none are available
            weights = np.ones(len(msa)) / len(msa)
            logger.debug("Using uniform sequence weights for conservation calculation")
        
        # Initialize conservation scores
        conservation = np.zeros(msa.width)
        
        # Calculate conservation for each column
        for i in range(msa.width):
            # Extract column
            col = msa_array[:, i]
            
            # Calculate weighted frequencies
            aa_freqs = {}
            for seq_idx, aa in enumerate(col):
                if aa not in aa_freqs:
                    aa_freqs[aa] = 0
                aa_freqs[aa] += weights[seq_idx]
            
            # Normalize frequencies to sum to 1
            total_weight = sum(aa_freqs.values())
            for aa in aa_freqs:
                aa_freqs[aa] /= total_weight
            
            # Convert to array for entropy calculation
            freq_array = np.array(list(aa_freqs.values()))
            
            # Calculate entropy
            X = freq_array[freq_array > 0]
            H = -np.sum(X * np.log2(X))
            
            # Normalize if requested
            if normalize:
                conservation[i] = 1 - (H / np.log2(len(freq_array)))
            else:
                conservation[i] = H
        
        logger.info(f"Conservation calculation complete: min={conservation.min():.4f}, max={conservation.max():.4f}, mean={conservation.mean():.4f}")
        return conservation

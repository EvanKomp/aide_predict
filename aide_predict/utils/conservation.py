# aide_predict/utils/conservation.py
'''
* Author: Evan Komp
* Created: 9/9/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from collections import Counter
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence


# see https://www.jalview.org/help/html/misc/aaproperties.html
JALVIEW_AA_ORDER = "ILVCAGMFYWHKREQDNSTPBZ"
PC_PROPERTIES_FROM_JALVIEW = {
    "Hydrophobic": "XXXXXXXXXXXX······X···",
    "Polar": "········XXXXXXXXXXX·XX",
    "Small": "··XXXX·········XXXXX··",
    "Proline": "···················X··",
    "Tiny": "····XX···········X····",
    "Aliphatic": "XXX················",
    "Aromatic": "·······XXXX···········",
    "Positive": "··········XXX·········",
    "Negative": "·············X·X······",
    "Charged": "··········XXXX·X······"
}
# convert to bool and then to dict
PC_PROPERTIES_MASKS = {prop: np.array([c == 'X' for c in mask]) for prop, mask in PC_PROPERTIES_FROM_JALVIEW.items()}
PC_PROPERRIES_DICT = {prop: set(JALVIEW_AA_ORDER[i] for i, is_prop in enumerate(mask) if is_prop) for prop, mask in PC_PROPERTIES_MASKS.items()}
# get the inverse of each property
PC_PROPERRIES_DICT.update({f"not_{prop}": set(JALVIEW_AA_ORDER[i] for i, is_prop in enumerate(mask) if not is_prop) for prop, mask in PC_PROPERTIES_MASKS.items()})


class ConservationAnalysis:
    """
    A class for analyzing amino acid property conservation in protein sequence alignments.

    This class provides methods to compute conservation scores and their statistical significance
    for various amino acid properties across aligned protein sequences. It can also compare
    conservation between two alignments.

    Attributes:
        PROPERTIES (Dict[str, set]): A dictionary mapping property names to sets of amino acids
            that possess that property.
        EXPECTED_FREQUENCIES (Dict[str, float]): A dictionary mapping property names to their
            expected frequencies based on the 20 standard amino acids.

    Args:
        protein_sequences (ProteinSequences): An aligned set of protein sequences.
        ignore_gaps (bool): Whether to ignore gaps in conservation calculations. Default is True.

    Raises:
        ValueError: If the input ProteinSequences object is not aligned.
    """

    PROPERTIES = PC_PROPERRIES_DICT
    EXPECTED_FREQUENCIES = {prop: len(aa_set) / 22 for prop, aa_set in PROPERTIES.items()}

    def __init__(self, protein_sequences: ProteinSequences, ignore_gaps: bool = True):
        if not protein_sequences.aligned:
            raise ValueError("Input ProteinSequences must be aligned.")
        self.sequences = protein_sequences
        self.ignore_gaps = ignore_gaps
        
        # Upper everything and replace any . with -
        array = protein_sequences.as_array()
        self.alignment_array = np.array([[aa.upper() if aa != '.' else '-' for aa in seq] for seq in array])
        valid_characters = JALVIEW_AA_ORDER + '-'
        # Raise error if any invalid characters
        if not all(aa in valid_characters for aa in np.unique(self.alignment_array)):
            raise ValueError(f"Invalid characters in alignment: {np.unique(self.alignment_array)}")

    def compute_conservation(self) -> Dict[str, np.ndarray]:
        """
        Compute conservation scores for each amino acid property across all alignment positions.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping property names to arrays of conservation
                scores. Each array has a length equal to the alignment width, with values
                between 0 and 1 representing the fraction of sequences that have the property
                at each position.
        """
        conservation_scores = {}
        
        for prop, aa_set in self.PROPERTIES.items():
            prop_mask = np.array([[aa in aa_set for aa in seq] for seq in self.alignment_array])

            if self.ignore_gaps:
                nongap_mask = self.alignment_array != '-'
                conservation_scores[prop] = np.sum(prop_mask & nongap_mask, axis=0) / np.sum(nongap_mask, axis=0)
            else:
                conservation_scores[prop] = np.mean(prop_mask, axis=0)
        
        return conservation_scores

    def compute_significance(self, alpha: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Compute the statistical significance of conservation for each property and position.

        This method uses a binomial test to compare the observed frequency of each property
        to its expected frequency based on amino acid composition.

        Args:
            alpha (float, optional): The significance level for the binomial test. Defaults to 0.01.

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: A tuple containing:
                1. A boolean array indicating significant positions (True if any property is significant).
                2. A dictionary mapping property names to arrays of p-values for each position.
        """
        p_values = {}
        
        for prop, expected_freq in self.EXPECTED_FREQUENCIES.items():
            observed_mask = np.array([[aa in self.PROPERTIES[prop] for aa in seq] for seq in self.alignment_array])
            
            if self.ignore_gaps:
                nongap_mask = self.alignment_array != '-'
                observed = np.sum(observed_mask & nongap_mask, axis=0)
                n_trials = np.sum(nongap_mask, axis=0)
            else:
                observed = np.sum(observed_mask, axis=0)
                n_trials = np.full(self.sequences.width, len(self.sequences))
            
            # Compute p-values using binomial test for each column
            p_values_ = np.array([
                stats.binom_test(x, n=n, p=expected_freq, alternative='greater')
                for x, n in zip(observed, n_trials)
            ])

            p_values[prop] = p_values_

        # Check if any p-values are significant column-wise
        p_values_vector = np.vstack(list(p_values.values()))
        significant_positions = np.any(p_values_vector < alpha, axis=0)
        
        return significant_positions, p_values

    @staticmethod
    def compare_alignments(alignment1: ProteinSequences, alignment2: ProteinSequences, ignore_gaps: bool = True, alpha: float = 0.01) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Compare conservation scores between two alignments and compute statistical significance.

        Args:
            alignment1 (ProteinSequences): The first aligned set of protein sequences.
            alignment2 (ProteinSequences): The second aligned set of protein sequences.
            ignore_gaps (bool): Whether to ignore gaps in conservation calculations. Default is True.
            alpha (float): The significance level for the binomial test. Default is 0.01.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: A tuple containing:
                1. A dictionary mapping property names to arrays of conservation score differences.
                2. A dictionary mapping property names to arrays of p-values for the differences.

        Raises:
            ValueError: If the two alignments have different lengths.
        """
        if alignment1.width != alignment2.width:
            raise ValueError("Alignments must have the same length")
        
        ca1 = ConservationAnalysis(alignment1, ignore_gaps=ignore_gaps)
        ca2 = ConservationAnalysis(alignment2, ignore_gaps=ignore_gaps)
        
        differences = {}
        p_values = {}
        
        for prop in ConservationAnalysis.PROPERTIES:
            prop_mask1 = np.array([[aa in ConservationAnalysis.PROPERTIES[prop] for aa in seq] for seq in ca1.alignment_array])
            prop_mask2 = np.array([[aa in ConservationAnalysis.PROPERTIES[prop] for aa in seq] for seq in ca2.alignment_array])
            
            if ignore_gaps:
                nongap_mask1 = ca1.alignment_array != '-'
                nongap_mask2 = ca2.alignment_array != '-'
                observed1 = np.sum(prop_mask1 & nongap_mask1, axis=0)
                observed2 = np.sum(prop_mask2 & nongap_mask2, axis=0)
                n_trials1 = np.sum(nongap_mask1, axis=0)
                n_trials2 = np.sum(nongap_mask2, axis=0)
            else:
                observed1 = np.sum(prop_mask1, axis=0)
                observed2 = np.sum(prop_mask2, axis=0)
                n_trials1 = np.full(alignment1.width, len(alignment1))
                n_trials2 = np.full(alignment2.width, len(alignment2))
            
            # Compute conservation score differences
            score1 = observed1 / n_trials1
            score2 = observed2 / n_trials2
            differences[prop] = score2 - score1

            # consider only positive conservation differences
            differences[prop][differences[prop] < 0] = 0
            
            # Compute p-values the second group having more conservation
            # use fisher exact
            p_values[prop] = np.array([
                stats.fisher_exact([[observed2[i], n_trials2[i] - observed2[i]], [observed1[i], n_trials1[i] - observed1[i]]], alternative='greater')[1]
                for i in range(alignment1.width)
            ])

        return differences, p_values
    
  
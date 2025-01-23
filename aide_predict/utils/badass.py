# aide_predict/utils/badass.py
'''
* Author: Evan Komp
* Created: 1/16/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Adaptation of the following work to use AIDE models as predictors:
https://www.biorxiv.org/content/10.1101/2024.10.25.620340v1
'''
import logging
from typing import Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from aide_predict.utils.common import MessageBool
from aide_predict.utils.data_structures import ProteinSequence, ProteinSequences
from aide_predict.bespoke_models.base import ProteinModelWrapper, CanRegressMixin
from dataclasses import dataclass, field
from typing import List, Tuple, Callable

logger = logging.getLogger(__name__)

try:
    from badass.mld.general_optimizer import GeneralProteinOptimizer
    AVAILABLE = MessageBool(True, "BADASS optimizer is available.")
except ImportError:
    AVAILABLE = MessageBool(False, "BADASS optimizer is not available. Install badass package.")

@dataclass
class BADASSOptimizerParams:
    """Parameters for the BADASS optimizer algorithm.
    
    Args:
        seqs_per_iter: Number of sequences to evaluate per iteration
        num_iter: Number of iterations to run
        init_score_batch_size: Batch size for initial scoring of single mutants
        temperature: Initial temperature for simulated annealing
        seed: Random seed
        gamma: Weight for variance boosting
        cooling_rate: Rate at which temperature decreases
        num_mutations: Number of mutations per sequence
        sites_to_ignore: Sites to exclude from mutation (1-indexed)
        normalize_scores: Whether to normalize scores
        simple_simulated_annealing: Use simple SA without adaptation
        cool_then_heat: Use cooling-then-heating schedule
        adaptive_upper_threshold: If float, use quantile. If int, use top N sequences
        n_seqs_to_keep: Number of sequences to keep in results
        score_threshold: Score threshold for phase transitions. If None, computed from data
        reversal_threshold: Score threshold for phase transition reversal. If None, computed
    """
    seqs_per_iter: int = 500
    num_iter: int = 200 
    init_score_batch_size: int = 500
    temperature: float = 1.5
    seed: int = 42
    gamma: float = 0.5
    cooling_rate: float = 0.92
    num_mutations: int = 5
    sites_to_ignore: List[int] = None  # 1-indexed
    normalize_scores: bool = True
    simple_simulated_annealing: bool = False
    cool_then_heat: bool = False
    adaptive_upper_threshold: Optional[Union[float, int]] = None
    n_seqs_to_keep: Optional[int] = None
    score_threshold: Optional[float] = None
    reversal_threshold: Optional[float] = None

    def __post_init__(self):
        # Convert None to empty list for sites_to_ignore
        if self.sites_to_ignore is None:
            self.sites_to_ignore = []

    def to_dict(self) -> dict:
        """Convert parameters to dictionary format expected by BADASS."""
        return {
            'seqs_per_iter': self.seqs_per_iter,
            'num_iter': self.num_iter,
            'init_score_batch_size': self.init_score_batch_size,
            'T': self.temperature,
            'seed': self.seed,
            'gamma': self.gamma,
            'cooling_rate': self.cooling_rate,
            'num_mutations': self.num_mutations,
            'sites_to_ignore': self.sites_to_ignore,
            'normalize_scores': self.normalize_scores,
            'simple_simulated_annealing': self.simple_simulated_annealing,
            'cool_then_heat': self.cool_then_heat,
            'adaptive_upper_threshold': self.adaptive_upper_threshold,
            'n_seqs_to_keep': self.n_seqs_to_keep,
            'score_threshold': self.score_threshold,
            'reversal_threshold': self.reversal_threshold
        }

class BADASSOptimizer:
    """Wrapper for the BADASS protein sequence optimizer.
    
    This class wraps the BADASS optimizer to work with AIDE's data structures
    and provides a simplified interface for optimization.
    
    The optimizer uses simulated annealing with adaptive temperature cycling
    to explore the protein sequence space, detecting phase transitions to
    balance exploration and exploitation.

    Args:
        predictor: Function that takes a list of sequences and returns scores
        reference_sequence: Reference/wild-type protein sequence
        params: Optimization parameters
        
    Example:
        ```python
        # Define predictor that takes list of sequences, returns scores
        def predict(sequences: List[str]) -> np.ndarray:
            return model.predict(sequences)
        
        # Create optimizer
        optimizer = BADASSOptimizer(
            predictor=predict,
            reference_sequence=wt,
            params=BADASSOptimizerParams(
                num_mutations=3,
                num_iter=100
            )
        )
        
        # Run optimization
        results, stats = optimizer.optimize()
        
        # Plot results
        optimizer.plot()
        ```
    """
    def __init__(
        self,
        predictor: Callable[[List[str]], np.ndarray],
        reference_sequence: ProteinSequence,
        params: BADASSOptimizerParams
    ):
        """Initialize the optimizer."""
        self.predictor = predictor
        self.reference_sequence = reference_sequence
        self.params = params
        
        # Create wrapped optimizer
        self._optimizer = GeneralProteinOptimizer(
            predictor=self._wrapped_predictor,
            ref_seq=str(reference_sequence),
            optimizer_params=params.to_dict()
        )

    def _wrapped_predictor(self, sequences: List[str]) -> List[float]:
        """Wrap the predictor to handle format conversion."""
        try:
            scores = self.predictor.predict(sequences)
            if isinstance(scores, np.ndarray):
                scores = scores.tolist()
            return scores
        except Exception as e:
            logger.error(f"Error in predictor: {str(e)}")
            raise

    def optimize(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the optimization process.
        
        Returns:
            tuple: (results_df, stats_df)
                - results_df: DataFrame with sequences and scores
                - stats_df: DataFrame with optimization statistics
        """
        results_df, stats_df = self._optimizer.optimize()
        
        # Convert relative mutation notation to full sequences
        results_df['full_sequence'] = results_df['sequences'].apply(
            lambda x: ProteinSequence(
                self._mutations_to_sequence(x),
                id=f"mut_{x}"  # Use mutations as ID
            )
        )
        
        return results_df, stats_df

    def _mutations_to_sequence(self, mutations: str) -> str:
        """Convert relative mutation notation to full sequence."""
        if not mutations:
            return str(self.reference_sequence)
            
        from badass.utils.sequence_utils import rel_sequences_to_dict, apply_mutations
        mutations_dict = rel_sequences_to_dict([mutations], sep='-')[0]
        return apply_mutations(str(self.reference_sequence), mutations_dict)

    def plot(self, save_figs: bool = True) -> None:
        """Generate visualization plots of the optimization process.
        
        Args:
            save_figs: Whether to save plots to files
        """
        self._optimizer.plot_scores(save_figs=save_figs)

    def save_results(self, filename: str = None) -> None:
        """Save optimization results to CSV files.
        
        Args:
            filename: Base filename for saving results
        """
        self._optimizer.save_results(filename=filename)

    @property 
    def results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get the latest optimization results.
        
        Returns:
            tuple: (results_df, stats_df) if optimize has been run,
                  (None, None) otherwise
        """
        if hasattr(self._optimizer, 'df'):
            return self._optimizer.df, self._optimizer.df_stats
        return None, None
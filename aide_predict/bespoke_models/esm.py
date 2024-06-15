# aide_predict/bespoke_models/esm.py
'''
* Author: Evan Komp
* Created: 6/14/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Using ESM as a zero shot evaluator.

ESM has a few methods for which to evaluate likelihoods, see the paper:
Meier, J. et al. Language models enable zero-shot prediction of the effects of mutations on protein function. Preprint at https://doi.org/10.1101/2021.07.09.450648 (2021).

The paper explored the following methods:
1. Masked Marginal Likelihood (masked_marginal) (Not yet implemented)
    Pass the wild type sequence L times, where L is the length of the sequence.
    Compute the likelihood of each AA at each position.
    Compare mutant vs wildtype AA at each position.

2. Mutant Marginal Likelihood (mutant_marginal) (Not yet implemented)
    Pass each variant sequence.
    N forward passes, where N is the count of variants.
    Compute the likelihood of mutated vs wildtype AA on each variant.
   
3. Wildtype Marginal Likelihood (wildtype_marginal)
    Pass the wild type sequence.
    1 forward pass, regardless of count of variants
    Compute the likelihood of mutated vs wildtype AA.

4. Psuedo-Likelihood (pseudo_likelihood) (Not implmented)
    No plans to implement, proved poor performance in the paper.

Since ESM is a transformer, it can output position specific scores. Recall that such a model must adhere to the following rules:
Inherits from PositionSpecificMixin, which enforces that `positions` is a parameter. We can use those positions to extract
likelihoods at specific positions. If `positions` is None, we will return all positions.

There is a lot of here. Let's lay out a logic table to determine how to be most efficient here.

| WT | Fixed Length | Positions passed | Pool | Method | N passes | Description
|----|--------------|------------------|------|--------|----| ----
| Y  | Y            | N                | Y    | masked | M unique mutated positions in whole set, < L | Traditional masked marginal as described in the paper. Take WT, mask each mutated position, compare to WT, pool
| Y  | Y            | N                | N    | masked | L | Can no longer only mask mutated positions since we are not pooling. Must mask all positions. This is L forward passes. Return comparison of mut to wt for each position individually. Many will be zero if they are not mutated anywhere.
| Y  | Y            | Y                | Y    | masked | Positions passed | Mask each position, compare to WT, pool
| Y  | Y            | Y                | N    | masked | Positions passed | Mask each position, compare to WT, no pooling output positions
| Y  | Y            | N                | Y    | wild_type | 1 | Traditional wild type marginal as described in the paper. Take WT and pass. Compare mutant likelihood to WT and pool only the mutated positions
| Y  | Y            | N                | N    | wild_type | 1 | Take WT and pass. Compare mutant likelihood to WT on WT probability vector. Many positions will be zero since they are unmutated
| Y  | Y            | Y                | Y    | wild_type | 1 | Take WT and pass. Compare mutant likelihood to WT on WT probability vector for only chosen positions. Pool.
| Y  | Y            | Y                | N    | wild_type | 1 | Take WT and pass. Compare mutant likelihood to WT on WT probability vector for only chosen positions. No pooling.
| Y  | Y            | N                | Y    | mutant | N | Traditional mutant marginal as described in the paper. Take each mutant and pass. Compare mutant likelihood to WT for only mutate positions on the mutant probability vector. Pool.
| Y  | Y            | N                | N    | mutant | N | Take each mutant and pass. Compare mutant likelihood to WT for all positions on the mutant probability vector many will be zero. No pooling.
| Y  | Y            | Y                | Y    | mutant | N | Take each mutant and pass. Compare mutant likelihood to WT on the mutant vector for positions specified.
| Y  | Y            | Y                | N    | mutant | N | Take each mutant and pass. Compare mutant likelihood to WT on the mutant vector for positions specified. No pooling.
| N  | Y            | N                | Y    | masked | L*N | Mask each position of each mutant, check probability of true AA at each position. Pool.
| N  | Y            | N                | N    | masked | L*N | Mask each position of each mutant, check probability of true AA at each position. No pooling.
| N  | Y            | Y                | Y    | masked | N * positions passed | Mask mutants on each position passed, check probability of true AA at each position. Pool.
| N  | Y            | Y                | N    | masked | N * positions passed | Mask mutants on each position passed, check probability of true AA at each position. No pooling.
| N  | Any          | Any              | Any  | wild_type | 0 | Not avialable. No wild type to compare to
| N  | Y            | N                | Y    | mutant | N | Pass each mutant, check probability of true AA at each position. Pool.
| N  | Y            | N                | N    | mutant | N | Pass each mutant, check probability of true AA at each position. No pooling.
| N  | Y            | Y                | Y    | mutant | N | Pass each mutant, check probability of true AA at only passed positions. Pool.
| N  | Y            | Y                | N    | mutant | N | Pass each mutant, check probability of true AA at only passed positions. No pooling.
| N  | N            | N                | Y    | masked | ~L*N | Mask each position of each mutant, check probability of true AA at each position. Pool.
| N  | N            | N                | N    | masked | 0 | Not available. Not pooling results in variabel length outputs.
| N  | N            | Y                | Y    | masked | 0 | Not available. Cannot specify positions with variable length sequences.
| N  | N            | Y                | N    | masked | 0 | Not available. Cannot specify positions with variable length sequences.
| N  | N            | N                | Y    | mutant | N | Pass each mutant, check probability of true AA at each position. Pool.
| N  | N            | N                | N    | mutant | 0 | Not available. Not pooling results in variabel length outputs.
| N  | N            | Y                | Y    | mutant | 0 | Not available. Cannot specify positions with variable length sequences.
| N  | N            | Y                | N    | mutant | 0 | Not available. Cannot specify positions with variable length sequences.
| Y  | N            | N                | Y    | masked | ~L*(N+1) | Mask each position of each mutant, check probability of true AA at each position. Pool. Repeat for WT and noramlize.
| Y  | N            | N                | N    | masked | 0 | Not available. Not pooling results in variabel length outputs.
| Y  | N            | Y                | Y    | masked | 0 | Not available. Cannot specify positions with variable length sequences.
| Y  | N            | Y                | N    | masked | 0 | Not available. Cannot specify positions with variable length sequences.
| Y  | N            | N                | Y    | wild_type | 0 | Not available. Wild type not same length as mutants, so you cannit look at mutant likelihood from wt pass.
| Y  | N            | N                | N    | wild_type | 0 | Not available. Wild type not same length as mutants, so you cannit look at mutant likelihood from wt pass.
| Y  | N            | Y                | Y    | wild_type | 0 | Not available. Wild type not same length as mutants, so you cannit look at mutant likelihood from wt pass.
| Y  | N            | Y                | N    | wild_type | 0 | Not available. Wild type not same length as mutants, so you cannit look at mutant likelihood from wt pass.
| Y  | N            | N                | Y    | mutant | N+1 | Pass each mutant, check probability of true AA at each position on its own probability vector. Pool. Normalize by WT value
| Y  | N            | N                | N    | mutant | 0 | Not available. Not pooling results in variabel length outputs.
| Y  | N            | Y                | Y    | mutant | 0 | Not available. Cannot specify positions with variable length sequences.
| Y  | N            | Y                | N    | mutant | 0 | Not available. Cannot specify positions with variable length sequences.

Conclusions:
  1. If Variable length sequences, must pool. Cannot pass positions. wild_type marginal not available
  2. If no wild type is given, only mutant or masked marginal is available.

Oh boy.

'''
from sklearn.base import TransformerMixin, RegressorMixin

from aide_predict.bespoke_models.base import ModelWrapper, PositionSpecificMixin

from aide_predict.utils.common import fixed_length_sequences

class ESMPredictorWrapper(PositionSpecificMixin, TransformerMixin, RegressorMixin, ModelWrapper):
    """Pretrained ESM as a log likelihood predictor.

    Params:
    - metadata_folder: str
        The folder containing the ESM model.
    - model_checkpoint: str
        Name of the ESM model to use
    - marginal_method: str
        The method to use for marginal likelihoods. One of 'masked_marginal', 'mutant_marginal', 'wildtype_marginal'
    - positions: list
        Not applicable unless sequences are fixed length. Species which positions in the sequence to use
    - pool: bool
        Whether to pool the likelihoods. Required for variable length sequences.
    - wt: str
        The wild type sequence. Required if comparing to wild type.
    - batch_size: int
        The batch size for inference. If you are hitting OOM, try reducing this.
    """
    _requires_wt_during_inference = True
    _requires_fixed_length = False
    _requires_msa = False
    _per_position_capable = True

    def __init__(
        self,
        metadata_folder: str=None,
        model_checkpoint: str='esm1v_t33_650M_UR90S_1',
        marginal_method: str='mutant_marginal',
        positions: list=None,
        pool: bool=True,
        wt: str=None,
        batch_size: int=4
    ):
        self.positions = positions
        self.pool = pool
        self.wt = wt
        self.model_checkpoint = model_checkpoint
        self.marginal_method = marginal_method
        self.batch_size = batch_size
        super().__init__(metadata_folder=metadata_folder)

    def _more_tags(self):
        return {'stateless': True,
                'preserves_dtype': [],
                }
    
    def fit(self, X, y=None):
        """Fit the model.
        
        Ignored. The model is already trained."""
        self.fitted_ = True
        return self


    def _check_fixed_length_sequences(self, X):
        if not fixed_length_sequences(X):
            return False
        if self.wt is not None and len(self.wt) != len(X[0]):
            return False
        return True

    def _assert_if_not_pooling_possible(self, X):
        """Check if not pooling is possible.
        
        If pooling is not possible, raise an error.
        """
        if not self.pool and not self._check_fixed_length_sequences(X):
            raise ValueError("Pooling is required with variable length sequences.")
        
    def _assert_if_positions_possible(self, X):
        """Check if positions are to specifify"""
        if self.positions is not None and not self._check_fixed_length_sequences(X):
            raise ValueError("Positions can only be specified with fixed length sequences.")

    def _call_esm(self, sequences):
        """Call the ESM model on sequences.
        
        Params:
        - sequences: list of amino acid sequences.
          These should be ready for the tokenizer
        """
        

    def _transform(self, X):
        """Get ESM scores for sequences.
        
        Params:
        - X: 2d np.ndarray of sequence amino acid strings
        """
        self._assert_if_not_pooling_possible(X)
        self._assert_if_positions_possible(X)

        # Start with the easier part to code... non fixed length sequences


        
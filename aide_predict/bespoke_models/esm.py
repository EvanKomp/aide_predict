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
  3. Masked marginal removed for the case where wt is not given or sequences are variable length.
     For these cases, masks will have to be applied to all sequences not just the WT, vastly increasing cost.

Oh boy.

'''
import warnings

from sklearn.base import TransformerMixin, RegressorMixin
import numpy as np
from tqdm import tqdm

from aide_predict.bespoke_models.base import PositionSpecificMixin, ProteinModelWrapper

from aide_predict.utils.common import fixed_length_sequences, mutated_positions

try:
    import torch
    from transformers import EsmForMaskedLM, AutoTokenizer
except ImportError:
    raise ImportError("You must install transformers to use this model. See `requirements-transformers.txt`.")

import logging
logger = logging.getLogger(__name__)

class ESMPredictorWrapper(PositionSpecificMixin, TransformerMixin, RegressorMixin, ProteinModelWrapper):
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
        model_checkpoint: str='esm2_t6_8M_UR50D',
        marginal_method: str='mutant_marginal',
        positions: list=None,
        pool: bool=True,
        wt: str=None,
        batch_size: int=2,
        device: str='cuda'
    ):
        self.positions = positions
        self.pool = pool
        self.model_checkpoint = model_checkpoint
        self.marginal_method = marginal_method
        self.batch_size = batch_size
        self.device = device

        self._model = None
        self._tokenizer = None
        
        super().__init__(metadata_folder=metadata_folder, wt=wt)
        logger.debug(f"ESM model inititalized with {self.__dict__}")

    @property
    def model_(self):
        if self._model is None:
            self._model = EsmForMaskedLM.from_pretrained('facebook/'+self.model_checkpoint).to(self.device)
        return self._model
    
    @property
    def tokenizer_(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained('facebook/'+self.model_checkpoint)
        return self._tokenizer          

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
    
    def _assert_if_marginal_available(self, X):
        """Check if the marginal method is available"""
        if self.marginal_method == 'wildtype_marginal' and not self._check_fixed_length_sequences(X):
            raise ValueError("Wildtype marginal is not available with variable length sequences.")
        if self.marginal_method == 'wildtype_marginal' and self.wt is None:
            raise ValueError("Wildtype marginal requires a wild type sequence.")
        if self.marginal_method == 'masked_marginal' and self.wt is None:
            raise ValueError("Masked marginal requires a wild type sequence.")
        if self.marginal_method == 'masked_marginal' and not self._check_fixed_length_sequences(X):
            raise ValueError("Masked marginal is not available with variable length sequences.")

    def _tokenize(self, sequences, on_device=True):
        """Tokenize the sequences.
        
        Params:
        - sequences: list of amino acid sequences. These should be ready for the tokenizer eg. including masks.
        """
        if on_device:
            return self.tokenizer_(sequences, add_special_tokens=True, return_tensors='pt', padding=True).to(self.device)
        else:
            return self.tokenizer_(sequences, add_special_tokens=False, return_tensors='np', padding=True)

    def _call_esm(self, sequences):
        """Call the ESM model on sequences.
        
        Params:
        - sequences: list of amino acid sequences. These should be ready for the tokenizer eg. indluding masks.
          These should be ready for the tokenizer
        """
        logger.info(f"Calling ESM on {len(sequences)} sequences, in {len(sequences)/self.batch_size} batches.")
        with torch.no_grad():
            with tqdm(total=len(sequences), desc="ESM forward passes...") as pbar:
                for i in range(0, len(sequences), self.batch_size):
                    tokenized = self._tokenize(sequences[i:i+self.batch_size], on_device=True)
                    # these tokens have ESM start tokens and end tokens
                    # get probabilities over amino acids
                    logits = self.model_(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits
                    # get log probabilities over amino acids
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    pbar.update(len(tokenized))

                    # return list of log probabilities, back to original sequence lengths
                    # use attention mask to get only the relevant positions
                    # if not all sequence lengths were the same, there will be masks
                    for j in range(len(tokenized.input_ids)):
                        log_probs_j = log_probs[j][tokenized.attention_mask[j].bool()].cpu().numpy()
                        # this has the extra index of the start and end tokens
                        # remove them
                        yield log_probs_j[1:-1]

    def _marginal_likelihood(self, X):
        """Get marginal likelihoods for sequences.

        This method is for full sequence marginals. For masked marginals, see `_masked_marginal_likelihood`.
        Here, returned log likelihoods are for each position of the passed sequence.
        
        Params:
        - X: 2d np.ndarray of sequence amino acid strings

        Returns:
        - list of np.ndarray of log likelihoods for each sequence
           each array is of shape (len sequence, vocab size)
        """
        # get ready for tokenizer
        new_sequences = [' '.join(x) for x in X]
        # call the model
        yield from self._call_esm(new_sequences)

    
    def _masked_marginal_likelihood(self, X, only_positions=None):
        """Get masked marginal likelihoods for sequences.

        This method is for masked marginals. For full sequence marginals, see `_marginal_likelihood`.
        Here, each position is masked individually. The log likelihood vector for each of those masks
        are concatenated. This requires a call for each position in each sequences.
        
        Params:
        - X: 2d np.ndarray of sequence amino acid strings
        - only_positions: list of int
            The positions to mask. If None, all positions are masked.

        Returns:
        - list of np.ndarray of log likelihoods for each sequence
           each array is of shape (len sequence, vocab size)
           if only_positions is not None, the shape is (len only_positions, vocab size)
        """
        # first, raise if not fixed length and only_positions is not None
        if only_positions is not None and not self._check_fixed_length_sequences(X):
            raise ValueError("Cannot specify positions to masked marginals with variable length sequences.")
        
        # since we have to run multiple passes for sequence, let's construct superbatches, one for
        # each input sequence
        for x in X:
            # prepare for tokenizer
            x = list(x)
            # determine positions to mask
            if only_positions is None:
                positions = range(len(x))
            else:
                positions = only_positions
            # construct the super batch
            super_batch = []
            for pos in positions:
                # mask the position
                masked = x.copy()
                masked[pos] = self.tokenizer_.mask_token
                super_batch.append(' '.join(masked))
            # call the model
            log_probs = list(self._call_esm(super_batch))
            log_probs = np.vstack([np.expand_dims(lp, axis=0) for lp in log_probs])
            # this is of shape (len positions, sequence length, vocab size)
            # we want to extract the probs from only the masked positions
            pivoted_log_probs = []
            for i, pos in enumerate(positions):
                pivoted_log_probs.append(log_probs[i, pos]) # these should be shape (vocab size,)
            # reshape so that it is an array of size (len positions, vocab size)
            yield np.vstack(pivoted_log_probs)
        

    def _transform(self, X):
        """Get ESM scores for sequences.
        
        Params:
        - X: 2d np.ndarray of sequence amino acid strings
        """
        self._assert_if_not_pooling_possible(X)
        self._assert_if_positions_possible(X)
        self._assert_if_marginal_available(X)

        fixed_length = self._check_fixed_length_sequences(X)

        # temporary warning for method testing
        if not self.marginal_method == 'masked_marginal' and not fixed_length:
            warnings.warn("The ESMWrapper has been tested using masked marginal likelihoods on fixed length sequences and it reproduced ProteinGym benchmark on ENVZ_ECOLI_Ghose_2023, however, it has not been tested on other methods or variable length sequences. Use at your own risk.")

        # start with easy one - WT marginal
        if self.marginal_method == 'wildtype_marginal':
            # we are gauranteed fixed length in this indent
            logger.info("Getting wild type marginal likelihoods, this requires 1 forward pass.")
            wild_type_log_probs = self._marginal_likelihood([self.wt])[0]
            # compare variants to wild type
            # first tokenize the sequences
            variants = [' '.join(x) for x in X]
            input_ids = self._tokenize(variants, on_device=False).input_ids
            # shape is (len X, sequence length)
            # index the wt loh probs with the input ids
            # this will give us the log likelihoods of the true AA at each position given the WT
            # output should be shape (len X, sequence length)
            rows = np.arange(wild_type_log_probs.shape[0])
            rows_expanded = np.expand_dims(rows, axis=0)
            variants_log_prob_vector = wild_type_log_probs[rows_expanded, input_ids]
            assert variants_log_prob_vector.shape == (len(X), len(X[0]))

            # subtract the wild type value
            wt_tokens = self._tokenize([self.wt], on_device=False).input_ids[0]
            wt_log_probs = wild_type_log_probs[wt_tokens]
            # subtract the wild type value
            variant_log_prob_vector -= wt_log_probs
            
            # if positions were passed, only return those positions
            if self.positions is not None:
                variants_log_prob_vector = variants_log_prob_vector[:, self.positions]
            
            if self.pool:
                return np.mean(variants_log_prob_vector, axis=1).reshape(-1, 1)
            else:
                return variants_log_prob_vector
            
        # mutant marginal
        elif self.marginal_method == 'mutant_marginal':
            # we are not guaranteed fixed length in this indent...
            # first tokenize the sequences and get log prob vectors for each
            logger.info(f"Getting mutant marginal likelihoods, this requires {len(X)} forward passes.")
            variants = [' '.join(x) for x in X]
            variant_log_probs = list(self._marginal_likelihood(variants))

            # tokenize WT if present, we will need it if comparing to WT
            if self.wt is not None:
                wt_ids = self._tokenize([self.wt], on_device=False).input_ids[0]

            # shape is (len X, sequence length, vocab size), note that 
            # sequence length may vary so it is actually a list of array
            # get the likelihood of the observed AA at each position
            variants_log_prob_vector = []
            for variant_log_prob in variant_log_probs:
                # tokenize the sequence
                variant_ids = self._tokenize(variants, add_special_tokens=False, return_tensors='np').input_ids
                assert len(variant_log_prob) == len(variant_ids)
                # get the log likelihood of the observed AA at each position
                rows = np.arange(len(variant_log_prob))
                rows_expanded = np.expand_dims(rows, axis=0)
                variant_log_prob_vector = variant_log_prob[rows_expanded, variant_ids]

                # if we have a wild type and we are fixed lenth, we need to determine
                # probability of wt AA on mutant vector
                if self.wt is not None and fixed_length:
                    # get the log likelihood of the observed AA at each position
                    wt_log_probs = variant_log_prob[rows_expanded, wt_ids]
                    # subtract the wild type value
                    variant_log_prob_vector -= wt_log_probs 
                
                assert variant_log_prob_vector.shape == (1, len(X[0]))

                # shape should be (1, sequence length)
                if self.positions is not None:
                    # we should be good here, if it is not None
                    # we have already checked fixed length sequences
                    variant_log_prob_vector = variant_log_prob_vector[:, self.positions]
                variants_log_prob_vector.append(variant_log_prob_vector)

            # this is a list of log prob values of the observed AA at each position
            # or a subset of positions
            # they arrays in the list need not be the same length
            # if sequences WERE the same length, we have already normalized
            # the log prob vector by WT. If they were not, we coulf not, so must now after pooling
            if self.wt is not None and not fixed_length:
                wt_on_wt_log_probs = self._marginal_likelihood([self.wt])[0]
                rows = np.arange(wt_on_wt_log_probs.shape[0])
                rows_expanded = np.expand_dims(rows, axis=0)
                wt_on_wt_log_probs_vector = wt_on_wt_log_probs[rows_expanded, wt_ids]
                wt_base_log_prob_pooled = np.mean(wt_on_wt_log_probs_vector)
                assert wt_base_log_prob_pooled.shape == (1,)
            else:
                wt_base_log_prob_pooled = 0

            # time to pool!
            if self.pool:
                # if fixed length, we have already normalized by WT
                variant_log_probs = np.array([np.mean(x, axis=1) for x in variants_log_prob_vector]).reshape(-1, 1)
                variant_log_probs -= wt_base_log_prob_pooled
                return variant_log_probs
            else:
                # we should be guaranteed fixed length here
                # if pool is False, we already checked that we are fixed length
                return np.vstack(variants_log_prob_vector)
            

        # masked marginal
        elif self.marginal_method == 'masked_marginal':
            # like WT marginal, we are guaranteed fixed length and wt is present
            # first get the positions that need to be masked
            if self.positions is not None:
                masked_positions = self.positions
            else:
                # determine where there are mutations
                masked_positions = mutated_positions(X)

            # get the masked marginal likelihoods
            logger.info(f"Getting masked marginal likelihoods for mutate positions, this requires {len(masked_positions)} forward passes.")
            masked_log_probs = list(self._masked_marginal_likelihood([self.wt], only_positions=masked_positions))[0]
            # this output is the size of the positions passed as opposed to the full length
            # rectify this so that they can be indexed by inserting zero rows
            masked_log_probs_full = np.zeros((len(masked_positions), masked_log_probs.shape[1]))
            masked_log_probs_full[masked_positions] = masked_log_probs

            # index the masked log probs with the input ids
            # of wt
            wt_ids = self._tokenize([self.wt], on_device=False).input_ids[0]
            rows = np.arange(masked_log_probs_full.shape[0])
            rows_expanded = np.expand_dims(rows, axis=0)
            wt_log_probs_vector = masked_log_probs_full[rows_expanded, wt_ids]

            # repeat for each variant
            variant_ids = self._tokenize([' '.join(x) for x in X], on_device=False).input_ids
            # shape is (len X, sequence length)
            # index the masked log probs with the input ids
            # this will give us the log likelihoods of the true AA at each position given the WT masked
            # output should be shape (len X, sequence length)
            rows = np.arange(masked_log_probs_full.shape[0])
            rows_expanded = np.expand_dims(rows, axis=0)
            variants_log_prob_vector = masked_log_probs_full[rows_expanded, variant_ids]
            assert variants_log_prob_vector.shape == (len(X), len(X[0]))

            # subtract the wild type value
            variants_log_prob_vector -= wt_log_probs_vector
            if self.pool:
                return np.mean(variants_log_prob_vector, axis=1).reshape(-1, 1)
            else:
                return variants_log_prob_vector
        

    def _predict(self, X):
        return self._transform(X)

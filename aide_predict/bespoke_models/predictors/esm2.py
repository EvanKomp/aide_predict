# aide_predict/bespoke_models/esm2.py
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

import numpy as np
from tqdm import tqdm

from aide_predict.bespoke_models.base import PositionSpecificMixin, ProteinModelWrapper, CanRegressMixin, CanHandleAlignedSequencesMixin, RequiresWTDuringInferenceMixin
from aide_predict.utils.data_structures import ProteinSequences
from aide_predict.utils.common import MessageBool

try:
    import torch
    from transformers import EsmForMaskedLM, AutoTokenizer
    AVAILABLE = MessageBool(True, "ESM2 model is available.")
except ImportError:
    AVAILABLE = MessageBool(False, "ESM2 model is not available. Please see the transformers requirements")

from typing import List, Optional, Tuple, Union

import logging
logger = logging.getLogger(__name__)

class ESM2LikelihoodWrapper(
    RequiresWTDuringInferenceMixin,
    CanHandleAlignedSequencesMixin,
    PositionSpecificMixin,
    CanRegressMixin,
    ProteinModelWrapper
):
    """
    Pretrained ESM as a log likelihood predictor.

    This class wraps the ESM2 model to use it as a log likelihood predictor for protein sequences.
    It supports multiple methods for calculating marginal likelihoods and can handle both fixed-length
    and variable-length sequences.

    Attributes:
        positions (Optional[List[int]]): Positions in the sequence to use for prediction.
        pool (bool): Whether to pool the likelihoods across positions.
        model_checkpoint (str): Name of the ESM model to use.
        marginal_method (str): Method to use for marginal likelihoods.
        batch_size (int): Batch size for inference.
        device (str): Device to use for computation ('cuda' or 'cpu').
    """
    _available = AVAILABLE

    def __init__(
        self,
        metadata_folder: str=None,
        model_checkpoint: str='esm2_t6_8M_UR50D',
        marginal_method: str='mutant_marginal',
        positions: list=None,
        pool: bool=True,
        flatten: bool=True,
        wt: str=None,
        batch_size: int=2,
        device: str='cuda'
    ):
        """
        Initialize the ESM2PredictorWrapper.

        Args:
            metadata_folder (Optional[str]): Folder to store metadata (not used in this implementation).
            model_checkpoint (str): Name of the ESM model to use.
            marginal_method (str): Method to use for marginal likelihoods ('masked_marginal', 'mutant_marginal', or 'wildtype_marginal').
            positions (Optional[List[int]]): Positions in the sequence to use for prediction.
            pool (bool): Whether to pool the likelihoods across positions.
            wt (Optional[Union[str, ProteinSequence]]): Wild type sequence.
            batch_size (int): Batch size for inference.
            device (str): Device to use for computation ('cuda' or 'cpu').
        """
        self.model_checkpoint = model_checkpoint
        self.marginal_method = marginal_method
        self.batch_size = batch_size
        self.device = device

        self.model_ = None
        self.tokenizer_ = None
        
        super().__init__(metadata_folder=metadata_folder, wt=wt, positions=positions, pool=pool, flatten=flatten)
        logger.debug(f"ESM model inititalized with {self.__dict__}")        

    def _more_tags(self):
        return {'stateless': True,
                'preserves_dtype': [],
                }
    
    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'ESM2PredictorWrapper':
        """
        Fit the model (no-op for pre-trained models).

        Args:
            X (ProteinSequences): The input sequences.
            y (Optional[np.ndarray]): Ignored. Present for API consistency.

        Returns:
            ESM2PredictorWrapper: The fitted model.
        """
        _fit_transformers_esm(X)
        self.model_ = EsmForMaskedLM.from_pretrained('facebook/'+self.model_checkpoint).to(self.device)
        self.tokenizer_ = AutoTokenizer.from_pretrained('facebook/'+self.model_checkpoint)
        return self

    def _assert_if_not_pooling_possible(self, X: ProteinSequences) -> None:
        """
        Check if pooling is possible.
        
        Args:
            X (ProteinSequences): The input sequences.

        Raises:
            ValueError: If pooling is not possible with variable length sequences.
        """
        if not self.pool and not self._check_fixed_length(X):
            raise ValueError("Pooling is required with variable length sequences.")
        
    def _assert_if_positions_possible(self, X: ProteinSequences) -> None:
        """
        Check if specifying positions is possible.

        Args:
            X (ProteinSequences): The input sequences.

        Raises:
            ValueError: If positions cannot be specified with unaligned variable length sequences.
        """
        if self.positions is not None and not self._check_fixed_length(X):
            if X.aligned:
                logger.warning("Positions were passed for aligned variable length sequences. Positions are being interpreted as aligned positions.")
            else:
                raise ValueError("Cannot specify positions with unaligned variable length sequences.")
    
    def _assert_if_marginal_available(self, X: ProteinSequences) -> None:
        """
        Check if the marginal method is available.

        Args:
            X (ProteinSequences): The input sequences.

        Raises:
            ValueError: If the chosen marginal method is not available for the given input.
        """
        if self.marginal_method == 'wildtype_marginal' and not self._check_fixed_length(X):
            raise ValueError("Wildtype marginal is not available with variable length sequences.")
        if self.marginal_method == 'wildtype_marginal' and self.wt is None:
            raise ValueError("Wildtype marginal requires a wild type sequence.")
        if self.marginal_method == 'masked_marginal' and self.wt is None:
            raise ValueError("Masked marginal requires a wild type sequence.")
        if self.marginal_method == 'masked_marginal' and not self._check_fixed_length(X):
            raise ValueError("Masked marginal is not available with variable length sequences.")

    def _tokenize(self, sequences: List[str], on_device: bool = True) -> Union[torch.Tensor, np.ndarray]:
        """
        Tokenize the sequences.
        
        Args:
            sequences (List[str]): List of amino acid sequences. These should be ready for the tokenizer, e.g., including masks.
            on_device (bool): Whether to put the tokenized sequences on the specified device.

        Returns:
            Union[torch.Tensor, np.ndarray]: Tokenized sequences.
        """
        if on_device:
            return self.tokenizer_(sequences, add_special_tokens=True, return_tensors='pt', padding=True).to(self.device)
        else:
            return self.tokenizer_(sequences, add_special_tokens=False, return_tensors='np', padding=True)

    def _call_esm(self, prepared_sequences: List[str]) -> np.ndarray:
        """
        Call the ESM model on prepared sequences.
        
        Args:
            prepared_sequences (List[str]): List of prepared amino acid sequences.

        Returns:
            np.ndarray: Log probabilities for each sequence.
        """
        logger.info(f"Calling ESM on {len(prepared_sequences)} sequences.")
        with torch.no_grad():
            tokenized = self._tokenize(prepared_sequences, on_device=True)
            logits = self.model_(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            results = []
            for j in range(len(tokenized.input_ids)):
                log_probs_j = log_probs[j][tokenized.attention_mask[j].bool()].cpu().numpy()
                results.append(log_probs_j[1:-1])  # Remove start and end tokens

        return results

    def _prepare_marginal(self, batch: ProteinSequences) -> List[str]:
        """
        Prepare sequences for wildtype marginal method.
        
        Args:
            batch (ProteinSequences): A batch of protein sequences.
        
        Returns:
            List[str]: Prepared sequences for ESM model input.
        """
        return [' '.join(str(seq)) for seq in batch]
    
    
    def _prepare_masked_marginal(self, batch: ProteinSequences) -> Tuple[List[str], List[List[int]]]:
        """
        Prepare sequences for masked marginal method.

        Note that sequences passed here are used only to determine mutated postitions.
        It is the wild type that is masked and passed to the model.
        
        Args:
            batch (ProteinSequences): A batch of protein sequences.
        
        Returns:
            Tuple[List[str], List[List[int]]]: Prepared sequences for ESM model input and mutated positions.
        """
        prepared_sequences_batch = []

        # loop through all sequences and capture all changing positions
        if self.positions is not None:
            positions_to_mask = self.positions
        else:
            positions_to_mask = []
            for seq in batch:
                positions_to_mask.extend(seq.mutated_positions(self.wt))
                positions_to_mask = list(set(positions_to_mask))

        batch_positions = []
        for pos in positions_to_mask:
            masked_seq = list(str(self.wt))
            masked_seq[pos] = self.tokenizer_.mask_token
            prepared_sequences_batch.append(' '.join(masked_seq))
            batch_positions.append(pos)

            if len(prepared_sequences_batch) == self.batch_size:
                yield prepared_sequences_batch, batch_positions
                prepared_sequences_batch = []
                batch_positions = []
        if len(prepared_sequences_batch) > 0:
            yield prepared_sequences_batch, batch_positions


    def _transform(self, X: ProteinSequences) -> np.ndarray:
        """
        Get ESM scores for sequences.
        
        Args:
            X (ProteinSequences): Input sequences.

        Returns:
            np.ndarray: Transformed sequences.

        Raises:
            ValueError: If the input does not meet the requirements for the chosen marginal method.
        """
        self._assert_if_not_pooling_possible(X)
        self._assert_if_positions_possible(X)
        self._assert_if_marginal_available(X)

        fixed_length = self._check_fixed_length(X)
        results = []


        if self.marginal_method == 'wildtype_marginal':
            logger.info("Getting wild type marginal likelihoods. This requires 1 forward pass.")
            prepared_wt = self._prepare_marginal([self.wt])
            wt_log_probs = self._call_esm(prepared_wt)[0]
            wt_tokens = self._tokenize([' '.join(str(self.wt))], on_device=False).input_ids[0]
            rows = np.arange(wt_log_probs.shape[0])
            rows_expanded = np.expand_dims(rows, axis=0)
            wt_log_probs_vector = wt_log_probs[rows_expanded, wt_tokens]

            # this is of shape (len wt, vocab size
            # compare each sequence to this
            for batch in X.iter_batches(self.batch_size):
                batch_prepared = self._prepare_marginal(batch)
                batch_input_ids = self._tokenize(batch_prepared, on_device=False).input_ids

                batch_log_prob_vector = wt_log_probs[rows_expanded, batch_input_ids]
                # of shape 
                batch_log_prob_vector -= wt_log_probs_vector

                # this should be of shape (len batch, sequence length)
                if self.positions is not None:
                    batch_log_prob_vector = batch_log_prob_vector[:, self.positions]
                
                if self.pool:
                    # we should only pool over changed positions, otherwise
                    # variants with more mutations have inherantly higher scores
                    # by virtue of having fewer zeros
                    masks = [x.mutated_positions(self.wt) for x in batch]
                    batch_log_prob_vector = [log_probs[mask] for log_probs, mask in zip(batch_log_prob_vector, masks)]
                    results.extend(list(np.mean(batch_log_prob_vector, axis=1)))
                else:
                    results.extend(list(batch_log_prob_vector))
        
        elif self.marginal_method == 'masked_marginal':
            logger.info("Getting masked marginal likelihoods for mutate positions. This requires a forward pass for each mutated position.")
            masked_wt_log_probs = np.zeros((len(self.wt), self.tokenizer_.vocab_size))
            wt_tokens = self._tokenize([' '.join(str(self.wt))], on_device=False).input_ids[0]
            for batch_wt_masked, masked_positions in self._prepare_masked_marginal(X):
                batch_log_probs = self._call_esm(batch_wt_masked)
                # of shape (len batch, sequence length, vocab size)
                for i, log_probs in enumerate(batch_log_probs):
                    masked_wt_log_probs[masked_positions[i]] = log_probs[masked_positions[i]]
            rows = np.arange(masked_wt_log_probs.shape[0])
            rows_expanded = np.expand_dims(rows, axis=0)
            wt_log_probs_vector = masked_wt_log_probs[rows_expanded, wt_tokens]
            
            # index the masked log probs with the input ids
            # of variant
            for batch in X.iter_batches(self.batch_size):
                batch_prepared = self._prepare_marginal(batch)
                batch_input_ids = self._tokenize(batch_prepared, on_device=False).input_ids

                batch_log_prob_vector = masked_wt_log_probs[rows_expanded, batch_input_ids]
                batch_log_prob_vector -= wt_log_probs_vector

                if self.positions is not None:
                    batch_log_prob_vector = batch_log_prob_vector[:, self.positions]

                if self.pool:
                    # we should only pool over changed positions, otherwise 
                    # variants with more mutations have inherantly higher scores
                    # by virtue of having fewer zeros
                    masks = [x.mutated_positions(self.wt) for x in batch]
                    batch_log_prob_vector = [log_probs[mask] for log_probs, mask in zip(batch_log_prob_vector, masks)]

                    results.extend(list(np.mean(batch_log_prob_vector, axis=1)))
                else:
                    results.extend(list(batch_log_prob_vector))

        elif self.marginal_method == 'mutant_marginal':
            logger.info("Getting mutant marginal likelihoods. This a forward pass for each sequence.")
            if self.wt is not None:
                wt_ids = self._tokenize([' '.join(str(self.wt))], on_device=False).input_ids[0]

            for batch in X.iter_batches(self.batch_size):
                batch_prepared = self._prepare_marginal(batch)
                batch_log_probs = self._call_esm(batch_prepared)
                # of shape (len batch, sequence length, vocab size)
                # these maye be different sequence lengths
                for i, log_probs in enumerate(batch_log_probs):
                    # get the ids so we can index as a vector
                    variant_input_ids = self._tokenize([batch_prepared[i]], on_device=False).input_ids[0]
                    rows = np.arange(log_probs.shape[0])
                    rows_expanded = np.expand_dims(rows, axis=0)
                    variant_log_prob_vector = log_probs[rows_expanded, variant_input_ids]
                    if self.wt is not None and fixed_length:
                        wt_log_probs_vector = log_probs[rows_expanded, wt_ids]
                        variant_log_prob_vector -= wt_log_probs
                    
                    if self.positions is not None:
                        variant_log_prob_vector = variant_log_prob_vector[self.positions]

                    if self.pool:
                        results.append(np.mean(variant_log_prob_vector))

                    else:
                        results.append(variant_log_prob_vector)

            # if we gave wt but not fixed length, we need to normalize by wt
            if self.wt is not None and not fixed_length:
                assert self.pool, "Something went wrong, we should only be here if we are pooling"
                wt_on_wt_log_probs = self._call_esm([' '.join(str(self.wt))])[0]
                rows = np.arange(wt_on_wt_log_probs.shape[0])
                rows_expanded = np.expand_dims(rows, axis=0)
                wt_on_wt_log_probs_vector = wt_on_wt_log_probs[rows_expanded, wt_ids]
                wt_base_log_prob_pooled = np.mean(wt_on_wt_log_probs_vector)
                results = [x - wt_base_log_prob_pooled for x in results]
        else:
            raise ValueError(f"Marginal method {self.marginal_method} not recognized.")

        return np.array(results)



def _fit_transformers_esm(X: ProteinSequences):
    """"""
    warnings.warn("Finetuning ESM models is not yet supported. Using Pretrained model weights.")
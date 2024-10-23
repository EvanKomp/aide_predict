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

from aide_predict.bespoke_models.base import RequiresFixedLengthMixin, CacheMixin
from aide_predict.bespoke_models.predictors.pretrained_transformers import LikelihoodTransformerBase, MarginalMethod
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
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


class ESM2LikelihoodWrapper(CacheMixin, RequiresFixedLengthMixin, LikelihoodTransformerBase):
    _available = AVAILABLE

    def __init__(
        self,
        metadata_folder: str = None,
        model_checkpoint: str = 'esm2_t6_8M_UR50D',
        marginal_method: MarginalMethod = MarginalMethod.MUTANT.value,
        positions: list = None,
        pool: bool = True,
        flatten: bool = True,
        wt: str = None,
        batch_size: int = 2,
        device: str = 'cpu',
        use_cache: bool = True
    ):
        super().__init__(
            metadata_folder=metadata_folder,
            marginal_method=marginal_method,
            positions=positions,
            pool=pool,
            flatten=flatten,
            wt=wt,
            batch_size=batch_size,
            device=device,
            use_cache=use_cache
        )
        self.model_checkpoint = model_checkpoint
        logger.debug(f"ESM2 model initialized with {self.__dict__}")

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'ESM2LikelihoodWrapper':
        self.fitted_ = True
        return self
    
    def _load_model(self) -> None:
        """Load and the model and other objects into memory on device such that they can be accessed in
        `_compute_log_likelihoods` and `_index_log_probs`.

        Required abstract class from `LikelihoodTransformerBase`.
        """
        self.model_ = EsmForMaskedLM.from_pretrained('facebook/'+self.model_checkpoint).to(self.device)
        self.tokenizer_ = AutoTokenizer.from_pretrained('facebook/'+self.model_checkpoint)
    
    
    def _cleanup_model(self) -> None:
        """
        Clean up the model and other objects loaded into memory in `_load_model`.

        Required abstract class from `LikelihoodTransformerBase`.
        """
        del self.model_
        del self.tokenizer_

    def _tokenize(self, sequences: List[str], on_device: bool=True) -> "torch.Tensor":
        if on_device:
            return self.tokenizer_(sequences, add_special_tokens=True, return_tensors='pt', padding=True).to(self.device)

        elif not on_device:
            return self.tokenizer_(sequences, add_special_tokens=False, return_tensors='np', padding=True)

    def _compute_log_likelihoods(self, X: ProteinSequences, mask_positions: List[List[int]] = None) -> List[np.ndarray]:
        prepared_sequences = [list(str(seq).upper()) for seq in X]

        if mask_positions:
            if not len(mask_positions) == len(X):
                raise ValueError("Mask positions must be the same length as the input sequences.")
            for i, seq in enumerate(prepared_sequences):
                positions = mask_positions[i]
                for pos in positions:
                    seq[pos] = self.tokenizer_.mask_token
        
        prepared_sequences = [' '.join(seq) for seq in prepared_sequences]
        
        with torch.no_grad():
            tokenized = self._tokenize(prepared_sequences, on_device=True)
            logits = self.model_(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits
            # shape (batch size, sequence length (padded), vocab size)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            results = []
            for j in range(len(tokenized.input_ids)):
                log_probs_j = log_probs[j][tokenized.attention_mask[j].bool()].cpu().numpy()
                results.append(log_probs_j[1:-1])  # Remove start and end tokens

        return results

    def _index_log_probs(self, log_probs: np.ndarray, sequences: ProteinSequences) -> np.ndarray:
        tokenized = self._tokenize([' '.join(str(seq)) for seq in sequences], on_device=False)
        token_ids = tokenized.input_ids
        
        rows = np.arange(log_probs.shape[0])
        rows_expanded = np.expand_dims(rows, axis=0)
        
        return log_probs[rows_expanded, token_ids]

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return super().get_feature_names_out(input_features)



def _fit_transformers_esm(X: ProteinSequences):
    """"""
    warnings.warn("Finetuning ESM models is not yet supported. Using Pretrained model weights.")

# aide_predict/bespoke_models/predictors/esm_if.py
'''
* Author: Evan Komp
* Created: 2026-05-26
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Wrapper around ESM-IF1 (ESM Inverse Folding), the GVP-Transformer from fair-esm.

ESM-IF1 is autoregressive and conditioned on backbone structure. Wildtype and
mutant marginals are well-defined; masked_marginal is not (no bidirectional
mask scoring mode) and is refused at __init__.

For multichain complexes, set ``context_chains`` on the attached
ProteinStructure (or call ``set_target_chain`` / ``with_target_chain``). The
wrapper concatenates context-chain coords with the standard 10-residue
NaN-padded gap used by fair-esm's ``score_sequence_in_complex``; the decoder
only emits predictions for the target chain.

See:
Hsu, C. et al. Learning inverse folding from millions of predicted structures.
ICML 2022. https://doi.org/10.1101/2022.04.10.487779
'''
import numpy as np
from typing import List, Optional, Union

from aide_predict.bespoke_models.base import RequiresStructureMixin, RequiresWTToFunctionMixin
from aide_predict.bespoke_models.predictors.pretrained_transformers import LikelihoodTransformerBase, MarginalMethod
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence, ProteinStructure
from aide_predict.utils.common import MessageBool

try:
    import torch
    import torch.nn.functional as F
    import esm
    from esm.inverse_folding.util import CoordBatchConverter
    from esm.inverse_folding.multichain_util import _concatenate_coords
    AVAILABLE = MessageBool(True, "ESM-IF is available.")
except ImportError:
    AVAILABLE = MessageBool(
        False,
        "ESM-IF requires fair-esm and torch_geometric. Install via requirements_files/requirements-esm-if.txt",
    )

import logging
logger = logging.getLogger(__name__)


class ESMIFLikelihoodWrapper(RequiresStructureMixin, RequiresWTToFunctionMixin, LikelihoodTransformerBase):
    """
    ESM-IF1 zero-shot variant scorer.

    Inherits ExpectsNoFitMixin, RequiresFixedLengthMixin, CanRegressMixin,
    RequiresWTDuringInferenceMixin, PositionSpecificMixin, CacheMixin from
    LikelihoodTransformerBase. Adds RequiresStructureMixin and
    RequiresWTToFunctionMixin so callers must supply a WT with attached
    ProteinStructure.
    """
    _available = AVAILABLE

    def __init__(
        self,
        metadata_folder: str = None,
        model_checkpoint: str = 'esm_if1_gvp4_t16_142M_UR50',
        marginal_method: Union[MarginalMethod, str] = MarginalMethod.WILDTYPE.value,
        positions: Optional[List[int]] = None,
        pool: bool = True,
        flatten: bool = False,
        wt: Optional[Union[str, ProteinSequence]] = None,
        batch_size: int = 1,
        use_cache: bool = False,
        device: str = 'cpu',
    ):
        if isinstance(marginal_method, MarginalMethod):
            marginal_method = marginal_method.value
        if marginal_method == MarginalMethod.MASKED.value:
            raise ValueError(
                "ESM-IF is autoregressive; masked_marginal is not defined. "
                "Use 'wildtype_marginal' or 'mutant_marginal'."
            )
        super().__init__(
            metadata_folder=metadata_folder,
            marginal_method=marginal_method,
            positions=positions,
            pool=pool,
            flatten=flatten,
            use_cache=use_cache,
            wt=wt,
            batch_size=batch_size,
            device=device,
        )
        self.model_checkpoint = model_checkpoint

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'ESMIFLikelihoodWrapper':
        self.fitted_ = True
        return self

    def _load_model(self) -> None:
        loader = getattr(esm.pretrained, self.model_checkpoint)
        self.model_, self.alphabet_ = loader()
        self.model_ = self.model_.eval().to(self.device)
        self.batch_converter_ = CoordBatchConverter(self.alphabet_)
        self._aa_to_index = {aa: self.alphabet_.get_idx(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
        self._vectorized_aa_to_index = np.vectorize(lambda x: self._aa_to_index.get(x, -1))

    def _cleanup_model(self) -> None:
        del self.model_
        del self.alphabet_
        del self.batch_converter_

    def _resolve_structure(self, seq: ProteinSequence) -> ProteinStructure:
        if seq.structure is not None:
            return seq.structure
        if self.wt is not None and self.wt.structure is not None:
            return self.wt.structure
        raise ValueError(
            "ESM-IF requires structure; neither the input sequence nor the WT carries a ProteinStructure."
        )

    def _build_coords(self, structure: ProteinStructure) -> np.ndarray:
        """Assemble coordinates with target chain first; multichain joins via fair-esm's NaN-gap convention."""
        target_coords = structure.get_chain_coords(structure.chain)
        if structure.context_chains is None:
            return target_coords
        coords_dict = {structure.chain: target_coords}
        for ctx in structure.context_chains:
            coords_dict[ctx] = structure.get_chain_coords(ctx)
        return _concatenate_coords(coords_dict, structure.chain)

    def _compute_log_likelihoods(
        self,
        X: ProteinSequences,
        mask_positions: Optional[List[List[int]]] = None,
    ) -> List[np.ndarray]:
        if mask_positions:
            for mp in mask_positions:
                if mp:
                    raise ValueError(
                        "ESM-IF cannot honor mask_positions; choose wildtype_marginal or mutant_marginal."
                    )

        results = []
        with torch.no_grad():
            for seq in X:
                structure = self._resolve_structure(seq)
                coords = self._build_coords(structure)
                seq_str = str(seq).upper()
                batch = [(coords, None, seq_str)]
                coords_b, conf_b, _, tokens, padding_mask = self.batch_converter_(batch, device=self.device)
                prev = tokens[:, :-1]
                logits, _ = self.model_(coords_b, padding_mask, conf_b, prev)
                # ESM-IF's alphabet sets append_eos=False, so logits length already
                # equals len(seq_str). Slicing to len(seq_str) is a no-op in that
                # case but stays correct if a future alphabet adds an EOS prediction.
                logits_seq = logits[:, :, :len(seq_str)]
                log_probs = F.log_softmax(logits_seq, dim=1)
                # → [L, V] in float64 (validator uses atol=1e-5 on exp().sum())
                lp = log_probs[0].transpose(0, 1).cpu().numpy().astype(np.float64)
                # Renormalize for float-precision safety so exp().sum(-1) hits 1.0 within atol.
                lp = lp - np.log(np.exp(lp).sum(axis=-1, keepdims=True))
                results.append(lp)
        return results

    def _index_log_probs(self, log_probs: np.ndarray, sequences: ProteinSequences) -> np.ndarray:
        seq_array = np.array([list(str(seq).upper()) for seq in sequences])
        aa_indices = self._vectorized_aa_to_index(seq_array)
        rows = np.arange(log_probs.shape[0])
        rows_expanded = np.expand_dims(rows, axis=0)
        return log_probs[rows_expanded, aa_indices]

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return super().get_feature_names_out(input_features)

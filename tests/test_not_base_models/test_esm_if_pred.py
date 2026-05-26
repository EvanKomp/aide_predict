# tests/test_not_base_models/test_esm_if_pred.py
'''
* Author: Evan Komp
* Created: 2026-05-26
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Integration tests for ESMIFLikelihoodWrapper against the ENVZ_ECOLI Ghose
DMS benchmark. Requires fair-esm + torch_geometric + torch_scatter +
torch_cluster + biotite<1.0; excluded from default CI by pytest.ini.
'''
import os
import tempfile

import pytest
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence, ProteinStructure, StructureMapper

import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def _build_two_chain_pdb(src_path: str, dst_path: str, offset: float = 100.0) -> None:
    """
    Append a copy of chain A as chain B, translated by `offset` Å along x.
    Used for multichain smoke tests; chain B is far enough that it should have
    negligible structural influence on chain A's per-residue predictions.
    """
    with open(src_path) as f:
        lines = f.readlines()

    a_atoms = [line for line in lines if line.startswith("ATOM") and line[21] == "A"]
    if not a_atoms:
        raise RuntimeError(f"No chain A atoms found in {src_path}")
    next_serial = max(int(line[6:11]) for line in a_atoms) + 1

    b_atoms = []
    for line in a_atoms:
        x = float(line[30:38]) + offset
        new_line = (
            line[:6]
            + f"{next_serial:>5}"
            + line[11:21]
            + "B"
            + line[22:30]
            + f"{x:8.3f}"
            + line[38:]
        )
        b_atoms.append(new_line)
        next_serial += 1

    with open(dst_path, "w") as f:
        for line in lines:
            if line.startswith("END"):
                f.writelines(b_atoms)
                f.write(f"TER   {next_serial:>5}\n")
            f.write(line)


def _run_proteingym_benchmark(pdb_path: str, csv_path: str, wt_sequence: str, marginal_method: str = 'mutant_marginal') -> float:
    """Run an ESM-IF marginal-method scoring against a ProteinGym-style CSV and return Spearman."""
    from aide_predict.bespoke_models.predictors.esm_if import ESMIFLikelihoodWrapper

    structure = ProteinStructure(pdb_path)
    wt = ProteinSequence(wt_sequence, structure=structure)

    assay_data = pd.read_csv(csv_path)
    sequences = ProteinSequences.from_list(assay_data['mutated_sequence'].tolist())
    scores = assay_data['DMS_score'].tolist()

    model = ESMIFLikelihoodWrapper(
        marginal_method=marginal_method,
        device=DEVICE,
        pool=True,
        wt=wt,
        metadata_folder='./tmp/esm_if',
    )
    model.fit(sequences)
    predictions = model.predict(sequences)
    assert np.isfinite(predictions).all()
    return float(spearmanr(scores, predictions)[0])


@pytest.mark.optional
def test_esm_if_envz_ecoli_spearman():
    """ENVZ_ECOLI Ghose 2023, mutant_marginal scoring. ProteinGym reference for
    ESM-IF1 on this assay is ~0.115; the wrapper's mutant_marginal now uses
    ProteinGym's exact formula (mean per-position log-likelihood of the variant
    under its own context, no WT subtraction) so the numbers should match."""
    wt_sequence = "LADDRTLLMAGVSHDLRTPLTRIRLATEMMSEQDGYLAESINKDIEECNAIIEQFIDYLR"
    spearman = _run_proteingym_benchmark(
        pdb_path='tests/data/ENVZ_ECOLI.pdb',
        csv_path=os.path.join('tests', 'data', 'ENVZ_ECOLI_Ghose_2023.csv'),
        wt_sequence=wt_sequence,
    )
    print(f"\n>>> ESM-IF mutant_marginal Spearman on ENVZ_ECOLI (Ghose 2023): {spearman:.4f}  (ProteinGym ref: 0.115)")
    assert abs(spearman - 0.115) < 0.03


def _proteingym_score_sequences(pdb_path: str, chain: str, sequences: list[str], device: str) -> "np.ndarray":
    """
    Replicate ProteinGym's compute_fitness_esm_if1.py scoring formula on a list
    of full variant sequences, using fair-esm directly (no aide wrapper).

    Returns one scalar per variant: ``ll_fullseq = -mean(cross_entropy)`` averaged
    over all non-pad target positions of the variant.
    """
    import esm
    from esm.inverse_folding.util import CoordBatchConverter, load_coords

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(device)
    coords, _ = load_coords(pdb_path, chain)
    batch_converter = CoordBatchConverter(alphabet)

    out = np.empty(len(sequences), dtype=np.float64)
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            batch = [(coords, None, seq)]
            coords_b, conf_b, _, tokens, padding_mask = batch_converter(batch, device=device)
            prev = tokens[:, :-1]
            target = tokens[:, 1:]
            target_padding_mask = (target == alphabet.padding_idx)
            logits, _ = model(coords_b, padding_mask, conf_b, prev)
            loss = torch.nn.functional.cross_entropy(logits, target, reduction='none')
            loss_np = loss[0].cpu().numpy()
            mask_np = target_padding_mask[0].cpu().numpy()
            ll = -np.sum(loss_np * ~mask_np) / np.sum(~mask_np)
            out[i] = ll
    del model, alphabet
    if device == 'cuda':
        torch.cuda.empty_cache()
    return out


@pytest.mark.optional
def test_proteingym_reference_envz_ecoli():
    """Bypass the wrapper and replicate ProteinGym's exact scoring formula on
    our ENVZ_ECOLI test data. If this reproduces ~0.115, the wrapper's lower
    Spearman is a scoring-formula difference (Meier-style WT marginal vs
    ProteinGym's full-sequence mean log-likelihood), not a wrapper bug."""
    assay = pd.read_csv(os.path.join('tests', 'data', 'ENVZ_ECOLI_Ghose_2023.csv'))
    seqs = assay['mutated_sequence'].tolist()
    scores = assay['DMS_score'].tolist()
    preds = _proteingym_score_sequences(
        pdb_path='tests/data/ENVZ_ECOLI.pdb', chain='A', sequences=seqs, device=DEVICE,
    )
    spearman = float(spearmanr(scores, preds)[0])
    print(f"\n>>> ProteinGym-formula Spearman on ENVZ_ECOLI: {spearman:.4f}  (reference: 0.115)")


@pytest.mark.optional
def test_proteingym_reference_gfp_aequvi():
    """Same as above for GFP_AEQVI. If this reproduces ~0.713, the wrapper's
    lower Spearman is purely a formula choice — not a wrapper bug."""
    assay = pd.read_csv(os.path.join('tests', 'data', 'GFP_AEQVI_Sarkisyan_2016.csv'))
    seqs = assay['mutated_sequence'].tolist()
    scores = assay['DMS_score'].tolist()
    preds = _proteingym_score_sequences(
        pdb_path='tests/data/GFP_AEQVI.pdb', chain='A', sequences=seqs, device=DEVICE,
    )
    spearman = float(spearmanr(scores, preds)[0])
    print(f"\n>>> ProteinGym-formula Spearman on GFP_AEQVI: {spearman:.4f}  (reference: 0.713)")


@pytest.mark.optional
def test_esm_if_gfp_aequvi_spearman():
    """GFP_AEQVI Sarkisyan 2016, mutant_marginal scoring. ProteinGym reference
    for ESM-IF1 on this assay is ~0.713; with the redefined mutant_marginal we
    expect to be within float-noise of that."""
    structure_for_wt = ProteinStructure('tests/data/GFP_AEQVI.pdb')
    wt_sequence = structure_for_wt.get_sequence()
    spearman = _run_proteingym_benchmark(
        pdb_path='tests/data/GFP_AEQVI.pdb',
        csv_path=os.path.join('tests', 'data', 'GFP_AEQVI_Sarkisyan_2016.csv'),
        wt_sequence=wt_sequence,
    )
    print(f"\n>>> ESM-IF mutant_marginal Spearman on GFP_AEQVI (Sarkisyan 2016): {spearman:.4f}  (ProteinGym ref: 0.713)")
    assert abs(spearman - 0.713) < 0.03


@pytest.mark.optional
def test_esm_if_mutant_marginal_subset():
    from aide_predict.bespoke_models.predictors.esm_if import ESMIFLikelihoodWrapper

    structure = ProteinStructure('tests/data/ENVZ_ECOLI.pdb')
    wt = ProteinSequence(
        "LADDRTLLMAGVSHDLRTPLTRIRLATEMMSEQDGYLAESINKDIEECNAIIEQFIDYLR",
        structure=structure,
    )
    assay_data = pd.read_csv(os.path.join('tests', 'data', 'ENVZ_ECOLI_Ghose_2023.csv'))
    # 50 variants is enough to verify the path; one forward pass per variant is ~100ms on CPU.
    subset = assay_data.head(50)
    sequences = ProteinSequences.from_list(subset['mutated_sequence'].tolist())

    model = ESMIFLikelihoodWrapper(
        marginal_method='mutant_marginal',
        device=DEVICE,
        pool=True,
        wt=wt,
        metadata_folder='./tmp/esm_if',
    )
    model.fit(sequences)
    predictions = model.predict(sequences)
    assert predictions.shape == (50, 1)
    assert np.all(np.isfinite(predictions))


@pytest.mark.optional
def test_esm_if_position_specific():
    from aide_predict.bespoke_models.predictors.esm_if import ESMIFLikelihoodWrapper

    structure = ProteinStructure('tests/data/ENVZ_ECOLI.pdb')
    wt = ProteinSequence(
        "LADDRTLLMAGVSHDLRTPLTRIRLATEMMSEQDGYLAESINKDIEECNAIIEQFIDYLR",
        structure=structure,
    )
    assay_data = pd.read_csv(os.path.join('tests', 'data', 'ENVZ_ECOLI_Ghose_2023.csv'))
    sequences = ProteinSequences.from_list(assay_data['mutated_sequence'].head(20).tolist())

    model = ESMIFLikelihoodWrapper(
        marginal_method='wildtype_marginal',
        positions=[8, 9, 10],
        pool=False,
        device=DEVICE,
        wt=wt,
        metadata_folder='./tmp/esm_if',
    )
    model.fit(sequences)
    predictions = model.predict(sequences)
    assert predictions.shape == (20, 3)


@pytest.mark.optional
def test_esm_if_multichain_smoke(tmp_path):
    """Adding a far-translated copy of chain A as chain B should leave the per-position
    deltas approximately unchanged (no structural interaction at 100 Å)."""
    from aide_predict.bespoke_models.predictors.esm_if import ESMIFLikelihoodWrapper

    single_chain_pdb = 'tests/data/ENVZ_ECOLI.pdb'
    multichain_pdb = str(tmp_path / "envz_2chain.pdb")
    _build_two_chain_pdb(single_chain_pdb, multichain_pdb, offset=100.0)

    wt_sequence = "LADDRTLLMAGVSHDLRTPLTRIRLATEMMSEQDGYLAESINKDIEECNAIIEQFIDYLR"

    # Single-chain reference.
    single_struct = ProteinStructure(single_chain_pdb)
    single_wt = ProteinSequence(wt_sequence, structure=single_struct)

    # Multichain version with the duplicated chain B as structural context.
    multi_struct = ProteinStructure(multichain_pdb, chain='A', context_chains=('B',))
    multi_wt = ProteinSequence(wt_sequence, structure=multi_struct)

    # Score a tiny subset of variants under both setups.
    assay_data = pd.read_csv(os.path.join('tests', 'data', 'ENVZ_ECOLI_Ghose_2023.csv'))
    sequences = ProteinSequences.from_list(assay_data['mutated_sequence'].head(10).tolist())

    common_kwargs = dict(
        marginal_method='wildtype_marginal',
        device=DEVICE,
        pool=True,
        metadata_folder='./tmp/esm_if',
    )
    single_model = ESMIFLikelihoodWrapper(wt=single_wt, **common_kwargs)
    single_model.fit(sequences)
    single_preds = single_model.predict(sequences).ravel()

    multi_model = ESMIFLikelihoodWrapper(wt=multi_wt, **common_kwargs)
    multi_model.fit(sequences)
    multi_preds = multi_model.predict(sequences).ravel()

    # Both should be finite and similarly ranked.
    assert np.all(np.isfinite(single_preds))
    assert np.all(np.isfinite(multi_preds))
    # Correlation between single- and multi-chain scores should be high since
    # chain B is far away and shouldn't change autoregressive context-conditional
    # log probs at chain A meaningfully.
    if len(single_preds) > 1 and np.std(single_preds) > 0 and np.std(multi_preds) > 0:
        rho = spearmanr(single_preds, multi_preds).correlation
        print(f"Single-vs-multichain Spearman: {rho}")
        assert rho > 0.7


@pytest.mark.optional
def test_esm_if_structure_mapper_smoke(tmp_path):
    """End-to-end: discover a structure folder, build ProteinSequences with multichain
    context, score variants with ESM-IF."""
    from aide_predict.bespoke_models.predictors.esm_if import ESMIFLikelihoodWrapper

    multichain_pdb = str(tmp_path / "envz_2chain.pdb")
    _build_two_chain_pdb('tests/data/ENVZ_ECOLI.pdb', multichain_pdb, offset=100.0)

    folder = tmp_path / "structures"
    folder.mkdir()
    target_path = folder / "envz.pdb"
    target_path.write_text(open(multichain_pdb).read())

    mapper = StructureMapper(str(folder))
    wts = mapper.get_protein_sequences(target_chain='A')
    assert len(wts) == 1
    wt = wts[0]
    assert wt.structure.chain == 'A'
    assert wt.structure.context_chains == ('B',)

    assay_data = pd.read_csv(os.path.join('tests', 'data', 'ENVZ_ECOLI_Ghose_2023.csv'))
    sequences = ProteinSequences.from_list(assay_data['mutated_sequence'].head(5).tolist())

    model = ESMIFLikelihoodWrapper(
        marginal_method='wildtype_marginal',
        device=DEVICE,
        pool=True,
        wt=wt,
        metadata_folder='./tmp/esm_if',
    )
    model.fit(sequences)
    predictions = model.predict(sequences).ravel()
    assert predictions.shape == (5,)
    assert np.all(np.isfinite(predictions))


if __name__ == "__main__":
    test_esm_if_wildtype_marginal()

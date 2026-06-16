# tests/test_bespoke_models/test_predictors/test_esm_if_unit.py
'''
* Author: Evan Komp
* Created: 2026-05-26
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Static unit tests for ESMIFLikelihoodWrapper that do not load the model.
Tests that exercise actual model forward passes live in
tests/test_not_base_models/test_esm_if_pred.py (excluded from default CI).
'''
import pytest
import tempfile
import os

from aide_predict.bespoke_models.predictors.esm_if import ESMIFLikelihoodWrapper
from aide_predict.bespoke_models.predictors.pretrained_transformers import MarginalMethod
from aide_predict.utils.data_structures import ProteinSequence, ProteinSequences, ProteinStructure
from aide_predict.utils.checks import check_model_compatibility


SAMPLE_PDB = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.362   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.702   2.144   0.907  1.00  0.00           O
ATOM      5  N   GLY A   2       2.831   1.687  -0.987  1.00  0.00           N
ATOM      6  CA  GLY A   2       3.396   3.037  -1.009  1.00  0.00           C
ATOM      7  C   GLY A   2       2.362   4.089  -1.408  1.00  0.00           C
ATOM      8  O   GLY A   2       2.730   5.261  -1.509  1.00  0.00           O
END
"""


@pytest.fixture
def temp_pdb():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(SAMPLE_PDB)
    yield f.name
    os.unlink(f.name)


class TestESMIFMixinFlags:
    """Class-level mixin flags don't require fair-esm to be installed."""

    def test_requires_structure(self):
        assert ESMIFLikelihoodWrapper._requires_structure is True

    def test_requires_wt_to_function(self):
        assert ESMIFLikelihoodWrapper._requires_wt_to_function is True

    def test_requires_wt_during_inference(self):
        assert ESMIFLikelihoodWrapper._requires_wt_during_inference is True

    def test_per_position_capable(self):
        assert ESMIFLikelihoodWrapper._per_position_capable is True

    def test_requires_fixed_length(self):
        assert ESMIFLikelihoodWrapper._requires_fixed_length is True

    def test_can_regress(self):
        assert ESMIFLikelihoodWrapper._can_regress is True

    def test_expects_no_fit(self):
        assert ESMIFLikelihoodWrapper._expects_no_fit is True


@pytest.mark.optional
class TestESMIFInit:
    """Initialization tests require fair-esm to be importable so AVAILABLE is True."""

    def test_masked_marginal_string_refused(self, temp_pdb):
        wt = ProteinSequence("AG", structure=ProteinStructure(temp_pdb))
        with pytest.raises(ValueError, match="masked_marginal is not defined"):
            ESMIFLikelihoodWrapper(marginal_method='masked_marginal', wt=wt)

    def test_masked_marginal_enum_refused(self, temp_pdb):
        wt = ProteinSequence("AG", structure=ProteinStructure(temp_pdb))
        with pytest.raises(ValueError, match="masked_marginal is not defined"):
            ESMIFLikelihoodWrapper(marginal_method=MarginalMethod.MASKED, wt=wt)

    def test_wildtype_marginal_accepted(self, temp_pdb):
        wt = ProteinSequence("AG", structure=ProteinStructure(temp_pdb))
        wrapper = ESMIFLikelihoodWrapper(marginal_method='wildtype_marginal', wt=wt)
        assert wrapper.marginal_method == 'wildtype_marginal'

    def test_mutant_marginal_accepted(self, temp_pdb):
        wt = ProteinSequence("AG", structure=ProteinStructure(temp_pdb))
        wrapper = ESMIFLikelihoodWrapper(marginal_method='mutant_marginal', wt=wt)
        assert wrapper.marginal_method == 'mutant_marginal'

    def test_enum_normalized_to_string(self, temp_pdb):
        wt = ProteinSequence("AG", structure=ProteinStructure(temp_pdb))
        wrapper = ESMIFLikelihoodWrapper(marginal_method=MarginalMethod.WILDTYPE, wt=wt)
        # Base class compares against MarginalMethod.*.value (strings), so
        # the enum must be normalized to its string value.
        assert wrapper.marginal_method == MarginalMethod.WILDTYPE.value


class TestESMIFCompatibility:
    """check_model_compatibility uses class attributes only — does not load fair-esm."""

    def test_incompatible_without_structure(self):
        wt = ProteinSequence("ACDEFGHIKLMNPQRSTVWY")
        result = check_model_compatibility(
            training_sequences=None,
            testing_sequences=None,
            wt=wt,
        )
        assert "ESMIFLikelihoodWrapper" in result["incompatible"]

    def test_compatible_with_structure(self, temp_pdb):
        # Tiny PDB has 'AG'; need a wt whose length matches the structure or compatibility
        # logic flags fixed-length mismatches separately. For this test we only verify
        # the structure-presence path doesn't immediately disqualify the model when
        # fair-esm is unavailable. If fair-esm is missing, _available is False which
        # also makes the model incompatible; skip in that case.
        if not ESMIFLikelihoodWrapper._available:
            pytest.skip("ESM-IF deps not installed; cannot exercise the compatible path.")
        wt = ProteinSequence("AG", structure=ProteinStructure(temp_pdb))
        result = check_model_compatibility(
            training_sequences=None,
            testing_sequences=None,
            wt=wt,
        )
        assert "ESMIFLikelihoodWrapper" in result["compatible"]

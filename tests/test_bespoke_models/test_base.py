# tests/test_bespoke_models/test_base.py
'''
* Author: Evan Komp
* Created: 7/3/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import pytest
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from aide_predict.utils.common import MessageBool

from aide_predict.utils.data_structures import (
    ProteinSequence, ProteinSequences,
)
from aide_predict.bespoke_models.base import (
    ProteinModelWrapper, RequiresMSAForFitMixin, RequiresFixedLengthMixin,
    CanRegressMixin, RequiresWTDuringInferenceMixin, PositionSpecificMixin,
    RequiresWTToFunctionMixin, CacheMixin, CanHandleAlignedSequencesMixin,
    ShouldRefitOnSequencesMixin, RequiresStructureMixin, is_jsonable
)


class TestProteinModelWrapper:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model = ProteinModelWrapper(metadata_folder=self.temp_dir)
        yield
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        assert os.path.exists(self.temp_dir)
        assert self.model.wt is None

    def test_unavailable(self):
        class TestModel(ProteinModelWrapper):
            _available = MessageBool(False, "This model is not available.")
        with pytest.raises(ValueError):
            TestModel()

    def test_validate_input(self):
        input_list = ["ACDE", "FGHI"]
        result = self.model._validate_input(input_list)
        assert isinstance(result, ProteinSequences)

        input_list = []
        result = self.model._validate_input(input_list)
        assert isinstance(result, ProteinSequences)
        assert len(result) == 0

        with pytest.raises(ValueError):
            self.model._validate_input(5)

        # check uppercase handling
        self.model._accepts_lower_case = False
        result = self.model._validate_input(["a"])
        assert result[0][0] == "A"

    def test_skip_fitting(self):
        class TestModel(ProteinModelWrapper):
            def _fit(self, X, y=None):
                self.fitted_ = True
                self.X_ = X
                return self
            
        model = TestModel(metadata_folder=self.temp_dir)
        model.fit(["ACDE", "FGHI"]) # fits the model
        assert len(model.X_) == 2

        # fit again, should skip
        model.fit(["ACDE", "FGHI", "LMNO"])
        assert len(model.X_) == 2

        # fit but force, should refit
        model.fit(["ACDE", "FGHI", "LMNO"], force=True)
        assert len(model.X_) == 3

    def test_assert_aligned(self):
        mock_sequences = MagicMock()
        mock_sequences.aligned = False
        self.model._requires_msa_for_fit = True
        with pytest.raises(ValueError):
            self.model._assert_aligned(mock_sequences)

    def test_assert_fixed_length(self):
        mock_sequences = MagicMock()
        mock_sequences.fixed_length = False
        self.model._requires_fixed_length = True
        with pytest.raises(ValueError):
            self.model._assert_fixed_length(mock_sequences)

        # different wt
        mock_sequences.width = 5
        mock_sequences.fixed_length = True  
        self.model.wt = ProteinSequence("ACDE")
        with pytest.raises(ValueError):
            self.model._assert_fixed_length(mock_sequences)

    def test_enforce_aligned(self):
        mock_sequences = MagicMock()
        mock_sequences.aligned = False
        mock_sequences.align_all.return_value = "aligned_sequences"
        result = self.model._enforce_aligned(mock_sequences)
        assert result == "aligned_sequences"

        mock_sequences.aligned = True
        result = self.model._enforce_aligned(mock_sequences)
        assert result is mock_sequences

    def test_abstract_methods(self):
        with pytest.raises(NotImplementedError):
            self.model._fit(None)
        with pytest.raises(NotImplementedError):
            self.model._transform(None)
        with pytest.raises(NotImplementedError):
            self.model._partial_fit(None)

    @patch('aide_predict.bespoke_models.base.ProteinModelWrapper._fit')
    @patch('aide_predict.bespoke_models.base.ProteinModelWrapper._validate_input')
    @patch('aide_predict.bespoke_models.base.ProteinModelWrapper._assert_aligned')
    def test_fit(self, mock_assert_aligned, mock_validate_input, mock_fit):
        mock_validate_input.return_value = MagicMock()
        self.model.fit(["ACDE", "FGHI"])
        mock_fit.assert_called_once()

    @patch('aide_predict.bespoke_models.base.ProteinModelWrapper._transform')
    @patch('aide_predict.bespoke_models.base.ProteinModelWrapper._validate_input')
    def test_transform(self, mock_validate_input, mock_transform):
        mock_validate_input.return_value = MagicMock()
        mock_transform.return_value = np.array([1, 2, 3])
        self.model.fitted_ = True  # Mock fitted state
        result = self.model.transform(["ACDE", "FGHI"])
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_predict_not_regressor(self):
        with pytest.raises(ValueError):
            self.model.predict(["ACDE", "FGHI"])

    def test_get_set_params(self):
        params = self.model.get_params()
        assert 'metadata_folder' in params
        assert 'wt' in params

        new_params = {'metadata_folder': '/new/path', 'wt': 'ACDE'}
        self.model.set_params(**new_params)
        assert self.model.metadata_folder == '/new/path'
        assert str(self.model.wt) == 'ACDE'

    def test_get_feature_names_out(self):
        self.model.fitted_ = True  # Mock fitted state
        feature_names = self.model.get_feature_names_out()
        assert feature_names == ['ProteinModelWrapper']

    @pytest.mark.parametrize("mixin_class,attribute,expected", [
        (RequiresMSAForFitMixin, 'requires_msa_for_fit', True),
        (RequiresFixedLengthMixin, 'requires_fixed_length', True),
        (CanRegressMixin, 'can_regress', True),
        (RequiresWTDuringInferenceMixin, 'requires_wt_during_inference', True),
    ])
    def test_mixins(self, mixin_class, attribute, expected):
        class TestModel(mixin_class, ProteinModelWrapper):
            pass
        tempdir = tempfile.mkdtemp()
        model = TestModel(metadata_folder=tempdir)
        assert getattr(model, attribute) == expected

    def test_position_specific_mixin(self):
        class TestModel(PositionSpecificMixin, ProteinModelWrapper):
            def __init__(self, pool=False, flatten=False, *args, **kwargs):
                super().__init__(pool=pool, flatten=flatten, positions=[1,2], *args, **kwargs)
        tempdir = tempfile.mkdtemp()
        model = TestModel(metadata_folder=tempdir)
        assert model.per_position_capable

        with patch.object(ProteinModelWrapper, 'transform') as mock_transform:
            mock_transform.return_value = np.array([[1, 2]])
            result = model.transform(["ACDE"])
            np.testing.assert_array_equal(result, np.array([[1, 2]]))

        with patch.object(ProteinModelWrapper, 'transform') as mock_transform:
            mock_transform.return_value = np.array([[1, 2, 3]])
            with pytest.raises(ValueError):
                model.transform(["ACDE"])

        model.fitted_ = True  # Mock fitted state
        feature_names = model.get_feature_names_out()
        # we passed flatten = False, so even though there is only one output above it still has dim
        assert feature_names == ['TestModel_1', 'TestModel_2']

        # retry with flatten on 
        model = TestModel(metadata_folder=tempdir, flatten=True)
        model.fitted_ = True  # Mock fitted state
        feature_names = model.get_feature_names_out()
        assert feature_names == ['TestModel_1_dim0', "TestModel_2_dim0"]

        with patch.object(ProteinModelWrapper, 'transform') as mock_transform:
            # we passed flatten, so the base class should handle this
            mock_transform.return_value = np.ones((2, 2, 3))
            result = model.transform(["ACDE", "FGHI"])
            assert result.shape == (2, 6)


    def test_wt_with_gaps(self):
        with pytest.raises(ValueError, match="Wild type sequence cannot have gaps."):
            ProteinModelWrapper(wt="AC-DE")

    def test_requires_wt_no_wt_provided(self):
        class TestModel(RequiresWTToFunctionMixin, ProteinModelWrapper):
            pass
        with pytest.raises(ValueError, match="This model requires a wild type sequence to function."):
            TestModel()

    def test_cache_mixin(self):
        class TestModel(CacheMixin, ProteinModelWrapper):
            def _transform(self, X):
                return np.array([len(seq) for seq in X]).reshape(-1, 1)

        model = TestModel(use_cache=True)
        model.fitted_ = True  # Mock fitted state
        
        # First transformation
        result1 = model.transform(["ACDE", "FGH"])
        np.testing.assert_array_equal(result1, np.array([[4], [3]]))
        
        # Second transformation (should use cache)
        result2 = model.transform(["ACDE", "FGH"])
        np.testing.assert_array_equal(result2, np.array([[4], [3]]))

        # third this time, also use cache, cache already open
        result3 = model.transform(["ACDE", "FGH"])
        
        # Check that cache was used
        # model._db_file should have two entries
        import sqlite3
        con = sqlite3.connect(model._db_file)
        assert con.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 2

        # test that results remain the same when shapes are multidimensional
        outs = np.random.random((1, 2, 3))
        class TestModel2(CacheMixin, ProteinModelWrapper):
            def _transform(self, X):
                return np.vstack([outs for _ in range(len(X))])
            
        model = TestModel2(use_cache=True)
        model.fitted_ = True  # Mock fitted state
        result1 = model.transform(["ACDE", "FGH"])
        np.testing.assert_array_equal(result1, np.vstack([outs, outs]))

        result2 = model.transform(["ACDE", "FGH"])
        np.testing.assert_array_equal(result2, np.vstack([outs, outs]))

        # check cache
        con = sqlite3.connect(model._db_file)
        assert con.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 2

    def test_can_handle_aligned_sequences(self):
        class TestModel(CanHandleAlignedSequencesMixin, ProteinModelWrapper):
            def _transform(self, X):
                return np.array([seq.count('-') for seq in X]).reshape(-1, 1)

        model = TestModel()
        model.fitted_ = True  # Mock fitted state
        
        result = model.transform(["AC-DE", "FG-H-"])
        np.testing.assert_array_equal(result, np.array([[1], [2]]))

    def test_check_fixed_length_mixin(self):
        model = ProteinModelWrapper()
        
        fixed_length_sequences = ProteinSequences.from_list(["ACDE", "FGHI"])
        assert model._check_fixed_length(fixed_length_sequences) == True
        
        variable_length_sequences = ProteinSequences.from_list(["ACDE", "FGH"])
        assert model._check_fixed_length(variable_length_sequences) == False

    def test_should_refit_on_sequences_mixin(self):
        class TestModel(ShouldRefitOnSequencesMixin, ProteinModelWrapper):
            def _fit(self, X, y=None):
                self.fitted_ = True
                return self
            
        model = TestModel(metadata_folder=self.temp_dir)
        model.fit(["ACDE", "FGHI"])

        # sklearn clone should reset this model
        from sklearn.base import clone
        from sklearn.utils.validation import check_is_fitted, NotFittedError
        model_clone = clone(model)
        with pytest.raises(NotFittedError):
            check_is_fitted(model_clone, 'fitted_')

        # sklearn clone should not reset this model without the mixin
        class TestModel(ProteinModelWrapper):
            def _fit(self, X, y=None):
                self.fitted_ = True
                return self
            
        model = TestModel(metadata_folder=self.temp_dir)
        model.fit(["ACDE", "FGHI"])
        model_clone = clone(model)
        check_is_fitted(model_clone, 'fitted_')

    def test_requires_structure(self):
        class TestModel(RequiresStructureMixin, ProteinModelWrapper):
            def _fit(self, X, y=None):
                self.fitted_ = True
                return self

            def _transform(self, X):
                return np.array([len(seq) for seq in X]).reshape(-1, 1)

        tempdir = tempfile.mkdtemp()
        model = TestModel(metadata_folder=tempdir)
        assert model.requires_structure

        mock_sequences = ProteinSequences.from_list(["ACDE", "FGHI"])

        # Test with sequences that do not have structure
        with pytest.raises(ValueError, match="This model requires structure information"):
            model.fit(mock_sequences)
            model.transform(mock_sequences)

        # Test with sequences that have structure
        tmp = tempfile.NamedTemporaryFile(suffix='.pdb')   
        # mock out so it does not check the file Bio.PDB.PDBParser.PDBParser
        with patch('aide_predict.utils.data_structures.structures.PDBParser') as pdb:
            with patch('aide_predict.utils.data_structures.structures.ProteinStructure.get_sequence') as get_seq:
                get_seq.return_value = "ACDE"
                for s in mock_sequences:
                    s.structure = str(tmp.name)

                model.fit(mock_sequences)
                model.transform(mock_sequences)

                mock_sequences = ProteinSequences.from_list(["ACDE", "FGHI"])
                # Test with wild type sequence that has structure
                wt= ProteinSequence("ACDE", structure=str(tmp.name))
                model = TestModel(metadata_folder=tempdir, wt=wt)
                model.fit(mock_sequences)
                model.transform(mock_sequences)

                # Test with wild type sequence that does not have structure
                with pytest.raises(ValueError, match="This model acts on structure but a wild type structure was not given."):
                    model = TestModel(metadata_folder=tempdir, wt="ACDE")
                    model(metadata_folder=tempdir, wt="ACDE")

                tmp.close()

    def test_requires_msa_mixin(self):
        class TestModel(RequiresMSAForFitMixin, ProteinModelWrapper):
            def _fit(self, X, y=None):
                self.fitted_ = True
                return self

        tempdir = tempfile.mkdtemp()
        model = TestModel(metadata_folder=tempdir)
        assert model.requires_msa_for_fit

        mock_sequences = ProteinSequences.from_list(["ACDE", "FGHI"])
        model.fit(mock_sequences)

        # now check that it is aligned if not
        model = TestModel(metadata_folder=tempdir)
        mock_sequences = MagicMock()
        mock_sequences.aligned = False
        model._validate_input = MagicMock(return_value=mock_sequences)
        model._enforce_aligned = MagicMock(return_value=mock_sequences)
        model.fit(mock_sequences)
        model._enforce_aligned.assert_called_once()

    def test__partial_fit(self):
        class TestModel(ProteinModelWrapper):
            _partial_fit = MagicMock()

        model = TestModel(metadata_folder=self.temp_dir)
        model.partial_fit(["ACDE", "FGHI"])
        model._partial_fit.assert_called_once()



def test_is_jsonable():
    assert is_jsonable({"key": "value"}) == True
    assert is_jsonable([1, 2, 3]) == True
    assert is_jsonable("string") == True
    assert is_jsonable(123) == True
    assert is_jsonable(12.34) == True
    assert is_jsonable(True) == True
    assert is_jsonable(None) == True

    class NonSerializable:
        pass

    assert is_jsonable(NonSerializable()) == False

    # Test with a complex object containing non-serializable elements
    complex_obj = {
        "key1": "value1",
        "key2": NonSerializable(),
        "key3": [1, 2, 3],
        "key4": {"nested_key": NonSerializable()}
    }
    assert is_jsonable(complex_obj) == False

        
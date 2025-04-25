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
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            
            def _fit(self, X, y=None):
                # Mock implementation
                self.fitted_ = True
                return self
                
            def _transform(self, X):
                # Mock implementation returning a test array
                # In real implementation, this would process sequences
                # Return a (samples, positions, features) shape array for testing
                return np.ones((len(X), 2, 3))
                
        tempdir = tempfile.mkdtemp()
        
        # Test initialization and property
        model = TestModel(metadata_folder=tempdir, positions=[1, 2], pool=False, flatten=False)
        assert model.per_position_capable
        assert model.positions == [1, 2]
        assert not model.pool
        assert not model.flatten
        
        # Fit the model to allow transform to work
        model.fit([])
        
        # Test case 1: Basic transform with no pooling or flattening
        seqs = ProteinSequences([ProteinSequence("ACDE"), ProteinSequence("FGHI")])
        result = model.transform(seqs)
        assert result.shape == (2, 2, 3)  # (samples, positions, features)
        
        # Test case 2: Transform with flattening
        model_flat = TestModel(metadata_folder=tempdir, positions=[1, 2], 
                            pool=False, flatten=True)
        model_flat.fit([])
        result = model_flat.transform(seqs)
        assert result.shape == (2, 6)  # (samples, positions*features)
        
        # Test case 3: Transform with pooling
        model_pool = TestModel(metadata_folder=tempdir, positions=[1, 2], 
                            pool=True, flatten=False)
        model_pool.fit([])
        result = model_pool.transform(seqs)
        assert result.shape == (2, 3)  # (samples, features) - pooled across positions
        
        # Test case 4: Pooling with specific function
        model_max_pool = TestModel(metadata_folder=tempdir, positions=[1, 2], 
                                pool='max', flatten=False)
        model_max_pool.fit([])
        result = model_max_pool.transform(seqs)
        assert result.shape == (2, 3)  # (samples, features) - max-pooled
        
        # Test feature names - no pooling, no flattening
        feature_names = model.get_feature_names_out()
        assert feature_names == ['TestModel_pos1', 'TestModel_pos2']
        
        # Test feature names - flattened
        model_flat.output_dim_ = 3  # Mock embedding dimension
        feature_names = model_flat.get_feature_names_out()
        assert feature_names == ['TestModel_pos1_dim0', 'TestModel_pos1_dim1', 'TestModel_pos1_dim2',
                            'TestModel_pos2_dim0', 'TestModel_pos2_dim1', 'TestModel_pos2_dim2']
        
        # Test feature names - pooled
        model_pool.embedding_dim_ = 3  # Mock embedding dimension
        feature_names = model_pool.get_feature_names_out()
        assert feature_names == ['TestModel_emb0', 'TestModel_emb1', 'TestModel_emb2']
        
        # Test validation of dimensions - should raise error when dimensions don't match
        class WrongDimModel(TestModel):
            def _transform(self, X):
                # Return wrong number of dimensions
                return np.ones((len(X), 3, 3))  # 3 positions instead of 2
                
        wrong_model = WrongDimModel(metadata_folder=tempdir, positions=[1, 2], pool=False, flatten=False)
        wrong_model.fit([])
        
        with pytest.raises(ValueError, match="output second dimension"):
            wrong_model.transform(seqs)
        
        # Test with ragged array output
        class RaggedModel(TestModel):
            def _transform(self, X):
                # Return a list of arrays with different shapes
                return [np.ones((3, 4)), np.ones((2, 5))]
                
        ragged_model = RaggedModel(metadata_folder=tempdir, positions=[1, 2], pool=False, flatten=True)
        ragged_model.fit([])
        
        with pytest.warns(UserWarning, match="ragged array"):
            result = ragged_model.transform(seqs)
        assert isinstance(result, list)


    def test_wt_with_gaps(self):
        with pytest.raises(ValueError, match="Wild type sequence cannot have gaps."):
            ProteinModelWrapper(wt="AC-DE")

    def test_requires_wt_no_wt_provided(self):
        class TestModel(RequiresWTToFunctionMixin, ProteinModelWrapper):
            pass
        with pytest.raises(ValueError, match="This model requires a wild type sequence to function."):
            TestModel()

    def test_cache_mixin(self):
        # Define test model with CacheMixin using new hook-based approach
        class TestModel(CacheMixin, ProteinModelWrapper):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.fitted_ = True  # Mock fitted state
                
            def _fit(self, X, y=None):
                self.fitted_ = True
                return self
                
            def _transform(self, X):
                if X is None:  # Handle case where all sequences are cached
                    return None
                return np.array([len(seq) for seq in X]).reshape(-1, 1)
        
        # Create model with cache enabled
        model = TestModel(use_cache=True)
        
        # Create protein sequences for testing
        sequences = ProteinSequences([
            ProteinSequence("ACDE", id="seq1"),
            ProteinSequence("FGH", id="seq2")
        ])
        
        # First transformation - should populate cache
        result1 = model.transform(sequences)
        np.testing.assert_array_equal(result1, np.array([[4], [3]]))
        
        # Second transformation - should use cache
        result2 = model.transform(sequences)
        np.testing.assert_array_equal(result2, np.array([[4], [3]]))
        
        # Third transformation - should still use cache
        result3 = model.transform(sequences)
        np.testing.assert_array_equal(result3, np.array([[4], [3]]))
        
        # Check that cache was used
        import sqlite3
        con = sqlite3.connect(model._db_file)
        assert con.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 2
        
        # Test partial cache usage by adding a new sequence
        extended_sequences = ProteinSequences([
            ProteinSequence("ACDE", id="seq1"),
            ProteinSequence("FGH", id="seq2"),
            ProteinSequence("WXYZ", id="seq3")
        ])
        
        # Should use cache for first two sequences and process the third
        result4 = model.transform(extended_sequences)
        np.testing.assert_array_equal(result4, np.array([[4], [3], [4]]))
        
        # Now all three should be cached
        con = sqlite3.connect(model._db_file)
        assert con.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 3

        # Test with multi-dimensional outputs
        outs = np.random.random((1, 2, 3))
        class TestModel2(CacheMixin, ProteinModelWrapper):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.fitted_ = True
                
            def _fit(self, X, y=None):
                self.fitted_ = True
                return self
                
            def _transform(self, X):
                if X is None:  # Handle case where all sequences are cached
                    return None
                return np.vstack([outs for _ in range(len(X))])
            
        # Create new model and test sequences
        model2 = TestModel2(use_cache=True)
        
        # First transformation - should populate cache
        result1 = model2.transform(sequences)
        np.testing.assert_array_equal(result1, np.vstack([outs, outs]))
        
        # Second transformation - should use cache
        result2 = model2.transform(sequences)
        np.testing.assert_array_equal(result2, np.vstack([outs, outs]))
        
        # Check cache
        con = sqlite3.connect(model2._db_file)
        assert con.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 2
        
        # Test cache with mixed results (some cached, some new)
        mixed_results = model2.transform(extended_sequences)
        expected_shape = (3,) + outs.shape[1:]
        assert mixed_results.shape == expected_shape

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

    def test_requires_msa_for_fit_mixin(self):
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
        mock_sequences = ProteinSequences.from_list(["ACDE", "FGH"])
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

        
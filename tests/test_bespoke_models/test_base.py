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
from unittest.mock import patch, MagicMock, PropertyMock
from aide_predict.utils.common import MessageBool

from aide_predict.utils.data_structures import (
    ProteinSequence, ProteinSequences, ProteinStructure
)
from aide_predict.bespoke_models.base import (
    ProteinModelWrapper, RequiresMSAForFitMixin, RequiresFixedLengthMixin,
    CanRegressMixin, RequiresWTDuringInferenceMixin, PositionSpecificMixin,
    RequiresWTToFunctionMixin, RequiresWTMSAMixin, RequiresMSAPerSequenceMixin, CacheMixin, CanHandleAlignedSequencesMixin,
    ShouldRefitOnSequencesMixin, RequiresStructureMixin, is_jsonable, ExpectsNoFitMixin
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
    def test_transform(self, mock_transform):
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
        
        # Test validation of dimensions - should index by positions if the model internal did not
        class OutputFullDimModel(TestModel):
            def _transform(self, X):
                # Return wrong number of dimensions
                return np.ones((len(X), 4, 3))  # all 4 positions on sequence
                
        full_dim_model = OutputFullDimModel(metadata_folder=tempdir, positions=[1, 2], pool=False, flatten=False)
        full_dim_model.fit([])

        outs = full_dim_model.transform(seqs)
        assert outs.shape == (2, 2, 3)  # (samples, positions chosen, features)

        # Same as above but model does not return shape equal to sequence length
        class OutputFullDimModel(TestModel):
            def _transform(self, X):
                # Return wrong number of dimensions
                return np.ones((len(X), 5, 3))  # 5 positions instead of 4
        full_dim_model = OutputFullDimModel(metadata_folder=tempdir, positions=[1, 2], pool=False, flatten=False)
        full_dim_model.fit([])

        with pytest.raises(ValueError):
            full_dim_model.transform(seqs)
        
        # Test with ragged array output without pooling
        class RaggedModel(TestModel):
            def _transform(self, X):
                # Return a list of arrays with different shapes
                return [np.ones((3, 4)), np.ones((2, 5))]
                
        ragged_model = RaggedModel(metadata_folder=tempdir, positions=[1, 2], pool=False, flatten=True)
        ragged_model.fit([])
        
        with pytest.raises(ValueError):
            result = ragged_model.transform(seqs)

        # positions are not none but input sequences are not fixed length, should raise error
        model = TestModel(metadata_folder=tempdir, positions=[1, 2], pool=False, flatten=False).fit([])
        seqs = ProteinSequences([ProteinSequence("ACDE"), ProteinSequence("FGH")])
        with pytest.raises(ValueError):
            model.transform(seqs)


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

def test_mixin_hooks_registration():
    """Test that mixin hooks are properly registered and called in the correct order."""
    # Create a set of test mixins that track hook calls
    class TestPreFitHook:
        call_order = []
        
        def _pre_fit_hook(self, X, y):
            TestPreFitHook.call_order.append("pre_fit")
            return X, y
    
    class TestPostFitHook:
        call_order = []
        
        def _post_fit_hook(self, X, y):
            TestPostFitHook.call_order.append("post_fit")
    
    class TestPreTransformHook:
        call_order = []
        
        def _pre_transform_hook(self, X):
            TestPreTransformHook.call_order.append("pre_transform")
            return X
    
    class TestPostTransformHook:
        call_order = []
        
        def _post_transform_hook(self, result, X):
            TestPostTransformHook.call_order.append("post_transform")
            return result
    
    class TestInitHandler:
        call_order = []
        
        def _init_handler(self, param1=None, **kwargs):
            TestInitHandler.call_order.append("init_handler")
            self.param1 = param1
    
    # Create a model with all the test mixins
    class TestAllHooksModel(
        TestPreFitHook,
        TestPostFitHook,
        TestPreTransformHook,
        TestPostTransformHook,
        TestInitHandler,
        ProteinModelWrapper
    ):
        def _fit(self, X, y=None):
            self.fitted_ = True
            return self
            
        def _transform(self, X):
            return np.array([len(seq) for seq in X]).reshape(-1, 1)
    
    # Reset call orders
    TestPreFitHook.call_order = []
    TestPostFitHook.call_order = []
    TestPreTransformHook.call_order = []
    TestPostTransformHook.call_order = []
    TestInitHandler.call_order = []
    
    # Test initialization with params
    model = TestAllHooksModel(param1="test_value")
    assert model.param1 == "test_value"
    assert TestInitHandler.call_order == ["init_handler"]
    
    # Test fit hooks
    sequences = ProteinSequences.from_list(["ACDE", "FGHI"])
    model.fit(sequences)
    assert TestPreFitHook.call_order == ["pre_fit"]
    assert TestPostFitHook.call_order == ["post_fit"]
    
    # Test transform hooks
    result = model.transform(sequences)
    assert TestPreTransformHook.call_order == ["pre_transform"]
    assert TestPostTransformHook.call_order == ["post_transform"]
    
    # Verify that hooks were collected in the class
    assert len(TestAllHooksModel._pre_fit_hooks) == 1
    assert len(TestAllHooksModel._post_fit_hooks) == 1
    assert len(TestAllHooksModel._pre_transform_hooks) == 1
    assert len(TestAllHooksModel._post_transform_hooks) == 1
    assert len(TestAllHooksModel._mixin_init_handlers) == 1

def test_process_wt():
    """Test the _process_wt method with various inputs and model requirements."""
    # 1. Test basic processing
    model = ProteinModelWrapper()
    wt = "acde"  # lowercase sequence
    processed_wt = model._process_wt(wt)
    assert str(processed_wt) == "ACDE"  # Should be converted to uppercase
    assert isinstance(processed_wt, ProteinSequence)
    
    # 2. Test None input
    processed_wt = model._process_wt(None)
    assert processed_wt is None
    
    # 3. Test with already ProteinSequence input
    wt_seq = ProteinSequence("ACDE")
    processed_wt = model._process_wt(wt_seq)
    assert processed_wt is not wt_seq  # Should be a new object
    assert str(processed_wt) == "ACDE"
    
    # 4. Test with gaps - should raise exception
    with pytest.raises(ValueError, match="Wild type sequence cannot have gaps"):
        model._process_wt("AC-DE")
    
    # 5. Test with RequiresStructureMixin
    class StructureModel(RequiresStructureMixin, ProteinModelWrapper):
        pass
    
    structure_model = StructureModel()
    
    # Without structure - should raise exception
    with pytest.raises(ValueError, match="This model acts on structure but a wild type structure was not given"):
        structure_model._process_wt("ACDE")
        
    # With structure - properly mock the structure property
    with patch.object(ProteinSequence, 'structure', new_callable=PropertyMock) as mock_structure:
        # Configure the mock to return True (indicating structure exists)
        mock_structure.return_value = True
        
        # Create a sequence with our mocked property
        wt_with_structure = ProteinSequence("ACDE")
        # The above patching should make this pass the structure check
        
        # We need to patch the has_structure property too
        with patch.object(ProteinSequence, 'has_structure', new_callable=PropertyMock) as mock_has_structure:
            mock_has_structure.return_value = True
            
            # Test that it passes validation
            processed_wt = structure_model._process_wt(wt_with_structure)
            assert processed_wt is not None
            assert str(processed_wt) == "ACDE"

    # 6. Test with RequiresWTMSAMixin
    class MSAModel(RequiresWTMSAMixin, RequiresWTToFunctionMixin, ProteinModelWrapper):
        pass
    msa_good = ProteinSequences([ProteinSequence("ACDE"), ProteinSequence("FGHI")])
    
    msa_model = MSAModel(wt=ProteinSequence("ACDE", msa=msa_good))
    
    # Without MSA - should raise exception
    with pytest.raises(ValueError):
        msa_model._process_wt(ProteinSequence("ACDE"))
    
    # With MSA but not matching width
    wt_with_msa = ProteinSequence("ACDE")
    msa = ProteinSequences([ProteinSequence("ACDEFG"), ProteinSequence("HIJKLM")])
    wt_with_msa.msa = msa
    
    with pytest.raises(ValueError):
        msa_model._process_wt(wt_with_msa)
    
    # With properly aligned MSA
    wt_with_msa = ProteinSequence("ACDE")
    wt_with_msa.msa = msa_good
    
    processed_wt = msa_model._process_wt(wt_with_msa)
    assert processed_wt.msa is msa_good

def test_cache_mixin_advanced():
    """Test advanced functionality of the CacheMixin."""
    test_dir = tempfile.mkdtemp()
    
    class TestCacheModel(CacheMixin, ProteinModelWrapper):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.fitted_ = True
            self._transform_call_count = 0
            
        def _fit(self, X, y=None):
            self.fitted_ = True
            return self
            
        def _transform(self, X):
            # Track number of actual transform calls
            self._transform_call_count += 1
            if X is None:  # Handle case where all sequences are cached
                return None
            # Return a simple array with sequence index and length
            return np.array([[i, len(seq)] for i, seq in enumerate(X)])
    
    # 1. Test model state hashing
    model = TestCacheModel(metadata_folder=test_dir, use_cache=True)
    state_hash1 = model._get_model_state_hash()
    
    # Change a parameter and check hash changes
    model._param2 = 10
    state_hash2 = model._get_model_state_hash()
    assert state_hash1 == state_hash2  # Should be same since _transform_call_count_ ends with underscore
    
    # Add a fitted attribute and check hash changes
    model.new_param_ = "test"
    state_hash3 = model._get_model_state_hash()
    assert state_hash2 != state_hash3  # Should be different
    
    # 2. Test protein hashing
    seq1 = ProteinSequence("ACDE", id="seq1")
    seq2 = ProteinSequence("ACDE", id="seq2")  # Same sequence, different ID
    
    hash1 = model._get_protein_hashes([seq1])[0]
    hash2 = model._get_protein_hashes([seq2])[0]
    assert hash1 != hash2  # Hashes should differ due to different IDs
    
    # Test with structure
    mock_structure = MagicMock()
    mock_structure.pdb_file = "test.pdb"
    mock_structure.chain = "A"
    mock_structure.plddt_file = None
    # modify mock so that isinsance returns true
    mock_structure.__class__ = ProteinStructure
    
    seq3 = ProteinSequence("ACDE", id="seq1")
    seq3.structure = mock_structure
    
    hash3 = model._get_protein_hashes([seq3])[0]
    assert hash1 != hash3  # Hashes should differ due to structure
    
    # 3. Test batch caching operations
    model = TestCacheModel(metadata_folder=test_dir, use_cache=True)
    sequences = ProteinSequences([
        ProteinSequence("ACDE", id="seq1"),
        ProteinSequence("FGHI", id="seq2"),
        ProteinSequence("JKLM", id="seq3")
    ])
    
    # First run - should cache all results
    result1 = model.transform(sequences)
    assert model._transform_call_count == 1
    
    # Second run - should use cache for all
    result2 = model.transform(sequences)
    assert model._transform_call_count == 1  # No additional calls
    np.testing.assert_array_equal(result1, result2)
    
    # 4. Test cache invalidation based on model state
    model.new_param_ = "changed"  # Change model state
    result3 = model.transform(sequences)
    assert model._transform_call_count == 2  # Should have called transform again
    
    # 5. Test partial cache usage with mixed sequences
    model = TestCacheModel(metadata_folder=test_dir, use_cache=True)
    # First cache some sequences
    result1 = model.transform(sequences[:2])
    assert model._transform_call_count == 1
    
    # Now add a new sequence
    extended_sequences = ProteinSequences([
        sequences[0], sequences[1], 
        ProteinSequence("NOPQ", id="seq4")  # New sequence
    ])
    
    # Should use cache for first two, compute only the third
    result2 = model.transform(extended_sequences)
    assert model._transform_call_count == 2  # Only one additional call
    assert result2.shape == (3, 2)  # Should have 3 sequences
    np.testing.assert_array_equal(result2[:2], result1)  # First two should match
    
    # Now all should be cached
    result3 = model.transform(extended_sequences)
    assert model._transform_call_count == 2  # No additional calls
    
    # 6. Test HDF5 storage and retrieval
    # Create a test array with complex shape
    complex_array = np.random.random((2, 3, 4))
    
    # Manually cache it
    protein_hashes = model._get_protein_hashes(sequences[:2])
    model_state = model._get_model_state_hash()
    protein_lengths = {ph: len(sequences[i]) for i, ph in enumerate(protein_hashes)}
    
    results_dict = {
        protein_hashes[0]: complex_array[0:1],
        protein_hashes[1]: complex_array[1:2]
    }
    
    model._batch_cache_results(
        results_dict,
        model_state,
        protein_lengths
    )
    
    # Retrieve and check
    cached_results = model._batch_get_cached_results(protein_hashes)
    for i, ph in enumerate(protein_hashes):
        np.testing.assert_array_equal(cached_results[ph], complex_array[i:i+1])
    
    # 7. Test _batch_is_cached function
    cache_status = model._batch_is_cached(protein_hashes, model_state)
    for ph in protein_hashes:
        assert cache_status[ph] == True
    
    # Check with wrong model state
    wrong_state = "wrong_state"
    cache_status = model._batch_is_cached(protein_hashes, wrong_state)
    for ph in protein_hashes:
        assert cache_status[ph] == False
    

def test_expects_no_fit_mixin():
    """Test the ExpectsNoFitMixin behavior."""
    class TestNoFitModel(ExpectsNoFitMixin, ProteinModelWrapper):
        def _fit(self, X, y=None):
            self.X_ = X
            self.fitted_ = True
            return self
        
        def _transform(self, X):
            return np.ones((len(X), 1))
    
    # Create model and test sequences
    model = TestNoFitModel()
    sequences = ProteinSequences.from_list(["ACDE", "FGHI"])
    
    # Should fit without input
    model.fit()
    assert model.fitted_ is True
    assert model.X_ is None
    
    # Should ignore input when fitting
    model = TestNoFitModel()
    with pytest.warns(UserWarning):
        model.fit(sequences)
    assert model.fitted_ is True
    assert model.X_ is None
    
    # Should still transform normally
    result = model.transform(sequences)
    assert result.shape == (2, 1)
        

def test_requires_msa_per_sequence_mixin():
    """Test the RequiresMSAPerSequenceMixin behavior."""
    class TestMSAPerSeqModel(RequiresMSAPerSequenceMixin, RequiresWTDuringInferenceMixin, ProteinModelWrapper):
        def _fit(self, X, y=None):
            self.fitted_ = True
            return self
        
        def _transform(self, X):
            return np.array([len(seq.msa) for seq in X]).reshape(-1, 1)
    
    # Create model
    model = TestMSAPerSeqModel()
    model.fitted_ = True  # For testing transform directly
    
    # Create sequences with MSAs
    msa1 = ProteinSequences.from_list(["ACDE", "FGHI"])
    msa2 = ProteinSequences.from_list(["JKLM", "NOPQ"])
    
    seq1 = ProteinSequence("ACDE", id="seq1")
    seq1.msa = msa1
    
    seq2 = ProteinSequence("FGHI", id="seq2")
    seq2.msa = msa2
    
    sequences = ProteinSequences([seq1, seq2])
    
    # Should work with MSAs present
    result = model.transform(sequences)
    np.testing.assert_array_equal(result, np.array([[2], [2]]))
    
    # Test with missing MSA - should raise ValueError
    seq3 = ProteinSequence("JKLM", id="seq3")  # No MSA
    sequences_missing_msa = ProteinSequences([seq1, seq3])
    
    with pytest.raises(ValueError):
        model.transform(sequences_missing_msa)
    
    # Test with WT that has MSA - should use WT MSA for missing ones
    wt = ProteinSequence("WTYZ", id="wt")
    wt.msa = ProteinSequences.from_list(["WTYZ", "RSTU", "VTYZ"])  # MSA with 3 sequences
    
    model_with_wt = TestMSAPerSeqModel(wt=wt)
    model_with_wt.fitted_ = True
    
    with pytest.warns(UserWarning):
        result = model_with_wt.transform(sequences_missing_msa)
    np.testing.assert_array_equal(result, np.array([[2], [3]]))  # seq3 gets wt MSA with 3 sequences
    
    # Test with non-matching MSA width
    seq4 = ProteinSequence("ACDE", id="seq4")
    seq4.msa = ProteinSequences.from_list(["ACDEFG", "HIJKLM"])  # Longer than seq4
    sequences_wrong_width = ProteinSequences([seq4])
    
    with pytest.raises(ValueError, match="Not all sequence MSAs have the same width"):
        model.transform(sequences_wrong_width)
    
    # Test with CanHandleAlignedSequencesMixin
    class TestMSAAlignedModel(RequiresMSAPerSequenceMixin, RequiresWTDuringInferenceMixin, CanHandleAlignedSequencesMixin, ProteinModelWrapper):
        def _fit(self, X, y=None):
            self.fitted_ = True
            return self
        
        def _transform(self, X):
            return np.array([len(seq.msa) for seq in X]).reshape(-1, 1)
    
    # Create aligned model
    aligned_model = TestMSAAlignedModel()
    aligned_model.fitted_ = True
    
    # Should work with non-matching MSA width when CanHandleAlignedSequencesMixin is present
    result = aligned_model.transform(sequences_wrong_width)
    np.testing.assert_array_equal(result, np.array([[2]]))


def test_requires_wt_during_inference_mixin():
    """Test the RequiresWTDuringInferenceMixin behavior."""
    class TestWTInferenceModel(RequiresWTDuringInferenceMixin, RequiresWTToFunctionMixin, ProteinModelWrapper):
        def _fit(self, X, y=None):
            self.fitted_ = True
            return self
        
        def _transform(self, X):
            # Don't normalize by WT in _transform since the mixin handles it
            return np.array([len(seq) for seq in X]).reshape(-1, 1)
    
    # Create model with WT
    wt = ProteinSequence("ACDE")
    model = TestWTInferenceModel(wt=wt)
    model.fitted_ = True
    
    # Transform sequences
    sequences = ProteinSequences.from_list(["FGHI", "JKLMN"])
    result = model.transform(sequences)
    
    # Check results - should not automatically normalize by WT since the mixin
    # indicates the model handles WT normalization internally
    np.testing.assert_array_equal(result, np.array([[4], [5]]))
    
    # Compare to a model without the mixin, which should automatically normalize
    class TestNoMixinModel(RequiresWTToFunctionMixin, ProteinModelWrapper):
        def _fit(self, X, y=None):
            self.fitted_ = True
            return self
        
        def _transform(self, X):
            return np.array([len(seq) for seq in X]).reshape(-1, 1)
    
    no_mixin_model = TestNoMixinModel(wt=wt)
    no_mixin_model.fitted_ = True
    
    # Transform sequences
    result_no_mixin = no_mixin_model.transform(sequences)
    
    # Check results - should automatically normalize by WT length (4)
    np.testing.assert_array_equal(result_no_mixin, np.array([[0], [1]]))
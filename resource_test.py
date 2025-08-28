# resource_test.py
"""
Benchmarking script for AIDE protein prediction models.

This script measures execution times for:
1. Zero-shot predictors (with different marginal methods where applicable)
2. Embedders

Both CPU and GPU performance are measured when device parameter is available.
Results are cached to avoid re-running expensive computations.
Timeouts can optionally be recorded to store ">X hours" resource benchmarks.

Usage:
    python benchmark_models.py --use_cache --force --output_dir ./benchmark_results --record_timeouts
"""

import os
import sys
import time
import json
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

import pandas as pd
import numpy as np
import torch

from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence, ProteinStructure
from aide_predict.bespoke_models import TOOLS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Timeout in seconds (30 minutes)
TIMEOUT = 1800

class TimeoutError(Exception):
    """Raised when a model takes too long to run."""
    pass

def load_test_data() -> Tuple[ProteinSequence, ProteinSequences]:
    """
    Load GFP test data for benchmarking.
    
    Returns:
        Tuple containing:
        - GFP wild-type sequence with MSA and structure
        - Test sequences for prediction
    """
    # Load GFP data
    gfp_pdb_path = os.path.join('tests', 'data', 'GFP_AEQVI_Sarkisyan_2016.pdb')
    gfp_msa_path = os.path.join('tests', 'data', 'GFP_AEQVI_Sarkisyan_2016.a3m')
    gfp_csv_path = os.path.join('tests', 'data', 'GFP_AEQVI_Sarkisyan_2016.csv')
    
    # Create GFP WT sequence with structure and MSA
    gfp_structure = ProteinStructure(pdb_file=gfp_pdb_path)
    gfp_wt = ProteinSequence.from_fasta(gfp_msa_path).upper()
    gfp_wt.msa = gfp_wt.msa.upper()
    gfp_wt.structure = gfp_structure
    id_ = "GFP_WT/1-238"
    gfp_wt.id = id_
    gfp_wt.msa[0].id = id_

    # Load test sequences
    gfp_assay_data = pd.read_csv(gfp_csv_path)
    gfp_test_sequences = ProteinSequences.from_list(
        gfp_assay_data['mutated_sequence'].tolist()[:50]  # Limit for faster benchmarking
    )
    
    return gfp_wt, gfp_test_sequences

def get_available_devices() -> List[str]:
    """Get list of available devices for testing."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:1")
    elif torch.backends.mps.is_available():
        devices.append("mps")
    return devices

def has_device_parameter(model_class) -> bool:
    """Check if model constructor has device parameter."""
    import inspect
    sig = inspect.signature(model_class.__init__)
    return 'device' in sig.parameters

def can_use_cache(model_class) -> bool:
    """Check if model supports caching."""
    from aide_predict.bespoke_models.base import CacheMixin
    return issubclass(model_class, CacheMixin)

def get_marginal_methods(model_class) -> List[str]:
    """Get available marginal methods for transformer models."""
    marginal_methods = []
    
    # Check if it's a transformer-based model with marginal methods
    if hasattr(model_class, '__bases__'):
        for base in model_class.__bases__:
            if 'LikelihoodTransformerBase' in str(base):
                marginal_methods = ["wildtype_marginal", "mutant_marginal", "masked_marginal"]
                break
    
    # Special cases for specific models
    model_name = model_class.__name__
    if model_name in ["ESM2LikelihoodWrapper", "SaProtLikelihoodWrapper", "MSATransformerLikelihoodWrapper"]:
        marginal_methods = ["wildtype_marginal", "mutant_marginal", "masked_marginal"]
    
    return marginal_methods if marginal_methods else [None]

def timeout_handler(func, timeout_duration=TIMEOUT):
    """Execute function with timeout."""
    import signal
    
    def timeout_signal_handler(signum, frame):
        raise TimeoutError(f"Function timed out after {timeout_duration} seconds")
    
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(timeout_duration)
    
    try:
        result = func()
        signal.alarm(0)  # Cancel timeout
        return result
    except TimeoutError:
        signal.alarm(0)  # Cancel timeout
        raise

def benchmark_predictor(model_class, wt_sequence: ProteinSequence, test_sequences: ProteinSequences, 
                       device: str, marginal_method: Optional[str], use_cache: bool, 
                       metadata_base_dir: str) -> Dict[str, Any]:
    """
    Benchmark a zero-shot predictor model.
    
    Returns:
        Dict with timing results and model info
    """
    model_name = model_class.__name__
    
    # Create metadata folder - only include device in name if device parameter is supported
    if has_device_parameter(model_class):
        metadata_folder = os.path.join(metadata_base_dir, f"{model_name}_{device}")
    else:
        metadata_folder = os.path.join(metadata_base_dir, model_name)
    
    if marginal_method:
        metadata_folder += f"_{marginal_method}"
    os.makedirs(metadata_folder, exist_ok=True)
    
    # Model configuration
    model_kwargs = {
        'metadata_folder': metadata_folder,
        'wt': wt_sequence,
    }
    if can_use_cache(model_class):
        model_kwargs['use_cache'] = use_cache
    
    
    # Add device parameter if supported
    if has_device_parameter(model_class):
        model_kwargs['device'] = device
    
    # Add marginal method if applicable
    if marginal_method:
        model_kwargs['marginal_method'] = marginal_method
    
    # Special case configurations
    if model_name == "VESPAWrapper":
        model_kwargs['light'] = True

        # we need to change the test sequences to be exactly 1 mutatuon from WT
        # it can't handle the gfp mutants with multiple mutations
        test_sequences = wt_sequence.saturation_mutagenesis()[:50]
    elif model_name == "EVEWrapper":
        model_kwargs['training_steps'] = 30000  # Reduced for faster benchmarking
        # the gfp data contains a bunch of mutations not in the MSA support for this subset MSA -
        # we should probably have the external call script just ignore these mutations but for now create valid mutants
        test_sequences = wt_sequence.saturation_mutagenesis()[500:550]
    elif model_name == "EVMutationWrapper":
        model_kwargs['iterations'] = 100  # Reduced for faster benchmarking
    elif model_name == "SSEmbWrapper":
        model_kwargs['gpu_id'] = 1
    elif model_name == "SaProtLikelihoodWrapper":
        model_kwargs['foldseek_path'] = 'foldseek'
    
    # Initialize model
    start_time = time.time()
    model = model_class(**model_kwargs)
    init_time = time.time() - start_time
    
    # Fit model
    start_time = time.time()
    def fit_func():
        model.fit()
    timeout_handler(fit_func)
    fit_time = time.time() - start_time
    
    # Predict
    start_time = time.time()
    def predict_func():
        return model.predict(test_sequences)
    predictions = timeout_handler(predict_func)
    predict_time = time.time() - start_time
    
    return {
        'status': 'success',
        'init_time': init_time,
        'fit_time': fit_time,
        'predict_time': predict_time,
        'total_time': init_time + fit_time + predict_time,
        'n_predictions': len(predictions) if predictions is not None else 0,
        'device': device if has_device_parameter(model_class) else None,
        'marginal_method': marginal_method
    }

def benchmark_embedder(model_class, wt_sequence, test_sequences: ProteinSequences, device: str, 
                      use_cache: bool, metadata_base_dir: str) -> Dict[str, Any]:
    """
    Benchmark an embedder model.
    
    Returns:
        Dict with timing results and model info
    """
    model_name = model_class.__name__
    
    # Create metadata folder - only include device in name if device parameter is supported
    if has_device_parameter(model_class):
        metadata_folder = os.path.join(metadata_base_dir, f"{model_name}_{device}")
    else:
        metadata_folder = os.path.join(metadata_base_dir, model_name)
    os.makedirs(metadata_folder, exist_ok=True)
    
    # Model configuration
    model_kwargs = {
        'metadata_folder': metadata_folder,
        'wt': wt_sequence
        # 'pool': True  # Pool embeddings for sequence-level representations
    }
    # check if model can use cache
    if can_use_cache(model_class):
        model_kwargs['use_cache'] = use_cache

    # Add device parameter if supported
    if has_device_parameter(model_class):
        model_kwargs['device'] = device
    
    # Special case configurations
    if model_name == "SSEmbEmbedding":
        model_kwargs['gpu_id'] = 1
    elif model_name == "MSATransformerEmbedding":
        model_kwargs['n_msa_seqs'] = 360
    elif model_name == "SaProtEmbedding":
        model_kwargs['foldseek_path'] = 'foldseek'
    elif model_name == "KmerEmbedding":
        model_kwargs['k'] = 3
    
    # Initialize model
    start_time = time.time()
    model = model_class(**model_kwargs)
    init_time = time.time() - start_time
    
    # Fit model
    start_time = time.time()
    def fit_func():
        model.fit(test_sequences)
    timeout_handler(fit_func)
    fit_time = time.time() - start_time
    
    # Transform (embed)
    start_time = time.time()
    def transform_func():
        return model.transform(test_sequences)
    embeddings = timeout_handler(transform_func)
    transform_time = time.time() - start_time
    
    return {
        'status': 'success',
        'init_time': init_time,
        'fit_time': fit_time,
        'transform_time': transform_time,
        'total_time': init_time + fit_time + transform_time,
        'n_sequences': len(test_sequences),
        'embedding_shape': embeddings.shape if embeddings is not None else None,
        'device': device if has_device_parameter(model_class) else None
    }

def load_existing_results(output_file: str) -> Dict[str, Any]:
    """Load existing benchmark results if they exist."""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return {}

def save_results(results: Dict[str, Any], output_file: str):
    """Save benchmark results to JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def create_test_key(model_name: str, device: str, marginal_method: Optional[str], 
                   has_device_param: bool) -> str:
    """Create a consistent test key based on model capabilities."""
    if has_device_param:
        key = f"{model_name}_{device}"
    else:
        key = model_name
    
    if marginal_method:
        key += f"_{marginal_method}"
    
    return key

def should_skip_test(model_name: str, device: str, marginal_method: Optional[str], 
                    existing_results: Dict[str, Any], force: bool, model_class, 
                    record_timeouts: bool) -> bool:
    """
    Check if test should be skipped based on existing results.
    
    Args:
        model_name: Name of the model
        device: Device being tested
        marginal_method: Marginal method if applicable
        existing_results: Previously saved results
        force: If True, never skip tests
        model_class: Model class for determining test key
        record_timeouts: If True, don't skip tests that previously timed out
    
    Returns:
        True if test should be skipped, False otherwise
    """
    if force:
        return False
    
    key = create_test_key(model_name, device, marginal_method, has_device_parameter(model_class))
    
    # If no existing result, don't skip
    if key not in existing_results:
        return False
    
    existing_result = existing_results[key]
    
    # If recording timeouts, don't skip timeout results (allow retry)
    if not record_timeouts and existing_result.get('status') == 'timeout':
        return False
    
    # Skip if we have a successful result or (if recording timeouts) any completed result
    return existing_result.get('status') in (['success'] if not record_timeouts else ['success', 'timeout'])

def is_predictor(model_class) -> bool:
    """Check if model is a predictor (vs embedder)."""
    # Check if the model has CanRegressMixin in its MRO
    from aide_predict.bespoke_models.base import CanRegressMixin
    return any(issubclass(base, CanRegressMixin) for base in model_class.__mro__ if base != object)

def is_embedder(model_class) -> bool:
    """Check if model is an embedder."""
    model_name = model_class.__name__
    return 'Embedding' in model_name

def main():
    parser = argparse.ArgumentParser(description='Benchmark AIDE protein prediction models')
    parser.add_argument('--use_cache', action='store_true', default=False,
                      help='Enable model-level caching (passed to model use_cache parameter)')
    parser.add_argument('--force', action='store_true', default=False,
                      help='Force re-running all tests, ignoring existing results')
    parser.add_argument('--output_dir', type=str, default='./benchmark_results',
                      help='Directory to store benchmark results')
    parser.add_argument('--predictors_only', action='store_true', default=False,
                      help='Only benchmark predictors, skip embedders')
    parser.add_argument('--embedders_only', action='store_true', default=False,
                      help='Only benchmark embedders, skip predictors')
    parser.add_argument('--dry_run', action='store_true', default=False,
                      help='Show what would be run without actually running tests')
    parser.add_argument('--log_level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='Set logging level')
    parser.add_argument('--record_timeouts', action='store_true', default=False,
                      help='Record timeout results (">X hours") for resource benchmarking. '
                           'If enabled, timeouts are saved and skipped in future runs.')
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Output files
    predictor_results_file = os.path.join(args.output_dir, 'predictor_benchmark_results.json')
    embedder_results_file = os.path.join(args.output_dir, 'embedder_benchmark_results.json')
    
    # Load existing results (always load unless --force is used)
    existing_predictor_results = load_existing_results(predictor_results_file)
    existing_embedder_results = load_existing_results(embedder_results_file)
    
    # Load test data
    logger.info("Loading test data...")
    try:
        gfp_wt, gfp_test_sequences = load_test_data()
        logger.info(f"Loaded GFP WT sequence (length: {len(gfp_wt)})")
        logger.info(f"Loaded {len(gfp_test_sequences)} test sequences")
        logger.debug(f"GFP WT has MSA: {gfp_wt.has_msa}")
        logger.debug(f"GFP WT has structure: {gfp_wt.structure is not None}")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        sys.exit(1)
    
    # Get available devices
    devices = get_available_devices()
    logger.info(f"Available devices: {devices}")
    
    # Log timeout recording setting
    if args.record_timeouts:
        logger.info(f"Timeout recording ENABLED - timeouts will be recorded as '>={TIMEOUT/3600:.1f} hours'")
    else:
        logger.info("Timeout recording DISABLED - timeouts will NOT be saved and can be retried")
    
    # Dry run mode - just show what would be executed
    if args.dry_run:
        logger.info("=== DRY RUN MODE - No tests will be executed ===")
        
        total_tests = 0
        
        # Count predictor tests
        if not args.embedders_only:
            logger.info("\nPredictor tests that would be run:")
            for model_class in TOOLS:
                if not is_predictor(model_class):
                    continue
                    
                model_name = model_class.__name__
                has_device_param = has_device_parameter(model_class)
                
                # Check if model is available
                if hasattr(model_class, '_available') and not model_class._available.value:
                    logger.debug(f"Would skip {model_name}: {model_class._available.message}")
                    continue
                
                marginal_methods = get_marginal_methods(model_class)
                
                # Determine which devices to test
                test_devices = devices if has_device_param else ["cpu"]
                
                for marginal_method in marginal_methods:
                    for device in test_devices:
                        test_key = create_test_key(model_name, device, marginal_method, has_device_param)
                        
                        if should_skip_test(model_name, device, marginal_method, existing_predictor_results, 
                                          args.force, model_class, args.record_timeouts):
                            logger.debug(f"Would skip {test_key} (already completed)")
                            continue
                        
                        logger.info(f"  Would run: {test_key}")
                        total_tests += 1
        
        # Count embedder tests
        if not args.predictors_only:
            logger.info("\nEmbedder tests that would be run:")
            for model_class in TOOLS:
                if not is_embedder(model_class):
                    continue
                    
                model_name = model_class.__name__
                has_device_param = has_device_parameter(model_class)
                
                # Check if model is available
                if hasattr(model_class, '_available') and not model_class._available.value:
                    logger.debug(f"Would skip {model_name}: {model_class._available.message}")
                    continue
                
                # Determine which devices to test
                test_devices = devices if has_device_param else ["cpu"]
                
                for device in test_devices:
                    test_key = create_test_key(model_name, device, None, has_device_param)
                    
                    if should_skip_test(model_name, device, None, existing_embedder_results, 
                                      args.force, model_class, args.record_timeouts):
                        logger.debug(f"Would skip {test_key} (already completed)")
                        continue
                    
                    logger.info(f"  Would run: {test_key}")
                    total_tests += 1
        
        logger.info(f"\nTotal tests to run: {total_tests}")
        logger.info("Run without --dry_run to execute these tests.")
        return
    
    # Benchmark predictors
    if not args.embedders_only:
        logger.info("Starting predictor benchmarking...")
        predictor_results = existing_predictor_results.copy()
        
        for model_class in TOOLS:
            if not is_predictor(model_class):
                continue
                
            model_name = model_class.__name__
            has_device_param = has_device_parameter(model_class)
            logger.info(f"Benchmarking predictor: {model_name}")
            
            # Check if model is available
            if hasattr(model_class, '_available') and not model_class._available.value:
                logger.warning(f"Skipping {model_name}: {model_class._available.message}")
                continue
            
            # Get marginal methods if applicable
            marginal_methods = get_marginal_methods(model_class)
            logger.debug(f"Marginal methods for {model_name}: {marginal_methods}")
            
            # Determine which devices to test
            test_devices = devices if has_device_param else ["cpu"]
            
            for marginal_method in marginal_methods:
                for device in test_devices:
                    # Create test key
                    test_key = create_test_key(model_name, device, marginal_method, has_device_param)
                    
                    if should_skip_test(model_name, device, marginal_method, existing_predictor_results, 
                                      args.force, model_class, args.record_timeouts):
                        logger.info(f"Skipping {test_key} (already completed)")
                        continue
                    
                    logger.info(f"Running {test_key}...")
                    
                    # Run benchmark
                    try:
                        result = benchmark_predictor(
                            model_class=model_class,
                            wt_sequence=gfp_wt,
                            test_sequences=gfp_test_sequences,
                            device=device,
                            marginal_method=marginal_method,
                            use_cache=args.use_cache,
                            metadata_base_dir=os.path.join(args.output_dir, 'metadata')
                        )
                        
                        # Store successful result
                        result['model_name'] = model_name
                        result['timestamp'] = datetime.now().isoformat()
                        predictor_results[test_key] = result
                        
                        logger.info(f"Completed {test_key}: {result['status']}")
                        if result['status'] == 'success':
                            logger.info(f"  Total time: {result['total_time']:.2f}s")
                            logger.debug(f"  Init: {result['init_time']:.2f}s, Fit: {result['fit_time']:.2f}s, Predict: {result['predict_time']:.2f}s")
                        elif result['status'] == 'error':
                            logger.error(f"  Error: {result['error']}")
                        
                        # Save results after each successful test
                        save_results(predictor_results, predictor_results_file)
                        
                    except TimeoutError:
                        if args.record_timeouts:
                            # Record the timeout as a benchmark result
                            timeout_result = {
                                'status': 'timeout',
                                'timeout_duration': TIMEOUT,
                                'timeout_hours': TIMEOUT / 3600,
                                'resource_benchmark': f">={TIMEOUT/3600:.1f} hours",
                                'device': device if has_device_param else None,
                                'marginal_method': marginal_method,
                                'model_name': model_name,
                                'timestamp': datetime.now().isoformat(),
                                'message': f"Model exceeded {TIMEOUT/3600:.1f} hour timeout limit"
                            }
                            predictor_results[test_key] = timeout_result
                            logger.warning(f"Test {test_key} timed out - recorded as resource benchmark (>={TIMEOUT/3600:.1f} hours)")
                            save_results(predictor_results, predictor_results_file)
                        else:
                            logger.warning(f"Test {test_key} timed out - will retry next run (timeout recording disabled)")
                        continue
                    except Exception as e:
                        # Handle other exceptions
                        result = {
                            'status': 'error',
                            'error': str(e),
                            'traceback': traceback.format_exc(),
                            'device': device if has_device_param else None,
                            'marginal_method': marginal_method,
                            'model_name': model_name,
                            'timestamp': datetime.now().isoformat()
                        }
                        predictor_results[test_key] = result
                        logger.error(f"Error in {test_key}: {str(e)}")
                        logger.debug(f"Full traceback: {traceback.format_exc()}")
                        save_results(predictor_results, predictor_results_file)
    
    # Benchmark embedders
    if not args.predictors_only:
        logger.info("Starting embedder benchmarking...")
        embedder_results = existing_embedder_results.copy()
        
        # Create test sequences for embedders (use GFP WT + test sequences)
        all_test_sequences = ProteinSequences([gfp_wt] + gfp_test_sequences.data[:20])
        logger.debug(f"Created {len(all_test_sequences)} sequences for embedder testing")
        
        for model_class in TOOLS:
            if not is_embedder(model_class):
                continue
                
            model_name = model_class.__name__
            has_device_param = has_device_parameter(model_class)
            logger.info(f"Benchmarking embedder: {model_name}")
            
            # Check if model is available
            if hasattr(model_class, '_available') and not model_class._available.value:
                logger.warning(f"Skipping {model_name}: {model_class._available.message}")
                continue
            
            # Determine which devices to test
            test_devices = devices if has_device_param else ["cpu"]
            
            for device in test_devices:
                test_key = create_test_key(model_name, device, None, has_device_param)
                
                if should_skip_test(model_name, device, None, existing_embedder_results, 
                                  args.force, model_class, args.record_timeouts):
                    logger.info(f"Skipping {test_key} (already completed)")
                    continue
                
                logger.info(f"Running {test_key}...")
                
                # Run benchmark
                try:
                    result = benchmark_embedder(
                        model_class=model_class,
                        test_sequences=all_test_sequences,
                        wt_sequence=gfp_wt,
                        device=device,
                        use_cache=args.use_cache,
                        metadata_base_dir=os.path.join(args.output_dir, 'metadata')
                    )
                    
                    # Store successful result
                    result['model_name'] = model_name
                    result['timestamp'] = datetime.now().isoformat()
                    embedder_results[test_key] = result
                    
                    logger.info(f"Completed {test_key}: {result['status']}")
                    if result['status'] == 'success':
                        logger.info(f"  Total time: {result['total_time']:.2f}s")
                        logger.debug(f"  Init: {result['init_time']:.2f}s, Fit: {result['fit_time']:.2f}s, Transform: {result['transform_time']:.2f}s")
                        if result.get('embedding_shape'):
                            logger.info(f"  Embedding shape: {result['embedding_shape']}")
                    elif result['status'] == 'error':
                        logger.error(f"  Error: {result['error']}")
                    
                    # Save results after each successful test
                    save_results(embedder_results, embedder_results_file)
                    
                except TimeoutError:
                    if args.record_timeouts:
                        # Record the timeout as a benchmark result
                        timeout_result = {
                            'status': 'timeout',
                            'timeout_duration': TIMEOUT,
                            'timeout_hours': TIMEOUT / 3600,
                            'resource_benchmark': f">={TIMEOUT/3600:.1f} hours",
                            'device': device if has_device_param else None,
                            'model_name': model_name,
                            'timestamp': datetime.now().isoformat(),
                            'message': f"Model exceeded {TIMEOUT/3600:.1f} hour timeout limit"
                        }
                        embedder_results[test_key] = timeout_result
                        logger.warning(f"Test {test_key} timed out - recorded as resource benchmark (>={TIMEOUT/3600:.1f} hours)")
                        save_results(embedder_results, embedder_results_file)
                    else:
                        logger.warning(f"Test {test_key} timed out - will retry next run (timeout recording disabled)")
                    continue

if __name__ == "__main__":
    main()
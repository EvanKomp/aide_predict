# external_calls/eve/_compute_evol_indices_one.py
'''
* Author: Evan Komp
* Created: 10/28/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology


Wrapper script for EVE. Expects to be called from EVE environment. 

Adapted from Script by P. Notin: https://github.com/OATML/EVE/tree/master
'''
import os
import sys
import json
import argparse
import pandas as pd
import torch

# Add EVE repo to Python path
eve_repo = os.environ.get('EVE_REPO')
if eve_repo:
    sys.path.insert(0, eve_repo)

from EVE import VAE_model
from utils import data_utils

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute evolutionary indices using EVE')
    
    # Core arguments for data and model
    parser.add_argument('--msa_file', type=str, required=True,
                        help='Path to the MSA file')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name for the model')
    parser.add_argument('--model_parameters', type=str, required=True,
                        help='Path to JSON file containing model parameters')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')

    # Processing parameters
    parser.add_argument('--theta_reweighting', type=float, default=0.2,
                        help='Parameter for MSA sequence re-weighting (default: 0.2)')
    parser.add_argument('--weights_folder', type=str, default='weights',
                        help='Folder to store sequence weights (default: weights/)')
    
    # Mutation analysis parameters
    parser.add_argument('--computation_mode', type=str, choices=['all_singles', 'input_mutations_list'], required=True,
                        help='Compute indices for all single mutations or from input list')
    parser.add_argument('--mutations_file', type=str,
                        help='Path to CSV file containing mutations to analyze (required if mode is input_mutations_list)')
    parser.add_argument('--output_folder', type=str, default='results',
                        help='Folder to store results (default: results/)')
    parser.add_argument('--output_suffix', type=str, default='',
                        help='Suffix to add to output filename')
    
    # Computational parameters
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to approximate delta ELBO (default: 10)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for computing indices (default: 256)')

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs(args.weights_folder, exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)

    print(f"Processing MSA file: {args.msa_file}")
    print(f"Using theta={args.theta_reweighting} for MSA re-weighting")
    print(f"Model name: {args.model_name}")

    # Process MSA data
    weights_path = os.path.join(args.weights_folder, 
                               f'{args.model_name}_theta_{args.theta_reweighting}.npy')
    data = data_utils.MSA_processing(
        MSA_location=args.msa_file,
        theta=args.theta_reweighting,
        use_weights=True,
        weights_location=weights_path
    )
    
    # Handle mutations based on computation mode
    if args.computation_mode == "all_singles":
        mutations_file = os.path.join(args.output_folder, f"{args.model_name}_all_singles.csv")
        data.save_all_singles(output_filename=mutations_file)
    else:
        if not args.mutations_file:
            raise ValueError("mutations_file must be provided when using input_mutations_list mode")
        mutations_file = args.mutations_file

    # Load model parameters and initialize model
    try:
        with open(args.model_parameters, 'r') as f:
            model_params = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading model parameters: {str(e)}")
        sys.exit(1)

    model = VAE_model.VAE_model(
        model_name=args.model_name,
        data=data,
        encoder_parameters=model_params["encoder_parameters"],
        decoder_parameters=model_params["decoder_parameters"],
        random_seed=42
    )
    model = model.to(model.device)

    # Load model checkpoint
    try:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Initialized VAE with checkpoint: {args.checkpoint}")
    except Exception as e:
        print(f"Error loading model checkpoint: {str(e)}")
        sys.exit(1)
    
    # Compute evolutionary indices
    list_valid_mutations, evol_indices, _, _ = model.compute_evol_indices(
        msa_data=data,
        list_mutations_location=mutations_file,
        num_samples=args.num_samples,
        batch_size=args.batch_size
    )

    # Create results dataframe
    results = pd.DataFrame({
        'protein_name': args.model_name,
        'mutations': list_valid_mutations,
        'evol_indices': evol_indices
    })
    
    # Save results
    output_file = os.path.join(
        args.output_folder,
        f"{args.model_name}_{args.num_samples}_samples{args.output_suffix}.csv"
    )
    
    # Append to existing file if it exists and isn't empty
    try:
        keep_header = os.stat(output_file).st_size == 0
    except:
        keep_header = True
        
    results.to_csv(output_file, index=False, mode='a', header=keep_header)
    print(f"Results saved to: {output_file}")


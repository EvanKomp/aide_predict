# external_calls/eve/_train_VAE_one.py
'''
* Author: Evan Komp
* Created: 10/28/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology


Wrapper script for EVE. Expects to be called from EVE environment. 

Adapted from Script by P. Notin: https://github.com/OATML/EVE/tree/master
'''
import os
import argparse
import json
import sys
import torch

# Add EVE repo to Python path
eve_repo = os.environ.get('EVE_REPO')
if eve_repo:
    sys.path.insert(0, eve_repo)

from EVE import VAE_model
from utils import data_utils

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train EVE VAE model on a single MSA')
    
    # Core arguments for single MSA processing
    parser.add_argument('--msa_file', type=str, required=True,
                        help='Path to the MSA file to process')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name for the model checkpoint')
    
    # Optional processing parameters
    parser.add_argument('--theta_reweighting', type=float, default=0.2,
                        help='Parameter for MSA sequence re-weighting (default: 0.2)')
    parser.add_argument('--weights_folder', type=str, default='weights',
                        help='Folder to store sequence weights (default: weights/)')
    
    # Model parameters and checkpoint locations
    parser.add_argument('--model_parameters', type=str, required=True,
                        help='Path to JSON file containing model parameters')
    parser.add_argument('--checkpoint_folder', type=str, default='checkpoints',
                        help='Folder to store model checkpoints (default: checkpoints/)')
    parser.add_argument('--logs_folder', type=str, default='logs',
                        help='Folder to store training logs (default: logs/)')
    
    args = parser.parse_args()

    # Create necessary directories if they don't exist
    os.makedirs(args.weights_folder, exist_ok=True)
    os.makedirs(args.checkpoint_folder, exist_ok=True)
    os.makedirs(args.logs_folder, exist_ok=True)

    # Construct paths
    weights_path = os.path.join(args.weights_folder, 
                               f'{args.model_name}_theta_{args.theta_reweighting}.npy')
    
    print(f"Processing MSA file: {args.msa_file}")
    print(f"Using theta={args.theta_reweighting} for MSA re-weighting")
    print(f"Model name: {args.model_name}")

    # Process MSA data
    data = data_utils.MSA_processing(
        MSA_location=args.msa_file,
        theta=args.theta_reweighting,
        use_weights=True,
        weights_location=weights_path
    )

    # Load model parameters
    try:
        with open(args.model_parameters, 'r') as f:
            model_params = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading model parameters from {args.model_parameters}: {str(e)}")
        exit(1)

    # Initialize model
    model = VAE_model.VAE_model(
        model_name=args.model_name,
        data=data,
        encoder_parameters=model_params["encoder_parameters"],
        decoder_parameters=model_params["decoder_parameters"],
        random_seed=42
    )
    if str(model.device) == 'cpu' and torch.backends.mps.is_available():
        model.device = 'mps'
        model.encoder.device = 'mps'
        model.decoder.device = 'mps'
    model = model.to(model.device)
    print("Using device:", model.device)

    # Update training parameters with new paths
    model_params["training_parameters"].update({
        'training_logs_location': args.logs_folder,
        'model_checkpoint_location': args.checkpoint_folder
    })

    # Train model
    print(f"Starting to train model: {args.model_name}")
    model.train_model(data=data, training_parameters=model_params["training_parameters"])

    # Save final model
    print(f"Saving model: {args.model_name}")
    final_checkpoint_path = os.path.join(args.checkpoint_folder, f"{args.model_name}_final")
    model.save(
        model_checkpoint=final_checkpoint_path,
        encoder_parameters=model_params["encoder_parameters"],
        decoder_parameters=model_params["decoder_parameters"],
        training_parameters=model_params["training_parameters"]
    )


# aide_predict/utils/plotting.py
'''
* Author: Evan Komp
* Created: 7/26/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Common plotting calls.
'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional

from aide_predict.utils.data_structures import ProteinSequences

def plot_protein_sequence_heatmap(sequences: ProteinSequences, 
                                  figsize: tuple = (20, 5),
                                  cmap: str = 'viridis',
                                  title: str = 'Protein Sequence Heatmap') -> plt.Figure:
    """
    Create a heatmap visualization of protein sequences with additional sequence properties.

    Args:
        sequences (ProteinSequences): A ProteinSequences object containing the protein sequences.
        figsize (tuple): Figure size (width, height) in inches.
        cmap (str): Colormap to use for the heatmap.
        title (str): Title of the plot.

    Returns:
        plt.Figure: The matplotlib Figure object containing the heatmap.
    """
    # Convert sequences to numeric representation
    aa_to_num = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    numeric_sequences = [[aa_to_num.get(aa, -1) for aa in str(seq)] for seq in sequences]

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(numeric_sequences, cmap=cmap, cbar=False, ax=ax)

    # Customize plot
    ax.set_xlabel('Amino Acid Position')
    ax.set_ylabel('Sequence Number')
    ax.set_title(title)

    # Add color legend
    cbar = plt.colorbar(ax.collections[0], ax=ax, orientation="vertical", pad=0.02)
    cbar.set_ticks(np.arange(20) + 0.5)
    cbar.set_ticklabels(list('ACDEFGHIKLMNPQRSTVWY'))
    cbar.set_label('Amino Acid', rotation=270, labelpad=15)

    plt.tight_layout()
    return fig

def plot_mutation_heatmap(mutations, scores):
    """
    Plot a heatmap of single point mutation scores.
    
    Parameters:
    mutations (list): List of mutation strings (e.g., ["L1V", "A2G", ...])
    scores (list): List of corresponding scores
    
    Returns:
    None (displays the plot)
    """
    # All possible amino acids
    all_aas = 'ACDEFGHIKLMNPQRSTVWY'
    
    # Extract residue positions and mutant amino acids
    positions = [int(m[1:-1]) for m in mutations]
    mutant_aas = [m[-1] for m in mutations]
    original_aas = [m[0] for m in mutations]
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Position': positions,
        'Mutant_AA': mutant_aas,
        'Original_AA': original_aas,
        'Score': scores
    })
    
    # Create a full matrix with all amino acids and positions
    full_matrix = pd.DataFrame(index=range(1, max(positions)+1), columns=list(all_aas))
    
    # Fill the matrix with scores
    for _, row in df.iterrows():
        full_matrix.at[row['Position'], row['Mutant_AA']] = row['Score']
    
    # Fill NaN values with a distinct value (e.g., -999) to color them differently
    full_matrix = full_matrix.fillna(-999)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 0.5*max(positions)))  # Adjust figure size
    sns.heatmap(full_matrix, center=0, ax=ax, cmap='coolwarm_r',
                cbar_kws={'label': 'Score'}, 
                mask=full_matrix == -999)  # Mask NaN values
    
    # Customize the plot
    plt.title('Single Point Mutation Scores')
    plt.xlabel('Mutant Amino Acid')
    plt.ylabel('Residue Position')
    
    # Add original amino acids to y-axis labels
    original_aa_dict = dict(zip(positions, original_aas))
    ax.set_yticks(range(len(full_matrix)))
    ax.set_yticklabels([f'{original_aa_dict.get(i+1, "?")} {i+1}' for i in range(len(full_matrix))])
    
    # Adjust aspect ratio to make cells square
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
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
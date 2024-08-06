# aide_predict/bespoke_models/predictors/vespa.py
'''
* Author: Evan Komp
* Created: 8/1/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Wrapper of VESPA:
Marquet, C. et al. Embeddings from protein language models predict conservation and variant effects. Hum Genet 141, 1629–1647 (2022).

This model embeds the sequences with a PLM, then uses the embeddings for a pretrained logistic regression model for conservation. These
are input into a model to predict single mutation effects.
'''
import os
from typing import Optional, Union, List
import numpy as np
import pandas as pd
import subprocess

from aide_predict.utils.data_structures import ProteinSequence, ProteinSequences
from aide_predict.bespoke_models.base import ProteinModelWrapper, CanRegressMixin, RequiresWTDuringInferenceMixin, RequiresWTToFunctionMixin
from aide_predict.utils.common import MessageBool

try:
    import vespa.predict.config
    from vespa.predict.vespa import VespaPred
    from vespa.predict import utils
    import torch
    AVAILABLE = MessageBool(True, "VESPA is available.")
except ImportError:
    AVAILABLE = MessageBool(False, "VESPA is not available. Please install it with requirements-vespa.txt.")


class VESPAWrapper(CanRegressMixin, RequiresWTDuringInferenceMixin, RequiresWTToFunctionMixin, ProteinModelWrapper):
    _available = AVAILABLE
    def __init__(self, metadata_folder: str=None, wt: Optional[Union[str, ProteinSequence]] = None, light: bool=True):
        super().__init__(metadata_folder=metadata_folder, wt=wt)
        self.light = light


    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'VESPA':
        self.fitted_ = True
        return self

    def _transform(self, X: ProteinSequences) -> np.ndarray:
        
        # check that each sequence is maximum a single point mutation from the wild type
        for seq in X:
            mutations = self.wt.get_mutations(seq)
            if len(mutations) != 1:
                raise ValueError(f"Sequence {seq} is not a single point mutation from the wild type sequence {self.wt}")

        mutation_file = os.path.join(self.metadata_folder, "mutations.txt")
        wt_fasta_file = os.path.join(self.metadata_folder, "wt.fasta")

        # Write mutations to file
        with open(mutation_file, 'w') as f:
            for seq in X:
                mutation = self.wt.get_mutations(seq)[0]
                fromAA, pos, toAA = mutation[0], mutation[1:-1], mutation[-1]
                f.write(f"{self.wt.id}_{fromAA}{pos}{toAA}\n")

        # create a temporary file for the wild type sequence
        ProteinSequences([self.wt]).to_fasta(wt_fasta_file)

        # run the model
        cmd = ["vespa", str(wt_fasta_file), "-m", str(mutation_file), "--prott5_weights_cache", self.metadata_folder]
        if self.light:
            cmd.append("--vespal")
        else:
            cmd.append("--vespa")
        stdo, stde = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.metadata_folder
        ).communicate()
        if stde:
            raise ValueError(f"VESPA failed with error: {stde.decode()}")
        
        results = []
        outpath = os.path.join(self.metadata_folder, "vespa_run_directory", "0.csv")
        column = "VESPA" if not self.light else "VESPAl"
        df = pd.read_csv(outpath, sep=';').set_index('Mutant')
        for seq in X:
            mutation = self.wt.get_mutations(seq)[0]
            fromAA, pos, toAA = mutation[0], mutation[1:-1], mutation[-1]
            result = df.loc[f"{fromAA}{pos-1}{toAA}"][column]
            results.append(result)

        return np.array(results).reshape(-1, 1)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return ["VESPA_score"]

---
title: API Examples
---

# API examples

The following should look and feel like canonical sklearn tasks/code. See the `demo` folder for more details and executable examples. Also see the [colab notebook](https://colab.research.google.com/drive/1baz4DdYkxaw6pPRTDscwh2o-Xqum5Krp#scrollTo=AV9VXhM6ebgI) to play with some if its capabilities in the cloud. Finally, checkout the notebooks in `showcase` where we conduct two full protein predictions optimization and scoring tasks on real data that are greater than small example sets. 

### 1. Checking which protein models are available given the data you have

```python
from aide_predict.utils.checks import check_model_compatability
seqs = ProteinSequences.from_csv('csv_file.csv', seq_col='sequence')
wt = seqs[0]

check_model_compatibility(
    training_msa=None,
    training_sequences=seqs,
    wt=wt,
)
>>>{'compatible': ['ESM2Embedding',
  'ESM2LikelihoodWrapper',
  'KmerEmbedding',
  'OneHotProteinEmbedding'],
 'incompatible': ['EVMutationWrapper',
  'HMMWrapper',
  'MSATransformerEmbedding',
  'MSATransformerLikelihoodWrapper',
  'OneHotAlignedEmbedding',
  'SaProtEmbedding',
  'SaProtLikelihoodWrapper',
  'VESPAWrapper']}
```

### 2. In silico mutagenesis using MSATransformer
```python
# data preparation
wt = ProteinSequence.from_fasta("data/msa.fasta") # assigns the msa attribute of the sequence
wt.has_msa
>>> True
wt.msa_same_width
>>> True
library = wt.saturation_mutagenesis()
mutations = library.ids
print(mutations[0])
>>> 'L1A'

# model fitting
model = MSATransformerLikelihoodWrapper(
   wt=wt,
   marginal_method="masked_marginal"
)
model.fit()

# make predictions for each mutated sequence
predictions = model.predict(library)

results = pd.DataFrame({'mutation': mutations, 'seqeunce': library,'prediction': predictions})
```

### 3. Compare a couple of zero shot predictors against experimental data
```python
# data preparation
X, y = ProteinSequences.from_csv("data/experimental_data.csv", seq_col='sequence', id_col='id', label_cols='experimental_value')
wt = X['my_id_for_WT']
msa = ProteinSequences.from_fasta("data/msa.fasta")
wt.msa = msa

# model defenitions
evmut = EVMutation(wt=wt, metadata_folder='./tmp/evm')
evmut.fit()
esm2 = ESM2LikelihoodWrapper(wt=wt, model_checkpoint='esm2_t33_650M_UR50S')
esm2.fit()
models = {'evmut': evmut, 'esm2': esm2}

# model fitting and scoring
for name, model in models.items():
    score = model.score(X, y)
    print(f"{name} score: {score}")
```

### 4. Train a supervised model to predict activity on an experimental combinatorial library, test on sequences with greater mutational depth than training
```python
# data preparation
sequences, y = ProteinSequences.from_csv("data/experimental_data.csv", seq_col='sequence', id_col='id', label_cols='experimental_value')
sequences.aligned
>>> True
sequences.fixed_length
>>> True

wt = sequences['my_id_for_WT']
mutational_depth = np.array([len(x.mutated_positions(wt)) for x in sequences])
test_mask = mutational_depth > 5
train_X = sequences[~test_mask]
train_y = y[~test_mask]
test_X = sequences[test_mask]
test_y = y[test_mask]

# embeddings protein sequences
# use mean pool embeddings of esm2
embedder = ESM2Embedding(pool='mean')
train_X = embedder.fit_transform(train_X)
test_X = embedder.transform(test_X)

# model fitting
model = RandomForestRegressor()
model.fit(train_X, train_y)

# model scoring
train_score = model.score(train_X, train_y)
test_score = model.score(test_X, test_y)
print(f"Train score: {train_score}, Test score: {test_score}")
```

### 5. Train a supervised predictor on a set of homologs, focusing only on positions of known importance, wrap the entire process into an sklearn pipeline including some standard sklearn transormers, and make predictions for a new set of homologs
```python
# data preparation
sequences, y_train = ProteinSequences.from_csv("data/training_data.csv", seq_col='sequence', id_col='id', label_cols='experimental_value')

wt = sequences['my_id_for_WT']
wt_important_positions = np.array([20, 21, 22, 33, 45]) # zero indexed, known from analysis elsewhere
sequences.aligned
>>> False
sequences.fixed_length
>>> False

# align the training sequences and get the important positions
msa = sequences.align_all()
msa.fixed_length
>>> False
msa.aligned
>>> True

wt_alignment_mapping = msa.get_alignment_mapping()['my_id_for_WT']
aligned_important_positions = wt_alignment_mapping[wt_important_positions]

# model defenitions
embedder = OneHotAlignedEmbedding(important_positions=aligned_important_positions).fit(msa)
scaler = StandardScaler()
feature_selector = VarianceThreshold(threshold=0.2)
predictor = RandomForestRegressor()
pipeline = Pipeline([
    ('embedder', embedder),
    ('scaler', scaler),
    ('feature_selector', feature_selector),
    ('predictor', predictor)
])

# model fitting
pipeline.fit(sequences, y_train)

# score new unaligned homologs
new_homologs = ProteinSequences.from_fasta("data/new_homologs.fasta")
y_pred = pipeline.predict(new_homologs)
```

### 6. Create new embedder or predictor within the aide framework

Here we create a K-mer counting embedding, except use Foldseek structure tokens instead of amino acids. We set the `_available` attribute to allow aide to dynamically check if the model is available to call.
```python
import numpy as np
from typing import List, Union, Optional
from collections import defaultdict

from aide_predict.bespoke_models.base import ProteinModelWrapper, CanHandleAlignedSequencesMixin, RequiresStructureMixin
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.utils.common import MessageBool
from aide_predict.bespoke_models.predictors.saprot import get_structure_tokens
#check if 'foldseek' is available to path for a terminal
if shutil.which('foldseek') is None:
    AVAILABLE = MessageBool(False, 'Foldseek is not available, please install and set environment variables.')
else:
    AVAILABLE = MessageBool(True, 'Foldseek is available')

class FoldseekKmerEmbedding(RequiresStructureMixin, CanHandleAlignedSequencesMixin, ProteinModelWrapper):
    _available=AVAILABLE
    def __init__(self, metadata_folder: str = None, 
                 k: int = 3, 
                 wt: ProteinSequence = None):
        super().__init__(metadata_folder=metadata_folder, wt=None)
        self.k = k
        self._kmer_to_index = {}

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> 'KmerEmbedding':
        unique_kmers = set()
        for seq in X:
            if seq.structure is None:
                raise ValueError("KmerEmbedding requires a structure to be present in each sequence.")

            struct_str = get_structure_tokens(seq.structure)
            unique_kmers.update(struct_str[i:i+self.k] for i in range(len(struct_str) - self.k + 1))
        
        self._kmer_to_index = {kmer: i for i, kmer in enumerate(sorted(unique_kmers))}
        self.n_features_ = len(self._kmer_to_index)
        self.fitted_ = True
        return self

    def _transform(self, X: ProteinSequences) -> np.ndarray:
        """
        Transform the protein sequences into K-mer embeddings.

        Args:
            X (ProteinSequences): The input protein sequences.

        Returns:
            np.ndarray: The K-mer embeddings for the sequences.
        """
        embeddings = np.zeros((len(X), self.n_features_), dtype=np.float32)

        for i, seq in enumerate(X):
            if seq.structure is None:
                raise ValueError("KmerEmbedding requires a structure to be present in each sequence.")
            
            struct_str = get_structure_tokens(seq.structure)
            for j in range(len(struct_str) - self.k + 1):
                kmer = struct_str[j:j+self.k]
                if kmer in self._kmer_to_index:
                    embeddings[i, self._kmer_to_index[kmer]] += 1

        return embeddings
```

Here we define an arbitrary predictor that calls a third party script and environment and communicates with aide via IO. We can check for model availability by checking for enviornment variables associated with the third party environment and location if necessary.


```python
import numpy as np
from typing import List, Union, Optional
from collections import defaultdict
import os
import subprocess
import tempfile
from aide_predict.bespoke_models.base import ProteinModelWrapper, RequiresStructureMixin, RequiresFixedLengthSequencesMixin
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.utils.common import MessageBool
from aide_predict.bespoke_models.predictors.saprot import get_structure_tokens

try:
    ENV_NAME = os.environ['BESPOKE_ENV_NAME']
    MODEL_BASE_DIR = os.environ['BESPOKE_MODEL_BASE_DIR']
    AVAILABLE = MessageBool(True, 'Model is available')
except KeyError:
    AVAILABLE = MessageBool(False, 'Model is not available, please install and set environment variables.')

class MyBespokeModel(ProteinModelWrapper, RequiresStructureMixin, RequiresFixedLengthSequencesMixin):
    _available = AVAILABLE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set params....

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None):
        # prepare data
        # call subprocess
        input_fasta = tempfile.NamedTemporaryFile(delete=False)
        X.to_fasta(input_fasta.name)
        # temdir for structures
        with tempfile.TemporaryDirectory() as tempdir:
            # pass structure files to the script
            structure_files = open(os.path.join(tempdir, 'structure_files.txt'), 'w')
            for seq in X:
                struct_path = seq.structure.pdb_file
                structure_files.write(f'{seq.id}\t{struct_path}\n')
            structure_files.close()

            # call subprocess in the environment and base dir
            subprocess.run(['python', 'path/to/train_script.py', '-s', structure_files, input_fasta.name, '-o', os.path.join(self.metadata_folder, 'model.pkl'), env=ENV_NAME, cwd=MODEL_BASE_DIR])
        return self

    def _transform(self, X: ProteinSequences) -> np.ndarray:
        # something similar to fit, call a predict script etc...

        return outputs.
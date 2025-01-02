---
title: API Examples
---

# API examples

The following should look and feel like canonical sklearn tasks/code. See the `demo` folder for more details and executable examples. Also see the [colab notebook](https://colab.research.google.com/drive/1baz4DdYkxaw6pPRTDscwh2o-Xqum5Krp#scrollTo=AV9VXhM6ebgI) to play with some if its capabilities in the cloud. Finally, checkout the notebooks in `showcase` where we conduct two full protein predictions optimization and scoring tasks on real data that are greater than small example sets. 

### Checking which protein models are available given the data you have

```python
from aide_predict.utils.checks import check_model_compatability
exp = pd.read_csv('exp.csv')
seqs = ProteinSequences.from_list(exp['sequence'].tolist())
wt = ProteinSequence("MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTELEVLFQGPLDPNSMATYEVLCEVARKLGTDDREVVLFLLNVFIPQPTLAQLIGALRALKEEGRLTFPLLAECLFRAGRRDLLRDLLHLDPRFLERHLAGTMSYFSPYQLTVLHVDGELCARDIRSLIFLSKDTIGSRSTPQTFLHWVYCMENLDLLGPTDVDALMSMLRSLSRVDLQRQVQTLMGLHLSGPSHSQHYRHTPLEHHHHHH", id='WT')

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

### In silico mutagenesis using MSATransformer
```python
# data preparation
wt = ProteinSequence(
    "LADDRTLLMAGVSHDLRTPLTRIRLATEMMSEQDGYLAESINKDIEECNAIIEQFIDYLR",
)
msa = ProteinSequences.from_fasta("data/msa.fasta")
library = wt.saturation_mutagenesis()
mutations = library.ids
print(mutations[0])
>>> 'L1A'

# model fitting
model = MSATransformerLikelihoodWrapper(
   wt=wt,
   marginal_method="masked_marginal"
)
model.fit(msa)

# make predictions for each mutated sequence
predictions = model.predict(library)

results = pd.DataFrame({'mutation': mutations, 'seqeunce': library,'prediction': predictions})
```

### Compare a couple of zero shot predictors against experimental data
```python
# data preparation
data = pd.read_csv("data/experimental_data.csv")
X = ProteinSequences.from_list(data['sequence'])
y = data['experimental_value']
wt = X['my_id_for_WT']
msa = ProteinSequences.from_fasta("data/msa.fasta")

# model defenitions
evmut = EVMutation(wt=wt, metadata_folder='./tmp/evm')
evmut.fit(msa)
esm2 = ESM2LikelihoodWrapper(wt=wt, model_checkpoint='esm2_t33_650M_UR50S')
esm2.fit([])
models = {'evmut': evmut, 'esm2': esm2}

# model fitting and scoring
for name, model in models.items():
    score = model.score(X, y)
    print(f"{name} score: {score}")
```

### Train a supervised model to predict activity on an experimental combinatorial library, test on sequences with greater mutational depth than training
```python
# data preparation
data = pd.read_csv("data/experimental_data.csv").set_index('id')
sequences = ProteinSequences.from_dict(data['sequence'])
sequences.aligned
>>> True
sequences.fixed_length
>>> True

wt = sequences['my_id_for_WT']
data['sequence'] = sequences
data['mutational_depth'] = data['sequence'].apply(lambda x: len(x.mutated_positions(wt)))
test = data[data['mutational_depth'] > 5]
train = data[data['mutational_depth'] <= 5]
train_X, train_y = train['sequence'], train['experimental_value']
test_X, test_y = test['sequence'], test['experimental_value']

# embeddings protein sequences
# use mean pool embeddings of esm2
embedder = ESM2Embedding(pool=True)
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

### Train a supervised predictor on a set of homologs, focusing only on positions of known importance, wrap the entire process into an sklearn pipeline including some standard sklearn transormers, and make predictions for a new set of homologs
```python
# data preparation
data = pd.read_csv("data/experimental_data.csv")
data.set_index('id', inplace=True)
sequences = ProteinSequences.from_dict(data['sequence'].to_dict())
y_train = data['experimental_value']

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

# score new analigned homologs
new_homologs = ProteinSequences.from_fasta("data/new_homologs.fasta")
y_pred = pipeline.predict(new_homologs)
```

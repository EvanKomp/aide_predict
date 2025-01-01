---
title: Combining models into scikit learn pipelines
---

# Building ML Pipelines

AIDE models can be combined with standard scikit-learn components into pipelines. Here's an example that combines one-hot encoding and ESM2 ZS predictions with a random forest:

```python
from aide_predict import OneHotProteinEmbedding, ESM2LikelihoodWrapper, ProteinSequence, ProteinSequences
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor

# Load data
sequences = ProteinSequences.from_fasta("sequences.fasta")
y = np.load("activity_values.npy")

# Create wild type reference
wt = sequences["wild_type"]

# Create feature union that combines raw OHE with scaled ESM2 scores
features = FeatureUnion([
    # One-hot encoding (keep as binary)
    ('ohe', OneHotProteinEmbedding(flatten=True)),
    
    # ESM2 features (apply scaling)
    ('esm2', Pipeline([
        ('predictor', ESM2LikelihoodWrapper(wt=wt, marginal_method="masked_marginal")),
        ('reshaper', FunctionTransformer(lambda x: x.reshape(-1, 1))),
        ('scaler', StandardScaler())
    ]))
])

# Create and train pipeline
pipeline = Pipeline([
    ('features', features),
    ('rf', RandomForestRegressor())
])

pipeline.fit(sequences, y)
predictions = pipeline.predict(sequences)
```

The pipeline can be saved and loaded like any scikit-learn model:

```python
from joblib import dump, load
dump(pipeline, 'protein_model.joblib')
```

All standard scikit-learn tools like `GridSearchCV` or `cross_val_score` can be used with these pipelines.

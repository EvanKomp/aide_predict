---
title: Supervised training and prediction
---
# Supervised Learning

## Overview

AIDE supports supervised machine learning by converting protein sequences into numerical features using embedding models. These features can then be used with any scikit-learn compatible model.

## Basic Example

Here's a complete example using ESM2 embeddings and random forest regression with hyperparameter optimization:

```python
from aide_predict import ESM2Embedding, ProteinSequences
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import numpy as np

# Load data
sequences = ProteinSequences.from_fasta("sequences.fasta")
y = np.load("activity_values.npy")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    sequences, y, test_size=0.2, random_state=42
)

# Create pipeline
pipeline = Pipeline([
    ('embedder', ESM2Embedding(pool='max', use_cache=True)),  # Create sequence-level embeddings
    ('rf', RandomForestRegressor(random_state=42))
])

# Define parameter space
param_distributions = {
    'rf__n_estimators': randint(100, 500),
    'rf__max_depth': [None] + list(range(10, 50, 10)),
    'rf__min_samples_split': randint(2, 20),
    'rf__min_samples_leaf': randint(1, 10)
}

# Random search
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=20,  # Number of parameter settings sampled
    cv=5,       # 5-fold cross-validation
    n_jobs=-1,  # Use all available cores
    scoring='r2',
    verbose=1
)

# Fit model
search.fit(X_train, y_train)

# Print results
print("\nBest parameters:", search.best_params_)
print("Best CV score:", search.best_score_)
print("Test score:", search.score(X_test, y_test))

# Make predictions on new sequences
new_sequences = ProteinSequences.from_fasta("new_sequences.fasta")
predictions = search.predict(new_sequences)
```

## Saving and loading models
Models can be dumped and loaded with joblib like any other scikit-learn model:

```python
import joblib

# Save the best model
joblib.dump(search.best_estimator_, 'protein_model.joblib')

# Load the model later
loaded_model = joblib.load('protein_model.joblib')
```

Note that this may currently break the `metadata_folder` attribute of models, unless it is loaded on the same machine in the same location.
In future, protocols to zip up this folder with the model during saving and loading will be provided.
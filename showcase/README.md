# A couple of tangible use cases for the porject.

For a brief and wide demo of what the package can do, see `demo`

## 1. Benchmark unsupervised, supervised, and combination models on a 4 site epistatic combinatorial library
Paper for the data: https://www.biorxiv.org/content/10.1101/2024.06.23.600144v1

Conducted in `epistatic_benchmarking.ipynb`. In this notebook, the dataset is tested in cross validation for:

1. Unsupervised models: MSATransformer, ESM2, EVMutation
2. Supervised models: One-hot, ESM2 embeddings, into Linear model or MLP. Random search conducted for hyperparameter optimization.


## 2. Creating a WT sequence PETase acitivity prediction model

Data from our recent paper: []

Conducted in `wt_petase_model_creation.ipynb`. In this notebook, we try a number of embedding and modeling strategies with extensive hyperparameter optimization to predict PETase activity at low pH on crystaline powder. The final model is dumped and can be opened to make predictions.
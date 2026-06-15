#!/usr/bin/env python
# coding: utf-8

# # Round 2: Filter with a more informed model, and "design" some sequences of higher order
# 
# In round 0 we did SSM, in round 1 we tested: additive enzymes of the best SSM hits, greedy search over some of the remaining (max order 3), and some UCB variants (like greedy but model uncertainty included. Here we are still working with formate.
# 
# Here, we will train a more performative model and then use it to produce a plate including:
# 1. Any remaining 2-3 order mutants not already tested above the 95%ile compared to observed activity
# 2. Order 5 mutants generated with BADASS - filter those scores also with ESM1v (?) something to limit pathology

# In[1]:


# builtins and utils
import os
import json
import hashlib
from tqdm.notebook import tqdm
import itertools

# DS stack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')
import scipy.stats as stats

# Downstream supervised tools
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, ElasticNet
from sklearn.feature_selection import SelectFromModel, VarianceThreshold 
import joblib

# aide predict
import aide_predict as ap
from aide_predict.utils.plotting import plot_mutation_heatmap, plot_protein_sequence_heatmap

# data handling
import datasets

# typing
from typing import List, Dict, Tuple, Union


# In[2]:


from sklearn.exceptions import ConvergenceWarning


# In[3]:


import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# In[4]:


# globals
POSITION_OFFSET = 10
MAX_MUTATION_COUNT = 4

MUTATION_COUNTS_OBSERVED = {}


# In[5]:


def get_importance_from_ttr(ttr):
    return ttr.regressor_.coef_


# In[6]:


def add_observed_mutations(mutations: List[str]):
    for mutation in mutations:
        if mutation in MUTATION_COUNTS_OBSERVED:
            MUTATION_COUNTS_OBSERVED[mutation] += 1
        else:
            MUTATION_COUNTS_OBSERVED[mutation] = 1


# In[7]:


# mutating wt to new variant
# the aide package does nto have an option for position offset
def mutate_wt(
        wild_type: ap.ProteinSequence,
        mutations: Union[str, List[str]],
        offset: int = POSITION_OFFSET,
        one_indexed: bool=True,
        return_str: bool=True
    ) -> ap.ProteinSequence:
    """"""
    if isinstance(mutations, str):
        mutations = [mutations]

    mutated = list(str(wild_type))
    for mutation in mutations:
        original_aa = mutation[0]
        position = int(mutation[1:-1])
        new_aa = mutation[-1]
        if one_indexed:
            position = position - 1
        if offset != 0:
            position = position - offset
        assert wild_type[position] == original_aa, f"Mutation {mutation} does not match wild type {wild_type[position]}"
        mutated[position] = new_aa
    if return_str:
        return "".join(mutated)
    else:
        return ap.ProteinSequence("".join(mutated))


# In[8]:


def get_mutations(
    wt: ap.ProteinSequence,
    variant: ap.ProteinSequence,
    offset: int = POSITION_OFFSET,
    one_indexed: bool=True
) -> List[str]:
    """Parse mutations from two protein sequences"""
    if not isinstance(wt, ap.ProteinSequence):
        wt = ap.ProteinSequence(wt)
    if not isinstance(variant, ap.ProteinSequence):
        variant = ap.ProteinSequence(variant)
    mutations = wt.get_mutations(variant)
    mutations_ = []
    for mutation in mutations:
        position = int(mutation[1:-1])
        if not one_indexed:
            position = position - 1
        if offset != 0:
            position = position + offset
        mutations_.append(f"{mutation[0]}{position}{mutation[-1]}")
    return mutations_


# In[9]:


# hashing protein sequences
def protein_hash(protein: str) -> str:
    return hashlib.md5(str(protein).encode()).hexdigest()


# In[10]:


kfold = KFold(n_splits=10, shuffle=True, random_state=42)


# In[11]:


def get_foldlike_iterator_from_df(df, split_mask_prefix):
    fold_index_iterator = []
    for col in df.columns:
        if col.startswith(split_mask_prefix):
            # the column is a bolean - 0 for train, 1 for test
            # we need to get indexes for train and test
            fold_index_iterator.append((
                df[df[col] == 0].index.tolist(),
                df[df[col] == 1].index.tolist()
            ))
    return fold_index_iterator


# In[12]:


class WeightedMAEScorer:
    def __init__(self, n_bins=20):
        self.n_bins = n_bins
        self.bin_edges = None
        self.bin_weights_map = None
    
    def fit(self, y):
        """Compute bin edges and weights from training data"""
        y = np.array(y)
        bin_counts, self.bin_edges = np.histogram(y, bins=self.n_bins)
        
        # fill in 0s with 1 to avoid division by 0
        bin_counts = np.maximum(bin_counts, 1)
        
        # Store the weight for each bin
        self.bin_weights_map = 1 / bin_counts
        
        return self
    
    def get_weights(self, y):
        """Get weights for new data using pre-computed bins"""
        if self.bin_edges is None:
            raise ValueError("Scorer must be fitted before getting weights")
            
        y = np.array(y)
        bin_index = np.clip(np.digitize(y, self.bin_edges) - 1, 0, self.n_bins - 1)
        
        # Use pre-computed weights
        weights = self.bin_weights_map[bin_index]
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return weights
    
    def score(self, y_true, y_pred):
        """Calculate weighted negative MAE using pre-computed bins"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        weights = self.get_weights(y_true)
        return -np.average(np.abs(y_true - y_pred), weights=weights)


# In[13]:


def random_hyperopt(model_pipeline, param_distributions, X, y, n_iter=100, cv=10, n_jobs=1, scorer='neg_mean_absolute_error', **kwargs):
    search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        n_jobs=n_jobs,
        verbose=10,
        random_state=42,
        scoring=scorer,
        error_score="raise",
    )
    search.fit(X, y, **kwargs)

    results_df = pd.DataFrame(search.cv_results_)
    best_params = search.best_params_
    best_score_mean = results_df[results_df['rank_test_score'] == 1]['mean_test_score'].values[0]
    best_score_std = results_df[results_df['rank_test_score'] == 1]['std_test_score'].values[0]
    return results_df, best_params, best_score_mean, best_score_std


# # 1. Model development
# 
# Test
# 
# 1) embedding methods: OHE, ESM mean pool, ESM max pool, ESM flatten with feature selection (LASSO)
# 2) Model head: linear, MLP, XGBoost

# ## 1.1 Data preparation

# In[14]:


alignment = ap.ProteinSequences.from_fasta('../data/msa/mda_hacl.a2m')
wt = alignment[0]


# In[15]:


len(alignment[0].with_no_gaps())


# In[16]:


ssm_data_raw = pd.read_excel('../data/experimental/round0_1_compiled.xlsx', sheet_name='Round 1 Data', skiprows=1)


# In[17]:


# remove column 0, 2 and row 1
ssm_data = ssm_data_raw.drop(ssm_data_raw.columns[0], axis=1)
ssm_data = ssm_data.drop(ssm_data.columns[1], axis=1)
ssm_data = ssm_data.drop(ssm_data.index[0])
# format long with columns [mutation_string, original, position, mutation, activity]
ssm_data.rename(columns={'Unnamed: 1': 'original_position_string'}, inplace=True)


# In[18]:


# parse positions/mutations
ssm_data['original_aa'] = ssm_data['original_position_string'].str[0]
ssm_data['position'] = ssm_data['original_position_string'].str[1:].astype(int)
ssm_data.drop(columns=['original_position_string'], inplace=True)
ssm_data = ssm_data.melt(id_vars=['original_aa', 'position'], var_name='mutation', value_name='activity').sort_values(by='position').reset_index(drop=True)


# In[19]:


# compute the position on our sequence, 0 indexed
ssm_data['position_on_wt_0_indexed'] = ssm_data['position'] - POSITION_OFFSET - 1
for _, row in ssm_data.iterrows():
    assert wt[row['position_on_wt_0_indexed']] == row['original_aa']


# In[20]:


ssm_data = ssm_data.dropna()

# add full sequence
ssm_data['mutation_string'] = ssm_data['original_aa'] + ssm_data['position'].astype(str) + ssm_data['mutation']
ssm_data['variant'] = ssm_data['mutation_string'].apply(lambda x: mutate_wt(wt, x))


# In[21]:


ssm_data['hash'] = ssm_data['variant'].apply(protein_hash)


# In[22]:


ssm_data['id'] = 'ssm_' + ssm_data['mutation_string']


# In[23]:


round1_data = pd.read_excel('../data/experimental/round0_1_compiled.xlsx', sheet_name='Round 2 data')


# In[24]:


round1_data.rename(columns={'FC relative to WT': 'activity', 'Mutations': 'id'}, inplace=True)
round1_data


# In[25]:


# convert mutations into variants
def get_variant_from_id(id: str) -> str:
    mutations = id.split('_')[-1]
    mutations = mutations.split('|')
    return mutate_wt(wt, mutations)

round1_data['variant'] = round1_data['id'].apply(get_variant_from_id)


# In[26]:


round1_data['hash'] = round1_data['variant'].apply(protein_hash)


# In[27]:


ssm_data


# In[28]:


round1_data


# In[29]:


# combine datasets
combined_data = pd.concat([ssm_data[['id', 'activity', 'variant', 'hash']], round1_data], axis=0)[['hash', 'variant', 'activity']].reset_index(drop=True)


# In[30]:


# aggregate over hash
combined_data = combined_data.groupby('hash').agg({
    'variant': 'first',
    'activity': 'mean'
}).reset_index()


# In[31]:


sns.histplot(combined_data['activity'])


# In[32]:


sns.histplot(np.log10(1+combined_data['activity']))


# In[33]:


combined_data['log_activity'] = np.log10(1+combined_data['activity'])


# In[34]:


(combined_data['log_activity'] > 0.6).sum()


# In[35]:


scorer = WeightedMAEScorer(n_bins=20)
scorer = scorer.fit(combined_data['log_activity'])
weights = scorer.get_weights(combined_data['log_activity'])


# In[36]:


# a way to map additive activity
ssm_activity_dict = ssm_data.set_index('mutation_string')['activity'].to_dict()


# In[37]:


ssm_activity_dict


# In[38]:


ssm_log_activity_dict = {k: np.log10(1+v) for k, v in ssm_activity_dict.items()}


# In[39]:


def get_additive_activity(mutations):
    mutation_strings = mutations
    fc_prediction = 1.0

    for mutation in mutation_strings:
        if mutation in ssm_activity_dict:
            fc_delta = ssm_activity_dict[mutation] - 1
            fc_prediction += fc_delta
        else:
            pass
    if fc_prediction < 0:
        return 0
    
    return fc_prediction


# ### Save data [SKIP IF RUN]

# In[39]:


combined_data.to_csv('../data/experimental/combined_data.csv', index=False)


# ## 1.2 create data splits [SKIP IF RUN]
# 
# consider random splits as well as stratified on a particular mutation split out

# In[40]:


combined_data = pd.read_csv('../data/experimental/combined_data.csv')


# ### 1.2.1 random splits

# In[42]:


for i, (train_index, test_index) in enumerate(kfold.split(combined_data)):
    combined_data[f'onrandom_{i}'] = 0
    combined_data.loc[test_index, f'onrandom_{i}'] = 1


# In[43]:


combined_data


# ### 1.2.2 stratified splits on a particular mutation
# 
# Our dataset is baised towards particular high performing mutations that dominate the activity. We want to create a predictor that can predict decently on other mutations, so we will test based on a mutation split.

# In[44]:


# get which mutations we tested and their counts
mutation_observed_counts = {}
for variant in combined_data['variant']:
    mutations = get_mutations(wt, variant)
    for mut in mutations:
        if mut in mutation_observed_counts:
            mutation_observed_counts[mut] += 1
        else:
            mutation_observed_counts[mut] = 1


# In[45]:


plot_mutation_heatmap(mutations=mutation_observed_counts.keys(), scores=mutation_observed_counts.values())


# In[46]:


for mutation in mutation_observed_counts:
    new_column = f"onmut_{mutation}"
    combined_data = combined_data.copy()
    combined_data[new_column] = combined_data['variant'].apply(lambda x: mutation in get_mutations(wt, x))


# In[47]:


combined_data = combined_data.copy()


# In[48]:


combined_data.to_csv('../data/experimental/combined_data.csv', index=False)


# ## 1.3 Embeddings sequences with model

# ### 1.3.1 Pretrained models

# In[61]:


combined_data = pd.read_csv('../data/experimental/combined_data.csv')
combined_data['weights'] = weights
mutfold = get_foldlike_iterator_from_df(combined_data, 'onmut_')


# In[62]:


all_seqs = ap.ProteinSequences.from_dict(combined_data.set_index('hash')['variant'].to_dict())
variable_positions = all_seqs.mutated_positions


# In[63]:


variable_positions


# In[64]:


len(mutfold)


# In[65]:


esm_maxpool = ap.ESM2Embedding(
    metadata_folder='.../embedders/esm2_maxpool',
    device='mps',
    model_checkpoint='esm2_t12_35M_UR50D',
    pool='max',
)


# In[66]:


esm_meanpool = ap.ESM2Embedding(
    metadata_folder='.../embedders/esm2_meanpool',
    device='mps',
    model_checkpoint='esm2_t12_35M_UR50D',
    pool='mean',
)


# #### 1.3.2 Flattening produces too many features to keep all, run an expensive LASSO to select features [SKIP IF RUN - found best alpha for lasso below]

# In[67]:


combined_data['log_activity'].describe()


# In[68]:


esm_high_res = ap.ESM2Embedding(
    metadata_folder='.../embedders/esm2_individual',
    device='mps',
    model_checkpoint='esm2_t12_35M_UR50D',
    pool=None,
    flatten=True,
    positions=variable_positions
)
esm_high_res.fit([])
high_res_features = esm_high_res.transform(combined_data['variant'])


# In[69]:


high_res_features.shape


# These are way too many features, it will make hyperopt extremely expensive. Let's assume the signal from important features can be seen via a linear model and run LASSO selection

# In[70]:


high_res_features = StandardScaler().fit_transform(high_res_features)


# In[59]:


test_model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
cross_validate(test_model, high_res_features, combined_data['log_activity'], cv=mutfold, scoring='neg_mean_squared_error')  


# In[62]:


# find the highest regularization we can conduct without losing performance
score_list = []
alpha_list = []
num_features_list = []
previous_score = -np.inf
improvement_frac_tol = -0.1
alpha = 1e-3
while True:
    est = TransformedTargetRegressor(Lasso(alpha=alpha), transformer=StandardScaler())
    # some folds have only a single example so use MSE
    scores = cross_validate(est, high_res_features, combined_data['log_activity'], cv=mutfold, scoring='neg_mean_absolute_error', verbose=0, n_jobs=4)
    score = np.mean(scores['test_score'])
    score_list.append(score)
    alpha_list.append(alpha)
    num_features = np.sum(est.fit(high_res_features, combined_data['log_activity']).regressor_.coef_ != 0)
    num_features_list.append(num_features)
    print(f"Alpha: {alpha}, Score: {score}, n_features: {num_features}")

    if alpha > 0.0:
        alpha = alpha * 2
    else:
        alpha = 1e-3
    score_frac_change = (score - previous_score) / np.abs(previous_score)
    if score_frac_change < improvement_frac_tol:
        break
    print(f"Score frac change: {score_frac_change}")
    previous_score = score


# In[63]:


fig, ax = plt.subplots(figsize=(5, 5))
ax2 = ax.twinx()
ax.plot(alpha_list, score_list, color='blue')
ax2.plot(alpha_list, num_features_list, color='red')
ax.set_xscale('log')
ax.set_xlabel('Alpha')
ax.set_ylabel('neg MAE')
ax2.set_ylabel('Number of features')
plt.show()
plt.savefig('../figures/round2/lasso_alpha_search.png')


# In[64]:


alpha_list[np.argmax(score_list)]


# ### 1.3.3 Setup lasso feature selector to be used in high res featurizer [SKIP IF RUN]

# In[65]:


chosen_alpha = 0.016


# In[66]:


esm_high_res = Pipeline([
    ('esm', ap.ESM2Embedding(
        metadata_folder='.../embedders/esm2_individual',
        device='mps',
        model_checkpoint='esm2_t12_35M_UR50D',
        pool=None,
        flatten=True,
        positions=variable_positions
    )),
    ('scaler', StandardScaler()),
    ('selector', SelectFromModel(
        TransformedTargetRegressor(Lasso(alpha=chosen_alpha), transformer=StandardScaler()),
        threshold=1e-10,
        importance_getter=get_importance_from_ttr
        )),
])
esm_high_res.fit(combined_data['variant'], combined_data['log_activity'])
joblib.dump(esm_high_res, '../embedders/esm_high_res_feature_selected.joblib')


# In[67]:


esm_high_res.transform(combined_data['variant']).shape


# ### 1.3.4 Define embedder dict to search over

# In[68]:


esm_high_res = joblib.load('../embedders/esm_high_res_feature_selected.joblib')


# In[69]:


embedders = {
    'meanpool': esm_meanpool,
    'maxpool': esm_maxpool,
    'high_res': esm_high_res,
    'ohe': ap.OneHotProteinEmbedding(positions=variable_positions, flatten=True)}


# ## 1.4 Hyperopt linear model over each embedding

# In[70]:


linear_head = Pipeline([
    ('scalar', StandardScaler()),
    ('pca', PCA(n_components=0.98)),
    ('regressor', TransformedTargetRegressor(ElasticNet(), transformer=StandardScaler()))
])


# In[71]:


param_space = {
    'head__regressor__regressor__alpha': stats.uniform(0.01, 10),
    'head__regressor__regressor__l1_ratio': stats.uniform(0, 1),
    'head__pca__n_components': stats.uniform(0.5, 0.5),
}


# ### 1.4.1 See if we can get away witha  cheaper CV strategy [SKIP IF RUN]
# The mutation held/ out fold iterator has 1200 folds, which is seriously expensive. Try some other strategies and see if they correlate well enough

# In[72]:


def generate_random_params(param_distributions, n_samples=50):
    """
    Generate random parameters from the parameter space.
    
    Args:
        param_distributions (dict): Dictionary of parameter distributions
        n_samples (int): Number of random parameter combinations to generate
        
    Returns:
        list: List of dictionaries containing parameter combinations
    """
    params_list = []
    for _ in range(n_samples):
        params = {}
        for param_name, param_dist in param_distributions.items():
            if hasattr(param_dist, 'rvs'):
                # If it's a scipy.stats distribution
                params[param_name] = param_dist.rvs(1)[0]
            elif isinstance(param_dist, list):
                # If it's a list of choices
                params[param_name] = np.random.choice(param_dist)
            elif isinstance(param_dist, dict):
                # If it's a dictionary of choices with probabilities
                choices = list(param_dist.keys())
                probs = list(param_dist.values())
                params[param_name] = np.random.choice(choices, p=probs)
        params_list.append(params)
    return params_list


# In[73]:


random_param_list = generate_random_params(param_space, 20)


# In[74]:


def evaluate_cv_strategies(model_pipeline, param_sets, X, y, cv1, cv2):
    """
    Evaluate parameter sets using two different CV strategies.
    
    Args:
        model_pipeline: Scikit-learn pipeline
        param_sets (list): List of parameter dictionaries
        X: Features
        y: Target
        cv1: First CV strategy
        cv2: Second CV strategy
        
    Returns:
        tuple: Arrays of mean scores for each CV strategy
    """
    scores_cv1 = []
    scores_cv2 = []
    
    for params in param_sets:
        model_pipeline.set_params(**params)
        
        # Evaluate using first CV strategy
        cv1_scores = []
        if hasattr(cv1, 'split'):
            cv1 = list(cv1.split(X))
        for train_idx, val_idx in cv1:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model_pipeline.fit(X_train, y_train)
            y_pred = model_pipeline.predict(X_val)
            cv1_scores.append(scorer.score(y_val, y_pred))
            print(f"CV1: {cv1_scores[-1]}")
        scores_cv1.append(np.mean(cv1_scores))
        print(f"CV1 mean: {scores_cv1[-1]}")
        
        # Evaluate using second CV strategy
        cv2_scores = []
        if hasattr(cv2, 'split'):
            cv2 = list(cv2.split(X))
        for train_idx, val_idx in cv2:
            train_idx = np.array(train_idx)
            val_idx = np.array(val_idx)
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model_pipeline.fit(X_train, y_train)
            y_pred = model_pipeline.predict(X_val)
            cv2_scores.append(scorer.score(y_val, y_pred))
            print(f"CV2: {cv2_scores[-1]}") 
        scores_cv2.append(np.mean(cv2_scores))
        print(f"CV2 mean: {scores_cv2[-1]}")
    
    return np.array(scores_cv1), np.array(scores_cv2)


# In[75]:


model = Pipeline([
    ('embedder', esm_meanpool),
    ('head', linear_head)
])


# In[76]:


scores_cv1, scores_cv2 = evaluate_cv_strategies(
    model,
    random_param_list,
    ap.ProteinSequences.from_list(combined_data['variant'].tolist()),
    combined_data['log_activity'],
    kfold, mutfold)


# In[77]:


fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(scores_cv1, scores_cv2)
ax.set_xlabel('5fold CV')
ax.set_ylabel('Held out Mut Validation')
spearman = stats.spearmanr(scores_cv1, scores_cv2)
ax.set_title(f"Spearman's R: {spearman.correlation:.2f}, p-value: {spearman.pvalue:.2f}")
plt.savefig('../figures/round2/mutcv_vs_kfold.png')


# It looks as if for the most part 5 fold = mutfold, however there is some cases where 5 fold overestimates performance, which is bad. That being said we really don'y have the time to do 1200 fold CV for hyperopt, so let's continue with the assumption that 5 fold is sufficient for evaluation.

# ### 1.4.2 Run hyperopt for linear model

# In[78]:


results = {}
for embedder_name, embedder in embedders.items():
    model = Pipeline([
        ('embedder', embedder),
        ('head', linear_head)
    ])
    if os.path.exists(f'../models/hyperopt/linear_{embedder_name}_hyperopt.joblib'):
        print(f"Loading hyperopt for {embedder_name}")
        results[embedder_name] = joblib.load(f'../models/hyperopt/linear_{embedder_name}_hyperopt.joblib')

    else:
        print(f"Running hyperopt for {embedder_name}")
        results[embedder_name] = random_hyperopt(
            model,
            param_space,
            combined_data['variant'].tolist(),
            combined_data['log_activity'].tolist(),
            n_iter=100,
            cv=kfold,
            n_jobs=4
        )
        joblib.dump(results[embedder_name], f'../models/hyperopt/linear_{embedder_name}_hyperopt.joblib')
    


# In[79]:


linear_results_df = pd.DataFrame({
    'embedder': [],
    'best_params': [],
    'best_score_mean': [],
    'best_score_std': []
})
for embedder_name, result in results.items():
    linear_results_df = linear_results_df.append(pd.Series({
        'embedder': embedder_name,
        'best_params': result[1],
        'best_score_mean': result[2],
        'best_score_std': result[3]
    }), ignore_index=True)


# In[80]:


linear_results_df['head'] = 'linear'


# ## 1.5 Hyperopt MLP model over each embedding

# In[205]:


mlp_ = MLPRegressor(
    max_iter=1000,
    early_stopping=True,
    random_state=42,
    validation_fraction=0.15,
    n_iter_no_change=10,
)

mlp_head = Pipeline([
    # remove zero variance
    ('var', VarianceThreshold()),
    ('scalar', StandardScaler()),
    ('pca', PCA(n_components=0.98)),
    ('regressor', TransformedTargetRegressor(
        mlp_,
        transformer=StandardScaler()))
])


# In[206]:


param_space = {
    'head__regressor__regressor__hidden_layer_sizes': [(50,), (50,50), (50,50,50)],
    'head__regressor__regressor__alpha': stats.uniform(0.01, 10),
    'head__regressor__regressor__learning_rate_init': stats.uniform(1e-5, 1e-2),
    'head__regressor__regressor__activation': ['relu', 'tanh'],
    'head__pca__n_components': stats.uniform(0.5, 0.5),
}


# In[207]:


results = {}
for embedder_name, embedder in embedders.items():
    model = Pipeline([
        ('embedder', embedder),
        ('head', mlp_head)
    ])
    if os.path.exists(f'../models/hyperopt/mlp_{embedder_name}_hyperopt.joblib'):
        print(f"Loading hyperopt for {embedder_name}")
        results[embedder_name] = joblib.load(f'../models/hyperopt/mlp_{embedder_name}_hyperopt.joblib')

    else:
        print(f"Running hyperopt for {embedder_name}")
        results[embedder_name] = random_hyperopt(
            model,
            param_space,
            combined_data['variant'].tolist(),
            combined_data['log_activity'].tolist(),
            n_iter=100,
            cv=kfold,
            n_jobs=2
        )
        joblib.dump(results[embedder_name], f'../models/hyperopt/mlp_{embedder_name}_hyperopt.joblib')


# In[208]:


mlp_results_df = pd.DataFrame({
    'embedder': [],
    'best_params': [],
    'best_score_mean': [],
    'best_score_std': []
})
for embedder_name, result in results.items():
    mlp_results_df = mlp_results_df.append(pd.Series({
        'embedder': embedder_name,
        'best_params': result[1],
        'best_score_mean': result[2],
        'best_score_std': result[3]
    }), ignore_index=True)


# In[216]:


mlp_results_df['head'] = 'mlp'


# ## 1.6 Run hyperopt for RandomForest model

# In[209]:


rf_head = Pipeline([
    ('var', VarianceThreshold()),
    ('scalar', StandardScaler()),
    ('pca', PCA(n_components=0.98)),
    ('regressor', TransformedTargetRegressor(
        RandomForestRegressor(
            n_estimators=10,
            random_state=42
        ),
        transformer=StandardScaler()))
])


# In[210]:


param_space = {
    'head__regressor__regressor__min_samples_split': stats.randint(2, 10),
    'head__regressor__regressor__min_samples_leaf': stats.randint(1, 10),
    'head__regressor__regressor__min_weight_fraction_leaf': stats.uniform(0, 0.5),
    'head__pca__n_components': stats.uniform(0.5, 0.5),
    'head__regressor__regressor__max_features': stats.uniform(0.1, 0.9),
}
    


# In[211]:


results = {}
for embedder_name, embedder in embedders.items():
    model = Pipeline([
        ('embedder', embedder),
        ('head', rf_head)
    ])
    if os.path.exists(f'../models/hyperopt/rf_{embedder_name}_hyperopt.joblib'):
        print(f"Loading hyperopt for {embedder_name}")
        results[embedder_name] = joblib.load(f'../models/hyperopt/rf_{embedder_name}_hyperopt.joblib')

    else:
        print(f"Running hyperopt for {embedder_name}")
        results[embedder_name] = random_hyperopt(
            model,
            param_space,
            combined_data['variant'].tolist(),
            combined_data['log_activity'].tolist(),
            n_iter=100,
            cv=kfold,
            n_jobs=2,
            head__regressor__sample_weight=(combined_data['weights']/combined_data['weights'].mean()).tolist()
        )
        joblib.dump(results[embedder_name], f'../models/hyperopt/rf_{embedder_name}_hyperopt.joblib')


# In[212]:


rf_results_df = pd.DataFrame({
    'embedder': [],
    'best_params': [],
    'best_score_mean': [],
    'best_score_std': []
})
for embedder_name, result in results.items():
    rf_results_df = rf_results_df.append(pd.Series({
        'embedder': embedder_name,
        'best_params': result[1],
        'best_score_mean': result[2],
        'best_score_std': result[3]
    }), ignore_index=True)
rf_results_df['head'] = 'rf'


# In[213]:


rf_results_df


# In[214]:


mlp_results_df


# In[215]:


linear_results_df


# ## 1.7 Combine results data and make plot

# In[217]:


# combine the results of all hyperparameter runs
results_df = pd.concat([linear_results_df, mlp_results_df, rf_results_df], axis=0)


# In[218]:


results_df


# In[219]:


results_df.to_csv('../models/hyperopt/hyperopt_results.csv', index=False)


# In[220]:


results_df = results_df.reset_index()


# In[221]:


# make a bar plot with stdev as error bar
# embedder as x, model head as hue
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=results_df,
    x='embedder',
    y='best_score_mean',
    hue='head',
    capsize=0.1,  # Size of error bar caps
    errorbar=('ci', None),  # Use raw standard deviation instead of confidence interval
    errwidth=1.5,  # Width of error bars,
    ax=ax,
    # gap=0.1,
    edgecolor='black',
    linewidth=2
)


# Calculate positions for error bars
n_embedders = len(results_df['embedder'].unique())
n_heads = len(results_df['head'].unique())
width = 0.8  # default bar width in seaborn
head_width = width / n_heads

# Add error bars manually
for idx, row in results_df.iterrows():
    # Calculate the bar position
    embedder_idx = list(results_df['embedder'].unique()).index(row['embedder'])
    head_idx = list(results_df['head'].unique()).index(row['head'])
    
    # Calculate center of each bar
    bar_center = embedder_idx + (head_idx * head_width) + (head_width / 2) - (width / 2)
    
    # Add error bars
    ax.vlines(
        x=bar_center,
        ymin=row['best_score_mean'] - row['best_score_std'],
        ymax=row['best_score_mean'] + row['best_score_std'],
        color='black',
        linewidth=2,
    )
    ax.set_xlabel('Embedder')

ax.set_ylabel('Negative MAE [log10(foldchange)]')
plt.legend(title='Head', loc='lower right', bbox_to_anchor=(0.8, 0.0))
plt.savefig('../figures/round2/hyperopt_results.png', bbox_inches='tight', dpi=300)


# I'll believe that when I see that...

# ## 1.8 Run  final model in CV and observe the parity

# In[222]:


best_model_row = results_df[results_df['best_score_mean'] == results_df['best_score_mean'].max()]


# In[223]:


best_params = best_model_row['best_params'].values[0]


# In[224]:


# create best model
embedder_ = embedders[best_model_row['embedder'].values[0]]
head_ = best_model_row['head'].values[0]
if head_ == 'mlp':
    head_ = mlp_
elif head_ == 'rf':
    head_ = rf_
elif head_ == 'linear':
    head_ = lin_


head = Pipeline([
    ('scalar', StandardScaler()),
    ('pca', PCA(n_components=0.98)),
    ('regressor', TransformedTargetRegressor(head_, transformer=StandardScaler()))
])

best_model_pipeline = Pipeline([
    ('embedder', embedder_),
    ('head', head)
])
best_model_pipeline.set_params(**best_params)


# In[225]:


combined_data.sort_values('log_activity', ascending=False).head(10)


# In[226]:


all_predictions = []
all_trues = []
for train_index, test_index in kfold.split(combined_data):
    X_train = combined_data.loc[train_index, 'variant'].values
    y_train = combined_data.loc[train_index, 'log_activity'].values
    X_test = combined_data.loc[test_index, 'variant'].values
    y_test = combined_data.loc[test_index, 'log_activity'].values
    best_model_pipeline.fit(X_train, y_train)
    y_pred = best_model_pipeline.predict(X_test)
    all_predictions.extend(y_pred)
    all_trues.extend(y_test)


# In[227]:


fig, ax = plt.subplots(figsize=(5, 5))
min_min = min(np.min(all_trues), np.min(all_predictions))
max_max = max(np.max(all_trues), np.max(all_predictions))
ax.scatter(all_trues, all_predictions, alpha=0.5)
ax.plot([min_min, max_max], [min_min, max_max], color='black', linestyle='--')
ax.set_xlabel('True log foldchange')
ax.set_ylabel('Predicted log foldchange')


# In[228]:


# convert back to foldchange
all_predictions_foldchange = np.power(10, all_predictions) - 1
all_trues_foldchange = np.power(10, all_trues) - 1

fig, ax = plt.subplots(figsize=(5, 5))
min_min = min(np.min(all_trues_foldchange), np.min(all_predictions_foldchange))
max_max = max(np.max(all_trues_foldchange), np.max(all_predictions_foldchange))
ax.scatter(all_trues_foldchange, all_predictions_foldchange, alpha=0.5)

ax.plot([min_min, max_max], [min_min, max_max], color='black', linestyle='--')
ax.set_xlabel('True foldchange')
ax.set_ylabel('Predicted foldchange')
plt.savefig('../figures/round2/best_parity.png', bbox_inches='tight', dpi=300)


# In[237]:


kendall_tau = stats.kendalltau(all_trues, all_predictions)
kendall_tau


# ## 1.9 Train and save final model

# In[229]:


best_model_trained = best_model_pipeline.fit(combined_data['variant'], combined_data['log_activity'])


# In[230]:


joblib.dump(best_model_trained, '../data/round2/best_model.joblib')


# In[71]:


best_model_trained = joblib.load('../data/round2/best_model.joblib')


# # 2 Generate next round sequences
# 
# 1. Go back through all 2nd and third order mutants the see if any are worth testing. Chose at least 3 fold better than WT

# #### Make selections based on molder threshold to medium list

# In[66]:


cutoff_threshold = np.log10(7) # 3 fold better than WT
cutoff_threshold


# In[232]:


tested_hashed = set(combined_data['hash'].values)


# In[233]:


variants_dataset = datasets.load_from_disk('../hf_datasets/variants')


# In[234]:


variants_dataset = variants_dataset.filter(lambda x: x['hash'] not in tested_hashed)


# In[235]:


def prediction_mapper(examples):
    predictions = list(best_model_trained.predict(list(examples['variant'])))
    examples['predicted_log_fold_activity'] = predictions
    return examples


# In[236]:


varaints_dataset = variants_dataset.map(prediction_mapper, batched=True, batch_size=5000)


# In[238]:


varaints_dataset.save_to_disk('../hf_datasets/variants_with_rd2_predictions')


# In[67]:


variants_dataset = datasets.load_from_disk('../hf_datasets/variants_with_rd2_predictions')


# In[68]:


fig, ax = plt.subplots(figsize=(5, 5))
sns.histplot(variants_dataset['predicted_log_fold_activity'], ax=ax)
plt.xlabel('Predicted log foldchange')
ax.vlines(cutoff_threshold, 0, ax.get_ylim()[1], color='red', linestyle='--', label='6 fold better than WT')
plt.legend()


# In[69]:


def map_threshold(example):
    if example['predicted_log_fold_activity'] > cutoff_threshold:
        example['passes_threshold'] = True
    else:
        example['passes_threshold'] = False
    return example


# In[70]:


variants_dataset = variants_dataset.map(map_threshold, batched=False)


# In[71]:


# filter the dataset to the medium list of variants of at least 3 fold prediction
variants_dataset = variants_dataset.filter(lambda x: x['passes_threshold'])
variants_dataset.save_to_disk('../hf_datasets/variants_with_rd2_passes')


# In[72]:


variants_dataset


# # 3. Higher order variants

# In[52]:


model = joblib.load('../data/round2/best_model.joblib')


# In[248]:


from aide_predict.utils.badass import BADASSOptimizer, BADASSOptimizerParams


# In[249]:


cutoff_threshold = np.quantile(combined_data['log_activity'], 0.9)
lower_threshold = np.quantile(combined_data['log_activity'], 0.5)


# In[250]:


# get only the 10 sites we are allowing to vary
s_ = ap.ProteinSequences.from_list(round1_data['variant'].tolist())


# In[251]:


s_.as_array()[:,np.array(s_.mutated_positions)]


# In[252]:


variable_positions_1_indexed = np.array(s_.mutated_positions)+1


# In[253]:


sites_to_ignore = [i for i in range(1, 1+len(wt)) if i not in variable_positions_1_indexed]


# In[254]:


import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


# In[259]:


params = BADASSOptimizerParams(
    seqs_per_iter=int(32*10),
    num_iter=1000,
    temperature=1.5,
    seed=43,
    gamma=1.0,
    normalize_scores=True,
    cooling_rate=0.93,
    num_mutations=5,
    reversal_threshold=lower_threshold,
    score_threshold=cutoff_threshold,
    sites_to_ignore=sites_to_ignore)
optimizer = BADASSOptimizer(
    predictor=model, params=params, reference_sequence=wt)
    


# In[260]:


results, stats = optimizer.optimize()


# In[262]:


optimizer.plot()


# In[270]:


def unnormalize_scores(badass_score):
    raw_score = badass_score * (optimizer._optimizer.ref_score_scale + 1.0) + optimizer._optimizer.ref_score_value
    return raw_score


# In[271]:


results['predicted_log_fold_activity'] = results['score'].apply(unnormalize_scores)


# In[276]:


results.to_csv('../data/round2/badass_results.csv')
stats.to_csv('../data/round2/badass_stats.csv')


# In[274]:


results.sort_values('iteration')


# In[243]:


results = pd.read_csv('../data/round2/badass_results.csv')


# In[245]:


# cutoff based on predicted activity
cutoff_threshold = np.log10(7)


# In[246]:


fig, ax = plt.subplots(figsize=(5, 5))
sns.histplot(results.sort_values('iteration')['predicted_log_fold_activity'], ax=ax)
plt.xlabel('Predicted log foldchange')
ax.vlines(cutoff_threshold, 0, ax.get_ylim()[1], color='red', linestyle='--', label='6 fold better than WT')
plt.legend()


# In[247]:


top12 = ssm_data.sort_values('activity', ascending=False).head(12)
top12['mutation_string_shifted'] = top12['mutation_string'].apply(lambda x: x[0]+str(int(x[1:-1])-POSITION_OFFSET)+x[-1])


# In[248]:


top12_mutation_strings_set = set(top12['mutation_string_shifted'].values)


# In[249]:


results['n_top_12_mutations'] = results['sequences'].apply(lambda x: sum([x_ in top12_mutation_strings_set for x_ in x.split('-')]))


# In[55]:


sns.kdeplot(results, x='n_top_12_mutations', y='predicted_log_fold_activity')


# In[250]:


round1_data['log_activity'] = np.log10(1+round1_data['activity'])
round1_data.sort_values('log_activity', ascending=False).head(10)


# In[251]:


results['predicted_greater_than_max_exp'] = results['predicted_log_fold_activity'] > np.max(combined_data['log_activity'])


# In[252]:


results['predicted_greater_than_max_exp'].sum()


# In[254]:


results.to_csv('../data/round2/badass_results.csv', index=False)


# In[255]:


cutoff_threshold = np.log10(7)


# In[256]:


results_pass_threshold = results[results['predicted_log_fold_activity'] > cutoff_threshold]
results_pass_threshold.to_csv('../data/round2/badass_results_filtered.csv', index=False)


# In[257]:


len(results_pass_threshold)


# # 4. Select variants from both libraries fore a plate
# 
# Two libraries: all 2-3 order mutatants, and 5 order mutatants from BADASS sampling.
# 
# Assuming 96 variants, split as follows:
# - 3 of the top predictions from each library (6)
# - Subset libraries by: predicted at least 3 fold better than wt. For the 5 fold, also subset by ESM
# - Compute distance matrix between all variants, remove all that are equivilent to those already seen before, just to make sure.
# - Iteratively select based on the network while relaxing the distance constrain, eg. select only those with distance 5 or greater, when there are none, move to 4 or greater.
# - Allow for use to require a certain number of  2-3 order variants

# #### First filter with ESM to attempt to remove pathologiv variants

# In[127]:


best_model_trained = joblib.load('../data/round2/best_model.joblib')


# In[128]:


cutoff_threshold = np.log10(7) # 6 fold better than WT


# In[129]:


esm1v = ap.ESM2LikelihoodWrapper(
    wt=wt,
    device='mps',
    model_checkpoint='esm1v_t33_650M_UR90S_1',
    marginal_method='masked_marginal',
    metadata_folder='../embedders/esm1v',
).fit([])


# How do the esm scores compare to the labels?

# In[130]:


combined_data['esm1v'] = esm1v.predict(combined_data['variant'].values)


# In[131]:


fig, ax = plt.subplots()
sns.regplot(data=combined_data, x='esm1v', y='log_activity', ax=ax)


# Determine the lowest decision threshold to get 100% recall of 3 fold better than WT

# In[132]:


four_fold_better = combined_data['log_activity'] > cutoff_threshold


# In[133]:


min_esm = combined_data['esm1v'].min()
max_max = combined_data['esm1v'].max()

from sklearn.metrics import  precision_recall_curve
# compute precision and recall curve
precision, recall, thresholds = precision_recall_curve(four_fold_better, combined_data['esm1v'])

# Get the last threshold that provides 100% recall
chosen_esm_thresh = thresholds[np.argmin(recall == 1)]

fig, ax = plt.subplots()
ax.plot(thresholds, precision[:-1], label='Precision')
ax.plot(thresholds, recall[:-1], label='Recall')
ax.vlines(chosen_esm_thresh, 0, 1, color='black', linestyle='--', label='Chosen threshold')
plt.legend()


# ### load the libraries and apply the ESM1v filter

# In[134]:


low_order_library = datasets.load_from_disk('../hf_datasets/variants_with_rd2_passes')


# In[135]:


len(low_order_library)


# In[136]:


def esm_mapper(examples):
    scores = esm1v.transform(list(examples['variant']))
    examples['esm1v_scores'] = scores
    return examples


# In[137]:


low_order_library = low_order_library.map(esm_mapper, batched=True, batch_size=5000)


# In[138]:


plt.hist(np.array(low_order_library['esm1v_scores']), bins=20)
plt.vlines(chosen_esm_thresh, 0, 1000, color='black', linestyle='--')


# High order library

# In[139]:


high_order_library = pd.read_csv('../data/round2/badass_results_filtered.csv')


# In[140]:


len(high_order_library)


# In[141]:


high_order_library['predicted_log_fold_activity'].min()


# In[142]:


high_order_library['esm1v_scores'] = esm1v.predict(high_order_library['full_sequence'].tolist())


# In[143]:


plt.hist(high_order_library['esm1v_scores'].values, bins=20)
plt.vlines(chosen_esm_thresh, 0, 1000, color='black', linestyle='--')


# In[144]:


low_order_library = low_order_library.filter(lambda x: x['esm1v_scores'] > chosen_esm_thresh)


# In[145]:


high_order_library = high_order_library[high_order_library['esm1v_scores'] > chosen_esm_thresh]


# ### Combine libraries into a common datastructure
# - columns: mutations, variant, hash, predicted_log_fold_activity, from

# In[146]:


low_order_library_ = low_order_library.to_pandas()[['mutations', 'hash', 'variant', 'predicted_log_fold_activity']]
low_order_library_['from'] = 'lol'


# In[147]:


high_order_library.columns


# In[148]:


high_order_library_ = high_order_library[['full_sequence', 'predicted_log_fold_activity']].rename(columns={'full_sequence': 'variant', 'predicted_log_fold_activity': 'predicted_log_fold_activity'})
high_order_library_['from'] = 'hol'
high_order_library_['mutations'] = high_order_library_.apply(lambda x: get_mutations(wt, x['variant']), axis=1)


# In[149]:


tested_library_ = combined_data[['variant', 'log_activity']].copy()
tested_library_['from'] = 'tested'
tested_library_['mutations'] = tested_library_['variant'].apply(lambda x: get_mutations(wt, x))
tested_library_['predicted_log_fold_activity'] = best_model_trained.predict(tested_library_['variant'])
tested_library_['hash'] = tested_library_['variant'].apply(protein_hash)


# In[150]:


libraries = pd.concat([low_order_library_, high_order_library_, tested_library_], axis=0)


# In[151]:


libraries['from'].value_counts()


# In[152]:


top12 = ssm_data.sort_values('activity', ascending=False).head(12)
top12_mutation_strings_set = set(top12['mutation_string'].values)


# In[153]:


libraries['n_mutations'] = libraries['mutations'].apply(lambda x: len(x))
libraries['n_top_12_mutations'] = libraries['mutations'].apply(lambda x: sum([x_ in top12_mutation_strings_set for x_ in x]))


# In[154]:


untested_libraries = libraries[libraries['from'] != 'tested']


# In[155]:


sns.jointplot(data=untested_libraries, x='n_top_12_mutations', y='predicted_log_fold_activity', hue='from', kind='kde')


# There are many examples in the library that do not contain purely top 12 mutations that the model thinks will do well

# In[156]:


libraries.reset_index(drop=True).to_csv('../data/round2/libraries.csv', index=False)


# ### Conduct network selection

# In[157]:


cutoff_threshold = np.log10(7)


# In[158]:


libraries = pd.read_csv('../data/round2/libraries.csv')
untested_libraries = libraries[libraries['from'] != 'tested']


# In[159]:


# get only the positions that we are changing
seqs = ap.ProteinSequences.from_list(untested_libraries['variant'].tolist())


# In[160]:


variable_positions = seqs.mutated_positions
variable_positions


# In[161]:


# compute smaller vectors iding the sequences for faster pairwise comaprison
seqs = ap.ProteinSequences.from_list(libraries['variant'].tolist())
from aide_predict import OneHotProteinEmbedding

ohe = OneHotProteinEmbedding(positions=variable_positions, flatten=False)
seq_array = ohe.fit_transform(seqs)


# In[162]:


seq_array = seq_array.astype(np.uint8)


# In[163]:


seq_array.shape


# In[164]:


libraries['from'].value_counts()


# In[165]:


seq_array = np.argmax(seq_array, axis=2)


# In[171]:


distances = np.sum(seq_array[:, np.newaxis, :].astype(np.uint8) != seq_array[np.newaxis, :, :].astype(np.uint8), axis=2, dtype=np.int8)
distances


# In[172]:


distances.shape


# In[173]:


len(ssm_data)


# In[174]:


libraries.iloc[-1]


# In[127]:


libraries.iloc[-2]


# In[176]:


seq_array[-1]


# In[177]:


seq_array[-2]


# In[178]:


distances[-1,-2]


# In[181]:


ssm_mask = ((libraries['from'] == 'tested') * (libraries['n_mutations'] == 1)).values


# In[182]:


print(list(distances[ssm_mask][:, ssm_mask][-1]))


# In[183]:


distances


# In[196]:


# tsne this space
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=50, random_state=42, metric='precomputed', init='random')
tsne_embedding = tsne.fit_transform(distances)
libraries['tsne_x'] = tsne_embedding[:, 0]
libraries['tsne_y'] = tsne_embedding[:, 1]


# In[197]:


libraries


# In[198]:


sns.scatterplot(data=libraries, x='tsne_x', y='tsne_y', hue='from', alpha=0.2)


# Begin selecting candidates. Start with the already tested ones for distance metrics.

# In[334]:


selected = libraries.where(libraries['from'] == 'tested').dropna()
selected_indexes = selected.index.tolist()


# In[335]:


len(selected_indexes)


# In[336]:


# now add the top 3 from each library
# only from hol and lol
top_3_each_indexes_to_test = untested_libraries.groupby('from').apply(lambda x: x.sort_values('predicted_log_fold_activity', ascending=False).head(3)).index.get_level_values(1).tolist()


# In[337]:


selected_indexes.extend(top_3_each_indexes_to_test)


# In[338]:


len(selected_indexes)


# In[339]:


def check_for_candidate(library, already_selected_indexes, all_distances, min_dist):
    found_one = False
    library_ = library.sort_values('predicted_log_fold_activity', ascending=False)
    selected_index = None
    for i, row in library_.iterrows():
        if i in already_selected_indexes:
            continue
        if np.min(all_distances[i, already_selected_indexes]) >= min_dist:
            selected_index = i
            found_one = True
            break
    
    return selected_index, found_one


# In[340]:


# now go through 2-3 order mutants and add candidates, enforcing distance from tested
# drop min distance as we become unable to find any more
min_distance = 4
n_added = 0
lo_added_indexes = []

while n_added < 18:
    
    index, found = check_for_candidate(libraries[libraries['from'] == 'lol'], selected_indexes, distances, min_distance)

    if found:
        print(f"Adding {index} at distance {min_distance} from existing candidates")
        n_added += 1
        selected_indexes.append(index)
        lo_added_indexes.append(index)
    else:
        print(f"Could not find any more candidates at distance {min_distance}")
        min_distance -= 1


# In[341]:


libraries.loc[lo_added_indexes]


# In[342]:


# repeat for high order library
min_distance = 7
n_added = 0
ho_added_indexes = []

while n_added < 64:
    
    index, found = check_for_candidate(libraries[libraries['from'] == 'hol'], selected_indexes, distances, min_distance)

    if found:
        print(f"Adding {index} at distance {min_distance} from existing candidates")
        n_added += 1
        selected_indexes.append(index)
        ho_added_indexes.append(index)
    else:
        print(f"Could not find any more candidates at distance {min_distance}")
        min_distance -= 1


# In[344]:


libraries.loc[ho_added_indexes].sort_values('predicted_log_fold_activity', ascending=False)


# In[219]:


distances[selected_indexes][:, selected_indexes]


# In[345]:


new_plate = libraries.loc[top_3_each_indexes_to_test + lo_added_indexes + ho_added_indexes]


# In[346]:


new_plate


# In[347]:


new_plate.to_csv('../data/round2/designed_plate.csv', index=False)


# In[354]:


new_plate['mutations'] = new_plate['variant'].apply(lambda x: get_mutations(wt, x))


# In[355]:


new_plate['id'] = new_plate.apply(lambda row: f'Rd2_{row["from"]}_{"-".join(row["mutations"])}', axis=1)


# In[ ]:


new_plate['V83_mutated'] = new_plate['mutations'].apply(lambda x: 'V83' in ''.join(x))


# In[363]:


new_plate['V83_mutated_to'] = new_plate['mutations'].apply(lambda x: ([None] + [s[-1] for s in x if s[0] == 'V' and s[1:-1] == '83'])[-1])


# In[366]:


new_plate = new_plate.sort_values(['V83_mutated', 'V83_mutated_to'])


# In[367]:


new_plate


# In[368]:


new_plate['predicted_foldchange'] = np.power(10, new_plate['predicted_log_fold_activity']) - 1


# In[370]:


new_plate.to_csv('../data/round2/designed_plate.csv', index=False)


# ### What are the distribution of mutations selected by the model, how much do they diverge from ssm?

# In[371]:


new_plate = pd.read_csv('../data/round2/designed_plate.csv')


# In[372]:


import ast
new_plate['mutations'] = new_plate['mutations'].apply(ast.literal_eval)


# In[373]:


new_plate_variants = ap.ProteinSequences.from_list(new_plate['variant'].tolist())


# In[374]:


new_plate_array = new_plate_variants.as_array()[:, np.array(variable_positions)]


# In[375]:


import matplotlib.colors as mcolors
from matplotlib.colors import Normalize


# In[377]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

output_dir = '../figures/round2/aa_preferences/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i, aas in enumerate(new_plate_array.T):
    position = variable_positions[i]
    best_ssm_mutation = ssm_data[ssm_data['position_on_wt_0_indexed'] == position].sort_values('activity', ascending=False).iloc[0]['mutation']
    original_aa = wt[position]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(f"Position {original_aa}{position+POSITION_OFFSET+1}, best SSM: {best_ssm_mutation}")

    counts = np.unique(aas, return_counts=True)
    colors = []
    barstyles = []

    # Create custom colormap with true white at center
    colors_under = plt.cm.coolwarm(np.linspace(0, 0.5, 128))
    colors_over = plt.cm.coolwarm(np.linspace(0.5, 1, 128))
    
    # Convert the middle color to pure white
    all_colors = np.vstack((colors_under, np.array([1, 1, 1, 1]), colors_over))
    custom_cmap = LinearSegmentedColormap.from_list('custom_coolwarm', all_colors)

    # Create normalizer centered at 1.0
    vmax = max(max(ssm_activity_dict.values()), 2.0)
    vmin = 0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    # Apply colors and styles
    for aa in counts[0]:
        ssm_activity = ssm_activity_dict.get(f'{original_aa}{position+POSITION_OFFSET+1}{aa}', 1.0)
        if aa == wt[position]:
            color = 'white'
        else:
            color = custom_cmap(norm(ssm_activity))
        
        colors.append(color)
        if aa == best_ssm_mutation:
            barstyles.append({'edgecolor': 'red', 'linewidth': 2})
        elif aa == wt[position]:
            barstyles.append({'edgecolor': 'black', 'linewidth': 2})
        else:
            barstyles.append({})

    # Create bars
    bars = ax.bar(counts[0], counts[1], color=colors)
    
    # Apply styles to each bar individually
    for bar, style in zip(bars, barstyles):
        bar.set(**style)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label='Activity')
    
    plt.savefig(f'{output_dir}position_{position+POSITION_OFFSET+1}.png', bbox_inches='tight', dpi=300)
    plt.close()


# In[229]:


# What number of selections contain a mutation that SSM says would break the protein eg fold change < 0.5
count = 0
for _, row in new_plate.iterrows():
    found = False
    bads = []
    for mutation in row['mutations']:
        if ssm_activity_dict.get(mutation, 1.0) < 0.5:
            found = True
            bads.append(mutation)
    if found:
        count += 1
        print(f"Found {bads} in {row['variant']}")
print(f'{count/len(new_plate)} of the selections contain a mutation that SSM says would break')


# SSM thinks A420 and S425 are not very mutable, but model is selecting some variants there. How does the additive model compare to predictions for the selections?

# In[230]:


new_plate['additive_log_activity'] = np.log10(new_plate['mutations'].apply(get_additive_activity)+1)


# In[231]:


new_plate['additive_activity'] = new_plate['mutations'].apply(get_additive_activity)


# In[232]:


fig, ax = plt.subplots(figsize=(5, 5))
min_min = min(np.min(new_plate['predicted_log_fold_activity']), np.min(new_plate['additive_log_activity']))
max_max = max(np.max(new_plate['predicted_log_fold_activity']), np.max(new_plate['additive_log_activity']))
ax.scatter(new_plate['additive_log_activity'], new_plate['predicted_log_fold_activity'], alpha=0.5)
ax.plot([min_min, max_max], [min_min, max_max], color='black', linestyle='--')
ax.set_xlabel('Additive log foldchange')
ax.set_ylabel('Predicted log foldchange')


# In[234]:


new_plate['predicted_fold_change'] = np.power(10, new_plate['predicted_log_fold_activity']) - 1


# To probe epistasis, we need the ML predictions over the entire dataset of possible variants so that we can compute per-mutation averages, alot like the BADASS score matrix

# In[258]:


# load badass results
badass_all_variants = pd.read_csv('../data/round2/badass_results.csv')
badass_all_variants['mutations'] = badass_all_variants['full_sequence'].apply(lambda x: get_mutations(wt, x))
badass_all_variants = badass_all_variants[['mutations', 'predicted_log_fold_activity']]


# In[259]:


badass_all_variants


# In[260]:


# load the 2-3 order variants
lo_all_variants = datasets.load_from_disk('../hf_datasets/variants_with_rd2_predictions')


# In[261]:


lo_all_variants = lo_all_variants.to_pandas()[['mutations', 'predicted_log_fold_activity']]


# In[262]:


all_variants = pd.concat([badass_all_variants, lo_all_variants], axis=0, ignore_index=True)


# In[263]:


all_variants['predicted_fold_change'] = np.power(10, all_variants['predicted_log_fold_activity']) - 1


# In[264]:


def analyze_epistasis(selected_df, all_variants_df, ssm_dict):
   # Calculate average prediction across ALL variants (not just selected)
   global_avg_prediction = all_variants_df['predicted_fold_change'].mean()
   print(f"Average prediction across all variants: {global_avg_prediction:.2f}")
   
   # Initialize storage for mutation statistics
   mutation_stats = {}
   
   # For each selected variant
   for idx, row in selected_df.iterrows():
       # Normalize prediction relative to global average from ALL variants
       normalized_pred = row['predicted_fold_change'] - global_avg_prediction
       mutations = row['mutations']
       
       # Update statistics for each mutation in this variant
       for mut in mutations:
           if mut not in mutation_stats:
               # Get all variants containing this mutation from the full dataset
               variants_with_mut = [
                   row['predicted_fold_change'] 
                   for idx, row in all_variants_df.iterrows() 
                   if mut in row['mutations']
               ]
               
               mutation_stats[mut] = {
                   'ml_predictions_all': variants_with_mut,
                   'ml_predictions_selected': [],
                   'normalized_predictions': [],
                   'ssm_effect': ssm_dict.get(mut),
                   'count_in_selected': 0
               }
           mutation_stats[mut]['ml_predictions_selected'].append(row['predicted_fold_change'])
           mutation_stats[mut]['normalized_predictions'].append(normalized_pred)
           mutation_stats[mut]['count_in_selected'] += 1
   
   # Convert to DataFrame for analysis
   analysis_data = []
   for mut, stats in mutation_stats.items():
       # Calculate average effect across ALL variants containing this mutation
       avg_ml_effect_all = np.mean(stats['ml_predictions_all']) - global_avg_prediction
       avg_ml_effect_selected = np.mean(stats['ml_predictions_selected'])
       avg_normalized_effect = np.mean(stats['normalized_predictions'])
       ssm_effect = stats['ssm_effect']
       count = stats['count_in_selected']
       
       # Only include mutations that have SSM data
       if ssm_effect is not None:
           # Calculate epistasis using predictions from all variants
           epistasis = avg_ml_effect_all - (ssm_effect - np.mean(list(ssm_dict.values())))
           
           analysis_data.append({
               'mutation': mut,
               'avg_effect_all_variants': avg_ml_effect_all,
               'avg_effect_selected': avg_ml_effect_selected,
               'normalized_ml_effect': avg_normalized_effect,
               'ssm_effect': ssm_effect,
               'epistasis_score': epistasis,
               'occurrence_count': count
           })
   
   df = pd.DataFrame(analysis_data)
   print(f"Found {len(df)} mutations with SSM data out of {len(mutation_stats)} total mutations")
   return df

epistasis_df = analyze_epistasis(new_plate, all_variants, ssm_activity_dict)


# In[265]:


epistasis_df.to_csv('../data/round2/epistasis.csv', index=False)


# In[267]:


def plot_epistasis_analysis(analysis_df):
   if len(analysis_df) == 0:
       print("No mutations with SSM data to plot!")
       return
   
   # Create figure with two subplots
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
   fig.subplots_adjust(hspace=0.3)  # Add space between plots since they're independent now
   
   # Upper plot: Scatter plot
   ax1.scatter(analysis_df['ssm_effect'], 
            analysis_df['avg_effect_all_variants'],
            alpha=0.6,
            s=analysis_df['occurrence_count']*50, label='Selected variants')
   
   # Add reference lines
   ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
   ax1.axvline(x=1, color='r', linestyle='--', alpha=0.5)
   # Show zero epistasis line
   xmin, xmax = ax1.get_xlim()
   ymin, ymax = ax1.get_ylim()
   min_min = min(xmin, ymin)
   max_max = max(xmax, ymax)
   ax1.plot([min_min, max_max], [min_min, max_max], color='black', linestyle='--', alpha=0.5, label='non-epistatic')
   
   ax1.set_ylabel('Average effect of mutation \n (Relative to global average prediction)')
   ax1.set_title('SSM vs ML Effects\n(point size indicates frequency in selected variants)')
   ax1.set_xlabel('SSM Effect')
   ax1.grid(True, alpha=0.3)
   ax1.legend()
   
   # Add labels for points with large epistasis
   if len(analysis_df) > 1:
       for idx, row in analysis_df.iterrows():
           if abs(row['epistasis_score']) > np.std(analysis_df['epistasis_score']):
               ax1.annotate(row['mutation'], 
                         (row['ssm_effect'], row['avg_effect_all_variants']),
                         xytext=(5, 5), textcoords='offset points')
   
   # Lower plot: Bar plot sorted by epistasis score
   sorted_df = analysis_df.sort_values('epistasis_score', ascending=True)
   bars = ax2.bar(range(len(sorted_df)), sorted_df['epistasis_score'])
   
   # Customize lower plot
   ax2.set_ylabel('Epistasis Score')
   ax2.grid(True, alpha=0.3, axis='y')
   
   # Add mutation labels
   ax2.set_xticks(range(len(sorted_df)))
   ax2.set_xticklabels(sorted_df['mutation'], rotation=90, ha='center', fontsize=9)
   
   # Color bars based on positive/negative epistasis
   for i, bar in enumerate(bars):
       if bar.get_height() < -0.5:
           bar.set_color('tab:red')
       elif bar.get_height() > 0.5:
           bar.set_color('tab:blue')
       else:
              bar.set_color('tab:gray')
   
   plt.tight_layout()
   return fig

fig = plot_epistasis_analysis(epistasis_df)


# Let's dive into V83A, which ssm did not think was particularly good but the model liked on average. Is its presence in high performing variants a function of the presence of another mutation?

# In[268]:


ssm_activity_dict['V83A']


# Create an epistatic model based on model predictions, eg. all single interactions and then pairwise intereactions.

# In[269]:


all_variants


# In[270]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


# In[271]:


def get_ind_pair_features(sequences, wt):
    ohe = ap.OneHotProteinEmbedding(positions=variable_positions, flatten=True)
    oh_features = ohe.fit_transform(combined_data['variant'].values)

    feature_names = ohe.get_feature_names_out()
    # these are 'pos<pos>_<AA>'
    # convert to '<wt_aa><pos+11><mut_aa>'
    feature_names_ = []
    for feature_name in feature_names:
        pos = int(feature_name[3:-2])
        aa = feature_name[-1]
        feature_names_.append(f"{wt[pos]}{pos+POSITION_OFFSET+1}{aa}")

    # now get pairwise features
    X = pd.DataFrame(oh_features, columns=feature_names_)
    poly = PolynomialFeatures(interaction_only=True, include_bias=False, degree=2)
    X_poly = poly.fit_transform(X)
    out_names = poly.get_feature_names_out(input_features=feature_names_)
    X_poly = pd.DataFrame(X_poly, columns=out_names)

    # remove zero variance columns
    X_poly = X_poly.loc[:, X_poly.std() != 0]
    return X_poly


# In[272]:


X_poly = get_ind_pair_features(combined_data['variant'], wt)
lin_model = Ridge(alpha=1.0).fit(X_poly, combined_data['activity'])


# In[273]:


X_poly.columns


# In[274]:


def create_interaction_matrix(coef_dict):
    # Extract unique individual mutations from feature names
    mutations = set()
    for feature in coef_dict.keys():
        terms = feature.split()
        for term in terms:
            if len(terms) == 1 or ' ' not in term:  # Single mutation
                mutations.add(term)
    
    mutations = sorted(mutations)
    n = len(mutations)
    
    # Create empty matrix
    matrix = np.zeros((n, n))
    mut_to_idx = {mut: idx for idx, mut in enumerate(mutations)}
    
    # Fill matrix
    for feature, coef in coef_dict.items():
        terms = feature.split()
        if len(terms) == 1:  # Single mutation
            idx = mut_to_idx[terms[0]]
            matrix[idx, idx] = coef
        else:  # Interaction term
            mut1, mut2 = terms
            idx1, idx2 = mut_to_idx[mut1], mut_to_idx[mut2]
            matrix[idx1, idx2] = coef
            matrix[idx2, idx1] = coef  # Matrix is symmetric
    
    return matrix, mutations

def plot_interaction_heatmap(matrix, mutations):
    plt.figure(figsize=(15, 12))
    
    # Create heatmap and capture the return value
    hm = sns.heatmap(matrix, 
                     cmap='coolwarm',
                     center=0,
                     xticklabels=mutations,
                     yticklabels=mutations,
                     square=True)
    
    plt.title('Mutation Interaction Coefficients')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Add colorbar label using the heatmap's colorbar
    hm.collections[0].colorbar.set_label('Coefficient value')
    
    plt.tight_layout()
    return plt


# In[275]:


interaction_matrix, positions_ = create_interaction_matrix(dict(zip(X_poly.columns, lin_model.coef_)))


# In[276]:


positions_ = np.array(positions_)


# In[277]:


plot_interaction_heatmap(interaction_matrix, positions_)


# In[278]:


def get_high_interactions(matrix, positions_, target_mutation):
    target_pos = np.argwhere(positions_ == target_mutation)[0][0]
    coefs = matrix[target_pos]
    out = pd.DataFrame({'mutation': positions_, 'coef': coefs})
    out['abs_coef'] = np.abs(out['coef'])
    out = out.sort_values('abs_coef', ascending=False).set_index('mutation')
    return out


# In[287]:


get_high_interactions(interaction_matrix, positions_, 'T561G')


# In[322]:


new_plate[new_plate['mutations'].apply(lambda x: 'T561G' in x)]


# T561G predicted good interaction with WT A399A, and indeed they A399 is only mutated in one example where T561G is present.

# In[323]:


new_plate[new_plate['mutations'].apply(lambda x: 'T561G' in x and 'V83G' in x)]


# Also likes to occur with V83G. What percent of the time does T561G occur without either?

# In[332]:


n_t561g = new_plate[new_plate['mutations'].apply(lambda x: 'T561G' in x)].shape[0]
n_t561g_no_v83g = new_plate[new_plate['mutations'].apply(lambda x: 'T561G' in x and 'V83G' not in x)].shape[0]
n_t561g_a399_mutated  = new_plate[new_plate['mutations'].apply(lambda x: 'T561G' in x and 'A399' in ''.join(x))].shape[0]
n_t561g_no_v83g_A399_mutated = new_plate[new_plate['mutations'].apply(lambda x: 'T561G' in x and 'V83G' not in x and 'A399' in ''.join(x))].shape[0]


# In[331]:


n_t561g_no_v83g / n_t561g


# In[333]:


n_t561g_a399_mutated / n_t561g


# In[330]:


n_t561g_no_v83g_A399_mutated / n_t561g


# Only 1 in 11 times do the "epistatic coefficients" not followed for T561G.

# In[295]:


get_high_interactions(interaction_matrix, positions_, 'V83A')


# Seems V83A likes to occur with S407A, S569A, and with NOT wt S569.
# 
# for V83A - how often does it occur with S569A, S407A, and overall?

# In[320]:


n_v83a = new_plate[new_plate['mutations'].apply(lambda x: 'V83A' in x)].shape[0]
n_v83a_and_s407A = new_plate[new_plate['mutations'].apply(lambda x: 'V83A' in x and 'S407A' in x)].shape[0]
n_v83a_and_s569 = new_plate[new_plate['mutations'].apply(lambda x: 'V83A' in x and 'S569' in ''.join(x))].shape[0]
n_v83a_and_no_s407a_and_no_s569 = new_plate[new_plate['mutations'].apply(lambda x: 'V83A' in x and 'S407A' not in x and 'S569' not in ''.join(x))].shape[0]


# In[318]:


n_v83a_and_s407A / n_v83a


# In[319]:


n_v83a_and_s569 / n_v83a


# In[321]:


n_v83a_and_no_s407a_and_no_s569 / n_v83a


# Yikes. Decent amount of support ... v83a rarely occurs without at least one of s407a and s569 mutated away.

#!/usr/bin/env python
# coding: utf-8

# # Use SSM data to inform a set of higher order variants

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
import json
import hashlib
from tqdm.notebook import tqdm

import aide_predict as ap
from aide_predict.utils.plotting import plot_mutation_heatmap, plot_protein_sequence_heatmap

# Downstream supervised tools
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
# get distributions to sample
from scipy.stats import loguniform, uniform
from sklearn.metrics import make_scorer, r2_score

import datasets


# In[9]:


# create a model class 
model_no_embedder = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.98)),
    # ('scaler2', StandardScaler()),
    ('regressor', BaggingRegressor(
        estimator=TransformedTargetRegressor(
            regressor=MLPRegressor(
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=10,
                activation='relu',
                hidden_layer_sizes=(100, 100),
                alpha=0.0001,
                learning_rate_init=0.001
            ),
            transformer=StandardScaler()
        ),
        n_estimators=10,
        max_samples=0.9,
        max_features=0.9,
        bootstrap=True,
        bootstrap_features=True,
    ))
])


# ## Load and visualize SSM

# In[2]:


alignment = ap.ProteinSequences.from_fasta('mda_hacl_msa.a2m')


# In[3]:


wt = alignment[0]
wt.structure = ap.ProteinStructure('structure')


# In[4]:


ssm_data = pd.read_excel('ssm_compiled.xlsx', header=1)


# In[47]:


# remove column 0, 2 and row 1
ssm_data = ssm_data.drop(ssm_data.columns[0], axis=1)
ssm_data = ssm_data.drop(ssm_data.columns[1], axis=1)
ssm_data = ssm_data.drop(ssm_data.index[0])
ssm_data


# In[50]:


# format long with columns [mutation_string, original, position, mutation, activity]
ssm_data.rename(columns={'Unnamed: 1': 'original_position_string'}, inplace=True)


# In[51]:


ssm_data['original_aa'] = ssm_data['original_position_string'].str[0]
ssm_data['position'] = ssm_data['original_position_string'].str[1:].astype(int)


# In[52]:


ssm_data.drop(columns=['original_position_string'], inplace=True)


# In[53]:


ssm_data = ssm_data.melt(id_vars=['original_aa', 'position'], var_name='mutation', value_name='activity').sort_values(by='position').reset_index(drop=True)


# ### Plot control value distribution

# In[38]:


ssm_data['control'] = ssm_data['original_aa'] == ssm_data['mutation']


# In[39]:


sns.kdeplot(ssm_data[ssm_data['control']]['activity'], label='control')


# In[40]:


control_data = ssm_data[ssm_data['control']]


# In[41]:


control_data.sort_values(by='activity')


# In[42]:


# drop failed control
control_data = control_data[control_data['activity'] > 0]


# In[43]:


control_mean, control_std = control_data['activity'].mean(), control_data['activity'].std()
ssm_data['significant'] = np.abs((ssm_data['activity'] - control_mean) / control_std) > 2


# In[44]:


ssm_data['significant'].mean()


# ### Plot the heatmap

# In[57]:


ssm_data['mutation_string'] = ssm_data['original_aa'] + ssm_data['position'].astype(str) + ssm_data['mutation']
ssm_data['position_0_index'] = ssm_data['position'] - 1


# In[46]:


from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable
sns.set_context('paper')

class CustomNorm(Normalize):
    def __init__(self, vmin, vcenter, vmax):
        super().__init__(vmin, vmax)
        self.vcenter = vcenter
        self.lower_norm = Normalize(vmin=vmin, vmax=vcenter)
        self.upper_norm = LogNorm(vmin=vcenter, vmax=vmax)

    def __call__(self, value, clip=None):
        lower = value <= self.vcenter
        upper = value > self.vcenter
        result = np.zeros_like(value, dtype=float)
        result[lower] = self.lower_norm(value[lower])/2.0
        result[upper] = 0.5 + 0.5 * self.upper_norm(value[upper])
        return result
    
    def inverse(self, value):
        lower = value <= 0.5
        upper = value > 0.5
        result = np.zeros_like(value, dtype=float)
        result[lower] = self.lower_norm.inverse(value[lower] * 2.0)
        result[upper] = self.upper_norm.inverse((value[upper] - 0.5) * 2.0)
        return result

def plot_mutation_heatmap(mutations, scores, all_positions=True, wt_activity=None, zero_minimum=False, log_scale_upper=False):
    """
    Plot a heatmap of single point mutation scores.
    
    Parameters:
    mutations (list): List of mutation strings (e.g., ["L1V", "A2G", ...])
    scores (list): List of corresponding scores
    all_positions (bool): If True, show all positions, even those without mutations
    wt_activity (float): Wild-type activity score for centering the colormap
    zero_minimum (bool): If True, set the minimum of the colormap to 0.0
    log_scale_upper (bool): If True, use log scale for upper half of colormap
    
    Returns:
    fig, ax: The created figure and axis objects
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
    original_aa_dict = dict(zip(df['Position'], df['Original_AA']))
    
    if all_positions:
        # Create a full matrix with all amino acids and positions
        full_matrix = pd.DataFrame(index=range(1, max(positions)+1), columns=list(all_aas))
    else:
        full_matrix = pd.DataFrame(index=np.sort(np.unique(positions)), columns=list(all_aas))
    
    # Fill the matrix with scores
    for _, row in df.iterrows():
        full_matrix.at[row['Position'], row['Mutant_AA']] = row['Score']
    
    # Create the heatmap
    width = 8  # Increased width to accommodate colorbar
    height = 6/20 * len(full_matrix)
    fig, (ax, cax) = plt.subplots(1, 2, figsize=(width, height), 
                                  gridspec_kw={'width_ratios': [20, 1]})
    
    # Determine colormap and normalization
    if wt_activity is None:
        cmap = sns.color_palette("viridis", as_cmap=True)
        vmin = 0.0 if zero_minimum else np.nanmin(full_matrix.values)
        vmax = np.nanmax(full_matrix.values)
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        cmap = sns.color_palette("vlag", as_cmap=True)
        vmin = 0.0 if zero_minimum else min(np.nanmin(full_matrix.values), wt_activity)
        vmax = max(np.nanmax(full_matrix.values), wt_activity)
        if log_scale_upper:
            norm = CustomNorm(vmin=vmin, vcenter=wt_activity, vmax=vmax)
        else:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=wt_activity, vmax=vmax)
    
    # Plot the heatmap
    sns.heatmap(full_matrix.astype(np.float64), ax=ax, cmap=cmap, norm=norm,
                cbar=False, square=True, linewidths=0.5, linecolor='black',
                mask=full_matrix.isnull())
    
    # Create custom colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, label='Score')

    # Customize the plot
    ax.set_title('Single Point Mutation Scores')
    ax.set_xlabel('Mutant Amino Acid')
    ax.set_ylabel('Residue Position')
    
    # Add original amino acids to y-axis labels
    ax.set_yticks(np.array(range(len(full_matrix)))+0.5)
    ax.set_yticklabels([f'{original_aa_dict.get(i, "?")} {i}' for i in full_matrix.index])
    
    # Adjust aspect ratio to make cells square
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    return fig, ax, full_matrix


# In[47]:


fig, ax, matrix = plot_mutation_heatmap(mutations = ssm_data['mutation_string'], scores=ssm_data['activity'], all_positions=False, wt_activity=control_mean, zero_minimum=True, log_scale_upper=True)


# In[48]:


# highlight top 10 boxes
top_10 = ssm_data.sort_values(by='activity', ascending=False).head(12)
for _, row in top_10.iterrows():
    pos, to_aa = row['position'], row['mutation'][-1]
    row_pos = matrix.index.get_loc(pos)
    col_pos = matrix.columns.get_loc(to_aa)
    ax.add_patch(plt.Rectangle((col_pos, row_pos), 1, 1, fill=False, edgecolor='red', lw=2))


# In[49]:


fig.savefig('ssm_heatmap.png', dpi=300, bbox_inches='tight')


# In[50]:


fig


# In[51]:


putative_mutatable_positions = top_10['position'].unique()
print(','.join(putative_mutatable_positions.astype(str)))


# In[52]:


putative_mutatable_positions_wt_offset_zero_index = putative_mutatable_positions - 10 - 1


# In[53]:


ssm_data['position_0_index_on_wt'] = ssm_data['position'] - 10 - 1


# ## Attempt to observe opportunity for epistasis over these positions

# In[196]:


evc = ap.EVMutationWrapper(wt=wt, metadata_folder='./evmutation_metadata')
evc.fit(alignment)


# In[201]:


dir(evc.model_)


# In[205]:


couplings = evc.model_.ecs


# In[219]:


target_couplings = couplings[couplings['i'].isin(putative_mutatable_positions_wt_offset_zero_index+1) & couplings['j'].isin(putative_mutatable_positions_wt_offset_zero_index+1)]


# In[220]:


couplings


# In[224]:


fig, ax = plt.subplots(figsize=(4, 4))
sns.kdeplot(couplings['cn'], ax=ax, color='grey', log_scale=True, label='All Couplings')
sns.kdeplot(target_couplings['cn'], ax=ax, color='red', log_scale=True, label='Target Couplings')


# In[231]:


couplings['is_target'] = couplings.index.isin(target_couplings.index)
target_cn_rankings = 1 - np.where(couplings['is_target'])[0]/len(couplings)


# In[237]:


target_cn_rankings


# In[236]:


print('Mean CN ranks for target couplings:', target_cn_rankings.mean())
print('quartiles:', np.quantile(target_cn_rankings, [0.25, 0.5, 0.75]))


# In[238]:


target_couplings


# > The couplings for our target sites are saturated with high coupling. Eg. 7 pairwise interactions are 90%ile couplings over whole MSA couplings. The 25%ile of target couplings is > 56% of bulk couplings. The structure also indicates that the target sites are in close proximity to each other, more opportunity for epistasis.
# 
# ![image.png](attachment:image.png)

# ## Plate 1: Highest ranked additive variants
# 
# Up to order 3, add single point mutations together to form higher order variants. Enforce that the variants explore each position equally regardless of additive ranking.

# In[25]:


# drop the ssm data down to only the positions we care about eg. the top 12 mutations
top_12_ssm_data = ssm_data.sort_values(by='activity', ascending=False).head(12).reset_index(drop=True)


# In[26]:


top_12_ssm_data


# In[27]:


# create all combinations of 3 mutations, and 2 mutations
from itertools import combinations
combination_indexes = list(combinations(
    top_12_ssm_data.index, 3
))

combination_indexes += list(combinations(
    top_12_ssm_data.index, 2
))
additive_df = pd.Series(name='top12_index_combinations', data=combination_indexes)
additive_df = pd.DataFrame(additive_df)


# In[28]:


# remove any combinations that occur at the same position
def check_positions(row):
    positions = top_12_ssm_data.loc[list(row), 'position']
    return len(positions) == len(set(positions))
additive_df = additive_df[additive_df['top12_index_combinations'].apply(check_positions)]


# In[29]:


# compute additive scores
def get_additive_score(row):
    top12_indexes = row['top12_index_combinations']
    score = top_12_ssm_data.loc[list(top12_indexes), 'activity'].sum()
    return score
additive_df['additive_activity'] = additive_df.apply(get_additive_score, axis=1)


# In[30]:


# sort by additive score
# chose the top plate, split into order 2 and order 3
# do not add the same mutation more than 96/12 = 8 times
# so that we explore
additive_df.sort_values(by='additive_activity', ascending=False, inplace=True)
additive_df['n_mutations'] = additive_df['top12_index_combinations'].apply(len)


# In[31]:


for top12_index in top_12_ssm_data.index:
    additive_df[f'has_{top12_index}'] = additive_df['top12_index_combinations'].apply(lambda x: top12_index in x)


# In[32]:


current_mutation_representative_count = np.zeros(12)
def get_representation_score(top_12_indexes):
    """Get a score that is how underrepresented the variant is"""
    # compute inverse fractions from the counts
    if sum(current_mutation_representative_count) == 0:
        return 1.0
    else:
        inverse_ratios = 1 - current_mutation_representative_count/sum(current_mutation_representative_count)

        return sum([inverse_ratios[ind] for ind in top_12_indexes])/len(top_12_indexes)

# initialize with best 3 and 2 mutant
selected_additive_indexes = [additive_df[additive_df['n_mutations'] == 3].index[0], additive_df[additive_df['n_mutations'] == 2].index[0]]
# update the representative count
for comb in additive_df.loc[selected_additive_indexes, 'top12_index_combinations']:
    for ind in comb:
        current_mutation_representative_count[ind] += 1


two_or_three_bool_switch = False
while len(selected_additive_indexes) < 96:

    if two_or_three_bool_switch:
        n_mutations = 3
    else:
        n_mutations = 2
    # switch
    two_or_three_bool_switch = not two_or_three_bool_switch

    # df subset to the correct mutation count and remove already chosen variants
    df_subset = additive_df[additive_df['n_mutations'] == n_mutations]
    df_subset = df_subset[~df_subset.index.isin(selected_additive_indexes)]

    # select the mutation with the current least representation
    # show only variants with that mutation
    selected_mutation_indexes_sorted = np.argsort(current_mutation_representative_count)
    for selected_mutation_index in selected_mutation_indexes_sorted:
        if df_subset[f'has_{selected_mutation_index}'].sum() > 0:
            break
    # select the variant that maximizes representation
    df_subset['representation_score'] = df_subset['top12_index_combinations'].apply(get_representation_score)
    # sort by representation score and select the first
    df_subset.sort_values(by='representation_score', ascending=False, inplace=True)

    selected_additive_index = df_subset.index[0]
    selected_variant_12_indexes = df_subset.loc[selected_additive_index, 'top12_index_combinations']

    # add the selected index to the list
    selected_additive_indexes.append(selected_additive_index)
    print(len(selected_additive_indexes))
    # update the representative count
    for ind in selected_variant_12_indexes:
        current_mutation_representative_count[ind] += 1

    print("Selected: ", selected_additive_index, selected_variant_12_indexes, "Additive score: ", df_subset.loc[selected_additive_index, 'additive_activity'])
    print("Current representative mutation ratio: ", current_mutation_representative_count/np.sum(current_mutation_representative_count))
    print("====================================")

    


# In[33]:


selected_additive = additive_df.loc[selected_additive_indexes]


# In[34]:


selected_additive


# In[35]:


# parse mutation strings and get variant scores
def get_variant_metadata_from_top12(row):
    top12_indexes = row['top12_index_combinations']

    mutations = []
    wt_0_index_positions = []
    for ind in top12_indexes:
        top12_row = top_12_ssm_data.loc[ind]
        mutation_string = top12_row['mutation_string']
        mutations.append(mutation_string)
        wt_0_index_positions.append(top12_row['position_0_index_on_wt'])
    
    sequence = list(wt)
    mutations_0_index = []
    for mutation, position in zip(mutations, wt_0_index_positions):
        original_aa = mutation[0]
        to_aa = mutation[-1]
        assert wt[position] == original_aa, f"Original amino acid mismatch: {wt[position]} != {original_aa}"
        sequence[position] = to_aa
        mutations_0_index.append(f'{original_aa}{position}{to_aa}')
    sequence = ap.ProteinSequence(''.join(sequence), id=f'Additive_{";".join(mutations)}')

    return {'variant': sequence, 'mutations': ';'.join(mutations), 'positions': ';'.join(map(str, wt_0_index_positions)), 'additive_activity': row['additive_activity'], 
            'id': f'Additive_{";".join(mutations)}'}

plate1_df = pd.DataFrame(list(selected_additive.apply(get_variant_metadata_from_top12, axis=1)))


# In[36]:


plate1_df.to_csv('plate1.csv', index=False)


# In[37]:


plate1_df


# ## Test ML model on SSM data
# 
# 1. Train a model on the data, hyperparameter optimize to maximize performance
# 2. With the chosen embedding, embed plate 1 so that we we can avoid sampling similar to plate 1
# 
# SEE NEXT SECTIONS
# 3. Enumerate 2 and 3 order mutants over all 20 amino acids at 10 positions: ~18,000 2 order variants, ~1 million order 3 variants
#     Remove any variants already seen. 
# 4. Predict UCB for variants, compute nearest neigbor in Plate 1 or SSM
# 5. Select a plates worth that maximizes UCB and diversity

# ### Model create
# pLM embedding is required to collect information across positions across the whole SSM that are not in the selected positions. Hopefully we can achieve decent performance with a small ESM model.

# In[63]:


# create X and y from SSM data
ssm_ml_df = ssm_data.copy()
def create_ssm_variant(row):
    original_aa = row['original_aa']
    position = row['position_0_index_on_wt']
    mutation = row['mutation'][-1]
    assert wt[position] == original_aa, f"Original amino acid mismatch: {wt[position]} != {original_aa}"
    sequence = list(wt)
    sequence[position] = mutation
    sequence = ''.join(sequence)
    
    return sequence
ssm_ml_df['seq_str'] = ssm_ml_df.apply(create_ssm_variant, axis=1)
ssm_ml_df['id'] = ['SSM_' + m for m in ssm_ml_df['mutation_string']]
ssm_ml_df.dropna(inplace=True)

# group by seq and take mean
seq_to_id = dict(zip(ssm_ml_df['seq_str'], ssm_ml_df['id']))
ssm_ml_df = ssm_ml_df.groupby('seq_str').mean().reset_index()
ssm_ml_df['id'] = ssm_ml_df['seq_str'].apply(lambda x: seq_to_id[x])


# In[64]:


ssm_ml_df['seq'] = ssm_ml_df.apply(lambda row: ap.ProteinSequence(row['seq_str'], id=row['id']), axis=1)


# In[65]:


ssm_ml_df = ssm_ml_df[['seq_str', 'seq', 'activity', 'id']]


# In[66]:


# re id wildtype
for row in ssm_ml_df.itertuples():
    if str(row.seq) == str(wt):
        ssm_ml_df.at[row.Index, 'id'] = 'WT'
        row.seq.id = 'WT'


# In[67]:


X_seq = ap.ProteinSequences(ssm_ml_df['seq'].to_list())
y = ssm_ml_df['activity'].to_numpy()


# In[68]:


plot_protein_sequence_heatmap(X_seq)


# In[69]:


# create a model class 
model_no_embedder = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.98)),
    # ('scaler2', StandardScaler()),
    ('regressor', BaggingRegressor(
        estimator=TransformedTargetRegressor(
            regressor=MLPRegressor(
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=10,
                activation='relu',
                hidden_layer_sizes=(100, 100),
                alpha=0.0001,
                learning_rate_init=0.001
            ),
            transformer=StandardScaler()
        ),
        n_estimators=10,
        max_samples=0.9,
        max_features=0.9,
        bootstrap=True,
        bootstrap_features=True,
    ))
])


# In[70]:


esm_full = ap.ESM2Embedding(
    model_checkpoint='esm2_t33_650M_UR50D',
    metadata_folder='esm2_embedder',
    pool=True,
    device='mps'
)
esm_full.fit([])


# In[206]:


X_big_embeddings = esm_full.transform(X_seq)


# #### Observe the embeddings, how information dense are we talking here

# In[216]:


scaler = StandardScaler()
X_ = scaler.fit_transform(X_big_embeddings)


# In[218]:


X_.mean(axis=0), X_.std(axis=0)


# In[219]:


pca = PCA(n_components=0.98)
X_ = pca.fit_transform(X_)


# In[220]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.scatterplot(x=X_[:, 0], y=X_[:, 1], hue=y, ax=ax, palette='viridis', alpha=0.8)


# There is a minor bit of label seperation. let's predict
# 
# #### Check for signal with a single train test split

# In[271]:


X_train, X_test, y_train, y_test = train_test_split(X_big_embeddings, y, test_size=0.2, random_state=42)


# In[272]:


oracle = model_no_embedder.fit(X_train, y_train)
oracle.score(X_test, y_test)


# In[273]:


# score is not bad.....
# how do predictions look
# get standard deviations and mean predictions from the bagging regressor
X_test_transformed = oracle[:-1].transform(X_test)


# In[275]:


all_predictions = []
for i, estimator in enumerate(oracle[-1].estimators_):
    feature_indexes = oracle[-1].estimators_features_[i]
    model_input = X_test_transformed[:, feature_indexes]
    y_pred_ = estimator.predict(model_input)
    all_predictions.append(y_pred_)
all_predictions = np.array(all_predictions)
means = np.mean(all_predictions, axis=0)
stds = np.std(all_predictions, axis=0)


# In[251]:


stds.max()


# In[252]:


fig, ax = plt.subplots(figsize=(8, 8))
min_min = min(y_test.min(), means.min())
max_max = max(y_test.max(), means.max())
ax.plot([min_min, max_max], [min_min, max_max], color='black', linestyle='--')

ax.errorbar(y_test, means, yerr=stds, fmt='o', alpha=0.5)


# ### Check size of embeddings required
# 
# This assumes that embeddings will behave similarly for downstream hyperopt.

# In[276]:


model_sizes = [
    'esm2_t33_650M_UR50D',
    'esm2_t30_150M_UR50D',
    'esm2_t12_35M_UR50D',
    'esm2_t6_8M_UR50D',
]

model_size_scores_dict = {}
for model_size in model_sizes:
    esm_full = ap.ESM2Embedding(
        model_checkpoint=model_size,
        metadata_folder='esm2_embedder',
        pool=True,
        device='mps'
    )
    esm_full.fit([])
    X_embeddings = esm_full.transform(X_seq)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_index, test_index in kfold.split(X_embeddings):
        X_train, X_test = X_embeddings[train_index], X_embeddings[test_index]
        y_train, y_test = y[train_index], y[test_index]

        oracle = model_no_embedder.fit(X_train, y_train)
        score = oracle.score(X_test, y_test)
        scores.append(score)
    model_size_scores_dict[model_size] = np.mean(scores), np.std(scores)


# In[279]:


model_size_scores


# In[278]:


fig, ax = plt.subplots(figsize=(8, 8))
model_size_scores = [v[0] for k, v in model_size_scores.items()]
model_size_std = [v[1] for k, v in model_size_scores.items()]
ax.bar(model_sizes, model_size_scores, yerr=model_size_std, color='grey')
                     


# In[ ]:





# ### Embedding all variants
# 
# The datasets schema:
# - `variant`: the variant sequence
# - `hash`: the hash of the variant
# - `id`: the id of the variant
# - `embedding`: the embedding of the variant
# 
# 1. Embed the SSM data and wild type.
#    ids like `SSM_<mutation_string>`
# 2. Embed plate 1
#    ids like `Additive_<mutation_string>`
# 3. Embed the 2 and 3 order variants
#    ids like `<mutation_string>`
#    check the existing data hashes to pass on duplicates

# In[78]:


embedder = ap.ESM2Embedding(
    model_checkpoint='esm2_t12_35M_UR50D',
    metadata_folder='esm2_embedder',
    pool=True,
    device='mps'
)
embedder.fit([])


# #### Start with SSM

# In[79]:


ssm_ml_df_ = ssm_ml_df[['seq_str', 'id', 'activity']].rename(columns={'seq_str': 'seq'})


# In[80]:


ssm_ml_df_['hash'] = ssm_ml_df_['seq'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())


# In[82]:


ssm_embeddings = embedder.transform(ssm_ml_df_['seq'].to_list()) 


# In[83]:


dataset_dict = {
    'id': ssm_ml_df_['id'].to_list(),
    'hash': ssm_ml_df_['hash'].to_list(),
    'variant': ssm_ml_df_['seq'].to_list(),
    'activity': ssm_ml_df_['activity'].to_list(),
    'embedding': ssm_embeddings
}


# In[84]:


ssm_dataset = datasets.Dataset.from_dict(dataset_dict)


# In[85]:


ssm_dataset.features


# In[86]:


ssm_dataset.save_to_disk('hf_ssm_dataset')


# #### Plate 1 embeddings

# In[56]:


plate1_seqs = ap.ProteinSequences(plate1_df['variant'].to_list())


# In[57]:


plate1_embeddings = embedder.transform(plate1_seqs)


# In[58]:


plate1_dataset = datasets.Dataset.from_dict({
    'id': plate1_df['id'].to_list(),
    'variant': [str(seq) for seq in plate1_df['variant']],
    'embedding': plate1_embeddings,
    'hash': [hashlib.md5(str(seq).encode()).hexdigest() for seq in plate1_df['variant']]
})


# In[59]:


plate1_dataset.save_to_disk('hf_plate1_dataset')


# #### Iterate over order 2 and 3 variants

# In[403]:


putative_mutatable_positions_wt_offset_zero_index


# In[414]:


import itertools


# In[411]:


def mutation_combinations(n_positions, n_options, max_order=2):

    to_yield = [None for _ in range(n_positions)]

    for positions in itertools.combinations(range(n_positions), max_order):
        for options in itertools.product(range(n_options), repeat=len(positions)):
            for i, pos in enumerate(positions):
                to_yield[pos] = options[i]
            yield tuple(to_yield)
            to_yield = [None for _ in range(n_positions)]


# In[416]:


def sequence_string_from_mutation_combination_iterator(
        wt_sequence, positions_mutating, max_order=2
):
    """Generate a sequence strings by iterating over mutation combinations

    Parameters
    ----------
    wt_sequence : str
        The wild-type sequence
    positions_mutating : list
        List of positions to mutate
    max_order : int
        Maximum order of mutations to consider
    """
    AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'
    n_positions = len(positions_mutating)
    n_options = len(AA_LIST)

    for mutation_combination in mutation_combinations(n_positions, n_options, max_order):
        sequence = list(wt_sequence)
        mutation_strings = []
        for sequence_index, mutation_index in zip(positions_mutating, mutation_combination):
            if mutation_index is None:
                continue
            if sequence[sequence_index] == AA_LIST[mutation_index]:
                continue
            sequence[sequence_index] = AA_LIST[mutation_index]
            mutation_strings.append(f'{wt_sequence[sequence_index]}{sequence_index+11}{AA_LIST[mutation_index]}')
        id_ = ';'.join(mutation_strings)
        hash_ = hashlib.md5(''.join(sequence).encode()).hexdigest()
        yield ''.join(sequence), id_, hash_


# In[430]:


from tqdm.notebook import tqdm


# In[425]:


existing_hashes = set(ssm_dataset['hash']).union(set(plate1_dataset['hash']))


# In[488]:


def multiple_mutation_orders_sequence_string_combination_iterator(
        wt_sequence, positions_mutating, max_order=3
):
    for order in range(2, max_order+1):
        for sequence, id_, hash_ in sequence_string_from_mutation_combination_iterator(wt_sequence, positions_mutating, order):
            yield sequence, id_, hash_


# In[489]:


# embed them
def dataset_generator():
    block_size = 1000
    bar = tqdm()

    block = {
        'id': [],
        'hash': [],
        'variant': [],
        'embedding': None
    }
    for i, (sequence, id_, hash_) in enumerate(multiple_mutation_orders_sequence_string_combination_iterator(wt, putative_mutatable_positions_wt_offset_zero_index, max_order=3)):
        if hash_ in existing_hashes:
            continue

        block['id'].append(id_)
        block['hash'].append(hash_)
        block['variant'].append(sequence)
        if len(block['id']) == block_size:
            block['embedding'] = embedder.transform(block['variant'])
            
            for i in range(len(block['id'])):
                yield {
                    'id': block['id'][i],
                    'hash': block['hash'][i],
                    'variant': block['variant'][i],
                    'embedding': block['embedding'][i]
                }

            block = {
                'id': [],
                'hash': [],
                'variant': [],
                'embedding': None
            }
            bar.update(block_size)

        


# In[490]:


variants_dataset = datasets.Dataset.from_generator(dataset_generator)


# In[491]:


variants_dataset.save_to_disk('hf_putative_variants_dataset')


# In[461]:


variants_dataset = datasets.load_from_disk('hf_putative_variants_dataset')


# ### Hyperparameter optimization
# 
# Use the SSM data to determine the best set of hyperparameters for the model.

# In[77]:


param_space = {
    'regressor__max_samples': uniform(0.5, 0.5),
    'regressor__max_features': uniform(0.5, 0.5),
    'regressor__estimator__regressor__alpha': loguniform(0.0001, 0.1),
    'regressor__estimator__regressor__learning_rate_init': loguniform(0.0001, 0.1),
    'regressor__estimator__regressor__hidden_layer_sizes': [(100, ), (100, 100), (100, 100, 100)],
    'regressor__estimator__regressor__activation': ['relu', 'tanh']
}


# In[109]:


# scores that balance model uncertainty and accuracy
def uncertainty_nmse(y_true, y_pred, y_std):
    """
    Compute the normalized mean squared error of the predictions.

    
    """
    squared_error = (y_true - y_pred)**2
    normalized_squared_error = squared_error / (y_std**2 + 1e-8)  # Add small epsilon to avoid division by zero
    return np.mean(normalized_squared_error)

def evaluate_model(model, X, y):
    predictions = []
    transform_pipeline = model[:-1]
    transformed_input = transform_pipeline.transform(X)

    for i, estimator in enumerate(model[-1].estimators_):
        feature_indexes = model[-1].estimators_features_[i]
        model_input = transformed_input[:, feature_indexes]
        y_pred = estimator.predict(model_input)
        predictions.append(y_pred)
    
    y_pred = np.mean(predictions, axis=0)
    y_std = np.std(predictions, axis=0)
    
    nmse = uncertainty_nmse(y, y_pred, y_std)
    r2 = r2_score(y, y_pred)
    
    return nmse, r2, y_pred, y_std

def custom_scorer(estimator, X, y):
    nmse, r2, _, _ = evaluate_model(estimator, X, y)
    return r2 + 1.0/nmse


# In[121]:


search = RandomizedSearchCV(
    model_no_embedder,
    param_space,
    n_iter=200,
    scoring=custom_scorer,
    cv=5,
    n_jobs=1,
    random_state=42,
    verbose=2
)


# In[73]:


ssm_dataset = datasets.load_from_disk('hf_ssm_dataset')


# In[74]:


ssm_dataset.set_format('numpy', columns=['embedding'])
X_ssm = ssm_dataset['embedding']
y_ssm = ssm_ml_df_['activity']


# In[69]:


X_ssm.shape


# In[122]:


search.fit(X_ssm, y_ssm)


# In[129]:


search.best_params_


# In[124]:


import json
with open('best_params.json', 'w') as f:
    json.dump(search.best_params_, f)
with open('best_params.json', 'r') as f:
    best_params = json.load(f)


# In[125]:


best_model = model_no_embedder.set_params(**best_params)


# In[126]:


kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_predictions = []
fold_targets = []
fold_stds = []

for train_index, test_index in kfold.split(X_ssm):
    X_train, X_test = X_ssm[train_index], X_ssm[test_index]
    y_train, y_test = y_ssm[train_index], y_ssm[test_index]

    model = best_model.fit(X_train, y_train)
    nmse, r2, y_pred, y_std = evaluate_model(model, X_test, y_test)
    
    fold_predictions.append(y_pred)
    fold_targets.append(y_test)
    fold_stds.append(y_std)

# combine all predictions
fold_predictions = np.concatenate(fold_predictions)
fold_targets = np.concatenate(fold_targets)
fold_stds = np.concatenate(fold_stds)


# In[127]:


fig, ax = plt.subplots(figsize=(8, 8))
min_min = min(fold_targets.min(), fold_predictions.min())
max_max = max(fold_targets.max(), fold_predictions.max())
ax.plot([min_min, max_max], [min_min, max_max], color='black', linestyle='--')
ax.errorbar(fold_targets, fold_predictions, yerr=fold_stds, fmt='o', alpha=0.5)
ax.set_xlabel('True Activity [fold WT]')
ax.set_ylabel('Predicted Activity [fold WT]')


# In[137]:


sns.set_style('whitegrid')
sns.set_context('talk')
fig, ax = plt.subplots(figsize=(8, 8))
min_min = min(fold_targets.min(), fold_predictions.min())
max_max = max(fold_targets.max(), fold_predictions.max())
sns.kdeplot(x=fold_targets, y=fold_predictions, ax=ax, fill=False, cmap='viridis', levels=50)
ax.plot([min_min, max_max], [min_min, max_max], color='black', linestyle='--')
ax.set_xlabel('True Activity [fold WT]')
ax.set_ylabel('Predicted Activity [fold WT]')


# In[131]:


# show model error vs uncertainty
fig, ax = plt.subplots(figsize=(8, 8))
model_error = np.abs(fold_targets - fold_predictions)
ax.scatter(model_error, fold_stds, alpha=0.5)
pearson = stats.pearsonr(model_error, fold_stds)
ax.set_xlabel('Model Error [fold WT]')
ax.set_ylabel('Model Uncertainty [fold WT]')
ax.title.set_text(f'Pearson R: {pearson[0]:.2f}')


# In[144]:


prediction_df = pd.DataFrame({
    'targets': fold_targets,
    'predictions': fold_predictions,
    'stds': fold_stds
})


# In[146]:


prediction_df['predicted_lt_1'] = prediction_df['predictions'] < 1
(prediction_df[prediction_df['predicted_lt_1']] > 1).mean()


# In[148]:


prediction_df['predicted_gt_1'] = prediction_df['predictions'] > 1
(prediction_df[prediction_df['predicted_gt_1']] <1).mean()


# In[149]:


(prediction_df['targets'] < 1.0).mean()


# ## Plate 2 and half of plate 3: Train final model and use to select variants

# In[5]:


variants_dataset = datasets.load_from_disk('hf_putative_variants_dataset')
variants_dataset.set_format('numpy', columns=['embedding'])


# #### Load ssm data and train

# In[187]:


ssm_dataset = datasets.load_from_disk('hf_ssm_dataset')
ssm_dataset.set_format('numpy', columns=['embedding', 'activity'])
X_ssm = ssm_dataset['embedding']
y_ssm = ssm_dataset['activity']


# In[188]:


with open('best_params.json', 'r') as f:
    best_params = json.load(f)


# In[189]:


final_model = model_no_embedder.set_params(**best_params)
final_model.fit(X_ssm, y_ssm)


# In[190]:


def predict_w_uncertainty(model, X):
    predictions = []
    transform_pipeline = model[:-1]
    transformed_input = transform_pipeline.transform(X)

    for i, estimator in enumerate(model[-1].estimators_):
        feature_indexes = model[-1].estimators_features_[i]
        model_input = transformed_input[:, feature_indexes]
        y_pred = estimator.predict(model_input)
        predictions.append(y_pred)
    
    y_pred = np.mean(predictions, axis=0)
    y_std = np.std(predictions, axis=0)
    
    return y_pred, y_std


# In[191]:


# create a batched HF mapping function
def model_prediction_mapping(examples):
    embeddings = examples['embedding']
    preds, stds = predict_w_uncertainty(final_model, embeddings)
    examples.update({'activity_pred': preds, 'activity_pred_std': stds})
    return examples


# In[192]:


# confirm mapped correctly with ssm data
ssm_dataset_ = ssm_dataset.map(model_prediction_mapping, batched=True, batch_size=1000)


# In[193]:


ssm_dataset_.set_format('pandas')
preds = ssm_dataset_['activity_pred']


# In[194]:


r2 = r2_score(ssm_dataset_['activity'], preds)
r2


# #### predict variant dataset

# In[195]:


# map to variants dataset
variants_dataset_ = variants_dataset.map(model_prediction_mapping, batched=True, batch_size=10000)


# In[196]:


# drop duplicates due to chosing wild type as a variant
# the has column tracks unique variants
# get the first index of each unique variant
hash_set = set()

indexes = []
for index, hash_ in enumerate(variants_dataset_['hash']):
    if hash_ in hash_set:
        continue
    hash_set.add(hash_)
    indexes.append(index)


# In[19]:


variants_dataset_ = variants_dataset_.select(indexes)


# #### map mutation list list for each variant

# In[197]:


# parse mutations for the plate
def map_mutation_string(example):
    id = example['id']
    example['mutations'] = id.split(';')
    return example


# In[198]:


variants_dataset_ = variants_dataset_.map(map_mutation_string)


# #### sort variants by prediction

# In[201]:


# sort by predicted activity
# reverse it
sorted_idx = np.array(list(reversed(np.argsort(variants_dataset_['activity_pred']))))


# In[202]:


variants_dataset_ = variants_dataset_.select(sorted_idx)


# #### iteratively add variants to plate greedily
# track the number of mutations explored so we do not saturate variants with a single mutation

# In[203]:


# start with plate 1 counts
plate1_df = pd.read_csv('plate1.csv')


# In[205]:


plate1_df['mutations'] = plate1_df['mutations'].apply(lambda s: s.split(';'))
mutation_count_dict = {}
for plate1_variant_mutations in plate1_df['mutations']:
    for mutation in plate1_variant_mutations:
        if mutation in mutation_count_dict:
            mutation_count_dict[mutation] += 1
        else:
            mutation_count_dict[mutation] = 1


# In[206]:


mutation_count_dict


# In[207]:


# add data greedily but track the number of times each mutation is added
# we do not want to observe the same mutation more than 3 times
MAX_MUTATION_OCCURANCE = 3

selected_greedy_indexes = []
# loop backwards
i = 0
bar = tqdm(len(variants_dataset_))
class Completed(Exception): pass
try:
    for i, row in enumerate(variants_dataset_):
        mutations = row['mutations']
        viable = True
        for mutation in mutations:
            if mutation_count_dict.get(mutation, 0) >= MAX_MUTATION_OCCURANCE:
                viable = False
                break
        if viable:
            selected_greedy_indexes.append(i)
            for mutation in mutations:
                if mutation in mutation_count_dict:
                    mutation_count_dict[mutation] += 1
                else:
                    mutation_count_dict[mutation] = 1

        i += 1
        bar.update(1)

        if len(selected_greedy_indexes) == 96:
            raise Completed
            
except Completed:
    pass


# In[208]:


mutation_count_dict


# In[209]:


greedy_selected  = variants_dataset_.select(selected_greedy_indexes)


# In[210]:


greedy_df = greedy_selected.to_pandas()
greedy_df


# In[211]:


plate2_single_mutation_counts = pd.DataFrame(pd.value_counts(np.hstack(greedy_df['mutations'])))


# In[212]:


plate2_single_mutation_counts['ssm_activity'] = ssm_data[ssm_data['mutation_string'].isin(plate2_single_mutation_counts.index)]['activity'].values


# In[213]:


# hist of mutations chosen and variants chosen
fig, ax = plt.subplots(figsize=(8, 8))
sns.histplot(plate2_single_mutation_counts['ssm_activity'], bins=20, ax=ax)
ax.title.set_text('Distribution of SSM activity scores for mutations chosen in Plate 2')


# In[214]:


# now hist predictions for plate 2 compared to actual ssm data
fig, ax = plt.subplots(figsize=(8, 8))
ssm_activities = ssm_data['activity'].values
pred_plate2 = greedy_df['activity_pred'].values
sns.kdeplot(ssm_activities, ax=ax, label='SSM Data', fill=False)
sns.kdeplot(pred_plate2, ax=ax, label='Plate 2 Predictions', fill=False)
plt.legend()
plt.title('SSM Activities vs Plate 2 Predictions')


# ### Plate 3 first half: use UCB

# #### sort by ucb

# In[215]:


ucb_sorted_idx = np.array(list(reversed(np.argsort(
    variants_dataset_['activity_pred'] + 2.0*variants_dataset_['activity_pred_std']
))))


# In[216]:


variants_dataset_ = variants_dataset_.select(ucb_sorted_idx)


# #### continue selecting variants, adhering to the mutation counts limit

# In[217]:


selected_ucb_index = []
# loop backwards
i = 0
bar = tqdm(len(variants_dataset_))
class Completed(Exception): pass
try:
    for i, row in enumerate(variants_dataset_):

        mutations = row['mutations']
        viable = True
        for mutation in mutations:
            if mutation_count_dict.get(mutation, 0) >= MAX_MUTATION_OCCURANCE:
                viable = False
                break
        if viable:
            selected_ucb_index.append(i)
            for mutation in mutations:
                if mutation in mutation_count_dict:
                    mutation_count_dict[mutation] += 1
                else:
                    mutation_count_dict[mutation] = 1

        i += 1
        bar.update(1)

        if len(selected_ucb_index) == 48:
            raise Completed
            
except Completed:
    pass


# In[222]:


ucb_selected = variants_dataset_.select(selected_ucb_index)
ucb_df = ucb_selected.to_pandas()


# In[233]:


greedy_df.to_csv('plate2_greedy.csv', index=False)
ucb_df.to_csv('plate3_ucb.csv', index=False)


# ## Second half of Plate 3: pure training set design/exploration using D optimality

# In[2]:


ssm_dataset = datasets.load_from_disk('hf_ssm_dataset')
plate1_dataset = datasets.load_from_disk('hf_plate1_dataset')
variants_dataset = datasets.load_from_disk('hf_putative_variants_dataset')
combined_dataset = datasets.concatenate_datasets([ssm_dataset, plate1_dataset, variants_dataset])
combined_dataset.set_format('torch', columns=['embedding'])


# In[3]:


greedy_df = pd.read_csv('plate2_greedy.csv')
ucb_df = pd.read_csv('plate3_ucb.csv')


# In[4]:


# label rows in the combined dataset that are already selected
ssm_hashes = set(ssm_dataset['hash'])
plate1_hashes = set(plate1_dataset['hash'])
ucb_hashes = set(ucb_df['hash'])
greedy_hashes = set(greedy_df['hash'])

assert len(ssm_hashes.intersection(plate1_hashes)) == 0
assert len(ssm_hashes.intersection(ucb_hashes)) == 0
assert len(ssm_hashes.intersection(greedy_hashes)) == 0
assert len(ucb_hashes.intersection(greedy_hashes)) == 0

all_hashes = ssm_hashes.union(plate1_hashes).union(ucb_hashes).union(greedy_hashes)


# In[14]:


print('Number of unique hashes:', len(all_hashes))


# In[22]:


# get the indexes of already selected sequences so we can start D optimality with the existing selected data
selected_indexes = np.isin(combined_dataset['hash'], list(all_hashes))


# In[25]:


sum(selected_indexes)


# In[28]:


selected_indexes = np.where(selected_indexes)[0]


# In[29]:


X = combined_dataset['embedding']


# In[32]:


pca = PCA(n_components=0.98)
X_pca = pca.fit_transform(X)


# In[33]:


import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial import distance_matrix
from scipy import stats


class ModifiedFrankWolfe:
    def __init__(self, X: np.ndarray, k: int, pre_evaluated_indices: list = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.n, self.d = self.X.shape
        self.k = k
        self.selected = set(pre_evaluated_indices) if pre_evaluated_indices else set()
        self.lamb = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.XXT = torch.bmm(self.X.unsqueeze(2), self.X.unsqueeze(1))
        self.A = torch.zeros((self.d, self.d), dtype=torch.float32, device=self.device)
        
        # Initialize with pre-evaluated examples
        if pre_evaluated_indices:
            for idx in pre_evaluated_indices:
                self.lamb[idx] = 1
                self.update_A(idx)
        
        # Metrics tracking
        self.D_opt_history = []
        self.uniformity_score_history = []

        # Compute feature ranges for normalization
        self.feature_min = torch.min(self.X, dim=0).values
        self.feature_max = torch.max(self.X, dim=0).values
    
    @property
    def D_opt(self):
        return torch.logdet(self.A).item()
    
    def update_A(self, index):
        self.A += self.XXT[index]
    
    def g_prime_i(self, indices):
        inv_A = torch.inverse(self.A)
        return torch.diagonal(torch.bmm(torch.bmm(inv_A.unsqueeze(0).expand(len(indices), -1, -1), 
                                                  self.XXT[indices]), 
                                        inv_A.unsqueeze(0).expand(len(indices), -1, -1))).sum(dim=1)
    
    def select_next(self):
        available = list(set(range(self.n)) - self.selected)
        gradients = self.g_prime_i(torch.tensor(available, device=self.device))
        best_index = available[torch.argmax(gradients).item()]
        return best_index
    
    def compute_uniformity_score(self):
        selected_X = self.X[list(self.selected)].cpu().numpy()
        distribution_scores = []
        
        for feature in range(self.d):
            feature_values = selected_X[:, feature]
            feature_min = self.feature_min[feature].item()
            feature_max = self.feature_max[feature].item()
            
            # Create bins across the entire feature range
            bins = np.linspace(feature_min, feature_max, num=20)
            
            # Compute observed frequencies
            observed_freq, _ = np.histogram(feature_values, bins=bins)
            observed_prob = observed_freq / len(feature_values)
            
            # Compute expected frequencies for a uniform distribution
            expected_prob = np.ones_like(observed_prob) / len(observed_prob)
            
            # Compute Chi-square statistic
            chi_square = np.sum((observed_prob - expected_prob)**2 / expected_prob)
            
            # Compute p-value (higher p-value indicates closer to uniform)
            p_value = 1 - stats.chi2.cdf(chi_square, df=len(bins)-1)
            distribution_scores.append(p_value)
        
        # Average p-value across all features
        return np.mean(distribution_scores)
    
    def compute_metrics(self):
        # D-optimality
        current_D_opt = self.D_opt
        self.D_opt_history.append(current_D_opt)

        # uniformity
        uniformity_score = self.compute_uniformity_score()
        self.uniformity_score_history.append(uniformity_score)
    
    def run(self):
        # Initial selection if needed
        if len(self.selected) < self.d:
            additional_needed = self.d - len(self.selected)
            available = list(set(range(self.n)) - self.selected)
            initial_indices = np.random.choice(available, size=additional_needed, replace=False)
            for idx in initial_indices:
                self.selected.add(idx)
                self.lamb[idx] = 1
                self.update_A(idx)
        
        # Main loop
        pbar = tqdm(total=self.k - len(self.selected))
        while len(self.selected) < self.k:
            next_index = self.select_next()
            self.selected.add(next_index)
            self.lamb[next_index] = 1
            self.update_A(next_index)
            self.compute_metrics()
            pbar.update(1)
            pbar.set_description(f"D-optimality: {self.D_opt:.4f}")
        
        pbar.close()
        return self.lamb.cpu().numpy(), self.D_opt


# In[42]:


fw = ModifiedFrankWolfe(X_pca, len(selected_indexes)+48, pre_evaluated_indices=list(selected_indexes), device='cpu')
lambdas, D_opt = fw.run()


# In[43]:


d_opt_selected = np.where(lambdas.astype(bool))[0]
d_opt_selected = [i for i in d_opt_selected if i not in selected_indexes]
d_opt_selected = variants_dataset.select(d_opt_selected)


# In[44]:


d_opt_selected


# In[ ]:





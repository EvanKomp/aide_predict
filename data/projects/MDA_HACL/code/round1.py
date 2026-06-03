#!/usr/bin/env python
# coding: utf-8

# # MDA Hydroxyacyle-CoA Lyase variant design round 1: Higher order variants
# 
# In Round 0, SSM over 64 residues was conducted. Here in Round 1, we use this data to inform some higher order combinatorial variants to test. These variants are design to target exploration and exploitation.
# 
# __Plate 1__: _The null hypothesis_, combine the top 12 SSM mutations at depths of 2 and 3 mutations. Each of the 12 mutations are explored evenely over the plate. The top performing variant according to an additive model of 3 mutations is included.
# 
# __Plate 2__: _Exploitation_ A supervised ML model is trained on the SSM data, using ESM2 embeddings. Under the assumption that the ESM2 embeddings capture global context, learnings from the SSM data provide __better than random__ (but do not promise much better than that...) predictions on higher order mutants. Two and three order mutants are selected greedily according to model predictions, enforcing that no single mutation is seen more than four times. Top 12 mutations explored additively in plate 1 are only allowed to be used once, and only alone in a variant.
# 
# __Plate 3__: _Exploration_, the plate is chosen according to the Upper Confidence Bound of the model, which incorporates approximate model uncertainty, forcing exploration of the protein landscape that the model has few information on. Mutation count is enforced as in plate 2. 

# ## 0. Imports and tools

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
import scipy.stats as stats

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
from sklearn.metrics import make_scorer, r2_score
import joblib

# aide predict
import aide_predict as ap
from aide_predict.utils.plotting import plot_mutation_heatmap, plot_protein_sequence_heatmap

# data handling
import datasets

# typing
from typing import List, Dict, Tuple, Union


# In[61]:


# globals
POSITION_OFFSET = 10
MAX_MUTATION_COUNT = 4

MUTATION_COUNTS_OBSERVED = {}


# In[3]:


def add_observed_mutations(mutations: List[str]):
    for mutation in mutations:
        if mutation in MUTATION_COUNTS_OBSERVED:
            MUTATION_COUNTS_OBSERVED[mutation] += 1
        else:
            MUTATION_COUNTS_OBSERVED[mutation] = 1


# In[4]:


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


# In[5]:


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


# In[6]:


# hashing protein sequences
def protein_hash(protein: str) -> str:
    return hashlib.md5(str(protein).encode()).hexdigest()


# In[7]:


# create a model class 
model_no_embedder = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.98)),
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


# In[8]:


def predict_ensemble(model, X):
    """Take average and std of predictions from all estimators in the ensemble."""
    all_predictions = []
    X_transformed = model[:-1].transform(X)
    for i, estimator in enumerate(model[-1].estimators_):
        feature_indexes = model[-1].estimators_features_[i]
        model_input = X_transformed[:, feature_indexes]
        y_pred_ = estimator.predict(model_input)
        all_predictions.append(y_pred_)
    all_predictions = np.array(all_predictions)
    means = np.mean(all_predictions, axis=0)
    stds = np.std(all_predictions, axis=0)
    return means, stds


# ## 1. Data preparation and initial analysis

# In[9]:


alignment = ap.ProteinSequences.from_fasta('../data/msa/mda_hacl.a2m')
wt = alignment[0]


# ### 1.1 Load and preprocess SSM data
# 
# Convert the excel into a pandas dataframe, rename columns, parse positions, melt what is originally a wide format.
# 
# NOTE: The positions in the excel are forward 10 residues because of His tag and 1 indexed.

# In[10]:


ssm_data = pd.read_excel('../data/experimental/ssm_compiled.xlsx', header=1)


# In[11]:


# remove column 0, 2 and row 1
ssm_data = ssm_data.drop(ssm_data.columns[0], axis=1)
ssm_data = ssm_data.drop(ssm_data.columns[1], axis=1)
ssm_data = ssm_data.drop(ssm_data.index[0])
# format long with columns [mutation_string, original, position, mutation, activity]
ssm_data.rename(columns={'Unnamed: 1': 'original_position_string'}, inplace=True)


# In[12]:


# parse positions/mutations
ssm_data['original_aa'] = ssm_data['original_position_string'].str[0]
ssm_data['position'] = ssm_data['original_position_string'].str[1:].astype(int)
ssm_data.drop(columns=['original_position_string'], inplace=True)
ssm_data = ssm_data.melt(id_vars=['original_aa', 'position'], var_name='mutation', value_name='activity').sort_values(by='position').reset_index(drop=True)


# In[13]:


# compute the position on our sequence, 0 indexed
ssm_data['position_on_wt_0_indexed'] = ssm_data['position'] - POSITION_OFFSET - 1
for _, row in ssm_data.iterrows():
    assert wt[row['position_on_wt_0_indexed']] == row['original_aa']


# In[14]:


# parse the mutation string
ssm_data['mutation_string'] = ssm_data['original_aa'] + ssm_data['position'].astype(str) + ssm_data['mutation']


# In[15]:


# get the variant and the hash, save to file
ssm_data['variant'] = ssm_data.apply(lambda x: mutate_wt(wt, x['mutation_string']), axis=1)
ssm_data['hash'] = ssm_data['variant'].apply(protein_hash)


# In[16]:


ssm_data.dropna().to_csv('../data/experimental/ssm_data.csv', index=False)


# In[17]:


ssm_data


# ### 1.2 Visualize SSM data

# #### 1.2.1 Statistical significance of hits
# 
# Determine how many and which single mutations are statistically significant based on the control values.

# In[18]:


# mark control mutations
ssm_data['control'] = ssm_data['original_aa'] == ssm_data['mutation']
sns.kdeplot(ssm_data[ssm_data['control']]['activity'], label='control')


# In[19]:


control_data = ssm_data[ssm_data['control']]
# drop failed control
control_data = control_data[control_data['activity'] > 0.01]
# stats
control_mean, control_std = control_data['activity'].mean(), control_data['activity'].std()
ssm_data['significant'] = np.abs((ssm_data['activity'] - control_mean) / control_std) > 2
print('Frac significant: ', ssm_data['significant'].mean())


# #### 1.2.2 SSM heatmap

# In[20]:


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


# In[21]:


fig, ax, matrix = plot_mutation_heatmap(mutations = ssm_data['mutation_string'], scores=ssm_data['activity'], all_positions=False, wt_activity=control_mean, zero_minimum=True, log_scale_upper=True)


# In[22]:


# highlight top 12 boxes
top_12 = ssm_data.sort_values(by='activity', ascending=False).head(12)
for _, row in top_12.iterrows():
    pos, to_aa = row['position'], row['mutation'][-1]
    row_pos = matrix.index.get_loc(pos)
    col_pos = matrix.columns.get_loc(to_aa)
    ax.add_patch(plt.Rectangle((col_pos, row_pos), 1, 1, fill=False, edgecolor='red', lw=2))


# In[23]:


fig.savefig('../figures/round1/ssm_heatmap.png', dpi=300)
fig


# In[24]:


putative_mutatable_positions = top_12['position'].unique()
print(','.join(putative_mutatable_positions.astype(str)))


# In[25]:


putative_mutatable_positions_wt_offset_zero_index = putative_mutatable_positions - POSITION_OFFSET - 1


# ### 1.3 Opportunity for Epistasis
# 
# Two methods: compute evolutionary couplings of the protein, check tcouplings between chosen sites based on top 12 SSM mutants to the overall pairwise couplings. We should expect that the couplings across our 10 sites are not on the low end compared to bulk.
# 
# Second method: look at the structure and see if some are proximal. This is qualitative. We want to see multiple nearby residue interactions. 

# #### 1.3.1 Evolutionary couplings

# In[42]:


evc = ap.EVMutationWrapper(wt=wt, metadata_folder='../evcouplings')
evc.fit(alignment)


# In[45]:


couplings = evc.model_.ecs
target_couplings = couplings[couplings['i'].isin(putative_mutatable_positions_wt_offset_zero_index+1) & couplings['j'].isin(putative_mutatable_positions_wt_offset_zero_index+1)]


# In[46]:


fig, ax = plt.subplots(figsize=(4, 4))
sns.kdeplot(couplings['cn'], ax=ax, color='grey', log_scale=True, label='All Couplings')
sns.kdeplot(target_couplings['cn'], ax=ax, color='red', log_scale=True, label='Target Couplings')


# In[47]:


couplings['is_target'] = couplings.index.isin(target_couplings.index)
target_cn_rankings = 1 - np.where(couplings['is_target'])[0]/len(couplings)
print('Mean CN ranks for target couplings:', target_cn_rankings.mean())
print('quartiles:', np.quantile(target_cn_rankings, [0.25, 0.5, 0.75]))


# > The couplings for our target sites are saturated with high coupling. Eg. 7 pairwise interactions are 90%ile couplings over whole MSA couplings. The 25%ile of target couplings is > 56% of bulk couplings. The structure also indicates that the target sites are in close proximity to each other, more opportunity for epistasis.

# #### 1.3.2 Structure

# In[ ]:





# ## 2. Plate 1: Additive Variant Selection (null hypothesis, no epistasis)
# 
# For the top 12 mutations, over 10 sites:
# 1. 9 Variants of top additive mutations
# 2. Remainder of the plates 2-3 mutations chosen from the top 12 mutations, each mutation is sampled equally over the plate.

# In[26]:


# drop the ssm data down to only the positions we care about eg. the top 12 mutations
top_12_ssm_data = ssm_data.sort_values(by='activity', ascending=False).head(12).reset_index(drop=True)


# ### 2.1 Generate additive variants

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
    score = (top_12_ssm_data.loc[list(top12_indexes), 'activity'].values - 1.0).sum() + 1.0
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


# create OHE of top 12
for top12_index in top_12_ssm_data.index:
    additive_df[f'has_{top12_index}'] = additive_df['top12_index_combinations'].apply(lambda x: top12_index in x)


# ### 2.2 Select top variants for plate 1

# #### 2.2.1 Top additive variants of size 1-10
# Here the same mutations will be sampled many times, not balanced.

# In[32]:


top_12_no_repeat_positions = top_12_ssm_data.drop_duplicates(subset='position')


# In[33]:


top_additive = []
for i in range(2,10):
    mutations = top_12_no_repeat_positions['mutation_string'].iloc[:i].tolist()
    activities = top_12_no_repeat_positions['activity'].iloc[:i].values - 1.0
    variant = mutate_wt(wt, mutations)
    hash_ = protein_hash(variant)
    top_additive.append({
        'mutations': mutations,
        'additive_activity': 1+ sum(activities),
        'variant': variant,
        'hash': hash_,
    })
top_additive = pd.DataFrame(top_additive)
top_additive['type'] = 'TopAdditive'


# #### 2.2.2 Sample from top 12 mutations max order 2-3
# 
# Here we sample mutations in order to maximize a representation score of individual mutants

# In[34]:


# remove from additive_df any rows that are the same as top_additive
def check_row(row):
    top12_indexes = row['top12_index_combinations']
    mutations = top_12_ssm_data.loc[list(top12_indexes), 'mutation_string']
    for top_mutations in top_additive['mutations']:
        if set(mutations) == set(top_mutations):
            return False
    return True
additive_df = additive_df[additive_df.apply(check_row, axis=1)]


# In[35]:


# starting with 0 representation of mutants
# add representation from top additive variants
current_mutation_representative_count = np.zeros(12)
for mutations in top_additive['mutations']:
    for mutation in mutations:
        index = top_12_ssm_data[top_12_ssm_data['mutation_string'] == mutation].index[0]
        current_mutation_representative_count[index] += 1


# In[36]:


additive_df


# In[37]:


def get_representation_score(top_12_indexes):
    """Get a score that is how underrepresented the variant is"""
    # compute inverse fractions from the counts
    if sum(current_mutation_representative_count) == 0:
        return 1.0
    else:
        inverse_ratios = 1 - current_mutation_representative_count/sum(current_mutation_representative_count)

        return sum([inverse_ratios[ind] for ind in top_12_indexes])/len(top_12_indexes)

selected_additive_indexes = []

two_or_three_bool_switch = False
while len(selected_additive_indexes) < (96 - len(top_additive)):

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


# In[38]:


current_mutation_representative_count


# In[39]:


# df of the combinations of mutants to chose
selected_additive = additive_df.loc[selected_additive_indexes]


# In[40]:


# parse mutation strings and get variant scores
# format into dataframe as expected for output
def get_variant_metadata_from_top12(row):
    top12_indexes = row['top12_index_combinations']

    mutations = []
    for ind in top12_indexes:
        top12_row = top_12_ssm_data.loc[ind]
        mutation_string = top12_row['mutation_string']
        mutations.append(mutation_string)
    
    variant = mutate_wt(wt, mutations)
    hash_ = protein_hash(variant)
    activity = row['additive_activity']
    return {
        'mutations': mutations,
        'additive_activity': activity,
        'variant': variant,
        'hash': hash_,
    }

plate1_df = pd.DataFrame(list(selected_additive.apply(get_variant_metadata_from_top12, axis=1)))
plate1_df['type'] = 'Additive'


# In[41]:


plate1_df = pd.concat([top_additive, plate1_df], ignore_index=True)


# In[42]:


plate1_df['mutations'] = plate1_df['mutations'].apply(lambda x: '|'.join(x))


# In[43]:


plate1_df.to_csv('../data/round1/designed_plates/plate1_additive.csv', index=False)


# ## 3. ML development
# 
# Check whether supervised models can predict the SSM data, optimize them for accuracy and error calibration.

# ### 3.1 Analysis of ESM embeddings - variance, PCA
# 
# Embed SSM data with ESM, check for signal between components and acitivity.

# In[42]:


ssm_ml_df = ssm_data.copy().sample(frac=1.0)
ssm_ml_df = ssm_ml_df[['activity', 'mutation_string', 'variant', 'hash']]

# group by hash and average (duplicate controls)
ssm_ml_df = ssm_ml_df.groupby('variant').mean().reset_index().dropna()


# In[43]:


X_seq = ap.ProteinSequences.from_list(ssm_ml_df['variant'].tolist())
y = ssm_ml_df['activity'].values


# #### 3.1.1 Embed sequences

# In[44]:


esm_full = ap.ESM2Embedding(
    model_checkpoint='esm2_t33_650M_UR50D',
    metadata_folder='../archive/esm2_embedder',
    pool=True,
    device='mps'
)
esm_full.fit([])


# In[45]:


X_embeddings = esm_full.transform(X_seq)


# #### 3.1.2 PCA and variance

# In[46]:


scaler = StandardScaler()
pca = PCA(n_components=0.98)


# In[47]:


X = scaler.fit_transform(X_embeddings)
X = pca.fit_transform(X)


# In[48]:


pca.explained_variance_ratio_


# In[49]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, ax=ax, palette='viridis', alpha=0.8)


# In[50]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.scatterplot(x=X[:, 2], y=X[:, 3], hue=y, ax=ax, palette='viridis', alpha=0.8)


# ### 3.2 Model training and optimization

# #### 3.2.1 Check for signal on a train test split

# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)


# In[52]:


oracle = model_no_embedder.fit(X_train, y_train)
oracle.score(X_test, y_test)


# R2 of 0.5 is not too bad. Hopefully we can increase this with hyperparameter optimization. How do predictions look?

# In[53]:


pred_mean, pred_std = predict_ensemble(oracle, X_test)


# In[58]:


fig, ax = plt.subplots(figsize=(8, 8))
min_min = min(y_test.min(), pred_mean.min())
max_max = max(y_test.max(), pred_mean.max())
ax.plot([min_min, max_max], [min_min, max_max], color='black', linestyle='--')

ax.errorbar(y_test, pred_mean, yerr=pred_std, fmt='o', alpha=0.5)


# #### 3.2.2 Hyperparameter optimization
# 
# Create best possible predictor. Random search is used.
# 
# Optimize over R2
# 
# Eg. error is penalized less when the model is less certain.

# In[59]:


# scores that balance model uncertainty and accuracy
def uncertainty_nmse(y_true, y_pred, y_std):
    """
    Compute the normalized mean squared error of the predictions.

    
    """
    squared_error = (y_true - y_pred)**2
    normalized_squared_error = squared_error / (y_std**2 + 1e-8)  # Add small epsilon to avoid division by zero
    return np.mean(normalized_squared_error)


# In[56]:


# function to score the model with uncertainty calibrated score
def evaluate_model(model, X, y):
    y_pred, y_std = predict_ensemble(model, X)
    nmse = uncertainty_nmse(y, y_pred, y_std)
    r2 = r2_score(y, y_pred)
    return nmse, r2

def custom_scorer(model, X, y, **kwargs):
    nmse, r2 = evaluate_model(model, X, y)
    return r2


# In[57]:


param_space = {
    'regressor__max_samples': stats.uniform(0.5, 0.5),
    'regressor__max_features': stats.uniform(0.5, 0.5),
    'regressor__estimator__regressor__alpha': stats.loguniform(0.0001, 0.1),
    'regressor__estimator__regressor__learning_rate_init': stats.loguniform(0.0001, 0.1),
    'regressor__estimator__regressor__hidden_layer_sizes': [(50, 50), (50, 50, 50), (50, 50, 50, 50)],
    'regressor__estimator__regressor__activation': ['relu', 'tanh']
}


# In[62]:


search = RandomizedSearchCV(
    model_no_embedder,
    param_space,
    n_iter=500,
    scoring=custom_scorer,
    cv=5,
    n_jobs=16,
    random_state=42,
    verbose=2
)


# In[63]:


search.fit(X_embeddings, y)


# In[64]:


sns.histplot(search.cv_results_['mean_test_score'])
plt.title('R2/NMSE Distribution over Random Search trials')


# In[65]:


# check the CV predictions of the best model
# worst_params = search.cv_results_['params'][np.argmin(search.cv_results_['mean_test_score'])]
# best_model = model_no_embedder.set_params(**worst_params)
best_model = model_no_embedder.set_params(**search.best_params_)
fold_predictions = []
fold_targets = []
fold_stds = []

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kfold.split(X_embeddings):
    X_train, X_test = X_embeddings[train_index], X_embeddings[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = best_model.fit(X_train, y_train)
    preds, stds = predict_ensemble(model, X_test)

    fold_predictions.append(preds)
    fold_targets.append(y_test)
    fold_stds.append(stds)

# combine all predictions and plot
fold_predictions = np.concatenate(fold_predictions)
fold_targets = np.concatenate(fold_targets)
fold_stds = np.concatenate(fold_stds)
fig, ax = plt.subplots(figsize=(8, 8))
min_min = min(fold_targets.min(), fold_predictions.min())
max_max = max(fold_targets.max(), fold_predictions.max())
ax.plot([min_min, max_max], [min_min, max_max], color='black', linestyle='--')
ax.errorbar(fold_targets, fold_predictions, yerr=fold_stds, fmt='o', alpha=0.5)
ax.set_xlabel('True Activity [fold WT]')
ax.set_ylabel('Predicted Activity [fold WT]')


# In[66]:


# use kdeplot because of too many points
r2 = r2_score(fold_targets, fold_predictions)
fig, ax = plt.subplots(figsize=(8, 8))
sns.kdeplot(x=fold_targets, y=fold_predictions, ax=ax, fill=False, palette='Blues', levels=10, bw_adjust=0.5)
ax.plot([min_min, max_max], [min_min, max_max], color='black', linestyle='--')
ax.set_xlabel('True Activity [fold WT]')
ax.set_ylabel('Predicted Activity [fold WT]')
ax.set_title(f'R2: {r2:.3f}')
plt.savefig('../figures/round1/model_parity.png', dpi=300, bbox_inches='tight')


# In[67]:


# let's look at actual error vs model uncertainty
spearman = stats.spearmanr(fold_stds, np.abs(fold_targets - fold_predictions))
fig, ax = plt.subplots(figsize=(8, 8))
sns.regplot(x=fold_stds, y=np.abs(fold_targets - fold_predictions), ax=ax)
ax.set_xlabel('Model Uncertainty')
ax.set_ylabel('Absolute Error')
ax.set_title(f'Spearman: {spearman.correlation:.3f}')
plt.savefig('../figures/round1/model_calibration.png', dpi=300, bbox_inches='tight')


# In[68]:


with open('../data/round1/best_params.json', 'w') as f:
    json.dump(search.best_params_, f)


# In[69]:


search.best_params_


# In[70]:


predict_df = pd.DataFrame({
    'predictions': fold_predictions,
    'targets': fold_targets})


# In[71]:


predict_df['is deleterious'] = predict_df['targets'] < 0.9
(predict_df['predictions'][predict_df['is deleterious']]>1.0).mean()


# In[72]:


predict_df['is beneficial'] = predict_df['targets'] > 1.1
(predict_df['predictions'][predict_df['is beneficial']]<0.5).mean()


# In[73]:


# kendal tau
stats.kendalltau(fold_targets, fold_predictions)


# #### 3.2.3 Compare ESM sizes for speed performance
# 
# This assumes that the hyperopt from the first model size applies to other sizes.

# In[175]:


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
        metadata_folder='../archive/esm2_embedder',
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

        oracle = model_no_embedder.set_params(**search.best_params_)
        oracle.fit(X_train, y_train)
        score = oracle.score(X_test, y_test)
        scores.append(score)
    model_size_scores_dict[model_size] = np.mean(scores), np.std(scores)


# In[186]:


model_size_scores_dict


# In[191]:


fig, ax = plt.subplots(figsize=(8, 8))
scores = np.array([v[0] for v in model_size_scores_dict.values()])
errors = np.array([v[1] for v in model_size_scores_dict.values()])
sizes = np.array(list(model_size_scores_dict.keys()))
ax.bar(x=sizes, height=scores, yerr=errors, capsize=5)


# Model size is near negligable, save maybe the smallest. Go with 35M param model.

# ### 3.3 Embedding all variants
# 
# Create datasets of schema:
# - `variant`: the variant sequence
# - `hash`: the hash of the variant
# - `embedding`: the embedding of the variant
# - `activity`: (optional) the activity of the variant
# - `mutations`: list of mutations comapared to WT.
# 
# 1. Embed the SSM data and wild type.
# 2. Embed plate 1
# 3. Embed the 2 and 3 order variants over 10 sites.

# In[74]:


embedder = ap.ESM2Embedding(
    model_checkpoint='esm2_t12_35M_UR50D',
    metadata_folder='../archive/esm2_embedder',
    pool=True,
    device='mps'
)
embedder.fit([])


# #### 3.3.1 Embed SSM, plate 1 into HF datasets

# In[75]:


# add hash, mutations strings back in
ssm_dataset = ssm_ml_df.copy()
ssm_dataset['hash'] = ssm_dataset['variant'].apply(protein_hash)
ssm_dataset['mutations'] = ssm_dataset['variant'].apply(lambda x: get_mutations(wt, x))


# In[76]:


ssm_seqs = ap.ProteinSequences.from_list(ssm_dataset['variant'].tolist())
X_ssm = embedder.transform(ssm_seqs)


# In[77]:


dataset_dict = {
    'hash': ssm_dataset['hash'].to_list(),
    'variant': ssm_dataset['variant'].to_list(),
    'activity': ssm_dataset['activity'].to_list(),
    'embedding': X_ssm,
    'mutations': ssm_dataset['mutations'].to_list(),
}
ssm_dataset = datasets.Dataset.from_dict(dataset_dict)
del X_ssm


# In[78]:


ssm_dataset.save_to_disk('../hf_datasets/ssm')


# In[79]:


plate1_seqs = ap.ProteinSequences.from_list(plate1_df['variant'].tolist())
X_plate1 = embedder.transform(plate1_seqs)


# In[80]:


dataset_dict = {
    'hash': plate1_df['hash'].to_list(),
    'variant': plate1_df['variant'].to_list(),
    'activity': plate1_df['additive_activity'].to_list(),
    'embedding': X_plate1,
    'mutations': plate1_df['mutations'].to_list(),
    'type': plate1_df['type'].to_list()
}
plate1_dataset = datasets.Dataset.from_dict(dataset_dict)
del X_plate1


# In[81]:


plate1_dataset.save_to_disk('../hf_datasets/plate1')


# #### 3.3.2 Produce all possible 2 and 3 order mutants and embed them
# 
# We define some iterators to iterate over combinations of mutations at 2 and 3 depth. We use these to generate variants and embeddings, skipping those that are already in the SSM data or plate 1. We lastly filter for redundant cariants due to selecting WT AA.

# In[82]:


putative_mutatable_positions_wt_offset_zero_index


# In[83]:


def mutation_combinations(n_positions, n_options, max_order=2):
    """Generate tuples of ints representing mutations at n_positions with n_options options.
    
    Ints are from 0. Map back to actual positions as post processing.
    """

    to_yield = [None for _ in range(n_positions)]

    for positions in itertools.combinations(range(n_positions), max_order):
        for options in itertools.product(range(n_options), repeat=len(positions)):
            for i, pos in enumerate(positions):
                to_yield[pos] = options[i]
            yield tuple(to_yield)
            to_yield = [None for _ in range(n_positions)]


# In[84]:


def sequence_string_from_mutation_combination_iterator(
        wt_sequence, positions_mutating, max_order=2
):
    """Generate a sequence strings by iterating over mutation combinations

    Parameters
    ----------
    wt_sequence : str
        The wild-type sequence
    positions_mutating : list
        List of positions to mutate, 0 indexed
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
            mutation_strings.append(f'{wt_sequence[sequence_index]}{sequence_index+POSITION_OFFSET+1}{AA_LIST[mutation_index]}')
        hash_ = hashlib.md5(''.join(sequence).encode()).hexdigest()
        yield ''.join(sequence), hash_, mutation_strings


# In[85]:


def multiple_mutation_orders_sequence_string_combination_iterator(
        wt_sequence, positions_mutating, max_order=3
):
    """Runs sequence_string_from_mutation_combination_iterator over multiple mutation depths"""
    for order in range(2, max_order+1):
        for sequence, hash_, mutation_strings in sequence_string_from_mutation_combination_iterator(wt_sequence, positions_mutating, order):
            yield sequence, hash_, mutation_strings


# In[87]:


# embed them
block_size = 1000
# get the hashes already in the dataset
existing_hashes = set(ssm_dataset['hash']).union(set(plate1_dataset['hash']))

def dataset_generator():
    """Generate a dataset of all possible 2-3 mutations not already sampled
    """
    
    bar = tqdm()

    block = {
        'mutations': [],
        'hash': [],
        'variant': [],
        'embedding': None
    }
    for i, (sequence, hash_, mutation_strings) in enumerate(multiple_mutation_orders_sequence_string_combination_iterator(wt, putative_mutatable_positions_wt_offset_zero_index, max_order=3)):
        if hash_ in existing_hashes:
            continue

        block['hash'].append(hash_)
        block['mutations'].append(mutation_strings)
        block['variant'].append(sequence)
        if len(block['hash']) == block_size:
            block['embedding'] = embedder.transform(block['variant'])
            
            for i in range(len(block['hash'])):
                example = {
                    'mutations': block['mutations'][i],
                    'hash': block['hash'][i],
                    'variant': block['variant'][i],
                    'embedding': block['embedding'][i]
                }
                yield example

            block = {
                'mutations': [],
                'hash': [],
                'variant': [],
                'embedding': None
            }
            bar.update(block_size)


# In[88]:


variants_dataset = datasets.Dataset.from_generator(dataset_generator)


# In[89]:


variants_dataset.features


# In[90]:


# drop duplicates due to chosing wild type as a variant
# the has column tracks unique variants
# get the first index of each unique variant
hash_set = set()

indexes = []
for index, hash_ in enumerate(variants_dataset['hash']):
    if hash_ in hash_set:
        continue
    hash_set.add(hash_)
    indexes.append(index)
variants_dataset = variants_dataset.select(indexes)


# In[91]:


variants_dataset.save_to_disk('../hf_datasets/variants')


# ## 4. Plate 2: Exploitation

# ### 4.1 Train Final model on all SSM data

# In[44]:


ssm_dataset = datasets.load_from_disk('../hf_datasets/ssm')
ssm_dataset.set_format('numpy', columns=['embedding', 'activity'])
X_ssm = ssm_dataset['embedding']
y_ssm = ssm_dataset['activity']


# In[45]:


with open('../data/round1/best_params.json', 'r') as f:
    best_params = json.load(f)


# In[46]:


final_model = model_no_embedder.set_params(**best_params)
final_model.fit(X_ssm, y_ssm)
final_model.score(X_ssm, y_ssm)


# ### 4.2 Run prediction for all 2 and 3 order mutants

# In[47]:


variants_dataset = datasets.load_from_disk('../hf_datasets/variants')
variants_dataset.set_format('numpy', columns=['embedding'])


# In[48]:


# create a batched HF mapping function
def model_prediction_mapping(examples):
    embeddings = examples['embedding']
    preds, stds = predict_ensemble(final_model, embeddings)
    examples.update({'activity_pred': preds, 'activity_pred_std': stds})
    return examples


# In[49]:


variants_dataset = variants_dataset.map(model_prediction_mapping, batched=True, batch_size=10000)


# ### 4.3 Greedy Selection

# #### 4.3.1 Sort by prediction mean

# In[50]:


# sort variants dataset by predicted activity
sorted_greedy_index = np.array(list(reversed(np.argsort(variants_dataset['activity_pred']))))


# In[51]:


variants_dataset_sorted = variants_dataset.select(sorted_greedy_index)


# In[52]:


variants_dataset_sorted[0]['activity_pred']


# #### 4.3.2 Iteratively add variants greedily, observing mutation counts
# 
# Including mutations from plate 1, we track how many times a particular mutation has been seen and cap it out to avoid saturating the plate with the same mutation over and over. Because of this, the model __must try to make the best variant possible without using the highest performing mutations too many times__.
# 
# Add the following exceptiuon: each top 12 mutant can be used once, in a variant without any other mtop 12 mutations.

# In[62]:


plate1_dataset = datasets.load_from_disk('../hf_datasets/plate1')
plate1_mutations = plate1_dataset['mutations']


# In[63]:


for mutations in plate1_mutations:
    add_observed_mutations(mutations)


# In[64]:


count_top_12_used = {
    k: 0 for k in top_12_ssm_data['mutation_string']
}


# In[65]:


count_top_12_used


# Let's actually do the loop. It cannot be parallelized because we have to check the current set of mutations at each evaluation. The dataset is sorted so we do not need to check predictions.

# In[57]:


variants_dataset_sorted.set_format('numpy', columns=['mutations'])


# In[66]:


selected_greedy_indexes = []

i = 0
bar = tqdm(len(variants_dataset_sorted))
class Completed(Exception): pass
try:
    for i, row in enumerate(variants_dataset_sorted):
        mutations = row['mutations']
        viable = True
        mutation_count_overwrite = False # for top 12 mutations

        # CHECK IF WE HAVE FILLED THE ALLOWABLE TOP 12
        n_top_12_in_variant = 0
        for mutation in mutations:
            if mutation in count_top_12_used:
                n_top_12_in_variant += 1
        if n_top_12_in_variant == 1:
            # here we have a variant with one of the top 12 in it
            # check if the one top 12 in this variant is already selected
            for mutation in mutations:
                if mutation in count_top_12_used:
                    if count_top_12_used[mutation] > 0:
                        viable = False
                        break
            if viable:
                # here we have a variant with one of the top 12 in it
                # and that top 12 hasn't been selected yet
                # for this plate
                mutation_count_overwrite = True
                for mutation in mutations:
                    if mutation in count_top_12_used:
                        count_top_12_used[mutation] += 1

        # check if we already have too many of the same mutation
        # observed in this variant
        if not mutation_count_overwrite:
            for mutation in mutations:
                if MUTATION_COUNTS_OBSERVED.get(mutation, 0) >= MAX_MUTATION_COUNT:
                    viable = False
                    break
        else:
            pass
        
        if viable:
            selected_greedy_indexes.append(i)
            add_observed_mutations(mutations)

        i += 1
        bar.update(1)

        if len(selected_greedy_indexes) == 96:
            raise Completed
            
except Completed:
    pass


# In[67]:


greedy_dataset = variants_dataset_sorted.select(selected_greedy_indexes)
def apply_type(example):
    example['type'] = 'Greedy'
    return example
greedy_dataset = greedy_dataset.map(apply_type)


# In[68]:


greedy_dataset.save_to_disk('../hf_datasets/greedy')


# In[69]:


plate2_df = greedy_dataset.to_pandas()
plate2_df = plate2_df[['hash', 'variant', 'mutations']]
plate2_df['type'] = 'Greedy'
plate2_df['mutations'] = plate2_df['mutations'].apply(lambda x: '|'.join(x))
plate2_df.to_csv('../data/round1/designed_plates/plate2_greedy.csv', index=False)


# In[71]:


count_top_12_used


# In[73]:


plate2_df['count_top_12_used'] = plate2_df['mutations'].apply(lambda x: sum([m in count_top_12_used for m in x.split('|')]))
plate2_df['count_top_12_used'].value_counts()


# ## 5. Plate 3: Exploration

# ### 5.1 UCB selection
# 
# Repeat what was done for Plate 2 but with UCB

# In[74]:


sorted_ucb_index = np.array(list(reversed(np.argsort(variants_dataset['activity_pred'] + 2*variants_dataset['activity_pred_std']))))
variants_dataset_sorted = variants_dataset.select(sorted_ucb_index)


# In[75]:


variants_dataset_sorted.set_format('numpy', columns=['mutations'])


# In[76]:


selected_ucb_indexes = []

i = 0
bar = tqdm(len(variants_dataset_sorted))
try:
    for i, row in enumerate(variants_dataset_sorted):
        mutations = row['mutations']
        viable = True
        # check if we already have too many of the same mutation
        # observed in this variant
        for mutation in mutations:
            if MUTATION_COUNTS_OBSERVED.get(mutation, 0) >= MAX_MUTATION_COUNT:
                viable = False
                break
        if viable:
            selected_ucb_indexes.append(i)
            add_observed_mutations(mutations)

        i += 1
        bar.update(1)

        if len(selected_ucb_indexes) == 96:
            raise Completed

except Completed:
    pass


# In[77]:


ucb_dataset = variants_dataset_sorted.select(selected_ucb_indexes)
def apply_type(example):
    example['type'] = 'UCB'
    return example
ucb_dataset = ucb_dataset.map(apply_type)


# In[78]:


np.max(ucb_dataset['activity_pred_std'])


# In[79]:


ucb_dataset.save_to_disk('../hf_datasets/ucb')


# In[80]:


ucb_df = ucb_dataset.to_pandas()
ucb_df = ucb_df[['hash', 'variant', 'mutations']]
ucb_df['type'] = 'UCB'


# In[81]:


ucb_df['mutations'] = ucb_df['mutations'].apply(lambda x: '|'.join(x))


# In[82]:


ucb_df.to_csv('../data/round1/designed_plates/plate3_ucb.csv', index=False)


# In[83]:


len(ucb_df)


# In[84]:


del variants_dataset_sorted


# ## 6. Combine plates into a uniform format

# ### 6.1 Load previous data

# In[184]:


plate1_df = pd.read_csv('../data/round1/designed_plates/plate1_additive.csv')
plate2_df = pd.read_csv('../data/round1/designed_plates/plate2_greedy.csv')
plate3_df = pd.read_csv('../data/round1/designed_plates/plate3_ucb.csv')


# In[185]:


plate1_df['plate'] = 1
plate2_df['plate'] = 2
plate3_df['plate'] = 3
combined_df = pd.concat([plate1_df, plate2_df, plate3_df], ignore_index=True)


# ### 6.2 Add columns

# In[186]:


# parse mutations back into a list
combined_df['mutations_string'] = combined_df['mutations']
combined_df['mutations'] = combined_df['mutations'].apply(lambda x: x.split('|'))


# In[187]:


combined_df['id'] = combined_df.apply(lambda x: f'{x["type"]}_{x["mutations_string"]}', axis=1)


# In[188]:


combined_df['has_V83'] = combined_df['mutations_string'].apply(lambda x: 'V83' in x)


# In[189]:


combined_df['has_V83G'] = combined_df['mutations_string'].apply(lambda x: 'V83G' in x)


# In[190]:


# get the model predictions
X_seq = ap.ProteinSequences.from_list(combined_df['variant'].tolist())

embedder = ap.ESM2Embedding(
    model_checkpoint='esm2_t12_35M_UR50D',
    metadata_folder='../archive/esm2_embedder',
    pool=True,
    device='mps'
)
embedder.fit([])
X_embeddings = embedder.transform(X_seq)
y_pred, y_std = predict_ensemble(final_model, X_embeddings)

combined_df['activity_pred'] = y_pred
combined_df['activity_pred_std'] = y_std


# In[191]:


# also get additive activity for Plate 2 and 3
ssm_data = pd.read_csv('../data/experimental/ssm_data.csv').set_index('mutation_string')['activity']
def get_additive_score(mutations):
    activities = []
    for mutation in mutations:
        try:
            activities.append(ssm_data[mutation] - 1.0)
        except KeyError:
            return np.nan
    return sum(activities) + 1.0


# In[192]:


combined_df['additive_activity'] = combined_df.apply(lambda x: get_additive_score(x['mutations']), axis=1)


# Count the number of top 12 mutants in the variant, add a column for the plate number.

# In[193]:


ssm_data = pd.read_csv('../data/experimental/ssm_data.csv').sort_values(by='activity', ascending=False).reset_index(drop=True).head(12)
ssm_data


# In[194]:


combined_df['n_mutations'] = combined_df['mutations'].apply(len)


# In[196]:


combined_df['n_top_12'] = combined_df['mutations'].apply(lambda x: sum([m in ssm_data['mutation_string'].values for m in x]))


# In[ ]:


# do a final check that no mutations occur outside the expected positions
wt_array = np.array(list(wt))
for variant in combined_df['variant']:
    variant_array = np.array(list(variant))
    variable_positions = np.where(variant_array != wt_array)[0]
    if not np.all(np.isin(variable_positions, putative_mutatable_positions_wt_offset_zero_index)):
        print('Mutation outside expected positions!')
        print(variant)
        print(variable_positions)
        print(putative_mutatable_positions_wt_offset_zero_index)


# In[233]:


# lowercase all but mutated residues
combined_df['variant'] = combined_df.apply(lambda x: ''.join([aa.lower() if i not in putative_mutatable_positions_wt_offset_zero_index else aa for i, aa in enumerate(x['variant'])]), axis=1)


# In[237]:


# order columns to make sense
combined_df = combined_df[['id', 'plate', 'type', 'mutations_string', 'has_V83', 'has_V83G', 'n_mutations', 'n_top_12', 
                           'variant', 'activity_pred', 'activity_pred_std', 'additive_activity']]
combined_df = combined_df.sort_values(by=['plate', 'type'])
combined_df.to_csv('../data/round1/all_variants.csv', index=False)


# In[238]:


combined_df['n_mutations'].value_counts()


# In[202]:





# ## 7. Analyze chosen data

# ### 7.1 Visualize data space explored
# 
# Look at PC's of embeddings vs types of data chosen

# #### 7.1.1 First embed full data space

# In[203]:


varaints_dataset = datasets.load_from_disk('../hf_datasets/variants')
ssm_dataset = datasets.load_from_disk('../hf_datasets/ssm')
combined_dataset = datasets.concatenate_datasets([varaints_dataset, ssm_dataset])
combined_dataset.set_format('numpy', columns=['embedding', 'activity'])

X = combined_dataset['embedding']
pca = PCA(n_components=0.95)
scaler = StandardScaler()
pca_pipe = Pipeline([
    ('scaler', scaler),
    ('pca', pca)
])
X_pca = pca_pipe.fit_transform(X)

joblib.dump(pca_pipe, '../data/intermediate/all_data_pca.pkl') 
joblib.dump(X_pca, '../data/intermediate/all_data_pca_embeddings.pkl')


# #### 7.1.2 Plot embedding of selected variants

# In[204]:


ssm_dataset = datasets.load_from_disk('../hf_datasets/ssm')
plate1_dataset = datasets.load_from_disk('../hf_datasets/plate1')
plate2_dataset = datasets.load_from_disk('../hf_datasets/greedy')
plate3_dataset = datasets.load_from_disk('../hf_datasets/ucb')


# In[205]:


def apply_type(example):
    example['type'] = 'SSM'
    return example
ssm_dataset = ssm_dataset.map(apply_type)


# In[206]:


combined_dataset = datasets.concatenate_datasets([ssm_dataset, plate1_dataset, plate2_dataset, plate3_dataset])
combined_dataset.set_format('numpy', columns=['embedding']) 


# In[207]:


chosen_pca = pca_pipe.transform(combined_dataset['embedding'])


# In[208]:


# select a sample of all PCS
all_pcs_ = X_pca[np.random.choice(X_pca.shape[0], 10000, replace=False)]


# In[209]:


chosen_df = pd.DataFrame(
    {'type': combined_dataset['type']}
)
for i in range(5):
    chosen_df[f'PC{i+1}'] = chosen_pca[:, i]


# In[210]:


fig, ax = plt.subplots(1,2, figsize=(12, 6))
sns.scatterplot(x='PC1', y='PC2', hue='type', data=chosen_df, ax=ax[0])
sns.scatterplot(x='PC3', y='PC4', hue='type', data=chosen_df, ax=ax[1])
sns.kdeplot(x=all_pcs_[:, 0], y=all_pcs_[:, 1], ax=ax[0], fill=False, levels=10, bw_adjust=0.5, color='black')
sns.kdeplot(x=all_pcs_[:, 2], y=all_pcs_[:, 3], ax=ax[1], fill=False, levels=10, bw_adjust=0.5, color='black')
plt.savefig('../figures/round1/pca_design_space.png', dpi=300, bbox_inches='tight')


# ### 7.2 View predicted activity vs additive activity for plate 1

# In[211]:


combined_df = pd.read_csv('../data/round1/all_variants.csv')


# In[212]:


plate1_df = combined_df[combined_df['plate'] == 1]


# In[213]:


g = sns.jointplot(data=plate1_df, x='activity_pred', y='additive_activity', kind='kde', bw_adjust=0.7)
ax = g.ax_joint
# plot the diagonal
min_min = min(plate1_df['activity_pred'].min(), plate1_df['additive_activity'].min())
max_max = max(plate1_df['activity_pred'].max(), plate1_df['additive_activity'].max())
ax.plot([min_min, max_max], [min_min, max_max], color='black', linestyle='--')
ax.set_xlabel('Predicted Activity')
ax.set_ylabel('Additive Activity')
plt.savefig('../figures/round1/plate1_pred_vs_additive.png', dpi=300, bbox_inches='tight')


# ### 7.3 Predicted activity for each plate

# In[214]:


from matplotlib.lines import Line2D


# In[215]:


# add a plates worth of random mutations at order 3
AA_LIST = list('ACDEFGHIKLMNPQRSTVWY')
random_variants = []
for i in range(96):
    positions = np.random.choice(putative_mutatable_positions_wt_offset_zero_index, 3, replace=False)
    mutations = []
    for position in positions:
        mutation = f'{wt[position]}{position}{np.random.choice(AA_LIST)}'
        mutations.append(mutation)
    variant = mutate_wt(wt, mutations, one_indexed=False, offset=0)
    random_variants.append(variant)


# In[216]:


random_embeddings = embedder.transform(ap.ProteinSequences.from_list(random_variants))


# In[217]:


random_predictions = predict_ensemble(final_model, random_embeddings)


# In[218]:


combined_df_ = combined_df.copy()
# add random variants
random_df = pd.DataFrame({
    'type': 'Random',
    'variant': random_variants,
    'activity_pred': random_predictions[0],
    'activity_pred_std': random_predictions[1],
    'plate': 'Random'
})
combined_df_ = pd.concat([combined_df_, random_df], ignore_index=True)


# In[219]:


g = sns.histplot(
    data=combined_df_, x='activity_pred', hue='plate',
    common_bins=True, common_norm=False, multiple='stack',
    stat='density', palette=sns.color_palette()[:4])
ax = plt.gca()
sns.kdeplot(x=ssm_dataset['activity'], color='black', label='SSM', fill=False, bw_adjust=0.7, ax=ax, )
ax.set_xlabel('Predicted Activity')
ax.legend(title='Type', handles=[
    Line2D([0], [0], color=sns.color_palette()[0], lw=4),
    Line2D([0], [0], color=sns.color_palette()[1], lw=4),
    Line2D([0], [0], color=sns.color_palette()[2], lw=4),
    Line2D([0], [0], color=sns.color_palette()[3], lw=4),
    Line2D([0], [0], color='black', lw=1)
], labels=['Plate 1', 'Plate 2', 'Plate 3', 'Random mutations', 'SSM Measured activity'])
plt.savefig('../figures/round1/plate_activity_distributions.png', dpi=300, bbox_inches='tight')


# ####  7.3.1 Compute some statistics on likely performance of variants according to model predictions

# In[220]:


combined_df_['norm_dist'] = combined_df_.apply(lambda x: stats.norm(x['activity_pred'], x['activity_pred_std']), axis=1)


# In[221]:


combined_df_['gt_50pc_better_than_wt'] = combined_df_['norm_dist'].apply(lambda x: x.cdf(1.0) < 0.5)


# In[222]:


combined_df_['gt_95pc_better_than_0.0'] = combined_df_['norm_dist'].apply(lambda x: x.cdf(0.0) < 0.05)


# In[223]:


combined_df_['p_gt_wt'] = combined_df_['norm_dist'].apply(lambda x: 1 - x.cdf(1.0))
combined_df_['p_gt_0.0'] = combined_df_['norm_dist'].apply(lambda x: 1 - x.cdf(0.0))


# In[224]:


x = np.linspace(-5, 5, 1000)
colors = dict(zip(combined_df_['plate'].unique(), sns.color_palette()[:4]))
for i, row in combined_df_.iterrows():
    c = colors[row['plate']]
    plt.plot(x, row['norm_dist'].pdf(x), alpha=0.2, color=c)
plt.legend(title='Plate', handles=[
    Line2D([0], [0], color=sns.color_palette()[0], lw=4),
    Line2D([0], [0], color=sns.color_palette()[1], lw=4),
    Line2D([0], [0], color=sns.color_palette()[2], lw=4),
    Line2D([0], [0], color=sns.color_palette()[3], lw=4),
], labels=['Plate 1', 'Plate 2', 'Plate 3', 'Random mutations'])
ax.set_xlabel('Predicted Activity')
ax.set_ylabel('Density')
plt.savefig('../figures/round1/plate_activity_distributions_pdf.png', dpi=300, bbox_inches='tight')


# In[225]:


combined_df_['gt_95pc_better_than_0.0'].groupby(combined_df_['plate']).mean()


# In[226]:


combined_df_['gt_50pc_better_than_wt'].groupby(combined_df_['plate']).mean()


# In[227]:


combined_df_['p_gt_wt'].groupby(combined_df_['plate']).mean()


# In[228]:


combined_df_['p_gt_0.0'].groupby(combined_df_['plate']).mean()


# ### 7.4 Mutation space explored

# In[229]:


combined_df['mutations'] = combined_df['mutations_string'].apply(lambda x: x.split('|'))


# In[230]:


mutations_tested_count = {}
for mutations in combined_df['mutations']:
    for mutation in mutations:
        mutations_tested_count[mutation] = mutations_tested_count.get(mutation, 0) + 1


# In[231]:


mutations_tested_count


# In[232]:


fig, _, _ = plot_mutation_heatmap(
    list(mutations_tested_count.keys()),
    list(mutations_tested_count.values()),
    zero_minimum=True,
    all_positions=False
)
axes = fig.get_axes()
axes[0].set_title('Mutations counts tested in 2-3 order combinatorial variants')
plt.savefig('../figures/round1/mutation_counts.png', dpi=300, bbox_inches='tight')


# In[ ]:





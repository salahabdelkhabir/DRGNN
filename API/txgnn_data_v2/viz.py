# Comprehensive path generation for top 200 drugs per disease
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import multiprocessing
import heapq

# Load initial data
with open(r'D:\Downloads\Drug_Explorer\txgnn_data_v2\full_graph_split1_eval.pkl', 'rb') as f:
    res = pd.compat.pickle_compat.load(f)
result = pd.DataFrame(res['result'])

# Load graph data
with open(r'D:\Downloads\Drug_Explorer\txgnn_data_v2\graphmask_output_indication.pkl', 'rb') as f:
    d_gm = pd.compat.pickle_compat.load(f)
with open(r'D:\Downloads\Drug_Explorer\txgnn_data_v2\attention_output_indication.pkl', 'rb') as f:
    d_att = pd.compat.pickle_compat.load(f)
with open(r'D:\Downloads\Drug_Explorer\txgnn_data_v2\gnnexplainer_output_indication.pkl', 'rb') as f:
    d_ge = pd.compat.pickle_compat.load(f)

# Preprocessing function
def preprocess(d):
    d = d[~d.y_name.str.contains('CYP')]
    d = d[~d.x_name.str.contains('CYP')]
    d = d.rename(columns={'indication_layer1_att': 'layer1_att', 'indication_layer2_att': 'layer2_att'})
    return d

d_gm = preprocess(d_gm)
d_ge = preprocess(d_ge)
d_att = preprocess(d_att)

# Import required libraries for graph and analysis
import sys
sys.path.append('../')
from txgnn import TxData
txdata = TxData(data_folder_path='../data')
txdata.prepare_split(split='random', seed=1)

# Retrieve mappings
mapping = txdata.retrieve_id_mapping
idx2id_disease = mapping['idx2id_disease']
idx2id_drug = mapping['idx2id_drug']
id2name_disease = mapping['id2name_disease']
id2name_drug = mapping['id2name_drug']

# Graph building and meta-path finding functions
def build_graph(df):
    G = nx.MultiDiGraph
    for _, row in tqdm(df.iterrows):
        G.add_node(row['x_name'], node_type=row['x_type'])
        G.add_node(row['y_name'], node_type=row['y_type'])
        G.add_edge(row['x_name'], row['y_name'], 
                   relation=row['relation'],
                   layer1_att=row['layer1_att'], 
                   layer2_att=row['layer2_att'])
    return G

def sigmoid(x):
    return 1/(1+np.exp(-x))

def calculate_relation_averages(G, layer_only=False):
    relation_sums = {}
    relation_counts = {}
    for u, v, data in tqdm(G.edges(data=True)):
        relation = data['relation']
        weight = data['layer' + str(layer_only) + '_att'] if layer_only else data['layer1_att'] + data['layer2_att']
        
        if relation in relation_sums:
            relation_sums[relation] += weight
            relation_counts[relation] += 1
        else:
            relation_sums[relation] = weight
            relation_counts[relation] = 1

    return {rel: relation_sums[rel] / relation_counts[rel] for rel in relation_sums}

# Build graphs and calculate relation averages
G_dict = {
    'att': build_graph(d_att),
    'gm': build_graph(d_gm),
    'ge': build_graph(d_ge)
}

relation_avg_dict = {
    'att': calculate_relation_averages(G_dict['att']),
    'gm': calculate_relation_averages(G_dict['gm']),
    'ge': calculate_relation_averages(G_dict['ge'])
}

# Path generation functions (using previous implementation)
def get_two_hop_neighborhood_enrichment_per_relation(G, node_id, K, K2, relation_averages, enrichment=True):
    # [Previous implementation remains the same]
    pass

def find_relation_specific_paths(G, start_id, end_id, max_depth=4):
    # [Previous implementation remains the same]
    pass

def score_path_enrichment(G, path, relation_averages, enrichment=True):
    # [Previous implementation remains the same]
    pass

def find_meta_paths(X_id, Y_id, G, not_cool_rel, relation_averages, enrichment=True):
    # [Previous implementation remains the same]
    pass

def get_path(X_id, Y_id, G, not_cool_rel, enrichment, label):
    # [Previous implementation remains the same]
    pass

# Prepare disease-drug pairs for top 200 drugs
disease_drug_pairs = []
for disease in result.Name.unique:
    ranked_list = result[result.Name == disease]['Ranked List'][0]
    truth_for_disease = result[result.Name == disease]['Hits@100'].values[0] + result[result.Name == disease]['Missed@100'].values[0]
    
    difference = np.setdiff1d(ranked_list, truth_for_disease)
    ordered_difference = np.array([item for item in ranked_list if item in difference])
    Y_ids = ordered_difference[:200]
    
    disease_drug_pairs += list(zip([disease] * len(Y_ids), Y_ids, ['Predicted Drugs'] * len(Y_ids)))

# Parallel path generation
def get_path_wrapper(X):
    return get_path(X[0], X[1], G_dict['gm'], 
                    ['rev_contraindication', 'contraindication', 'drug_drug', 
                     'rev_off-label use', 'off-label use', 
                     'anatomy_protein_absent', 'rev_anatomy_protein_absent'], 
                    False, label=X[2])

with multiprocessing.Pool(30) as p:
    r = list(tqdm(p.imap(get_path_wrapper, disease_drug_pairs), total=len(disease_drug_pairs)))

# Combine paths
all_path = pd.DataFrame
for i in r:
    all_path = all_path.append(i)

# Gene parsing and processing
def parse_genes(x):
    idx_gene = []
    for idx, i in enumerate(x['Meta-Path'].split('->')):
        if i.strip == 'gene/protein':
            idx_gene.append(idx)
    return [x['Path'].split('->')[gene].strip for gene in idx_gene]

tqdm.pandas
all_path['genes'] = all_path.progress_apply(lambda x: parse_genes(x), axis=1)

# Gene occurrence calculation
gene_count_per_path = {}
for gene in tqdm(all_path.genes):
    for i in gene:
        if i in gene_count_per_path:
            gene_count_per_path[i] += 1
        else:
            gene_count_per_path[i] = 1

# Add gene occurrence information
all_path['gene occurences across all paths'] = all_path.genes.apply(lambda x: {i: gene_count_per_path[i] for i in x})

# Calculate gene occurrences per disease
gene_count_per_path_per_disease = {}
path_count_per_disease = {}
for disease, gene in tqdm(all_path[['Disease','genes']].values):
    if disease not in gene_count_per_path_per_disease:
        gene_count_per_path_per_disease[disease] = {}
        path_count_per_disease[disease] = 0
    for i in gene:
        if i in gene_count_per_path_per_disease[disease]:
            gene_count_per_path_per_disease[disease][i] += 1
        else:
            gene_count_per_path_per_disease[disease][i] = 1
    path_count_per_disease[disease] += 1

# Add additional columns
all_path['gene occurences for this disease'] = all_path.apply(
    lambda x: {i: gene_count_per_path_per_disease[x.Disease][i] for i in x.genes}, axis=1
)
all_path['number of paths for this disease'] = all_path.Disease.apply(lambda x: path_count_per_disease[x])

# Save to CSV
all_path.to_csv('paths_top200_drugs.csv', index=False)
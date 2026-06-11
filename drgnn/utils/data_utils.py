import math
import numpy as np
import argparse
import copy
import pickle
import os
from collections import Counter
from random import choice
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import requests

try:
    import dgl
    from dgl.ops import edge_softmax
    import dgl.function as fn
except ImportError:
    dgl = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


def get_device():
    if torch is not None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return None


def dataverse_download(url: str, save_path: str) -> None:
    if os.path.exists(save_path):
        print('Found local copy...')
    else:
        print("Local copy not detected... Downloading...")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()


def data_download_wrapper(url: str, save_path: str) -> None:
    if os.path.exists(save_path):
        print('Found local copy...')
    else:
        dataverse_download(url, save_path)
        print("Done!")


def preprocess_kg(path, split, test_size=0.05, one_hop=False, mask_ratio=0.1):
    if split in ['cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland', 'autoimmune',
                  'metabolic_disorder', 'diabetes', 'neurodigenerative']:
        print('Generating disease area using ontology... might take several minutes...')
        name2id = {
            'cell_proliferation': '14566',
            'mental_health': '150',
            'cardiovascular': '1287',
            'anemia': '2355',
            'adrenal_gland': '9553',
            'autoimmune': '417',
            'metabolic_disorder': '655',
            'diabetes': '9351',
            'neurodigenerative': '1289'
        }
        ds = DataSplitter(kg_path=path)
        test_kg = ds.get_test_kg_for_disease(name2id[split], test_size=test_size, one_hop=one_hop, mask_ratio=mask_ratio)
        all_kg = ds.kg
        all_kg['split'] = 'train'
        test_kg['split'] = 'test'
        df = pd.concat([all_kg, test_kg]).drop_duplicates(subset=['x_index', 'y_index'], keep='last').reset_index(
            drop=True)
        print('test size: ', test_size)
        if test_size != 0.05:
            folder_name = split + '_kg_frac' + str(test_size)
        elif one_hop:
            folder_name = split + '_kg' + '_one_hop_ratio' + str(mask_ratio)
        else:
            folder_name = split + '_kg'
        path = os.path.join(path, folder_name)
        if not os.path.exists(path):
            os.mkdir(path)
        print('save kg.csv to ', os.path.join(path, 'kg.csv'))
        df.to_csv(os.path.join(path, 'kg.csv'), index=False)
        df = df[['x_type', 'x_id', 'relation', 'y_type', 'y_id', 'split']]
    else:
        df = pd.read_csv(os.path.join(path, 'kg.csv'))
        df = df[['x_type', 'x_id', 'relation', 'y_type', 'y_id']]
    unique_relation = np.unique(df.relation.values)
    undirected_index = []
    print('Iterating over relations...')
    for i in tqdm(unique_relation):
        if ('_' in i) and (i.split('_')[0] == i.split('_')[1]):
            df_temp = df[df.relation == i]
            df_temp['check_string'] = df_temp.apply(
                lambda row: '_'.join(sorted([str(row['x_id']), str(row['y_id'])])), axis=1)
            undirected_index.append(df_temp.drop_duplicates('check_string').index.values.tolist())
        else:
            d_off = df[df.relation == i]
            undirected_index.append(d_off[d_off.x_type == d_off.x_type.iloc[0]].index.values.tolist())
    flat_list = [item for sublist in undirected_index for item in sublist]
    df = df[df.index.isin(flat_list)]
    unique_node_types = np.unique(np.append(np.unique(df.x_type.values), np.unique(df.y_type.values)))
    df['x_idx'] = np.nan
    df['y_idx'] = np.nan
    df['x_id'] = df.x_id.apply(lambda x: convert2str(x))
    df['y_id'] = df.y_id.apply(lambda x: convert2str(x))
    idx_map = {}
    print('Iterating over node types...')
    for i in tqdm(unique_node_types):
        names = np.unique(np.append(df[df.x_type == i]['x_id'].values, df[df.y_type == i]['y_id'].values))
        names2idx = dict(zip(names, list(range(len(names)))))
        df.loc[df.x_type == i, 'x_idx'] = df[df.x_type == i]['x_id'].apply(lambda x: names2idx[x])
        df.loc[df.y_type == i, 'y_idx'] = df[df.y_type == i]['y_id'].apply(lambda x: names2idx[x])
        idx_map[i] = names2idx
    print('save kg_directed.csv...')
    df.to_csv(os.path.join(path, 'kg_directed.csv'), index=False)


def random_fold(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()
    for i in df.relation.unique():
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac=test_frac, replace=False, random_state=fold_seed)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac=val_frac / (1 - test_frac), replace=False, random_state=1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = df_train._append(train)
        df_valid = df_valid._append(val)
        df_test = df_test._append(test)
    return {'train': df_train.reset_index(drop=True),
            'valid': df_valid.reset_index(drop=True),
            'test': df_test.reset_index(drop=True)}


def disease_eval_fold(df, fold_seed, disease_idx):
    if not isinstance(disease_idx, list):
        disease_idx = np.array([disease_idx])
    else:
        disease_idx = np.array(disease_idx)
    dd_rel_types = ['contraindication', 'indication', 'off-label use']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)]
    unique_diseases = df_dd.y_idx.unique()
    train_diseases = np.setdiff1d(unique_diseases, disease_idx)
    df_dd_train_val = df_dd[df_dd.y_idx.isin(train_diseases)]
    df_dd_test = df_dd[df_dd.y_idx.isin(disease_idx)]
    df_dd_val = df_dd_train_val.sample(frac=0.05, replace=False, random_state=fold_seed)
    df_dd_train = df_dd_train_val[~df_dd_train_val.index.isin(df_dd_val.index)]
    df_train = pd.concat([df_not_dd, df_dd_train])
    df_valid = df_dd_val
    df_test = df_dd_test
    return {'train': df_train.reset_index(drop=True),
            'valid': df_valid.reset_index(drop=True),
            'test': df_test.reset_index(drop=True)}


def complex_disease_fold(df, fold_seed, frac):
    dd_rel_types = ['contraindication', 'indication', 'off-label use']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)]
    unique_diseases = df_dd.y_idx.unique()
    np.random.seed(fold_seed)
    np.random.shuffle(unique_diseases)
    train, valid, test = np.split(unique_diseases,
                                   [int(frac[0] * len(unique_diseases)), int((frac[0] + frac[1]) * len(unique_diseases))])
    df_dd_train = df_dd[df_dd.y_idx.isin(train)]
    df_dd_valid = df_dd[df_dd.y_idx.isin(valid)]
    df_dd_test = df_dd[df_dd.y_idx.isin(test)]
    df = df_not_dd
    train_frac, val_frac, test_frac = frac
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()
    for i in df.relation.unique():
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac=test_frac, replace=False, random_state=fold_seed)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac=val_frac / (1 - test_frac), replace=False, random_state=1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = df_train._append(train)
        df_valid = df_valid._append(val)
        df_test = df_test._append(test)
    df_train = pd.concat([df_train, df_dd_train])
    df_valid = pd.concat([df_valid, df_dd_valid])
    df_test = pd.concat([df_test, df_dd_test])
    return {'train': df_train.reset_index(drop=True),
            'valid': df_valid.reset_index(drop=True),
            'test': df_test.reset_index(drop=True)}


def few_edges_to_kg_fold(df, fold_seed, frac):
    dd_rel_types = ['contraindication', 'indication', 'off-label use']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)]
    disease2num_neighbors_1 = dict(df_not_dd[df_not_dd.x_type == 'disease'].groupby('x_idx').y_id.agg(len))
    disease2num_neighbors_2 = dict(df_not_dd[df_not_dd.y_type == 'disease'].groupby('y_idx').x_id.agg(len))
    disease2num_neighbors = {}
    for key in set(disease2num_neighbors_1).union(disease2num_neighbors_2):
        disease2num_neighbors[key] = disease2num_neighbors_1.get(key, 0) + disease2num_neighbors_2.get(key, 0)
    disease_with_less_than_3_connections_in_kg = np.array([i for i, j in disease2num_neighbors.items() if j <= 3])
    unique_diseases = df_dd.y_idx.unique()
    train_val_diseases = np.setdiff1d(unique_diseases, disease_with_less_than_3_connections_in_kg)
    test = np.intersect1d(unique_diseases, disease_with_less_than_3_connections_in_kg)
    print('Number of testing diseases: ', len(test))
    np.random.seed(fold_seed)
    np.random.shuffle(train_val_diseases)
    train, valid = np.split(train_val_diseases, [int(frac[0] * len(unique_diseases))])
    print('Number of train diseases: ', len(train))
    print('Number of valid diseases: ', len(valid))
    df_dd_train = df_dd[df_dd.y_idx.isin(train)]
    df_dd_valid = df_dd[df_dd.y_idx.isin(valid)]
    df_dd_test = df_dd[df_dd.y_idx.isin(test)]
    df = df_not_dd
    train_frac, val_frac, test_frac = frac
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()
    for i in df.relation.unique():
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac=test_frac, replace=False, random_state=fold_seed)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac=val_frac / (1 - test_frac), replace=False, random_state=1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = df_train._append(train)
        df_valid = df_valid._append(val)
        df_test = df_test._append(test)
    df_train = pd.concat([df_train, df_dd_train])
    df_valid = pd.concat([df_valid, df_dd_valid])
    df_test = pd.concat([df_test, df_dd_test])
    return {'train': df_train.reset_index(drop=True),
            'valid': df_valid.reset_index(drop=True),
            'test': df_test.reset_index(drop=True)}


def few_edges_to_indications_fold(df, fold_seed, frac):
    dd_rel_types = ['contraindication', 'indication', 'off-label use']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)]
    disease2num_indications = dict(
        df_dd[(df_dd.y_type == 'disease') & (df_dd.relation == 'indication')].groupby('x_idx').y_id.agg(len))
    disease_with_less_than_3_indications_in_kg = np.array([i for i, j in disease2num_indications.items() if j <= 3])
    unique_diseases = df_dd.y_idx.unique()
    train_val_diseases = np.setdiff1d(unique_diseases, disease_with_less_than_3_indications_in_kg)
    test = np.intersect1d(unique_diseases, disease_with_less_than_3_indications_in_kg)
    print('Number of testing diseases: ', len(test))
    np.random.seed(fold_seed)
    np.random.shuffle(train_val_diseases)
    train, valid = np.split(train_val_diseases, [int(frac[0] * len(unique_diseases))])
    print('Number of train diseases: ', len(train))
    print('Number of valid diseases: ', len(valid))
    df_dd_train = df_dd[df_dd.y_idx.isin(train)]
    df_dd_valid = df_dd[df_dd.y_idx.isin(valid)]
    df_dd_test = df_dd[df_dd.y_idx.isin(test)]
    df = df_not_dd
    train_frac, val_frac, test_frac = frac
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()
    for i in df.relation.unique():
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac=test_frac, replace=False, random_state=fold_seed)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac=val_frac / (1 - test_frac), replace=False, random_state=1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = df_train._append(train)
        df_valid = df_valid._append(val)
        df_test = df_test._append(test)
    df_train = pd.concat([df_train, df_dd_train])
    df_valid = pd.concat([df_valid, df_dd_valid])
    df_test = pd.concat([df_test, df_dd_test])
    return {'train': df_train.reset_index(drop=True),
            'valid': df_valid.reset_index(drop=True),
            'test': df_test.reset_index(drop=True)}


def create_fold_cv(df, split_num, num_splits):
    dd_rel_types = ['contraindication', 'indication', 'off-label use']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)]
    unique_diseases = df_dd.y_idx.unique()
    np.random.seed(42)
    np.random.shuffle(unique_diseases)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=num_splits)
    split_num_idx = {}
    for i, (train_index, test_index) in enumerate(kf.split(unique_diseases)):
        train_index, valid_index, _ = np.split(train_index,
                                                [int(0.9 * len(train_index)), int(len(train_index))])
        split_num_idx[i + 1] = {'train': unique_diseases[train_index],
                                'valid': unique_diseases[valid_index],
                                'test': unique_diseases[test_index]
                                }
    train, valid, test = split_num_idx[split_num]['train'], split_num_idx[split_num]['valid'], \
    split_num_idx[split_num]['test']
    df_dd_train = df_dd[df_dd.y_idx.isin(train)]
    df_dd_valid = df_dd[df_dd.y_idx.isin(valid)]
    df_dd_test = df_dd[df_dd.y_idx.isin(test)]
    df = df_not_dd
    train_frac, val_frac, test_frac = [0.83125, 0.11875, 0.05]
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()
    for i in df.relation.unique():
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac=test_frac, replace=False, random_state=split_num)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac=val_frac / (1 - test_frac), replace=False, random_state=1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = df_train._append(train)
        df_valid = df_valid._append(val)
        df_test = df_test._append(test)
    df_train = pd.concat([df_train, df_dd_train])
    df_valid = pd.concat([df_valid, df_dd_valid])
    df_test = pd.concat([df_test, df_dd_test])
    return df_train.reset_index(drop=True), df_valid.reset_index(drop=True), df_test.reset_index(drop=True)


def create_fold(df, fold_seed=100, frac=None, method='random', disease_idx=0.0):
    if frac is None:
        frac = [0.7, 0.1, 0.2]
    if method == 'random':
        out = random_fold(df, fold_seed, frac)
    elif method == 'complex_disease':
        out = complex_disease_fold(df, fold_seed, frac)
    elif method == 'few_edges_to_kg':
        out = few_edges_to_kg_fold(df, fold_seed, [0.7, 0.1, 0.2])
    elif method == 'few_edges_to_indications':
        out = few_edges_to_indications_fold(df, fold_seed, [0.7, 0.1, 0.2])
    elif method == 'downstream_pred':
        out = disease_eval_fold(df, fold_seed, disease_idx)
    elif method == 'disease_eval':
        out = disease_eval_fold(df, fold_seed, disease_idx)
    elif method == 'full_graph':
        out = random_fold(df, fold_seed, [0.95, 0.05, 0.0])
        out['test'] = out['valid']
    else:
        train_val = df[df.split == 'train'].reset_index(drop=True)
        test = df[df.split == 'test'].reset_index(drop=True)
        out = random_fold(train_val, fold_seed, [0.875, 0.125, 0.0])
        out['test'] = test
    return out['train'], out['valid'], out['test']


def create_split(df, split, disease_eval_index, split_data_path, seed):
    print('split_data_path: ', split_data_path)
    if split == 'complex_disease_cv':
        if seed < 1 or seed > 20:
            raise ValueError('Complex disease cross validation 20 folds, select seed from 1-20.')
        df_train, df_valid, df_test = create_fold_cv(df, split_num=seed, num_splits=20)
    else:
        df_train, df_valid, df_test = create_fold(df, fold_seed=seed, frac=[0.83125, 0.11875, 0.05],
                                                    method=split, disease_idx=disease_eval_index)
    unique_rel = df[['x_type', 'relation', 'y_type']].drop_duplicates()
    df_train = reverse_rel_generation(df, df_train, unique_rel)
    df_valid = reverse_rel_generation(df, df_valid, unique_rel)
    df_test = reverse_rel_generation(df, df_test, unique_rel)
    df_train.to_csv(os.path.join(split_data_path, 'train.csv'), index=False)
    df_valid.to_csv(os.path.join(split_data_path, 'valid.csv'), index=False)
    df_test.to_csv(os.path.join(split_data_path, 'test.csv'), index=False)
    return df_train, df_valid, df_test


def construct_negative_graph_each_etype(graph, k, etype, method, weights, device):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    if method == 'corrupt_dst':
        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,))
        neg_src = neg_src.to(device)
        neg_dst = neg_dst.to(device)
    elif method == 'corrupt_src':
        neg_dst = dst.repeat_interleave(k)
        neg_src = torch.randint(0, graph.number_of_nodes(utype), (len(dst) * k,))
    elif method == 'corrupt_both':
        neg_src = torch.randint(0, graph.number_of_nodes(utype), (len(dst) * k,))
        neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,))
    elif method in ['multinomial_src', 'inverse_src', 'fix_src']:
        neg_dst = dst.repeat_interleave(k)
        try:
            neg_src = weights[etype].multinomial(len(neg_dst), replacement=True)
        except:
            neg_src = torch.Tensor([]).to(device)
    elif method in ['multinomial_dst', 'inverse_dst', 'fix_dst']:
        neg_src = src.repeat_interleave(k)
        try:
            neg_dst = weights[etype].multinomial(len(neg_src), replacement=True)
        except:
            neg_dst = torch.Tensor([]).to(device)
    return {etype: (neg_src.to(device), neg_dst.to(device))}


def construct_negative_graph(graph, k, device):
    out = {}
    for etype in graph.canonical_etypes:
        out.update(construct_negative_graph_each_etype(graph, k, etype, method, weights, device))
    return dgl.heterograph(
        out,
        num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes}
    ).to(device)


class Minibatch_NegSampler(object):
    def __init__(self, g, k, method):
        self.g = g.to(g.device)
        if method == 'multinomial_dst':
            self.weights = {
                etype: g.in_degrees(etype=etype).float() ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == 'fix_dst':
            self.weights = {
                etype: (g.in_degrees(etype=etype) > 0).float()
                for etype in g.canonical_etypes
            }
        self.k = k

    def __call__(self, g, eids_dict):
        result_dict = {}
        for etype, eids in eids_dict.items():
            src, _ = g.find_edges(eids, etype=etype)
            src = src.repeat_interleave(self.k)
            dst = self.weights[etype].multinomial(len(src), replacement=True)
            result_dict[etype] = (src, dst)
        return result_dict


class Full_Graph_NegSampler:
    def __init__(self, g, k, method, device):
        self.device = device
        if method == 'multinomial_src':
            self.weights = {
                etype: g.out_degrees(etype=etype).float().to(device) ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == 'multinomial_dst':
            self.weights = {
                etype: g.in_degrees(etype=etype).float().to(device) ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == 'inverse_dst':
            self.weights = {
                etype: -g.in_degrees(etype=etype).float().to(device) ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == 'inverse_src':
            self.weights = {
                etype: -g.out_degrees(etype=etype).float().to(device) ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == 'fix_dst':
            self.weights = {
                etype: (g.in_degrees(etype=etype) > 0).float().to(device)
                for etype in g.canonical_etypes
            }
        elif method == 'fix_src':
            self.weights = {
                etype: (g.out_degrees(etype=etype) > 0).float().to(device)
                for etype in g.canonical_etypes
            }
        else:
            self.weights = {}
        self.k = k
        self.method = method

    def __call__(self, graph):
        out = {}
        for etype in graph.canonical_etypes:
            temp = construct_negative_graph_each_etype(graph, self.k, etype, self.method, self.weights, self.device)
            if etype in temp and len(temp[etype][0]) != 0:
                out.update(temp)
        if not out:
            raise ValueError("No negative edges were constructed for the given graph.")
        return dgl.heterograph(out,
                               num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes},
                               device=self.device)


def evaluate_graph_construct(df_valid, g, neg_sampler, k, device):
    g = g.to(device)
    out = {}
    df_in = df_valid[['x_idx', 'relation', 'y_idx']]
    for etype in g.canonical_etypes:
        try:
            df_temp = df_in[df_in.relation == etype[1]]
            src = torch.Tensor(df_temp.x_idx.values).to(device).to(dtype=torch.int64)
            dst = torch.Tensor(df_temp.y_idx.values).to(device).to(dtype=torch.int64)
            out.update({etype: (src, dst)})
        except:
            print(etype[1])
    g_valid = dgl.heterograph(out, num_nodes_dict={ntype: g.number_of_nodes(ntype) for ntype in g.ntypes},
                               device=device)
    g_valid = g_valid.to(device)
    ng = Full_Graph_NegSampler(g_valid, k, neg_sampler, device)
    g_neg_valid = ng(g_valid)
    g_neg_valid = g_neg_valid.to(device)
    return g_valid, g_neg_valid


def get_all_metrics(y, pred, rels):
    edge_dict_ = {v: k for k, v in edge_dict.items()}
    auroc_rel = {}
    auprc_rel = {}
    for rel in np.unique(rels):
        index = np.where(rels == rel)
        y_ = y[index]
        pred_ = pred[index]
        try:
            auroc_rel[edge_dict_[rel]] = roc_auc_score(y_, pred_)
            auprc_rel[edge_dict_[rel]] = average_precision_score(y_, pred_)
        except:
            pass
    micro_auroc = roc_auc_score(y, pred)
    micro_auprc = average_precision_score(y, pred)
    macro_auroc = np.mean(list(auroc_rel.values()))
    macro_auprc = np.mean(list(auprc_rel.values()))
    return auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc


def get_all_metrics_fb(pred_score_pos, pred_score_neg, scores, labels, G, full_mode=False):
    auroc_rel = {}
    auprc_rel = {}
    if full_mode:
        etypes = G.canonical_etypes
    else:
        etypes = [('drug', 'contraindication', 'disease'),
                  ('drug', 'indication', 'disease'),
                  ('drug', 'off-label use', 'disease'),
                  ('disease', 'rev_contraindication', 'drug'),
                  ('disease', 'rev_indication', 'drug'),
                  ('disease', 'rev_off-label use', 'drug')]
    for etype in etypes:
        try:
            out_pos = pred_score_pos[etype].reshape(-1,).detach().cpu().numpy()
            out_neg = pred_score_neg[etype].reshape(-1,).detach().cpu().numpy()
            pred_ = np.concatenate((out_pos, out_neg))
            y_ = [1] * len(out_pos) + [0] * len(out_neg)
            auroc_rel[etype] = roc_auc_score(y_, pred_)
            auprc_rel[etype] = average_precision_score(y_, pred_)
        except:
            pass
    micro_auroc = roc_auc_score(labels, scores)
    micro_auprc = average_precision_score(labels, scores)
    macro_auroc = np.mean(list(auroc_rel.values()))
    macro_auprc = np.mean(list(auprc_rel.values()))
    return auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc


def evaluate_fb(model, g_pos, g_neg, G, dd_etypes, device, return_embed=False, mode='valid'):
    model.eval()
    pred_score_pos, pred_score_neg, pos_score, neg_score = model(G, g_neg, g_pos, pretrain_mode=False, mode=mode)
    pos_score = torch.cat([pred_score_pos[i] for i in dd_etypes])
    neg_score = torch.cat([pred_score_neg[i] for i in dd_etypes])
    scores = torch.sigmoid(torch.cat((pos_score, neg_score)).reshape(-1,))
    labels_list = [1] * len(pos_score) + [0] * len(neg_score)
    loss = F.binary_cross_entropy(scores, torch.tensor(labels_list, dtype=torch.float32, device=device))
    if return_embed:
        return get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(),
                                   labels_list, G, True), loss.item(), pred_score_pos, pred_score_neg
    else:
        return get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(),
                                   labels_list, G, True), loss.item()


def evaluate_graphmask(model, G, g_valid_pos, g_valid_neg, only_relation, epoch, etypes_train, allowance,
                        penalty_scaling, device, mode='validation', weight_bias_track=False, wandb=None, no_base=False):
    model.eval()
    G = G.to(device)
    with torch.no_grad():
        loss_fct = nn.MSELoss()
        g_valid_pos = g_valid_pos.to(device)
        g_valid_neg = g_valid_neg.to(device)
        original_predictions_pos, original_predictions_neg, _, _ = model.graphmask_forward(
            G, g_valid_pos, g_valid_neg, graphmask_mode=False, only_relation=only_relation, no_base=no_base)
        pos_score = torch.cat([original_predictions_pos[i] for i in etypes_train])
        neg_score = torch.cat([original_predictions_neg[i] for i in etypes_train])
        original_predictions = torch.sigmoid(torch.cat((pos_score, neg_score)))
        original_predictions = original_predictions.to(device)
        updated_predictions_pos, updated_predictions_neg, penalty, num_masked = model.graphmask_forward(
            G, g_valid_pos, g_valid_neg, graphmask_mode=True, only_relation=only_relation, no_base=no_base)
        pos_score = torch.cat([updated_predictions_pos[i] for i in etypes_train])
        neg_score = torch.cat([updated_predictions_neg[i] for i in etypes_train])
        updated_predictions = torch.sigmoid(torch.cat((pos_score, neg_score)))
        labels_list = [1] * len(pos_score) + [0] * len(neg_score)
        loss_pred = F.binary_cross_entropy(updated_predictions, torch.Tensor(labels_list).float().to(device)).item()
        loss_pred_ori = F.binary_cross_entropy(original_predictions, torch.Tensor(labels_list).float().to(device)).item()
        loss = loss_fct(original_predictions, updated_predictions)
        g_val = torch.relu(loss - allowance).mean()
        f_val = penalty * penalty_scaling
        print("----- " + mode + " Result -----")
        print("Epoch {0:n}, Mean divergence={1:.4f}, mean penalty={2:.4f}, bce_update={3:.4f}, bce_original={4:.4f}, "
              "num_masked_l1={5:.4f}, num_masked_l2={6:.4f}".format(
            epoch, float(loss.mean().item()), float(f_val), loss_pred, loss_pred_ori,
            num_masked[0] / G.number_of_edges(), num_masked[1] / G.number_of_edges()))
        print("-------------------------------")
        if mode == 'testing':
            test_metrics = {
                'test auroc original': roc_auc_score(np.array(labels_list),
                                                      original_predictions.detach().cpu().numpy()),
                'test auprc original': average_precision_score(np.array(labels_list),
                                                                original_predictions.detach().cpu().numpy()),
                'test auroc update': roc_auc_score(np.array(labels_list),
                                                    updated_predictions.detach().cpu().numpy()),
                'test auprc update': average_precision_score(np.array(labels_list),
                                                              updated_predictions.detach().cpu().numpy()),
                'test %masked_L1': num_masked[0] / G.number_of_edges(),
                'test %masked_L2': num_masked[1] / G.number_of_edges()
            }
            if weight_bias_track and wandb:
                wandb.log(test_metrics)
            return float(loss.mean().item()) + float(f_val), test_metrics
        if weight_bias_track and wandb:
            wandb.log({mode + ' divergence': float(loss.mean().item()),
                       mode + ' penalty': float(f_val),
                       mode + ' bce_masked': loss_pred,
                       mode + ' bce_original': loss_pred_ori,
                       mode + ' %masked_L1': num_masked[0] / G.number_of_edges(),
                       mode + ' %masked_L2': num_masked[1] / G.number_of_edges()})
        return float(loss.mean().item()) + float(f_val)


def disable_all_gradients(module):
    for param in module.parameters():
        param.requires_grad = False


def print_dict(x, required_relations=None):
    if required_relations is None:
        required_relations = list(x.keys())
    for relation in required_relations:
        if relation in x:
            print(f"{relation}: {x[relation]}")
        else:
            print(f"Missing: {relation}")


def get_wandb_log_dict(auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, mode):
    results = {
        mode + " Micro AUROC": micro_auroc,
        mode + " Micro AUPRC": micro_auprc,
        mode + " Macro AUROC": macro_auroc,
        mode + " Macro AUPRC": macro_auprc
    }
    relations = [('drug', 'contraindication', 'disease'),
                 ('drug', 'indication', 'disease'),
                 ('drug', 'off-label use', 'disease'),
                 ('disease', 'rev_contraindication', 'drug'),
                 ('disease', 'rev_indication', 'drug'),
                 ('disease', 'rev_off-label use', 'drug')]
    name_mapping = {('drug', 'contraindication', 'disease'): ' Contraindication ',
                    ('drug', 'indication', 'disease'): ' Indication ',
                    ('drug', 'off-label use', 'disease'): ' Off-Label ',
                    ('disease', 'rev_contraindication', 'drug'): ' Rev-Contraindication ',
                    ('disease', 'rev_indication', 'drug'): ' Rev-Indication ',
                    ('disease', 'rev_off-label use', 'drug'): ' Rev-Off-Label '
                    }
    for i in relations:
        if i in auroc_rel:
            results.update({mode + name_mapping[i] + "AUROC": auroc_rel[i]})
        if i in auprc_rel:
            results.update({mode + name_mapping[i] + "AUPRC": auprc_rel[i]})
    return results


def sim_matrix(a, b, eps=1e-8):
    a_n = a.norm(dim=1)[:, None]
    b_n = b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def convert2str(x):
    try:
        if '_' in str(x):
            pass
        else:
            x = float(x)
    except:
        pass
    return str(x)


def reverse_rel_generation(df, df_valid, unique_rel):
    for i in unique_rel.values:
        temp = df_valid[df_valid.relation == i[1]]
        temp = temp.rename(columns={"x_type": "y_type",
                                     "x_id": "y_id",
                                     "x_idx": "y_idx",
                                     "y_type": "x_type",
                                     "y_id": "x_id",
                                     "y_idx": "x_idx"})
        if i[0] != i[2]:
            temp["relation"] = 'rev_' + i[1]
        df_valid = df_valid._append(temp)
    return df_valid.reset_index(drop=True)


def create_dgl_graph(df_train, df):
    unique_graph = df_train[['x_type', 'relation', 'y_type']].drop_duplicates()
    DGL_input = {}
    for i in unique_graph.values:
        o = df_train[df_train.relation == i[1]][['x_idx', 'y_idx']].values.T
        DGL_input[tuple(i)] = (o[0].astype(int), o[1].astype(int))
    temp = dict(df.groupby('x_type')['x_idx'].max())
    temp2 = dict(df.groupby('y_type')['y_idx'].max())
    temp['effect/phenotype'] = 0.0
    output = {}
    for d in (temp, temp2):
        for k, v in d.items():
            output.setdefault(k, float('-inf'))
            output[k] = max(output[k], v)
    g = dgl.heterograph(DGL_input, num_nodes_dict={i: int(output[i]) + 1 for i in output.keys()})
    node_dict = {}
    edge_dict_local = {}
    for ntype in g.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in g.etypes:
        edge_dict_local[etype] = len(edge_dict_local)
        g.edges[etype].data['id'] = torch.ones(g.number_of_edges(etype), dtype=torch.long) * edge_dict_local[etype]
    return g


def initialize_node_embedding(g, n_inp):
    device = get_device()
    for ntype in g.ntypes:
        emb = nn.Parameter(torch.empty(g.number_of_nodes(ntype), n_inp, device=device), requires_grad=False)
        emb = emb.to(g.device)
        nn.init.xavier_uniform_(emb)
        g.nodes[ntype].data['inp'] = emb
    return g


def disease_centric_evaluation(df, df_train, df_valid, df_test, data_path, G, model, device, disease_ids=None,
                                 relation=None, weight_bias_track=False, wandb=None, show_plot=False, verbose=False,
                                 return_raw=False, simulate_random=True, only_prediction=False):
    return {}


def randomize_edges(hetero_graph):
    randomized_edges = {}
    for etype in hetero_graph.canonical_etypes:
        src_type, rel_type, dst_type = etype
        num_src_nodes = hetero_graph.number_of_nodes(src_type)
        num_dst_nodes = hetero_graph.number_of_nodes(dst_type)
        num_edges = hetero_graph.number_of_nodes(etype)
        np.random.seed(42)
        src_random_nodes = np.random.randint(0, num_src_nodes, num_edges)
        dst_random_nodes = np.random.randint(0, num_dst_nodes, num_edges)
        src_tensor = torch.tensor(src_random_nodes, dtype=torch.int64).to(hetero_graph.device)
        dst_tensor = torch.tensor(dst_random_nodes, dtype=torch.int64).to(hetero_graph.device)
        edge_ids = torch.arange(num_edges, dtype=torch.int64).to(hetero_graph.device)
        hetero_graph.remove_edges(edge_ids, etype=etype)
        hetero_graph.add_edges(src_tensor, dst_tensor, etype=etype)
        randomized_edges[etype] = (src_tensor, dst_tensor)
    return hetero_graph, randomized_edges


def remove_random_edges(hetero_graph, K):
    removed_edges = {}
    for etype in hetero_graph.canonical_etypes:
        num_existing_edges = hetero_graph.number_of_edges(etype)
        num_edges_to_remove = int(K / 100 * num_existing_edges)
        np.random.seed(42)
        edges_to_remove = np.random.choice(num_existing_edges, num_edges_to_remove, replace=False)
        edge_ids = torch.tensor(edges_to_remove, dtype=torch.int64).to(hetero_graph.device)
        hetero_graph.remove_edges(edge_ids, etype=etype)
        removed_edges[etype] = edge_ids
    return hetero_graph, removed_edges


def add_random_edges(hetero_graph, K):
    added_edges = {}
    for etype in hetero_graph.canonical_etypes:
        src_type, rel_type, dst_type = etype
        num_src_nodes = hetero_graph.number_of_nodes(src_type)
        num_dst_nodes = hetero_graph.number_of_nodes(dst_type)
        num_existing_edges = hetero_graph.number_of_edges(etype)
        num_edges_to_add = int(K / 100 * num_existing_edges)
        np.random.seed(42)
        src_random_nodes = np.random.randint(0, num_src_nodes, num_edges_to_add)
        dst_random_nodes = np.random.randint(0, num_dst_nodes, num_edges_to_add)
        src_tensor = torch.tensor(src_random_nodes, dtype=torch.int64).to(hetero_graph.device)
        dst_tensor = torch.tensor(dst_random_nodes, dtype=torch.int64).to(hetero_graph.device)
        hetero_graph.add_edges(src_tensor, dst_tensor, etype=etype)
        added_edges[etype] = (src_tensor, dst_tensor)
    return hetero_graph, added_edges

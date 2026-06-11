import os
import warnings

try:
    from drgnn.utils.data_utils import (
        create_fold, create_dgl_graph
    )
except ImportError:
    warnings.warn("DataControl requires dgl and torch (not installed)")


class DataControl:
    def __init__(self, data_folder: str, split: str = 'complex_disease', seed: int = 42):
        self.data_folder = data_folder
        self.split = split
        self.seed = seed
        self.disease_eval_idx = None
        self.no_kg = False
        self.kg_path = os.path.join(data_folder, 'kg.csv')

        self.df = None
        self.df_train = None
        self.df_valid = None
        self.df_test = None
        self.G = None

    def prepare_split(self, split: str = None, seed: int = None):
        if split is not None:
            self.split = split
        if seed is not None:
            self.seed = seed
        import pandas as pd
        self.df = pd.read_csv(os.path.join(self.data_folder, 'kg_directed.csv'))
        self.df_train, self.df_valid, self.df_test = create_fold(
            self.df, fold_seed=self.seed, method=self.split
        )
        self.G = create_dgl_graph(self.df_train, self.df)

    def load_preprocessed(self, data_folder: str):
        import pandas as pd
        self.data_folder = data_folder
        self.df = pd.read_csv(os.path.join(data_folder, 'kg_directed.csv'))
        self.df_train = pd.read_csv(os.path.join(data_folder, 'train.csv'))
        self.df_valid = pd.read_csv(os.path.join(data_folder, 'valid.csv'))
        self.df_test = pd.read_csv(os.path.join(data_folder, 'test.csv'))
        self.G = create_dgl_graph(self.df_train, self.df)

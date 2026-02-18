import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import csv
from collections import defaultdict
import pandas as pd 


def _normalize_node_name(name: str):
    return ''.join(ch.lower() for ch in str(name) if ch.isalnum())


def is_excluded_fx_node(name: str):
    key = _normalize_node_name(name)
    excluded_exact = {
        'cnfx',
        'ukfx',
        'cntradeweighteddollarindex',
        'uktradeweighteddollarindex',
    }
    return key in excluded_exact

def create_columns(file_path):
    if not os.path.exists(file_path):
        return []

    # Read the CSV file of the dataset
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            # Read the first row
            col = [c for c in next(reader)]
            if col and col[0].strip() == 'Date':
                return col[1:]
            return col
        except StopIteration:
            return []

def build_predefined_adj(columns, graph_file='data/graph.csv'):
    # Initialize an empty dictionary
    graph = defaultdict(list)

    # Read the graph CSV file
    try:
        with open(graph_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                key_node = row[0]
                # Extract adjacent nodes (skipping empty strings)
                adjacent_nodes = [node for node in row[1:] if node]
                graph[key_node].extend(adjacent_nodes)
        print('Graph loaded with', len(graph), 'attacks...')
    except FileNotFoundError:
        print(f"Warning: Graph file not found at {graph_file}. Returning zero matrix.")
        return torch.zeros((len(columns), len(columns)))

    print(len(columns), 'columns loaded')
    n_nodes = len(columns)
    
    if n_nodes == 0:
        return torch.zeros(0, 0)

    col_to_idx = {col: i for i, col in enumerate(columns)}

    row_indices = []
    col_indices = []

    for node_name, neighbors in graph.items():
        if is_excluded_fx_node(node_name):
            continue
        if node_name in col_to_idx:
            i = col_to_idx[node_name]
            for neighbor_name in neighbors:
                if is_excluded_fx_node(neighbor_name):
                    continue
                if neighbor_name in col_to_idx:
                    j = col_to_idx[neighbor_name]
                    # Undirected graph (Symmetric)
                    row_indices.append(i); col_indices.append(j)
                    row_indices.append(j); col_indices.append(i)

    if not row_indices:
        print("No edges found in the graph.")
        return torch.zeros(n_nodes, n_nodes)
    
    data = np.ones(len(row_indices), dtype=np.float32)
    adj_sp = sp.coo_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes)).astype(np.float32)
    
    adj_dense = adj_sp.todense()
    adj_dense = np.where(adj_dense > 1, 1, adj_dense)
    
    adj = torch.from_numpy(adj_dense).float()
    print('Adjacency created...')

    return adj

def normal_std(x):
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))


class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2, out=1, col_file=None, fixed_eval_periods=1, valid_year=2024, test_year=2025):
        self.P = window
        self.h = horizon
        self.out_len = out
        self.device = device
        self.fixed_eval_periods = int(fixed_eval_periods)
        self.valid_year = int(valid_year)
        self.test_year = int(test_year)
        self.date_index = None

        try:
            print(f"Loading data from {file_name}...")
            df = pd.read_csv(file_name)

            date_col = None
            if 'Date' in df.columns:
                date_col = 'Date'
            elif len(df.columns) > 0 and str(df.columns[0]).lower() == 'date':
                date_col = df.columns[0]

            if date_col is not None:
                parsed_dates = pd.to_datetime(df[date_col], errors='coerce')
                if parsed_dates.notna().sum() > 0:
                    self.date_index = pd.DatetimeIndex(parsed_dates)
                df = df.drop(columns=[date_col])

            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.fillna(0)
            self.rawdat_np = df.values.astype(float)
            print("Data loaded and converted to numeric successfully.")

        except Exception as e:
            print(f"Pandas load failed: {e}. Trying np.loadtxt fallback...")
            try:
                self.rawdat_np = np.loadtxt(file_name, delimiter=',', skiprows=1)
            except Exception as e2:
                raise ValueError(f"Failed to load data: {e2}")

        self.rawdat = torch.from_numpy(self.rawdat_np).float()

        self.min_data = torch.min(self.rawdat)
        if(self.min_data < 0):
            self.shift = (self.min_data * -1) + 1
        elif (self.min_data == 0):
            self.shift = torch.tensor(1.0)
        else:
            self.shift = torch.tensor(0.0)

        self.dat = torch.zeros_like(self.rawdat)
        self.n, self.m = self.dat.shape
        self.normalize = normalize
        
        self.train_end, self.valid_end = self._resolve_split_points(train, valid)
        self._normalized(normalize)
        self.scale = self.scale.to(device)

        self._split(self.train_end, self.valid_end, self.n)

        target_col_file = col_file if col_file else file_name
        try:
            self.col = create_columns(target_col_file)
            if len(self.col) != self.m:
                 self.col = [str(i) for i in range(self.m)]
        except:
             self.col = [str(i) for i in range(self.m)]

        # graph.csv는 데이터 파일과 같은 디렉토리에 있다고 가정
        graph_path = os.path.join(os.path.dirname(file_name), 'graph.csv')
        self.adj = build_predefined_adj(self.col, graph_path)

        # Calculate metrics using Test set
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.test[1].size(1), self.m)
        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

    def _resolve_split_points(self, train_ratio, valid_ratio):
        ratio_train_end = int(train_ratio * self.n)
        ratio_valid_end = int((train_ratio + valid_ratio) * self.n)

        min_train_end = self.P + self.h + self.out_len
        if self.fixed_eval_periods == 1:
            if self.date_index is not None and len(self.date_index) == self.n:
                years = self.date_index.year
                valid_idx = np.where(years == self.valid_year)[0]
                test_idx = np.where(years == self.test_year)[0]
                if len(valid_idx) > 0 and len(test_idx) > 0:
                    train_end = int(valid_idx[0])
                    valid_end = int(test_idx[0])
                    if train_end >= min_train_end and valid_end > train_end:
                        print(f"[split] fixed periods applied from date column: valid={self.valid_year}, test={self.test_year}")
                        return train_end, valid_end

            if self.n >= (min_train_end + 24):
                train_end = self.n - 24
                valid_end = self.n - 12
                if valid_end > train_end:
                    print("[split] fallback fixed periods without date column: valid=last 24~13, test=last 12")
                    return train_end, valid_end

        train_end = max(ratio_train_end, min_train_end)
        valid_end = max(ratio_valid_end, train_end + self.out_len + 1)
        valid_end = min(valid_end, self.n - 1)
        if valid_end <= train_end:
            valid_end = min(self.n, train_end + self.out_len + 1)
        print(f"[split] ratio-based split applied: train_end={train_end}, valid_end={valid_end}")
        return train_end, valid_end


    def _normalized(self, normalize):
        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            train_max = torch.max(self.rawdat[:self.train_end, :])
            self.dat = self.rawdat / train_max

        # normalized by the maximum value of each row(sensor).
        if (normalize == 2):
            # Optimized: Vectorized operation using torch
            train_data = self.rawdat[:self.train_end, :]
            max_abs_val = torch.max(torch.abs(train_data), dim=0).values

            max_abs_val[max_abs_val == 0] = 1.0
            
            # [수정] 학습 데이터 기준의 scale을 저장 (나중에 복원할 때 사용)
            self.scale = max_abs_val.to(self.device)
            self.dat = self.rawdat.clone()
            # Avoid division by zero
            self.dat = self.rawdat / max_abs_val.cpu()

    def _split(self, train, valid, test):
        # util.py Logic: Strictly separates Train / Valid / Test ranges
        train_set = range(self.P + self.h - 1, train) 
        valid_set = range(train, valid) 
        test_set = range(valid, self.n)
        
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test =  self._batchify(test_set, self.h)

        # Year-aligned rolling windows for fair validation/testing comparison
        valid_start = max(0, train - self.P)
        test_start = max(0, valid - self.P)
        self.valid_window = self.dat[valid_start:valid, :].clone()
        self.test_window = self.dat[test_start:test, :].clone()

    def _batchify(self, idx_set, horizon):
        n = len(idx_set) 
        if n <= self.out_len:
            return [torch.zeros((0, self.P, self.m)), torch.zeros((0, self.out_len, self.m))]
        X = torch.zeros((n - self.out_len, self.P, self.m)) 
        Y = torch.zeros((n - self.out_len, self.out_len, self.m)) 

        for i in range(n - self.out_len): 
            end = idx_set[i] - self.h + 1 
            start = end - self.P 
            
            # Optimized: Direct tensor slicing
            X[i, :, :] = self.dat[start:end, :]
            Y[i, :, :] = self.dat[idx_set[i]:idx_set[i]+self.out_len, :] 
            
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield X, Y 
            start_idx += batch_size
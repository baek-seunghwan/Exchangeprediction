import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import csv
from collections import defaultdict
import pandas as pd
from pathlib import Path
import re

# Helper to resolve data paths relative to this module's data directory
_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR / "data"

def _resolve_data_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    # If path starts with 'data' treat it as relative to this module
    if p.parts and p.parts[0] == 'data':
        return str(_THIS_DIR / p)
    return str(_DATA_DIR / p)

def _is_excluded_node(name: str) -> bool:
    """정규식 기반 CN/UK 노드 필터링 (더 강력한 정제)"""
    s = str(name).strip().lower()
    # uk / cn이 어디에 붙어있든 걸러버림
    if re.search(r'(^|[^a-z])uk([^a-z]|$)', s): 
        return True
    if re.search(r'(^|[^a-z])cn([^a-z]|$)', s): 
        return True
    # 보수적으로 fx 표기도 같이
    if "uk_fx" in s or "cn_fx" in s: 
        return True
    return False

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

def build_predefined_adj(columns, graph_file='data/graph.csv', exclude_nodes=None):
    # CN/UK 노드 제외 (정규식 기반 강력한 필터링)
    graph = defaultdict(list)

    try:
        with open(graph_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                key_node = row[0]
                # 정규식 기반 제외 + 기본 exclude_nodes도 확인
                if key_node and (_is_excluded_node(key_node)):
                    continue
                # neighbor에서도 CN/UK 제외
                adjacent_nodes = [node for node in row[1:] if node and (not _is_excluded_node(node))]
                graph[key_node].extend(adjacent_nodes)
        print('Graph loaded with', len(graph), 'nodes...')
    except FileNotFoundError:
        print(f"Warning: Graph file not found at {graph_file}. Returning zero matrix.")
        clean_columns = [c for c in columns if not _is_excluded_node(c)]
        return torch.zeros((len(clean_columns), len(clean_columns)))

    # 컬럼에서도 CN/UK 제외 (최종 보조 필터)
    clean_columns = [c for c in columns if not _is_excluded_node(c)]
    print(len(clean_columns), 'columns loaded after exclusion')

    n_nodes = len(clean_columns)
    if n_nodes == 0:
        return torch.zeros(0, 0)

    col_to_idx = {col: i for i, col in enumerate(clean_columns)}

    row_indices = []
    col_indices = []

    for node_name, neighbors in graph.items():
        if node_name in col_to_idx:
            i = col_to_idx[node_name]
            for neighbor_name in neighbors:
                if neighbor_name in col_to_idx:
                    j = col_to_idx[neighbor_name]
                    row_indices.append(i); col_indices.append(j)
                    row_indices.append(j); col_indices.append(i)

    if not row_indices:
        print("No edges found in the graph.")
        return torch.zeros(n_nodes, n_nodes)

    data = np.ones(len(row_indices), dtype=np.float32)
    adj_sp = sp.coo_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes)).astype(np.float32)

    adj_dense = np.asarray(adj_sp.todense())
    adj_dense[adj_sp.toarray() > 1] = 1

    adj = torch.from_numpy(adj_dense).float()
    print('Adjacency created...')
    return adj

def normal_std(x):
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))


class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2, out=1, col_file=None):
        self.P = window
        self.h = horizon
        self.out_len = out
        self.device = device

        # Resolve paths relative to B-MTGNN/data when a relative path is provided
        file_name = _resolve_data_path(file_name)
        if col_file:
            col_file = _resolve_data_path(col_file)

        try:
            print(f"Loading data from {file_name}...")
            df = pd.read_csv(file_name, parse_dates=["Date"])
            self.dates_all = df["Date"].tolist()  # 실제 시점
            df = df.drop(columns=["Date"])
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.fillna(0)
            self.rawdat_np = df.values.astype(float)
            print("Data loaded and converted to numeric successfully.")

        except Exception as e:
            print(f"Pandas load failed: {e}. Trying np.loadtxt fallback...")
            try:
                # 헤더에서 컬럼 개수 파악 (Date 포함)
                with open(file_name, 'r') as f:
                    header = f.readline()
                ncol = len(header.strip().split(','))
                # 첫 번째 컬럼(Date) 제외하고 데이터만 읽기
                self.rawdat_np = np.loadtxt(file_name, delimiter=',', skiprows=1, usecols=range(1, ncol))
            except Exception as e2:
                raise ValueError(f"Failed to load data: {e2}")

        self.rawdat = torch.from_numpy(self.rawdat_np).float()

        self.dat = torch.zeros_like(self.rawdat)
        self.n, self.m = self.dat.shape
        self.normalize = 2
        
        self.scale = torch.ones(self.m)
        self.scale = self.scale.to(device)

        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        target_col_file = col_file if col_file else file_name
        try:
            self.col = create_columns(target_col_file)
            if len(self.col) != self.m:
                self.col = [str(i) for i in range(self.m)]
        except:
            self.col = [str(i) for i in range(self.m)]

        # Build adjacency using graph file inside module data dir by default
        graph_path = _resolve_data_path('data/graph.csv')
        # Use build_predefined_adj default (which excludes CN/UK case-insensitively)
        self.adj = build_predefined_adj(self.col, graph_file=graph_path)

        # Calculate metrics using Test set
        scale_expanded = self.scale.view(1, 1, self.m).expand(self.test[1].size(0), self.test[1].size(1), self.m)
        tmp_test = self.test[1].to(self.device)
        test_actual = self.test[1] * self.scale.view(1, 1, -1)
        self.rse = torch.std(tmp_test)
        self.rae = torch.mean(torch.abs(tmp_test - torch.mean(tmp_test)))

    def _normalized(self, normalize):

        eps = 1e-8

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / (torch.max(self.rawdat) + 1e-8)

        # normalized by the maximum value of each row(sensor).
        if (normalize == 2):
            # Optimized: Vectorized operation using torch
            max_abs_val = torch.max(torch.abs(self.rawdat), dim=0).values
            max_abs_val[max_abs_val == 0] = 1e-8
            self.scale = max_abs_val.to(self.device)

            mask = max_abs_val > 0
            self.dat = self.rawdat.clone()
            # Avoid division by zero
            self.dat[:, mask] = self.rawdat[:, mask] / max_abs_val[mask]  
            self.dat = (self.rawdat / max_abs_val.view(1, -1)).to(self.device)
    def _split(self, train, valid, test):
        # 날짜 기반 split을 "강제" (validation=2024, test=2025)
        if not (hasattr(self, "dates_all") and len(self.dates_all) == self.n):
            raise ValueError("Date 컬럼(dates_all)이 없어서 2024/2025 고정 split을 할 수 없습니다.")

        # 월초로 정규화 (혹시 일자 섞여 있어도 월 기준으로 정확히 자름)
        dates_m = [pd.Timestamp(d).to_period("M").to_timestamp() for d in self.dates_all]

        idx_2024 = [i for i, d in enumerate(dates_m) if d.year == 2024]
        idx_2025 = [i for i, d in enumerate(dates_m) if d.year == 2025]

        if not idx_2024:
            raise ValueError("2024년 데이터가 없습니다. validation split 불가")
        if not idx_2025:
            raise ValueError("2025년 데이터가 없습니다. test split 불가")

        valid_start, valid_end = min(idx_2024), max(idx_2024) + 1
        test_start, test_end   = min(idx_2025), max(idx_2025) + 1

        # train은 valid_start 이전까지 (슬라이딩 윈도우 고려: 최소 P+h-1부터 시작)
        train_set = range(self.P + self.h - 1, valid_start)
        valid_set = range(valid_start, valid_end)
        test_set  = range(test_start, test_end)

        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

        # test_window: 1-step 예측용 (입력 P개 + 출력 1개)
        # 12개월 forecast는 evaluate_sliding_window에서 롤링으로 구성됨
        test_window_start = max(0, test_start - self.P)
        test_window_end = test_end
        self.test_window = self.dat[test_window_start:test_window_end, :].clone()

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        # n - self.out_len >= 1이어야 최소 1개 배치 생성 가능
        # 즉, n > self.out_len이어야 함
        if n <= self.out_len:
            # 너무 작으면 경고하되, validation/test는 작을 수 있으므로
            # 최소 1개 샘플이라도 반환하도록 함
            print(f"[경고] Split 구간이 작음: n={n}, out_len={self.out_len}. 배치 생성 불가능")
            # 빈 배치 반환 (또는 에러 발생)
            if n <= self.out_len:
                # 1개 샘플이라도 만들기: n=1, out_len이 커도 일단 시도
                pass
        
        num_samples = max(1, n - self.out_len)  # 최소 1개
        X = torch.zeros((num_samples, self.P, self.m))
        Y = torch.zeros((num_samples, self.out_len, self.m))

        for i in range(num_samples):
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
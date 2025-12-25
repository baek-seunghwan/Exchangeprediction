import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import csv
from collections import defaultdict
import pandas as pd 

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
        if node_name in col_to_idx:
            i = col_to_idx[node_name]
            for neighbor_name in neighbors:
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
    adj_dense[adj_sp > 1] = 1 # Clip values to 1
    
    adj = torch.from_numpy(adj_dense).float()
    print('Adjacency created...')

    return adj

def normal_std(x):
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    # [2순위 안정화] 빈 배열 방어 - ZeroDivisionError 방지
    if len(x) == 0:
        return 1.0  # 최소 방어 (0이면 이후 스케일링/metric에서 또 터질 수 있음)
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))


def load_csv_safely(path: str) -> pd.DataFrame:
    """
    안전하게 CSV를 로드하고 불필요한 인덱스/날짜 컬럼을 자동 제거합니다.
    
    제거 대상:
    1. "Unnamed" 패턴의 컬럼 (pandas 인덱스)
    2. Date, DateTime, Time, Timestamp 같은 날짜 컬럼
    3. 첫 컬럼이 object 타입일 경우 (날짜/문자)
    """
    # UTF-8로 시도, 실패하면 latin-1로 폴백
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(path, encoding='latin-1')
        except:
            df = pd.read_csv(path)  # 기본값으로 재시도
    
    original_cols = len(df.columns)
    
    # 1) "Unnamed" 인덱스 컬럼 제거
    drop_candidates = [c for c in df.columns if c.lower().startswith("unnamed")]
    if drop_candidates:
        print(f"  -> Removing Unnamed columns: {drop_candidates}")
        df = df.drop(columns=drop_candidates)
    
    # 2) 날짜/시간 컬럼 제거
    date_like = {"date", "datetime", "time", "timestamp"}
    drop_candidates = [c for c in df.columns if c.strip().lower() in date_like]
    if drop_candidates:
        print(f"  -> Removing date-like columns: {drop_candidates}")
        df = df.drop(columns=drop_candidates)
    
    # 3) 첫 컬럼이 object면 날짜/인덱스일 확률 높음: 제거
    if len(df.columns) > 0 and df.dtypes.iloc[0] == "object":
        first_col = df.columns[0]
        print(f"  -> Removing first column (object type): '{first_col}'")
        df = df.drop(columns=[first_col])
    
    print(f"  [OK] Original columns: {original_cols}, After cleanup: {len(df.columns)}")
    
    return df


class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, adj=None, normalize=2, out=1, col_file=None):
        self.P = window
        self.h = horizon
        self.out_len = out

        try:
            print(f"Loading data from {file_name}...")
            # 안전한 CSV 로딩 (자동 정제)
            df = load_csv_safely(file_name)
            
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # [개선 1] fillna(0) 금지 → ffill/bfill 사용 (환율 데이터에 적합)
            df = df.ffill().bfill()
            
            # [개선 2] 상수 컬럼 제거 (변동이 0인 컬럼)
            nunique = df.nunique(dropna=False)
            const_cols = nunique[nunique <= 1].index.tolist()
            if const_cols:
                print(f"  → Dropping constant columns: {const_cols[:10]}{'...' if len(const_cols)>10 else ''}")
                df = df.drop(columns=const_cols)
            
            # [진단 출력]
            print("  → Data loaded, diagnostic info:")
            print(f"    cols: {list(df.columns)[:10]}")
            print(f"    shape: {df.shape}")
            print(f"    first col unique count: {df.iloc[:,0].nunique()}")
            print(f"    first row:\n{df.head(1).to_string()}")
            
            # [개선 3] 컬럼명 동기화: create_columns() 대신 df에서 직접 가져오기
            self.col = list(df.columns)
            
            # [2순위 추가] FX 컬럼 인덱스 저장 (가중치 학습용)
            self.fx_idx = [i for i, c in enumerate(self.col) if 'fx' in c.lower()]
            print(f"  → FX columns found: {len(self.fx_idx)}")
            if self.fx_idx:
                print(f"    FX indices: {self.fx_idx}, names: {[self.col[i] for i in self.fx_idx[:5]]}")
            
            self.rawdat_np = df.values.astype(float)
            print("  [OK] Data converted to numeric successfully.")

        except Exception as e:
            print(f"Pandas load failed: {e}. Trying np.loadtxt fallback...")
            try:
                self.rawdat_np = np.loadtxt(file_name, delimiter=',', skiprows=1)
            except Exception as e2:
                raise ValueError(f"Failed to load data: {e2}")

        self.rawdat = torch.from_numpy(self.rawdat_np).float()

        self.shift = 0
        self.min_data = torch.min(self.rawdat)
        if(self.min_data < 0):
            self.shift = (self.min_data * -1) + 1
        elif (self.min_data == 0):
            self.shift = 1

        self.dat = torch.zeros_like(self.rawdat)
        self.n, self.m = self.dat.shape
        
        # train/valid split 인덱스를 먼저 계산해서 정규화에 사용
        train_end = int(train * self.n)
        train_end = max(train_end, self.P + self.h)  # 최소 window+horizon 이상
        
        train_raw = self.rawdat[:train_end, :]
        
        # train 기준 평균/표준편차 계산 (train-only z-score)
        self.bias = train_raw.mean(dim=0)  # [m]
        self.scale = train_raw.std(dim=0)  # [m]
        
        # 상수열(std=0) 보호
        self.scale[self.scale < 1e-8] = 1.0
        
        # 전체 구간에 동일 변환 적용
        self.dat = (self.rawdat - self.bias.unsqueeze(0)) / self.scale.unsqueeze(0)

        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        # ====== rse/rae 기준값 계산: test를 "원 스케일"로 복원해서 계산 ======
        scale_3d = self.scale.view(1, 1, -1).expand(self.test[1].size(0), self.test[1].size(1), self.m)
        bias_3d  = self.bias.view(1, 1, -1).expand_as(scale_3d)
        tmp = self.test[1] * scale_3d + bias_3d  # 원 스케일

        self.rse = normal_std(tmp.detach().cpu())
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp))).detach().cpu()

        self.device = device
        
        # device로 전송
        self.scale = self.scale.to(device)
        self.bias = self.bias.to(device)
        
        # [컬럼명 동기화 완료] self.col은 이미 설정됨 (CSV 로딩 시)
        # 확인: 컬럼 수 일치 체크
        if len(self.col) != self.m:
            print(f"  Column count mismatch: self.col={len(self.col)}, self.m={self.m}")
            self.col = [str(i) for i in range(self.m)]
        
        # If adj is not provided, build it from graph.csv
        if adj is None:
            graph_path = os.path.join(os.path.dirname(__file__), "..", "data", "graph.csv")
            adj = build_predefined_adj(self.col, graph_file=graph_path)
        
        self.adj = adj



    def _split(self, train, valid, test):
        # util.py Logic: Strictly separates Train / Valid / Test ranges
        train_set = range(self.P + self.h - 1, train) 
        valid_set = range(train, valid) 
        test_set = range(valid, self.n)
        
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test =  self._batchify(test_set, self.h)
        
        # [개선 3] test_window를 스플릿 인덱스 기반으로 재정의 (평가 창 정확화)
        test_start = valid
        test_len = min(36, self.n - test_start)
        
        start = max(0, test_start - self.P)
        end = test_start + test_len
        
        self.test_window = self.dat[start:end, :].clone()

    def _batchify(self, idx_set, horizon):
        n = len(idx_set) 
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
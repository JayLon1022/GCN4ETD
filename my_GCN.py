import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from time import perf_counter
import time

# 1. 图卷积层
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            stdv = 1. / math.sqrt(out_features)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):
        support = input
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# 2. GCNETD模型
class GCNETA(nn.Module):
    def __init__(self, nfeat, nhid, num_layers=2):
        super(GCNETA, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphConvolution(nfeat, nhid, bias=False))
        for _ in range(num_layers - 1):
            self.convs.append(GraphConvolution(nhid, nhid, bias=False))

    def forward(self, x, adj):
        for i in range(self.num_layers):
            x = self.convs[i](x, adj)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x

# 3. 数据加载与预处理函数
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj_symmetric(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(path='e:/Learning/项目/GCN-ETA/data/', dataset="ETA"):
    print(f'Loading {dataset} dataset from path: {path}')
    content_file = f'{path}{dataset}.content.csv'
    adj_file = f'{path}{dataset}.adjacency.csv'
    try:
        idx_features_labels = pd.read_csv(content_file)[['ID', 'in_packets', 'in_bytes', 'out_packets', 'out_bytes', 'duration', 'label']]
    except FileNotFoundError:
        print(f"Error: Content file not found at {content_file}")
        return None, None, None

    in_bytes = idx_features_labels['in_bytes']
    out_bytes = idx_features_labels['out_bytes']
    in_bytes_deal = []
    out_bytes_deal = []
    for i in in_bytes:
        val = 0.0
        if isinstance(i, str):
            i_lower = i.lower()
            if 'bytes' in i_lower:
                try:
                    val = float(i_lower.replace('bytes','').strip()) / 1024
                except ValueError:
                    pass
            elif 'kb' in i_lower:
                try:
                    val = float(i_lower.replace('kb','').strip())
                except ValueError:
                    pass
            else:
                try:
                    val = float(i)
                except ValueError:
                    pass
        elif isinstance(i, (int, float)):
            val = float(i)
        in_bytes_deal.append(val)

    for i in out_bytes:
        val = 0.0
        if isinstance(i, str):
            i_lower = i.lower()
            if 'bytes' in i_lower:
                try:
                    val = float(i_lower.replace('bytes','').strip()) / 1024
                except ValueError:
                    pass
            elif 'kb' in i_lower:
                try:
                    val = float(i_lower.replace('kb','').strip())
                except ValueError:
                    pass
            else:
                try:
                    val = float(i)
                except ValueError:
                    pass
        elif isinstance(i, (int, float)):
            val = float(i)
        out_bytes_deal.append(val)

    idx_features_labels['in_bytes'] = in_bytes_deal
    idx_features_labels['out_bytes'] = out_bytes_deal
    idx_features_labels.fillna(0, inplace=True)

    features = sp.csr_matrix(idx_features_labels[['in_packets', 'in_bytes', 'out_packets', 'out_bytes', 'duration']], dtype=np.float32)
    labels = encode_onehot(idx_features_labels['label'])

    idx = np.array(idx_features_labels['ID'], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    try:
        edges_unordered = pd.read_csv(adj_file)
    except FileNotFoundError:
        print(f"Error: Adjacency file not found at {adj_file}")
        return None, None, None

    print(f'原始边数: {edges_unordered.shape[0]}')
    valid_edges = edges_unordered[edges_unordered.iloc[:, 0].isin(idx_map) & edges_unordered.iloc[:, 1].isin(idx_map)]
    print(f'有效边数 (节点存在于content中): {valid_edges.shape[0]}')

    edges = np.array(list(map(idx_map.get, valid_edges.values.flatten())), dtype=np.int32).reshape(valid_edges.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = normalize_features(features)
    adj_normalized = normalize_adj_symmetric(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj_normalized)

    return adj, features, labels

# 5. GCN-ETA预计算特征
def our_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        if i < degree - 1:
            features = F.relu(torch.spmm(adj, features))
        else:
            features = torch.spmm(adj, features)
    precompute_time = perf_counter() - t
    return features, precompute_time

# 6. 主训练与评估函数
def main():
    np.random.seed(222)
    torch.manual_seed(222)

    adj, features, labels = load_data(path='e:/Learning/项目/GCN-ETA-main/data/')
    if adj is None:
        print("数据加载失败，请检查路径和文件。")
        return

    print("数据加载完成.")
    print(f"邻接矩阵形状 (sparse): {adj.shape}")
    print(f"特征矩阵形状: {features.shape}")
    print(f"标签数量: {labels.shape[0]}")

    print("开始GCN-ETA特征预计算...")
    exfeatures, precompute_time = our_precompute(features, adj, 2)
    print(f"特征提取完成，耗时: {precompute_time:.4f}s")
    print(f"提取的特征形状: {exfeatures.size()}")

    X = exfeatures.cpu().numpy()
    Y = labels.cpu().numpy()

    print('\n开始使用决策树进行5折交叉验证评估...')
    scorings = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=322)
    results = {scoring: [] for scoring in scorings}
    dttimes = []

    fold_count = 0
    for train_idx, test_idx in kfold.split(X, Y):
        fold_count += 1
        print(f'--- Fold {fold_count}/5 ---')
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        print(f'训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}')

        start = time.process_time()
        clf = DecisionTreeClassifier(criterion='gini', random_state=42)
        clf.fit(X_train, y_train)
        end = time.process_time()
        fold_time = end - start
        dttimes.append(fold_time)
        print(f'决策树训练耗时: {fold_time:.4f}s')

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError as e:
            print(f"无法计算AUC: {e}.")
            auc = 0.5

        results['accuracy'].append(acc)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['f1'].append(f1)
        results['roc_auc'].append(auc)
        print(f'Fold {fold_count} 结果: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}')

    print('\n--- 5折交叉验证平均结果 ---')
    for scoring in scorings:
        mean_score = np.mean(results[scoring])
        std_dev = np.std(results[scoring])
        print(f"{scoring.capitalize()}: {mean_score:.4f} (+/- {std_dev:.4f})")

    avg_dt_time = np.mean(dttimes)
    total_classification_time = np.sum(dttimes)
    total_time_paper = precompute_time + total_classification_time
    v_flow_paper = X.shape[0] / total_time_paper if total_time_paper > 0 else 0

    avg_test_size = X.shape[0] / 5
    avg_total_fold_time = precompute_time / 5 + avg_dt_time
    v_flow_fold_avg = avg_test_size / avg_total_fold_time if avg_total_fold_time > 0 else 0

    print(f"\n--- 检测速度估算 ---")
    print(f"GCN-ETA 特征预计算总时间: {precompute_time:.4f}s")
    print(f"决策树平均每折训练时间: {avg_dt_time:.4f}s")
    print(f"决策树总训练时间 (5折): {total_classification_time:.4f}s")
    print(f"估算检测速度: {v_flow_fold_avg:.0f} flows/s")


if __name__ == '__main__':
    main()
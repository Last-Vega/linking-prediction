from preprocess import create_synthetic_dataset
from config import *
from scipy.sparse import coo_matrix, lil_matrix, csr_matrix
import time

import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv
from torch_geometric.utils import train_test_split_edges

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# データの作成
data, adj_base = create_synthetic_dataset(num_users, num_items, user_feature_dim, item_feature_dim, feature_dim)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# dataを学習用とテスト用に分割する
all_edge_index = data.edge_index

transform = T.Compose([
    # T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, split_labels=True)
])

data = transform(data)
print(data)
train_data, val_data, test_data = data[0], data[1], data[2]

print(train_data.x)
print(train_data.edge_index)


model = VGAE(VariationalGCNEncoder(train_data.num_features, feature_dim)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(variational=True):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    if variational:
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward(retain_graph=True)
    optimizer.step()
    return float(loss)

def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    # link_probs = model.decode(z, data.edge_index)
    auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    link_probs = torch.sigmoid(z @ z.t())
    return auc, ap, link_probs

for i in range(1, epoch+1):
    loss = train()
    val_auc, val_ap, adj_pred = test(val_data)
    print(f'Epoch: {i:03d}, Loss: {loss:.4f}, Val_AUC: {val_auc:.4f}, Val_AP: {val_ap:.4f}')

test_auc, test_ap, adj_pred = test(test_data)
print(f'Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}')

# userがitemを購入するかどうかの確率だけが気になるので，隣接行列のuser×itemの部分だけを取り出す
print(adj_pred.shape)
print(adj_pred[1])
adj_pred = adj_pred[:num_users, num_users:]
print(adj_pred.shape)
print(adj_pred[1])

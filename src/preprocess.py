# pytorch geometricでのデータの取り扱い方について確認する

import torch
import torch.nn as nn
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt


# ユーザとアイテムの2部グラフを作成する
# ユーザは行動の特徴量を，アイテムは価格の特徴量を持つ

# ユーザ数
num_users = 5
user_feature_dim = 20
# アイテム数
num_items = 10
item_feature_dim = 1

# ユーザの特徴量
user_features = torch.randn(num_users, user_feature_dim)

# アイテムの特徴量(価格は1000円〜10000円までの一様分布)
item_features = torch.randint(1000, 10000, (num_items, item_feature_dim)).float()

print(user_features)
print(item_features)

# ユーザとアイテムの2部グラフを作成する
# ユーザとアイテムの間にエッジを張る（ユーザがアイテムをある価格で購入すればエッジが貼られる）

# ユーザとアイテムのエッジのリスト
edges = []

# ユーザが購入しない場合は6番目のアイテムにエッジを貼る（6=非購買），購入する場合は購入したアイテムにエッジを貼る
for i in range(num_users):
    if i % 2 == 0:
        edges.append([i, 6])
    else:
        edges.append([i, i+torch.randint(0, num_items, (1,)).item()])


# グラフデータを作成する
# x: ノードの特徴量
# edge_index: エッジのインデックス

# ノードの特徴量を線形変換して次元数を合わせる
user_transform = nn.Linear(user_feature_dim, 5)
item_transform = nn.Linear(item_feature_dim, 5)

user_features = user_transform(user_features)
item_features = item_transform(item_features)

# ノードの特徴量を結合する
x = torch.cat([user_features, item_features])

# エッジのインデックスを作成する
edge_index = torch.tensor(edges).t().contiguous()

# グラフデータを作成する
data = Data(x=x, edge_index=edge_index)

# データの中身を確認する
print(data)

# データを可視化する
G = to_networkx(data)
nx.draw(G, with_labels=True)

plt.show()



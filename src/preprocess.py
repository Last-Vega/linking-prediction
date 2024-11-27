# pytorch geometricでのデータの取り扱い方について確認する

import torch
import torch.nn as nn
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import scipy as sp

# ユーザとアイテムの2部グラフを作成する
# ユーザは行動の特徴量を，アイテムは価格の特徴量を持つ

# ユーザ数
num_users = 1000
user_feature_dim = 50
# アイテム数
num_items = 10
item_feature_dim = 1

# ユーザの特徴量
user_features = torch.randn(num_users, user_feature_dim)

# アイテムの価格 (アイテムの特徴量．4000〜20000の間でランダムに価格を設定)
not_purchase = torch.zeros(1, item_feature_dim)
item_features = torch.cat([not_purchase, torch.randint(4000, 20000, (num_items-1, item_feature_dim)).float()])



print(user_features)
print(item_features)

# ユーザとアイテムの2部グラフを作成する
# ユーザとアイテムの間にエッジを張る（ユーザがアイテムをある価格で購入すればエッジが貼られる）

# ユーザとアイテムのエッジのリスト
edges = []

# ユーザが購入しない場合(3で割り切れない場合)は，not_purchaseのインデックスとエッジを張る
for i in range(num_users):
    if i % 3 != 0:
        edges.append([i, num_users])
    else:
        # ユーザが購入する場合は，ランダムにアイテムを選ぶ
        item_index = torch.randint(1, num_items, (1,)).item()
        edges.append([i, num_users+item_index])



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
print(data.x)
print(data.edge_index)

g = to_networkx(data)
nx.draw(g, with_labels=True)

plt.show()
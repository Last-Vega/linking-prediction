# pytorch geometricでのデータの取り扱い方について確認する

import torch
import torch.nn as nn
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import scipy as sp
from config import prices

# データの作成
def create_synthetic_dataset(num_users, num_items, user_feature_dim, item_feature_dim, feature_dim=5):
    # ユーザの特徴量
    user_features = torch.randn(num_users, user_feature_dim)

    # アイテムの価格 (アイテムの特徴量．4000〜20000の間でランダムに価格を設定)
    not_purchase = torch.zeros(1, item_feature_dim)
    # item_features = torch.cat([not_purchase, torch.randint(4000, 20000, (num_items-1, item_feature_dim)).float()])
    item_features = torch.cat([not_purchase, torch.tensor(prices).view(-1, 1).float()])
    print(item_features)


    # ユーザとアイテムの2部グラフを作成する
    # ユーザとアイテムの間にエッジを張る（ユーザがアイテムをある価格で購入すればエッジが貼られる）

    # ユーザとアイテムのエッジのリスト
    edges = []

    # ユーザが購入しない場合(3で割り切れない場合)は，not_purchaseのインデックスとエッジを張る
    # 正方行列の隣接行列を作成するために，(num_users+num_items)x(num_users+num_items)の行列を作成する．ただし，
    
    adj_base = torch.zeros(num_users+num_items, num_users+num_items)
    for i in range(num_users):
        if i % 3 != 0:
            edges.append([i, num_users])
            adj_base[i, num_users] = 1
        else:
            # ユーザが購入する場合は，ランダムにアイテムを選ぶ
            item_index = torch.randint(1, num_items, (1,)).item()
            edges.append([i, num_users+item_index])
            adj_base[i, num_users+item_index] = 1
    

    # グラフデータを作成する
    # x: ノードの特徴量
    # edge_index: エッジのインデックス

    # ノードの特徴量を線形変換して次元数を合わせる
    user_transform = nn.Linear(user_feature_dim, feature_dim)
    item_transform = nn.Linear(item_feature_dim, feature_dim)

    user_features = user_transform(user_features)
    item_features = item_transform(item_features)

    # ノードの特徴量を結合する
    x = torch.cat([user_features, item_features])

    # エッジのインデックスを作成する
    edge_index = torch.tensor(edges).t().contiguous()

    # グラフデータを作成する
    data = Data(x=x, edge_index=edge_index)

    return data, adj_base

# data, adj_base = create_synthetic_dataset(1000, 10, 50, 1)
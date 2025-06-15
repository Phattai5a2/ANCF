# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 23:33:22 2025

@author: PC
"""

# -*- coding: utf-8 -*-
"""
Meta-learning Hybrid Recommendation System
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor

# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đọc dữ liệu
df = pd.read_csv("classified_and_rated_tweets.csv")
user_ids = df["ID"].astype("category").cat.codes.values
item_ids = df["Categories"].astype("category").cat.codes.values
ratings = df["Ratting"].values

# Chuẩn hóa rating về khoảng 0-1
ratings = ratings / ratings.max()

# Chia tập dữ liệu thành tập huấn luyện và kiểm tra
train_user, test_user, train_item, test_item, train_ratings, test_ratings = train_test_split(
    user_ids, item_ids, ratings, test_size=0.2, random_state=42
)

# Convert dữ liệu sang tensor
train_user = torch.tensor(train_user, dtype=torch.long).to(device)
train_item = torch.tensor(train_item, dtype=torch.long).to(device)
train_ratings = torch.tensor(train_ratings, dtype=torch.float32).to(device)
test_user = torch.tensor(test_user, dtype=torch.long).to(device)
test_item = torch.tensor(test_item, dtype=torch.long).to(device)
test_ratings = torch.tensor(test_ratings, dtype=torch.float32).to(device)

# Định nghĩa mô hình Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, num_items, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(num_items, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, num_items)
        
    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.sigmoid(self.decoder(x))
        return x

# Khởi tạo mô hình Autoencoder
num_items = len(np.unique(item_ids))
encoding_dim = 64
autoencoder = Autoencoder(num_items, encoding_dim).to(device)

# Huấn luyện Autoencoder
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

user_item_matrix = torch.zeros((len(np.unique(user_ids)), num_items)).to(device)
for u, i, r in zip(train_user, train_item, train_ratings):
    user_item_matrix[u, i] = r

num_epochs = 20
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = autoencoder(user_item_matrix)
    loss = criterion(output, user_item_matrix)
    loss.backward()
    optimizer.step()
    print(f"Autoencoder Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Dự đoán từ Autoencoder
with torch.no_grad():
    reconstructed_matrix = autoencoder(user_item_matrix)

# Định nghĩa mô hình NCF (Neural Collaborative Filtering)
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        
    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        
        x = torch.cat([user_embed, item_embed], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        return x

# Khởi tạo mô hình NCF
num_users = len(np.unique(user_ids))
ncf = NCF(num_users, num_items).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(ncf.parameters(), lr=0.001)

# Huấn luyện NCF
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = ncf(train_user, train_item).squeeze()
    loss = criterion(output, train_ratings)
    loss.backward()
    optimizer.step()
    print(f"NCF Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Dự đoán từ NCF
with torch.no_grad():
    pred_ncf = ncf(test_user, test_item).squeeze().cpu().numpy()

# Xây dựng mô hình meta-learning
meta_features = np.vstack([reconstructed_matrix[test_user.cpu(), test_item.cpu()].cpu().numpy(), pred_ncf]).T
meta_labels = test_ratings.cpu().numpy()

meta_learner = RandomForestRegressor(n_estimators=100, random_state=42)
meta_learner.fit(meta_features, meta_labels)

# Dự đoán bằng meta-learning
meta_predictions = meta_learner.predict(meta_features)

# Chuyển thành dạng nhị phân (>=0.5 là positive, <0.5 là negative)
meta_binary_predictions = (meta_predictions >= 0.5).astype(int)
test_binary_labels = (meta_labels >= 0.5).astype(int)

# Đánh giá mô hình meta-learning
accuracy = accuracy_score(test_binary_labels, meta_binary_predictions)
precision = precision_score(test_binary_labels, meta_binary_predictions)
recall = recall_score(test_binary_labels, meta_binary_predictions)
f1 = f1_score(test_binary_labels, meta_binary_predictions)

print(f"Meta-learning Accuracy: {accuracy:.4f}")
print(f"Meta-learning Precision: {precision:.4f}")
print(f"Meta-learning Recall: {recall:.4f}")
print(f"Meta-learning F1 Score: {f1:.4f}")

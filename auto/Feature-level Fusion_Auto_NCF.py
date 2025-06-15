# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 01:07:33 2024

@author: PC
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Kiểm tra nếu GPU có sẵn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Sử dụng thiết bị: {device}')

# Đọc dataset
file_path = 'classified_and_rated_tweets.csv'  # Đường dẫn tới file của bạn
df = pd.read_csv(file_path)


# Chuẩn bị dữ liệu
user_ids = df['ID'].astype('category').cat.codes.values
item_ids = df['Categories'].astype('category').cat.codes.values
ratings = df['Ratting'].values

# Chuẩn hóa dữ liệu để các giá trị nằm trong khoảng từ 0 đến 1
ratings = ratings / ratings.max()

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_user, test_user, train_item, test_item, train_ratings, test_ratings = train_test_split(user_ids, item_ids, ratings, test_size=0.2, random_state=42)

train_user = torch.tensor(train_user, dtype=torch.long).to(device)
train_item = torch.tensor(train_item, dtype=torch.long).to(device)
train_ratings = torch.tensor(train_ratings, dtype=torch.float32).to(device)
test_user = torch.tensor(test_user, dtype=torch.long).to(device)
test_item = torch.tensor(test_item, dtype=torch.long).to(device)
test_ratings = torch.tensor(test_ratings, dtype=torch.float32).to(device)

# Định nghĩa mô hình Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_dim)
        
    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.sigmoid(self.decoder(x))
        return x

# Định nghĩa mô hình NCF
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        
    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        
        x = torch.cat([user_embed, item_embed], dim=1)
        x = torch.relu(self.fc1(x))
        features = torch.relu(self.fc2(x))
        return features

# Định nghĩa mô hình kết hợp (Hybrid Model) với Feature-level Fusion
class HybridModel(nn.Module):
    def __init__(self, autoencoder, ncf, num_users, num_items, encoding_dim):
        super(HybridModel, self).__init__()
        self.autoencoder = autoencoder
        self.ncf = ncf
        self.fc = nn.Linear(encoding_dim + 64, 1)  # Kết hợp output của Autoencoder và NCF
        
    def forward(self, user, item, user_item_matrix):
        # Lấy features từ NCF
        ncf_features = self.ncf(user, item)
        
        # Lấy features từ Autoencoder
        user_vector = user_item_matrix[user]
        autoencoder_output = self.autoencoder.encoder(user_vector)
        
        # Kết hợp các features
        combined_features = torch.cat([ncf_features, autoencoder_output], dim=1)
        
        # Đưa qua lớp fully connected để dự đoán cuối cùng
        output = torch.sigmoid(self.fc(combined_features))
        return output.squeeze()

# Khởi tạo mô hình
input_dim = len(np.unique(item_ids))
encoding_dim = 64

autoencoder = Autoencoder(input_dim, encoding_dim).to(device)
ncf = NCF(len(np.unique(user_ids)), len(np.unique(item_ids))).to(device)

hybrid_model = HybridModel(autoencoder, ncf, len(np.unique(user_ids)), len(np.unique(item_ids)), encoding_dim).to(device)

# Huấn luyện mô hình kết hợp (Hybrid Model)
criterion = nn.BCELoss()
optimizer = optim.Adam(hybrid_model.parameters(), lr=0.001)

num_epochs = 50
batch_size = 256

# Tạo ma trận user-item cho Autoencoder
user_item_matrix = torch.zeros((len(np.unique(user_ids)), input_dim)).to(device)
for i in range(len(train_user)):
    user_item_matrix[train_user[i], train_item[i]] = train_ratings[i]

losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    for i in range(0, len(train_user), batch_size):
        batch_user = train_user[i:i+batch_size]
        batch_item = train_item[i:i+batch_size]
        batch_ratings = train_ratings[i:i+batch_size]
        
        output = hybrid_model(batch_user, batch_item, user_item_matrix)
        loss = criterion(output, batch_ratings)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    losses.append(epoch_loss / len(train_user))
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Vẽ đồ thị loss
plt.plot(losses, label='Hybrid Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Hybrid Model Training Loss')
plt.legend()
plt.show()

# Đánh giá mô hình
with torch.no_grad():
    user_item_matrix = torch.zeros((len(np.unique(user_ids)), input_dim)).to(device)
    for i in range(len(test_user)):
        user_item_matrix[test_user[i], test_item[i]] = test_ratings[i]
        
    pred_ratings_hybrid = hybrid_model(test_user, test_item, user_item_matrix).squeeze()

# Chuyển đổi dự đoán thành nhãn phân loại cho mô hình lai
pred_labels_hybrid = (pred_ratings_hybrid.cpu().numpy() > 0.5).astype(int).flatten()
true_labels_hybrid = (test_ratings.cpu().numpy() > 0.5).astype(int)

# Tính toán các chỉ số đánh giá cho mô hình lai
accuracy_hybrid = accuracy_score(true_labels_hybrid, pred_labels_hybrid)
recall_hybrid = recall_score(true_labels_hybrid, pred_labels_hybrid)
precision_hybrid = precision_score(true_labels_hybrid, pred_labels_hybrid)
f1_hybrid = f1_score(true_labels_hybrid, pred_labels_hybrid)

print(f"Hybrid Model Accuracy: {accuracy_hybrid}")
print(f"Hybrid Model Recall: {recall_hybrid}")
print(f"Hybrid Model Precision: {precision_hybrid}")
print(f"Hybrid Model F1 Score: {f1_hybrid}")

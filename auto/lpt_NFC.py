# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 01:09:46 2024

@author: PC
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 16:37:09 2024

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

# Đọc dataset
file_path = 'D:\DT\\auto\classified_and_rated_tweets.csv'  # Đường dẫn tới file của bạn
df = pd.read_csv(file_path)

# Chuẩn bị dữ liệu
user_ids = df['ID'].astype('category').cat.codes.values
item_ids = df['Categories'].astype('category').cat.codes.values
ratings = df['Ratting'].values

# Chuẩn hóa dữ liệu để các giá trị nằm trong khoảng từ 0 đến 1
ratings = ratings / ratings.max()

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_user, test_user, train_item, test_item, train_ratings, test_ratings = train_test_split(user_ids, item_ids, ratings, test_size=0.2, random_state=42)

train_user = torch.tensor(train_user, dtype=torch.long)
train_item = torch.tensor(train_item, dtype=torch.long)
train_ratings = torch.tensor(train_ratings, dtype=torch.float32)
test_user = torch.tensor(test_user, dtype=torch.long)
test_item = torch.tensor(test_item, dtype=torch.long)
test_ratings = torch.tensor(test_ratings, dtype=torch.float32)

# Định nghĩa mô hình NCF
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

num_users = len(np.unique(user_ids))
num_items = len(np.unique(item_ids))

model = NCF(num_users, num_items)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình NCF
num_epochs = 50
batch_size = 256

losses = []  # Danh sách để lưu giá trị loss

for epoch in range(num_epochs):
    epoch_loss = 0
    for i in range(0, len(train_user), batch_size):
        batch_user = train_user[i:i+batch_size]
        batch_item = train_item[i:i+batch_size]
        batch_ratings = train_ratings[i:i+batch_size]
        
        output = model(batch_user, batch_item)
        loss = criterion(output.squeeze(), batch_ratings)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_loss /= len(train_user) // batch_size
    losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Vẽ biểu đồ Loss qua các Epoch
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Biểu đồ Loss qua các Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Dự đoán và đánh giá mô hình NCF
with torch.no_grad():
    pred_ratings = model(test_user, test_item).squeeze()

# Chuyển đổi dự đoán thành nhãn phân loại
threshold = 0.5  # Ngưỡng để phân loại
pred_labels = (pred_ratings.numpy() > threshold).astype(int).flatten()
true_labels = (test_ratings.numpy() > threshold).astype(int)

# Tính toán các chỉ số đánh giá
accuracy = accuracy_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

print(f"NCF Accuracy: {accuracy}")
print(f"NCF Recall: {recall}")
print(f"NCF Precision: {precision}")
print(f"NCF F1 Score: {f1}")

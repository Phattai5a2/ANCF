# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 01:00:30 2024

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

# Chuyển đổi dữ liệu thành ma trận người dùng - sản phẩm
user_item_matrix = df.pivot(index='ID', columns='Categories', values='Ratting').fillna(0)

# Chuẩn hóa dữ liệu để các giá trị nằm trong khoảng từ 0 đến 1
user_item_matrix = user_item_matrix / user_item_matrix.max().max()

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_data, test_data = train_test_split(user_item_matrix, test_size=0.2, random_state=42)

train_data = torch.tensor(train_data.values, dtype=torch.float32)
test_data = torch.tensor(test_data.values, dtype=torch.float32)

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

input_dim = train_data.shape[1]
encoding_dim = 64  # Kích thước biểu diễn nén

model = Autoencoder(input_dim, encoding_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình Autoencoder
num_epochs = 50
batch_size = 256

losses = []  # Danh sách để lưu giá trị loss

for epoch in range(num_epochs):
    epoch_loss = 0
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        output = model(batch)
        loss = criterion(output, batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_loss /= len(train_data) // batch_size
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

# Sử dụng mô hình Autoencoder để dự đoán
with torch.no_grad():
    predicted_ratings = model(test_data)

# Chuyển đổi dự đoán thành nhãn phân loại
threshold = 0.5  # Ngưỡng để phân loại
pred_labels = (predicted_ratings.numpy() > threshold).astype(int).flatten()
true_labels = (test_data.numpy() > threshold).astype(int).flatten()

# Tính toán các chỉ số đánh giá
accuracy = accuracy_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

print(f"Autoencoder Accuracy: {accuracy}")
print(f"Autoencoder Recall: {recall}")
print(f"Autoencoder Precision: {precision}")
print(f"Autoencoder F1 Score: {f1}")

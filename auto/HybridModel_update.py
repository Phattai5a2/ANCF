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
        self.output = nn.Linear(64, 1)
        
    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        
        x = torch.cat([user_embed, item_embed], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        return x

# Thiết lập các tham số và mô hình
input_dim = len(np.unique(item_ids))
encoding_dim = 64

autoencoder = Autoencoder(input_dim, encoding_dim).to(device)
ncf = NCF(len(np.unique(user_ids)), len(np.unique(item_ids))).to(device)

# Huấn luyện mô hình Autoencoder
criterion_autoencoder = nn.BCELoss()
optimizer_autoencoder = optim.Adam(autoencoder.parameters(), lr=0.001)

num_epochs = 50
batch_size = 256

# Tạo ma trận user-item cho Autoencoder
user_item_matrix = torch.zeros((len(np.unique(user_ids)), input_dim)).to(device)
for i in range(len(train_user)):
    user_item_matrix[train_user[i], train_item[i]] = train_ratings[i]

autoencoder_losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    for i in range(0, len(user_item_matrix), batch_size):
        batch = user_item_matrix[i:i+batch_size]
        output = autoencoder(batch)
        loss = criterion_autoencoder(output, batch)
        
        optimizer_autoencoder.zero_grad()
        loss.backward()
        optimizer_autoencoder.step()
        
        epoch_loss += loss.item()
    
    autoencoder_losses.append(epoch_loss / len(user_item_matrix))
    print(f'Autoencoder Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Vẽ đồ thị Loss của Autoencoder
plt.plot(autoencoder_losses, label='Autoencoder Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Autoencoder Training Loss')
plt.legend()
plt.show()

# Huấn luyện mô hình NCF
criterion_ncf = nn.BCELoss()
optimizer_ncf = optim.Adam(ncf.parameters(), lr=0.001)

ncf_losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    for i in range(0, len(train_user), batch_size):
        batch_user = train_user[i:i+batch_size]
        batch_item = train_item[i:i+batch_size]
        batch_ratings = train_ratings[i:i+batch_size]
        
        output = ncf(batch_user, batch_item).squeeze()
        loss = criterion_ncf(output, batch_ratings)
        
        optimizer_ncf.zero_grad()
        loss.backward()
        optimizer_ncf.step()
        
        epoch_loss += loss.item()
    
    ncf_losses.append(epoch_loss / len(train_user))
    print(f'NCF Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Vẽ đồ thị Loss của NCF
plt.plot(ncf_losses, label='NCF Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('NCF Training Loss')
plt.legend()
plt.show()

# Định nghĩa mô hình kết hợp Autoencoder và NCF
# Định nghĩa mô hình kết hợp Autoencoder và NCF với trọng số động
class HybridModel(nn.Module):
    def __init__(self, autoencoder, ncf, num_users, num_items):
        super(HybridModel, self).__init__()
        self.autoencoder = autoencoder
        self.ncf = ncf
        self.weight = nn.Parameter(torch.tensor(0.5))  # Trọng số động, có thể học được
    
    def forward(self, user, item, user_item_matrix):
        ncf_output = self.ncf(user, item).squeeze()
        user_vector = user_item_matrix[user]
        autoencoder_output = self.autoencoder(user_vector)
        autoencoder_item_output = autoencoder_output.gather(1, item.view(-1, 1)).squeeze()
        
        # Kết hợp hai giá trị dự đoán theo công thức
        combined_output = self.weight * ncf_output + (1 - self.weight) * autoencoder_item_output
        return combined_output

# Khởi tạo mô hình Hybrid với trọng số động
hybrid_model = HybridModel(autoencoder, ncf, len(np.unique(user_ids)), len(np.unique(item_ids))).to(device)

# Huấn luyện Hybrid Model với trọng số có thể học
criterion_hybrid = nn.MSELoss()  # Chuyển từ BCELoss sang MSELoss để phù hợp với bài toán dự đoán liên tục
optimizer_hybrid = optim.Adam(hybrid_model.parameters(), lr=0.001)

hybrid_losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    for i in range(0, len(train_user), batch_size):
        batch_user = train_user[i:i+batch_size]
        batch_item = train_item[i:i+batch_size]
        batch_ratings = train_ratings[i:i+batch_size]
        
        # Cập nhật ma trận user-item cho batch hiện tại
        user_item_matrix = torch.zeros((len(np.unique(user_ids)), input_dim)).to(device)
        for j in range(len(batch_user)):
            user_item_matrix[batch_user[j], batch_item[j]] = batch_ratings[j].clone().detach()
        
        output = hybrid_model(batch_user, batch_item, user_item_matrix)
        loss = criterion_hybrid(output, batch_ratings)
        
        optimizer_hybrid.zero_grad()
        loss.backward()
        optimizer_hybrid.step()
        
        epoch_loss += loss.item()
    
    hybrid_losses.append(epoch_loss / len(train_user))
    print(f'Hybrid Model Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Weight: {hybrid_model.weight.item():.4f}')



# Vẽ đồ thị Loss của Hybrid
plt.plot(hybrid_losses, label='Hybrid Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Hybrid Model Training Loss')
plt.legend()
plt.show()

# Đánh giá mô hình Autoencoder
with torch.no_grad():
    test_user_item_matrix = torch.zeros((len(np.unique(user_ids)), input_dim)).to(device)
    for i in range(len(test_user)):
        test_user_item_matrix[test_user[i], test_item[i]] = test_ratings[i]
    
    predicted_ratings_autoencoder = autoencoder(test_user_item_matrix).cpu().numpy()
    pred_labels_autoencoder = (predicted_ratings_autoencoder > 0.5).astype(int).flatten()
    true_labels_autoencoder = (test_user_item_matrix.cpu().numpy() > 0.5).astype(int).flatten()
    
    accuracy_autoencoder = accuracy_score(true_labels_autoencoder, pred_labels_autoencoder)
    recall_autoencoder = recall_score(true_labels_autoencoder, pred_labels_autoencoder)
    precision_autoencoder = precision_score(true_labels_autoencoder, pred_labels_autoencoder)
    f1_autoencoder = f1_score(true_labels_autoencoder, pred_labels_autoencoder)
    
    print(f"Autoencoder Accuracy: {accuracy_autoencoder}")
    print(f"Autoencoder Recall: {recall_autoencoder}")
    print(f"Autoencoder Precision: {precision_autoencoder}")
    print(f"Autoencoder F1 Score: {f1_autoencoder}")

# Đánh giá mô hình NCF
with torch.no_grad():
    pred_ratings_ncf = ncf(test_user, test_item).squeeze().cpu().numpy()
    pred_labels_ncf = (pred_ratings_ncf > 0.5).astype(int).flatten()
    true_labels_ncf = (test_ratings.cpu().numpy() > 0.5).astype(int)
    
    accuracy_ncf = accuracy_score(true_labels_ncf, pred_labels_ncf)
    recall_ncf = recall_score(true_labels_ncf, pred_labels_ncf)
    precision_ncf = precision_score(true_labels_ncf, pred_labels_ncf)
    f1_ncf = f1_score(true_labels_ncf, pred_labels_ncf)
    
    print(f"NCF Accuracy: {accuracy_ncf}")
    print(f"NCF Recall: {recall_ncf}")
    print(f"NCF Precision: {precision_ncf}")
    print(f"NCF F1 Score: {f1_ncf}")

# Đánh giá mô hình lai (Hybrid)
with torch.no_grad():
    user_item_matrix = torch.zeros((len(np.unique(user_ids)), input_dim)).to(device)
    for i in range(len(test_user)):
        user_item_matrix[test_user[i], test_item[i]] = test_ratings[i].clone().detach()
        
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

# Tạo bảng so sánh các kết quả đánh giá
results = {
    "Model": ["Autoencoder", "NCF", "Hybrid"],
    "Accuracy": [accuracy_autoencoder, accuracy_ncf, accuracy_hybrid],
    "Recall": [recall_autoencoder, recall_ncf, recall_hybrid],
    "Precision": [precision_autoencoder, precision_ncf, precision_hybrid],
    "F1 Score": [f1_autoencoder, f1_ncf, f1_hybrid]
}

results_df = pd.DataFrame(results)
print("\nSo sánh các mô hình:")
print(results_df)

# Hàm khuyến nghị các tweet cho người dùng cụ thể
def recommend_tweets(user_id, model, user_item_matrix, num_recommendations=5):
    user_index = user_item_matrix.index.get_loc(user_id)
    user_vector = torch.tensor(user_item_matrix.iloc[user_index].values, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructed_vector = model.autoencoder(user_vector).cpu().numpy()
    recommended_tweets_indices = np.argsort(reconstructed_vector)[-num_recommendations:]
    recommended_tweets = user_item_matrix.columns[recommended_tweets_indices]
    return recommended_tweets

# Chuyển đổi user_item_matrix thành DataFrame cho hàm recommend_tweets
user_item_matrix_df = pd.DataFrame(user_item_matrix.cpu().numpy(), index=pd.Index(np.unique(user_ids), name='User_ID'), columns=pd.Index(np.unique(item_ids), name='Item_ID'))

# Thử nghiệm khuyến nghị cho một người dùng cụ thể
sample_user_id = user_item_matrix_df.index[0]
recommended_tweets = recommend_tweets(sample_user_id, hybrid_model, user_item_matrix_df)
print(f"Recommended tweets for user {sample_user_id}: {recommended_tweets}")

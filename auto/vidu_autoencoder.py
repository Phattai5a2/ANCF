import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Dữ liệu đầu vào: Xếp hạng phim của nhiều người dùng (0 là giá trị bị thiếu)
ratings = np.array([
    [5, 0, 3, 0, 4],
    [4, 2, 0, 3, 0],
    [0, 5, 4, 0, 3],
    [3, 0, 0, 5, 2]
])

# Chuẩn hóa dữ liệu về khoảng [0,1] để dễ huấn luyện
max_rating = np.max(ratings)
norm_ratings = ratings / max_rating

# Chuyển đổi sang Tensor
ratings_tensor = torch.tensor(norm_ratings, dtype=torch.float32)
mask = (ratings_tensor != 0).float()  # Mask để bảo vệ giá trị đã có

# Định nghĩa Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()  # Đảm bảo đầu ra trong khoảng [0,1]
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

# Khởi tạo mô hình
input_size = ratings.shape[1]  # Số lượng phim
hidden_size = 3  # Kích thước không gian tiềm ẩn
model = Autoencoder(input_size, hidden_size)

# Định nghĩa loss function và optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Huấn luyện mô hình
num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(ratings_tensor)
    loss = criterion(output * mask, ratings_tensor * mask)  # Chỉ tính loss cho giá trị đã có
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

# Dự đoán giá trị bị thiếu
with torch.no_grad():
    predicted_ratings = model(ratings_tensor).numpy()

# Chuyển giá trị về thang điểm gốc
predicted_ratings = predicted_ratings * max_rating

# Lấy các giá trị bị thiếu trong đầu vào và thay bằng dự đoán
ratings_filled = ratings.copy()
missing_indices = ratings == 0
ratings_filled[missing_indices] = predicted_ratings[missing_indices]

print("Xếp hạng sau khi điền giá trị thiếu:")
print(ratings_filled)

# Đưa ra khuyến nghị: Gợi ý bộ phim có điểm dự đoán cao nhất cho mỗi người dùng
recommended_movies = np.argmax(ratings_filled, axis=1)
print("Phim được khuyến nghị cho mỗi người dùng:", recommended_movies)

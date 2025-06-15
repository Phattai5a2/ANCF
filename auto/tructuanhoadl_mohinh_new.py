# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 23:55:17 2025

@author: PC
"""

import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu
models = ['Autoencoder', 'NCF', 'Hybrid']
accuracy = [0.7934, 0.8309, 0.9790]
recall   = [0.0566, 0.1170, 0.9170]
precision= [0.1667, 0.4921, 0.9567]
f1_score = [0.0845, 0.1890, 0.9364]

# Vị trí cho các cột
x = np.arange(len(models))
width = 0.2  # độ rộng cột

# Tạo figure và axes
fig, ax = plt.subplots(figsize=(10,6))

# Vẽ các cột cho từng chỉ số
rects1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy')
rects2 = ax.bar(x - 0.5*width, recall, width, label='Recall')
rects3 = ax.bar(x + 0.5*width, precision, width, label='Precision')
rects4 = ax.bar(x + 1.5*width, f1_score, width, label='F1 Score')

# Thêm nhãn và tiêu đề
ax.set_ylabel('Giá trị')
ax.set_title('Đánh giá các mô hình')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Thiết lập giới hạn trục y (0 đến 1)
plt.ylim(0,1)

# Hiển thị biểu đồ
plt.show()

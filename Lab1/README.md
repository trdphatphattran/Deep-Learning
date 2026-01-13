# LAB 1: PYTORCH  
## Thông tin  
Sinh viên: Trần Đại Phát  
MSSV: 2374802010379  
Khóa: 29  
Trường Đại học Văn Lang  
Môn học: Giới thiệu về học sâu  
GVHD: Nguyễn Thái Anh  
Năm học: 2025 - 2026  

## Phần 1: Giới thiệu về Pytorch  
### 1. Pytroch là gì?  
PyTorch là một thư viện mã nguồn mở dùng để lập trình trí tuệ nhân tạo (AI), cụ thể là trong lĩnh vực Học sâu (Deep Learning). Nó được phát triển chủ yếu bởi phòng nghiên cứu AI của Meta (Facebook) và hiện là công cụ yêu thích nhất của các nhà khoa học dữ liệu trên toàn thế giới.  
### 2. Các chức năng chính của Pytorch  
- Tính toán Tensor tối ưu (Thay thế NumPy trên GPU).
- Tự động tính Đạo hàm (Autograd).
- Xây dựng Mạng thần kinh.
- Tối ưu hóa mô hình.
### 3. Cách hoạt động của Pytorch  
import torch  
torch.cuda.is_available()  
import torch  
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  

df = pd.read_csv('Iris.csv')  
df  
df.shape  
from sklearn.preprocessing import LabelEncoder  
from sklearn.model_selection import train_test_split  
le = LabelEncoder()  
X = df.drop(['Species'], axis=1).values  
y = le.fit_transform(df['Species'].values)  

# Chia dữ liệu với test size = 20%  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
X_train = torch.FloatTensor(X_train)  
X_test = torch.FloatTensor(X_test)  
y_train = torch.LongTensor(y_train).reshape(-1, 1)  
y_test = torch.LongTensor(y_test).reshape(-1, 1)  
len(y_train)  
labels, counts = y_train.unique(return_counts=True)  
print(labels, counts)  





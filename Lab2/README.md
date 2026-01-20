# LAB 2: PANDAS, NUMPY, MATPLOTLIB  
## Thông tin  
Sinh viên: Trần Đại Phát  
MSSV: 2374802010379  
Khóa: 29  
Trường Đại học Văn Lang  
Môn học: Giới thiệu về học sâu  
GVHD: Nguyễn Thái Anh  
Năm học: 2025 - 2026  

## Phần 1: Giới thiệu về Numpy  
### 1. Numpy là gì?  
NumPy (viết tắt của Numerical Python) là thư viện lõi và quan trọng nhất cho việc tính toán khoa học trong ngôn ngữ lập trình Python.  

### 2. Các tính năng của Numpy?  
NumPy không chỉ là nơi chứa dữ liệu mà còn cung cấp một hệ sinh thái các công cụ toán học:  
- Broadcasting: Cơ chế mạnh mẽ cho phép thực hiện các phép toán giữa các mảng có kích thước khác nhau mà không cần viết vòng lặp thủ công.  
- Toán học & Đại số tuyến tính: Cung cấp sẵn các hàm tính sin, cos, log, căn bậc hai, nhân ma trận, tìm định thức, nghịch đảo ma trận...  
- Số ngẫu nhiên (Random): Khả năng tạo ra các bộ số ngẫu nhiên theo nhiều phân phối khác nhau (Gaussian, Uniform, v.v.).  
- Biến đổi Fourier (FFT): Hỗ trợ đắc lực trong xử lý tín hiệu.

### 3. Một số ví dụ với Numpy  
#### Trước hết, để sử dụng cần tải thư viện:  
pip install numpy  

#### Import thư viện trước khi sử dụng:  
import numpy as np  

#### Ví dụ tạo một mảng với Numpy:  
<img width="284" height="25" alt="image" src="https://github.com/user-attachments/assets/5047e4d9-071f-41c9-9aa3-44b621d6fd80" />  

#### Lấy từ số 3 đến số 8  
<img width="194" height="91" alt="image" src="https://github.com/user-attachments/assets/7ac7cafc-030b-4604-98f2-d12799756487" />  

#### Lấy số 3 và số 5  
<img width="201" height="76" alt="image" src="https://github.com/user-attachments/assets/59bc33f1-2c9f-4576-8001-916d099aac2e" />  

#### Ví dụ cho một danh sách:  
<img width="209" height="200" alt="image" src="https://github.com/user-attachments/assets/83d6ac54-ae14-4816-bcf2-b643210e5c71" />  

#### Lấy số 2, 6, 10, 14  
<img width="178" height="76" alt="image" src="https://github.com/user-attachments/assets/afdcd60e-fa6e-4255-8043-4e2ad600d5c0" />  

### 4. Một số bài tập về Numpy  
#### Bài 1:  
<img width="464" height="436" alt="image" src="https://github.com/user-attachments/assets/25d48849-f3c6-40de-bf17-4d3cd3f5ea72" />  

- Cách làm:  
1. Khởi tạo và in ra bàn cờ ban đầu
<img width="206" height="94" alt="image" src="https://github.com/user-attachments/assets/c58835b8-e285-43c4-b4ac-21e460cbbdda" />

2. Kiểm tra hàng ngang, hàng dọc, đường chéo
<img width="346" height="236" alt="image" src="https://github.com/user-attachments/assets/1dbe2a27-844e-46d4-8183-7181e6b94551" />

3. In ra bàn cờ và điền vị trí vào
<img width="362" height="199" alt="image" src="https://github.com/user-attachments/assets/ffc55980-e74c-46eb-b38b-91733fb5b892" />

4. Điền các giá trị vào bàn cờ và kiểm tra thắng thua
<img width="309" height="126" alt="image" src="https://github.com/user-attachments/assets/a24022f4-ad79-414f-8881-7534bb47ad0a" />

- Ví dụ kết quả:
<img width="106" height="549" alt="image" src="https://github.com/user-attachments/assets/32d346f9-84f6-4eb8-879d-886d512f6eaa" />

#### Bài 2:  
<img width="126" height="256" alt="image" src="https://github.com/user-attachments/assets/0d2e1067-6a01-44e2-944f-83b2eeed559e" />  

y = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])


























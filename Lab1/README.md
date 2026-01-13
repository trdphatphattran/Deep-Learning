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
### 1. Pytorch là gì?  
PyTorch là một thư viện mã nguồn mở dùng để lập trình trí tuệ nhân tạo (AI), cụ thể là trong lĩnh vực Học sâu (Deep Learning). Nó được phát triển chủ yếu bởi phòng nghiên cứu AI của Meta (Facebook) và hiện là công cụ yêu thích nhất của các nhà khoa học dữ liệu trên toàn thế giới.  
### 2. Các chức năng chính của Pytorch  
- Tính toán Tensor tối ưu (Thay thế NumPy trên GPU).
- Tự động tính Đạo hàm (Autograd).
- Xây dựng Mạng thần kinh.
- Tối ưu hóa mô hình.
### 3. Cách hoạt động của Pytorch  
Giả sử ta có đoạn code sau  
<img width="694" height="234" alt="image" src="https://github.com/user-attachments/assets/ed9b0018-b4ad-4c6f-bd07-27513278d132" />  

Pytorch làm các nhiệm vụ chính sau:  
#### 1. Chuyển đổi kiểu dữ liệu  
Chức năng quan trọng nhất là biến các mảng Numpy (X_train, y_train, ...) thành các Pytorch Tensors.  
- torch.FloatTensor: chuyển các đặc trưng (features) như chiều dài, chiều rộng cánh hoa về kiểu số thực 32-bit.
- torch.LongTensor: chuyển các nhãn (labels) như 0, 1, 2 (loài hoa) về kiểu số nguyên 64-bit.

#### 2. Quản lý cấu trúc dữ liệu  
Dòng lệnh y_train.reshape(-1, 1) sử dụng khả năng thay đổi hình dạng của PyTorch:  
- Nó biến một mảng phẳng thành một cột dọc.  
- Việc này cực kỳ quan trọng để khi đưa vào mô hình, mỗi nhãn sẽ tương ứng chính xác với một hàng dữ liệu đặc trưng.  

## Phần 2: Các bài tập với Pytorch  
### Bài 1: 
$$
y = 5x^5 + 6x^3 - 3x + 1
$$

Cho biết độ dốc của đa thức trên ở điểm nào?  

<img width="522" height="203" alt="image" src="https://github.com/user-attachments/assets/78ab1a50-df8f-434f-9f33-350781c361c3" />  

- Khi có requires_grad=True, Pytorch đính kèm một bộ lưu trữ dữ liệu đặc biệt. Nó bắt đầu xây dựng một Đồ thị tính toán.  
- Với y.backward(): đây là lúc pytorch thực hiện chức năng chính. Nó đọc ngược từ y về x. Tại mỗi mắt xích, nó áp dụng công thức đạo hàm đã có trong thư viện.
- Quy trình tính toán:
+ 5 x (5.1^4) = 25.
+ 6 x (3.1^2) = 18.
+ -3x = -3.
+ 1' = 0.
Pytorch cộng dồn lại ra kết quả là 40.
<img width="291" height="47" alt="image" src="https://github.com/user-attachments/assets/375c5885-0659-4dcc-b574-89c8b07892f8" />  











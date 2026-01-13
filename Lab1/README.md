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

### Bài 2:  
Tạo một tensor ban đầu có giá trị là 2  
Định nghĩa hàm số và tính gradient  

$$  
y = x^3 + 2x^2 + 5x + 1  
$$  

Hãy tính dy/dx tại giá trị của x?  
Dùng phương pháp Gradient Descent với Learning Rate alpha = 0.1 để cập nhập giá trị x trong 10 vòng lặp  

<img width="696" height="481" alt="image" src="https://github.com/user-attachments/assets/ac808b98-8563-4cc4-8b9b-69b5c932682c" />  

- Pytorch tạo ra một Tensor x = 2.0.
- Tại mỗi vòng lặp, Pytorch tính giá trị của y dựa trên x hiện tại.
<img width="173" height="45" alt="image" src="https://github.com/user-attachments/assets/88b45f20-0c54-46f6-8758-b6d1991634b7" />
- Tại lệnh này, nếu không xóa thì gradient của vòng lặp mới sẽ bị cộng dồn với gradient cũ. Lệnh .zero() giúp chúng xóa sạch dữ liệu cũ.
- Tại lệnh y.backward(), pytorch thực hiện tính đạo hàm tại giá trị x hiện tại. Tại vòng lặp 1 (x = 2): y' = 3(2^2) + 4(2) + 5 = 25. Lưu giá trị 25 vào x.grad.
- Cập nhật giá trị với câu lệnh sau:
<img width="174" height="36" alt="image" src="https://github.com/user-attachments/assets/5b0052dc-b7f0-41c8-a75d-b359b926d21e" />

Kết quả:  
<img width="386" height="263" alt="image" src="https://github.com/user-attachments/assets/77fe11c6-81cc-4188-a025-3d37429c4fbc" />  
















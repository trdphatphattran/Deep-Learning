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

### Bài 3:  
Tạo một tập dữ liệu giả lập với x là số giờ học ngẫu nhiên từ 1 đến 10 và y là số điểm được tính theo công thức: y = 3x + 5 + noise (với noise là một giá trị ngẫu nhiên nhỏ).  

1. Khởi tạo tham số w và b ngẫu nhiên với requires_grad = True.  
2. Tính MSE.  
3. Tính gradient.  
4. Cập nhật tham số w và b bằng Gradient Descent với Learning Rate alpha = 0.01.  
5. Lặp lại quá trình trên trong 100 vòng lặp để xem sự hội tụ của mô hình.  

<img width="824" height="716" alt="image" src="https://github.com/user-attachments/assets/b0f3af04-2a78-4d60-b0b8-ba7277ecf097" />  

- Trước hết, cần khởi tạo các dữ liệu:
+ Dữ liệu thực tế (x, y): tạo ra 100 điểm dữ liệu mà bạn đã biết trước quy luật là y = 3x + 5.  
+ Tham số học (w, b): Đây là cái mà máy tính cần tìm. Lúc đầu, ta cho máy tính đoán đại một con số ngẫu nhiên (randn). requires_grad=True báo cho PyTorch phải theo dõi mọi biến đổi của $w$ và $b$ để tính đạo hàm.
- y_pred = x * w + b: dự đoán, máy tính lấy x (số giờ học) * w rôi cộng sai số b để đưa ra dự đoán (ở vòng lặp đầu tiên w và b ngẫu nhiên nên dự đoán thường sai).
- mse_loss = torch.mean((y_pred - y)**2): đo lường sự sai sót, nó tính khoảng cách giữa điểm dự đoán và điểm thực tế, sau đó bình phương lên lấy giá trị dương.
- mse_loss.backward(): nếu ta thay đổi w hoặc b, thì sai số mse_loss sẽ tăng hoặc giảm.

<img width="189" height="120" alt="image" src="https://github.com/user-attachments/assets/bf8ae940-de21-4f06-9644-d6af10f821e1" />  

- Máy tính thực hiện một bước đi nhỏ (kích thước bước là alpha = 0.01) ngược hướng với độ dốc.
+ Nếu độ dốc dương, nó sẽ giảm w.
+ Nếu độ dốc âm, nó sẽ tăng w.
+ zero_(): sau khi xong, nó sẽ xóa hết các tính năng cũ để chuẩn bị cho vòng lặp mới với dữ liệu mới.

Kết quả:  
<img width="379" height="296" alt="image" src="https://github.com/user-attachments/assets/e5350d6f-3428-4d9e-a42d-35a5e8f25496" />  

### Bài 4:  
#### TH1:  
<img width="194" height="42" alt="image" src="https://github.com/user-attachments/assets/454a1955-0159-411f-b6f2-b01e1e02248a" />  
<img width="93" height="37" alt="image" src="https://github.com/user-attachments/assets/2832debf-cac1-4cd4-8f92-29836bbfae26" />  

#### TH2:  
<img width="164" height="57" alt="image" src="https://github.com/user-attachments/assets/dcdeb287-e731-41f2-ae81-adc977dbc47f" />  
<img width="91" height="41" alt="image" src="https://github.com/user-attachments/assets/5bd926f0-e0af-46e7-b252-a1420c5f2e23" />  

Hãy giải thích sự khác nhau của 2 trường hợp trên?  
























# LAB 4: ANN    
## Thông tin  
Sinh viên: Trần Đại Phát  
MSSV: 2374802010379  
Khóa: 29  
Trường Đại học Văn Lang  
Môn học: Giới thiệu về học sâu  
GVHD: Nguyễn Thái Anh  
Năm học: 2025 - 2026  

## Phần 1: Giới thiệu về ANN  
### 1. ANN là gì?  
- Mạng Nơ-ron nhân tạo (ANN) là một cách để máy tính học từ dữ liệu, giống như dạy một đứa trẻ nhận biết chó và mèo thông qua nhiều hình ảnh.
- Tại sao dùng ANN: Vì nó giỏi tìm quy luật trong dữ liệu phức tạp, như phân biệt điểm thuộc vòng tròn hay vành đai - thứ mà đường thẳng đơn giản không làm được.
- Ví dụ: Nếu chúng ta có 300 điểm và muốn nhận biết điểm nào thuộc nhóm nào, ANN tự học từ dữ liệu thay vì chúng ta phải viết quy tắc bằng tay.

### 2. Cấu trúc của ANN?  
- ANN giống như một đội làm việc có 3 nhóm:
  - Lớp đầu vào: Nhóm nhận thông tin. Ở đây có 2 người (2 nút), một người nhận biết x, một người nhận biết y.
    - Ví dụ: Điểm (1, 2) vào, nút 1 nhận 1, nút 2 nhận 2.
  - Lớp ẩn: Nhóm phân tích thông tin, có 4 người (4 nút) để "nghĩ" sâu hơn.
    - Có thể thay đổi số nút tùy vào bài toán chứ không nhất thiết phải là 4 nút.
    - Ví dụ: Họ nhìn (1, 2) và tính xem nó gần gốc tọa độ hay xa.
  - Lớp đầu ra: Nhóm đưa ra kết quả cuối, chỉ cần 1 người (1 nút) nói "lớp 0" hay "lớp 1".
    - Ví dụ: Kết quả 0.7 nghĩa là nghiêng về lớp 1.  
   
- Hình ảnh minh họa:  
  - Sơ đồ:  
    - Bên trái: 2 vòng tròn (đầu vào x và y).  
    - Giữa: 4 vòng tròn nối với 2 vòng tròn trước bằng mũi tên (lớp ẩn).  
    - Bên phải: 1 vòng tròn nối với 4 vòng tròn giữa (đầu ra).  
  - Mỗi mũi tên là "trọng số" - độ quan trọng của thông tin.  
  - Hình minh họa:  
<img width="658" height="446" alt="image" src="https://github.com/user-attachments/assets/81bfb306-fb8b-4680-9d84-cd7815f7f7b7" />

## Phần 2: Bài tập với ANN  
### Bài 1:  
1. **Tăng số nút trong lớp ẩn**:  
   - Sửa lớp ẩn từ 4 nút thành 8 nút trong code định nghĩa `ANN`.  
   - Huấn luyện lại mô hình với cùng dữ liệu `X_train`, `y_train` từ lab (100 epochs).  
   - Ghi lại giá trị mất mát cuối cùng (`loss`) và độ chính xác trên tập kiểm tra (`X_test`, `y_test`).

Cách làm:  
- Tăng trọng số trong lớp ẩn:  
<img width="274" height="256" alt="image" src="https://github.com/user-attachments/assets/a685e6c1-013b-4bee-a137-251d3eb2e3ef" />

- Huấn luyện mô hình ở 100 epochs:
<img width="510" height="184" alt="image" src="https://github.com/user-attachments/assets/bedfda01-4093-45d6-9271-4341be8b9ebc" />

--> Kết quả:  
<img width="223" height="119" alt="image" src="https://github.com/user-attachments/assets/96b6550d-2489-4670-8b3c-385e8a981bd4" />  

2. **Thêm một lớp ẩn**:  
   - Thêm một lớp ẩn thứ hai với 6 nút, dùng ReLU làm hàm kích hoạt. Cấu trúc mới:  
     - Đầu vào: 2 nút.  
     - Lớp ẩn 1: 8 nút (ReLU).  
     - Lớp ẩn 2: 6 nút (ReLU).  
     - Đầu ra: 1 nút (Sigmoid).  
   - Huấn luyện lại mô hình (100 epochs).  
   - Ghi lại giá trị mất mát cuối cùng và độ chính xác.
  
Cách làm:  
- Thêm vào một lớp ẩn:
<img width="275" height="329" alt="image" src="https://github.com/user-attachments/assets/6c72d69f-c3f6-4b48-89da-79e9a964967a" />

- Huấn luyện mô hình ở 100 epochs:
<img width="507" height="182" alt="image" src="https://github.com/user-attachments/assets/92a99a4d-3284-43e2-9c4b-b81d4043a454" />

--> Kết quả:  
<img width="221" height="111" alt="image" src="https://github.com/user-attachments/assets/7ec84e64-7e39-44f6-a821-89e80f213056" />  

3. **Câu hỏi**:  
   - So sánh kết quả của 3 mô hình (4 nút, 8 nút, 8+6 nút):  
     - Mất mát cuối cùng thay đổi thế nào?  
     - Độ chính xác có cải thiện không? Tại sao bạn nghĩ vậy?
<img width="1340" height="171" alt="image" src="https://github.com/user-attachments/assets/0be7f02f-0dff-4bf2-8fd2-0f25f21731f4" />

### Bài 2:  
1. **Dùng BCEWithLogitsLoss thay cho BCELoss**:  
   - Thay `nn.BCELoss()` bằng `nn.BCEWithLogitsLoss()`.  
   - Xóa hàm Sigmoid khỏi lớp đầu ra của mô hình (vì `BCEWithLogitsLoss` tự xử lý).  
   - Huấn luyện lại mô hình với cấu trúc ban đầu (2-4-1, 100 epochs).  
   - Ghi lại mất mát cuối cùng và độ chính xác.
  
Cách làm:  
- Thay đổi nn.BCELoss() thành nn.BCEWithLogitLoss():
<img width="373" height="41" alt="image" src="https://github.com/user-attachments/assets/6c022ace-0aa2-4180-90c0-ee0d313b307f" />

- Xóa hàm sigmoid khỏi lớp đầu vào:
<img width="278" height="216" alt="image" src="https://github.com/user-attachments/assets/be465cf4-fb26-4271-8e63-9c5cda6fa21b" />

- Huấn luyện theo cấu trúc ban đầu:
<img width="508" height="188" alt="image" src="https://github.com/user-attachments/assets/6cd8de4f-c660-441f-96b9-50d6c3088471" />

--> Kết quả:  
<img width="224" height="119" alt="image" src="https://github.com/user-attachments/assets/8da27249-8603-4b26-bf68-b2127b050796" />  

2. **Thay Adam bằng SGD**:  
   - Dùng lại cấu trúc ban đầu (2-4-1) với `nn.BCELoss()`.  
   - Thay `optim.Adam` bằng `optim.SGD` với `lr=0.01`.  
   - Huấn luyện lại (100 epochs).  
   - Ghi lại mất mát cuối cùng và độ chính xác.

3. **Câu hỏi**:  
   - So sánh kết quả:  
     - `BCEWithLogitsLoss` có khác gì so với `BCELoss` về mất mát và độ chính xác? Tại sao?  
     - `SGD` so với `Adam`: Mất mát giảm nhanh hơn hay chậm hơn? Độ chính xác thay đổi ra sao?  





  

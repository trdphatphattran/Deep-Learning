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
  
Cách làm:  
- Thay optim.Adam thành optim.SGD với learning rate = 0.01:
<img width="368" height="39" alt="image" src="https://github.com/user-attachments/assets/0b4b1093-ef9e-4f10-9065-68f400c5720e" />

- Huấn luyện với 100 epochs:
<img width="510" height="183" alt="image" src="https://github.com/user-attachments/assets/53af417d-ffd1-45b4-af3f-cf38776c9270" />

--> Kết quả:  
<img width="223" height="109" alt="image" src="https://github.com/user-attachments/assets/1de9966f-971e-45b4-ae0e-8531c47b95d1" />  

3. **Câu hỏi**:  
   - So sánh kết quả:  
     - `BCEWithLogitsLoss` có khác gì so với `BCELoss` về mất mát và độ chính xác? Tại sao?  
     - `SGD` so với `Adam`: Mất mát giảm nhanh hơn hay chậm hơn? Độ chính xác thay đổi ra sao?
<img width="1365" height="172" alt="image" src="https://github.com/user-attachments/assets/a60194b4-cf45-4001-b822-5df8d7e332dc" />

### Bài 3:  
1. **Vẽ đồ thị mất mát**:  
   - Sửa code huấn luyện để lưu giá trị mất mát (`loss`) sau mỗi epoch vào một danh sách.  
   - Vẽ đồ thị mất mát theo epoch cho 3 trường hợp:  
     - Cấu trúc ban đầu (2-4-1, Adam, BCELoss).  
     - Cấu trúc 2-8-1 (Adam, BCELoss).  
     - Cấu trúc 2-4-1 (SGD, BCELoss).  
   - Dùng `matplotlib` để vẽ 3 đường trên cùng một đồ thị, thêm chú thích (`legend`).

Cách làm:  
- Vẽ đồ thị 2-4-1, Adam, BCELoss:
<img width="919" height="41" alt="image" src="https://github.com/user-attachments/assets/dd052ef9-c587-4acc-8e4e-8f75ec4430db" />

- Vẽ đồ thị 2-8-1, Adam, BCELoss:
<img width="920" height="39" alt="image" src="https://github.com/user-attachments/assets/35abf103-1de6-46d9-b347-472788c20b7e" />

- Vẽ đồ thị 2-4-1, SGD, BCELoss:
<img width="887" height="40" alt="image" src="https://github.com/user-attachments/assets/f1a07bb2-5a58-47f5-8f67-541d35fa88ce" />

- Dùng matplotlib để vẽ đồ thị, kết quả:
<img width="858" height="549" alt="image" src="https://github.com/user-attachments/assets/9c40babd-aede-4557-b13e-d5c3202be3fd" />  

2. **Câu hỏi**:  
   - Quan sát đồ thị:  
     - Mất mát giảm nhanh nhất ở trường hợp nào? Chậm nhất ở đâu?  
     - Có trường hợp nào mất mát không giảm đều không (dao động)? Giải thích tại sao.
<img width="1397" height="104" alt="image" src="https://github.com/user-attachments/assets/d078f3b8-8be3-477a-acf2-a2dfec4bde43" />

## Phần 3: Cách sử dụng  
1. Cài pytorch (dùng terminal hoặc cmd):  
pip3 install numpy
pip3 install pandas  

2. Import thư viện  
import numpy as np
import pandas as pd

3. Thay đổi tham số
- Ở mỗi bài tập, ví dụ có thể thay đổi hàm, tham số, ... khác nhau để xem kết quả.

## Phần 5: Tài liệu tham khảo  
1. Tài liệu lý thuyết + thực hành Van Lang University.
2. NumPy Documentation (https://numpy.org/doc/).
3. Pandas Documentation (https://pandas.pydata.org/docs/).  
4. Tài liệu hướng dẫn của W3Schools (https://www.w3schools.com/python/numpy/default.asp). 







  

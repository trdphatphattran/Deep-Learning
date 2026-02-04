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

  

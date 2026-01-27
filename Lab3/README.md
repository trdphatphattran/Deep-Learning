# LAB 3: PANDAS  
## Thông tin  
Sinh viên: Trần Đại Phát  
MSSV: 2374802010379  
Khóa: 29  
Trường Đại học Văn Lang  
Môn học: Giới thiệu về học sâu  
GVHD: Nguyễn Thái Anh  
Năm học: 2025 - 2026  

## Phần 1: Giới thiệu về Pandas  
- Pandas là một thư viện tương đối mới, được xây dựng dựa trên NumPy, và cung cấp một cách cài đặt hiệu quả cho cấu trúc dữ liệu DataFrame.  
- DataFrame về bản chất là các mảng đa chiều (multidimensional arrays) có gắn nhãn hàng (row labels) và nhãn cột (column labels), đồng thời thường chứa dữ liệu không đồng nhất về kiểu (heterogeneous types) và/hoặc giá trị bị thiếu (missing data).  
- Bằng việc cung cấp một giao diện lưu trữ thuận tiện cho dữ liệu có nhãn (labeled data), Pandas triển khai nhiều phép toán xử lý dữ liệu mạnh mẽ, quen thuộc với người dùng các hệ quản trị cơ sở dữ liệu (database frameworks) cũng như các chương trình bảng tính (spreadsheet programs).  
- Mặc dù cấu trúc dữ liệu ndarray của NumPy cung cấp các tính năng nền tảng quan trọng, nhưng những hạn chế của nó trở nên rõ ràng khi cần sự linh hoạt cao hơn, chẳng hạn như gắn nhãn cho dữ liệu, xử lý dữ liệu bị thiếu, v.v.  
- Pandas cung cấp khả năng truy cập và xử lý hiệu quả các tác vụ “data munging” (tiền xử lý, làm sạch và biến đổi dữ liệu) — những công việc chiếm phần lớn thời gian làm việc.

## Phần 2: Một số lý thuyết với Pandas  
### Series  
- Pandas Series là một mảng một chiều (one-dimensional array) có chỉ mục (index).  
- Series có thể được khởi tạo từ một danh sách (list) hoặc một mảng (array) như sau:
<img width="339" height="177" alt="image" src="https://github.com/user-attachments/assets/74991a09-90ac-4374-8467-6809d9875fee" />

- Tạo Series từ Numpy array:
<img width="256" height="231" alt="image" src="https://github.com/user-attachments/assets/4cc6a42f-85e2-460f-8886-c54385cc7315" />

- Attributes:
<img width="265" height="210" alt="image" src="https://github.com/user-attachments/assets/10f13c8f-fdaa-4b9e-b6e6-b0831b1c5068" />

- Indexing:
<img width="130" height="249" alt="image" src="https://github.com/user-attachments/assets/ad437685-0f6b-4b42-acd4-3778ed2b6cdd" />

- Letter indexing:
<img width="570" height="264" alt="image" src="https://github.com/user-attachments/assets/d23f4bb3-13eb-4910-8a53-84e95ed962e2" />

- Combined indexing:
<img width="366" height="194" alt="image" src="https://github.com/user-attachments/assets/925aa043-a8f5-428c-9bbb-fbc0788ae2f7" />

- Pandas and dictionary:

Hệ thống chỉ mục của Pandas có cấu trúc tương tự dictionary.  
Việc tạo một đối tượng Pandas từ dictionary là điều hoàn toàn dễ hiểu.  








  

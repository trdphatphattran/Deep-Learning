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

... Ngoài ra, có thể xem thêm các ví dụ trong file code đính kèm.  

## Phần 3: Một số bài tập với Pandas  
### Bài 1: Cho 2 ví dụ sau và giải thích sự khác nhau  
#### Ví dụ 1:  
```python
import numpy as np
numpy_arr = np.arange(5)
data_pd = pd.Series(numpy_arr)
data_pd
data_pd[-1]
```
--> Kết quả ra lỗi  
#### Ví dụ 2:  
```python
data_pd = pd.Series([1.25, 2.0, 0.75, 1.0], index = ['a', 'b', 'c', 'd'])
data_pd[-1]
```
--> Kết quả: 1.0  
#### Giải thích:  
Ở ví dụ 1: Series với index là số nguyên:  
- Khi tạo một series mà không chỉ định index thì pandas sẽ mặc định tạo ra các số nguyên (0, 1, 2, 3, 4).
- Khi index là các số nguyên, pandas sẽ ưu tiên hiểu các con số trong [] là Label-based indexing.
- Pandas sẽ tìm kiếm label có tên là -1. Vì trong các số nguyên không có số -1 nên nó sẽ báo lỗi.

Ở ví dụ 2: Series với index là kí tự  
- Khi index của series không phải là số nguyên ('a', 'b', 'c', 'd'), pandas sẽ linh hoạt hơn.
- Nếu label không phải là số, pandas sẽ hiểu sử dụng position-based indexing.
- Trong các số, -1 là đại diện cho số cuối, nên nó trả về kết quả là 1.

## Phần 4: Cách sử dụng  
1. Cài pytorch (dùng terminal hoặc cmd):  
pip3 install numpy
pip3 install pandas  

3. Import thư viện  
import numpy as np
import pandas as pd

4. Thay đổi tham số
- Ở mỗi bài tập, ví dụ có thể thay đổi hàm, tham số, ... khác nhau để xem kết quả.

## Phần 5: Tài liệu tham khảo  
1. Tài liệu lý thuyết + thực hành Van Lang University.
2. NumPy Documentation (https://numpy.org/doc/).
3. Pandas Documentation (https://pandas.pydata.org/docs/).  
4. Tài liệu hướng dẫn của W3Schools (https://www.w3schools.com/python/numpy/default.asp). 


  












  

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
### Cho 2 ví dụ sau và giải thích sự khác nhau  
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

### Bài 1:  
Tải file howlongwelive.csv và thực hiện các yêu cầu bên dưới:  
#### 1. In ra 2 dòng đầu và 2 dòng cuối của DataFrame  
- Chúng ta sử dụng head() để in ra các dòng đầu và tail() để in ra các dòng cuối.
<img width="128" height="37" alt="image" src="https://github.com/user-attachments/assets/4af80aec-de75-4b79-96c1-1523a2f6cbed" />

--> Kết quả:  
<img width="597" height="531" alt="image" src="https://github.com/user-attachments/assets/15f29e74-c235-4e80-9167-7bc765abda98" />  

#### 2. In ra kích thước (shape) của DataFrame  
- Chúng ta sử dụng shape để in ra kích thước.
<img width="114" height="23" alt="image" src="https://github.com/user-attachments/assets/ac08732f-8ce5-4df3-af9e-3218d85cc8b4" />

--> Kết quả:  
<img width="80" height="24" alt="image" src="https://github.com/user-attachments/assets/51743d52-ed16-44b2-ba32-9332d098fb4b" />  

#### 3. In ra tên các đặc trưng (các cột) của DataFrame.  
- Chúng ta sử dụng columns để in ra các cột đặc trưng.
<img width="131" height="21" alt="image" src="https://github.com/user-attachments/assets/6f514e25-4fee-45fd-b9cd-48828ad8dcbd" />

--> Kết quả:  
<img width="580" height="134" alt="image" src="https://github.com/user-attachments/assets/d10fabb1-4ac4-42ff-864c-01c072a65c95" />  

#### 4. In ra bảng thống kê mô tả bằng hàm .describe().  
- Chúng ta sử dụng describe() để in ra bảng thống kê mô tả.
<img width="151" height="22" alt="image" src="https://github.com/user-attachments/assets/4c669b03-ec56-4cdc-9692-709e1de36fbf" />

--> Kết quả:  
<img width="560" height="549" alt="image" src="https://github.com/user-attachments/assets/294d4048-f6c9-487c-9832-67a8017ab80a" />  

#### 5. Vì cột Hepatitis B có rất nhiều giá trị thiếu (NaN) và có mức tương quan cao với Diphtheria, hãy xóa cột Hepatitis B. Đồng thời xóa cột Population do có quá nhiều giá trị NaN.  
- Chúng ta sử dụng drop để xóa các cột không cần thiết.
<img width="375" height="20" alt="image" src="https://github.com/user-attachments/assets/cbf8987f-caa6-46b0-bbcd-a5a5fa75a470" />

#### 6. Chuyển đổi cột Status sang dạng số 0 cho Developing và 1 cho Developed  
- Chúng ta sử dụng replace để thay đổi định dạng.
<img width="523" height="23" alt="image" src="https://github.com/user-attachments/assets/ce63a353-c851-4047-9283-6cc04e439cbd" />

#### 7. Đổi tên cột thinness 1-19 years thành thinness 10-19 years.  
- Chúng ta sử dụng rename để thay đổi tên cột.
<img width="520" height="19" alt="image" src="https://github.com/user-attachments/assets/66105a03-5206-4882-a4f2-4db475a14d96" />

#### 8. Lấy tất cả các cột ngoại trừ Life Expectancy, chuyển sang mảng NumPy và lưu vào biến X.  
- Chúng ta sử dụng drop để loại bỏ cột Life Expectancy đi và dùng values để chuyển sang mảng numpy.  
<img width="358" height="20" alt="image" src="https://github.com/user-attachments/assets/7e69bc75-3fe9-4ab5-acca-8d008ab10b9c" />

--> Kết quả:  
<img width="82" height="29" alt="image" src="https://github.com/user-attachments/assets/f16f6d9d-7518-4071-be0b-62a53436157b" />  

#### 9. Lấy cột Life Expectancy, chuyển sang mảng NumPy và lưu vào biến y.  
- Chúng ta lấy cột Life expectancy và dùng values để chuyển sang mảng numpy.
<img width="246" height="18" alt="image" src="https://github.com/user-attachments/assets/803bd552-1bf5-4baf-9973-783bf53a6499" />

--> Kết quả:  
<img width="57" height="26" alt="image" src="https://github.com/user-attachments/assets/2dd8c69b-45ba-4bef-bc61-e30a32dc2654" />  

### Bài 2:  
#### 1. Kiểm tra xem mỗi cột có bao nhiêu giá trị bị thiếu (missing data/ NaN)  
- Chúng ta sử dụng isnull() để xác định ô trống và sum() để cộng tất cả cột.
<img width="178" height="19" alt="image" src="https://github.com/user-attachments/assets/146b9806-1969-4aa3-87ae-3c4f40dce286" />

--> Kết quả:  
<img width="288" height="393" alt="image" src="https://github.com/user-attachments/assets/2a5dc61b-6fb1-4771-ac99-e5ae8f39a8dc" />  

#### 2. Xử lý tất cả dữ liệu thiếu bằng cách thay thế bằng giá trị trung bình (mean) của từng cột tương ứng  
- Chúng ta sử dụng mean để tính giá trị trung bình của từng cột tương ứng.
<img width="310" height="23" alt="image" src="https://github.com/user-attachments/assets/ff559d35-d6d6-48e0-b1fa-94e1eed6ed70" />

#### 3. Thực hiện groupby theo quốc gia (Country): Quốc gia nào có tuổi thọ trung bình thấp nhất? Quốc gia nào có tuổi thọ trung bình cao nhất?  
- Chúng ta sử dụng groupby theo Country và tìm giá trị min/max của cột Life expectancy.
<img width="431" height="77" alt="image" src="https://github.com/user-attachments/assets/2dea1d0d-57ec-46f6-bdbe-b50fa0c1d162" />

--> Kết quả:  
<img width="205" height="45" alt="image" src="https://github.com/user-attachments/assets/e0bfbf4a-092d-41fa-9051-b33229607ffc" />  

#### 4. Thực hiện groupby theo Status (Developed / Developing). Có sự khác biệt rõ rệt nào về tuổi thọ trung bình giữa các quốc gia phát triển và đang phát triển hay không?  
- Vẫn tiếp tục sử dụng groupby nhưng theo Status.
<img width="413" height="20" alt="image" src="https://github.com/user-attachments/assets/1a996636-40af-465b-a56a-e315f19d6688" />

--> Kết quả:  
<img width="295" height="81" alt="image" src="https://github.com/user-attachments/assets/a9d189e9-b408-463c-800f-9d5be2a00e49" />  

- Các quốc gia developed (1) có độ tuổi trung bình cao hơn so với developing (0).

#### 5. Tạo một DataFrame mới bằng tay gồm 2 cột: Cột thứ nhất là ID, có giá trị giống với cột Country, Cột thứ hai là Noise_level, điền giá trị ngẫu nhiên (tùy ý)  
- Chúng ta dùng unique để lấy danh sách các quốc gia duy nhất và random để tạo giá trị ngẫu nhiên.
<img width="361" height="110" alt="image" src="https://github.com/user-attachments/assets/088e28d7-8dda-41a3-b058-4332a828707b" />

--> Kết quả:  
<img width="400" height="265" alt="image" src="https://github.com/user-attachments/assets/72f57a42-b97c-4916-8e4c-1a7313a16f54" />  

#### 6. Gộp (merge) hai DataFrame lại với nhau dựa trên cột ID.  
- Dùng merge để thực hiện việc gộp 2 dataframe lại với nhau, đó là df và df_noise.
<img width="321" height="61" alt="image" src="https://github.com/user-attachments/assets/c6a0d30b-c715-48e3-a9c7-72d3afe722a6" />

--> Kết quả:  
<img width="333" height="256" alt="image" src="https://github.com/user-attachments/assets/e95c5940-e506-4908-a11b-13642a6d2007" />  

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


  












  

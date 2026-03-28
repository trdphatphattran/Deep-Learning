# LAB 6: MẠNG NƠ-RON HỒI QUY (RNN)  
## Thông tin:  
Sinh viên: Trần Đại Phát  
MSSV: 2374802010379  
Môn học: Giới thiệu về học sâu  
GVHD: Nguyễn Thái Anh  
Năm học: 2025 - 2026    

## Phần 1: Lý thuyết về RNN  
## 1. Giới thiệu về RNN

RNN là một loại mạng nơ-ron nhân tạo được thiết kế để xử lý *dữ liệu tuần tự* (sequential data), như chuỗi thời gian, văn bản, hoặc âm thanh. Điểm đặc biệt của RNN là khả năng "ghi nhớ" thông tin từ các bước trước đó nhờ cấu trúc vòng lặp.

### Tại sao cần RNN?

- Trong mạng nơ-ron truyền thống, mỗi đầu vào là độc lập, không có mối quan hệ với các đầu vào trước đó.
- Với RNN, thông tin từ bước trước được truyền sang bước hiện tại, cho phép học các mẫu trong chuỗi dữ liệu.

### Ứng dụng của RNN

- Dự đoán chuỗi thời gian (ví dụ: giá cổ phiếu).
- Xử lý ngôn ngữ tự nhiên (NLP): dịch máy, sinh văn bản.
- Nhận diện giọng nói.

## 2. Cấu trúc của RNN

RNN có một vòng lặp trong kiến trúc, cho phép trạng thái ẩn (hidden state) được cập nhật qua từng bước thời gian. Các thành phần cơ bản:

- **Đầu vào:** $x_t$ tại thời điểm $t$.
- **Trạng thái ẩn:** $h_t$, lưu trữ thông tin từ các bước trước.
- **Đầu ra:** $y_t$ tại thời điểm $t$.

### Sơ đồ cơ bản

<img width="1058" height="661" alt="image" src="https://github.com/user-attachments/assets/ad9fad3b-3d43-41ed-8f90-363ed6cb0b37" />  



## 3. Công thức toán học của RNN

Trong RNN, tại mỗi thời điểm $t$, mô hình nhận đầu vào $x_t$, kết hợp với trạng thái ẩn từ bước trước $h_{t-1}$, rồi tạo ra trạng thái ẩn mới $h_t$. Từ trạng thái ẩn này, mô hình tiếp tục tính đầu ra $y_t$.

### 3.1. Công thức tính trạng thái ẩn

Trạng thái ẩn tại thời điểm $t$ được tính theo công thức:

$$
h_t = f(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
$$

### 3.2. Công thức tính đầu ra

Sau khi có trạng thái ẩn $h_t$, đầu ra tại thời điểm $t$ được tính theo công thức:

$$
y_t = g(W_{hy}h_t + b_y)
$$

## 4. Huấn luyện RNN (Backpropagation Through Time - BPTT)

RNN được huấn luyện bằng thuật toán **lan truyền ngược qua thời gian** (*Backpropagation Through Time - BPTT*).  
Ý tưởng chính là "mở" mạng RNN theo từng bước thời gian, sau đó áp dụng lan truyền ngược tương tự như trong mạng nơ-ron truyền thống, nhưng trên toàn bộ chuỗi.

Giả sử chuỗi có độ dài $T$, tổng hàm mất mát được tính như sau:

$$
L = \sum_{t=1}^{T} L_t
$$

Về mặt trực quan, gradient đối với ma trận trọng số hồi quy $W_{hh}$ có thể được biểu diễn như sau:

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}
$$

Điều này cho thấy trọng số $W_{hh}$ được chia sẻ qua tất cả các bước thời gian, nên gradient cuối cùng là tổng đóng góp từ toàn bộ chuỗi.  

## Phần 2: Bài tập với RNN  
### Bài 1:  
Thực hiện các yêu cầu sau:

- Chuẩn hóa dữ liệu về khoảng `[0, 1]` hoặc dùng chuẩn hóa z-score.
- Tạo các chuỗi con với độ dài `seq_length = 20`.
- Chia dữ liệu theo tỉ lệ:
  - 70% cho tập huấn luyện
  - 15% cho tập validation
  - 15% cho tập kiểm tra
 
```python
data = (data - data.min()) / (data.max() - data.min())

data = torch.FloatTensor(data).unsqueeze(1)

seq_length = 20
X, y = create_sequences(data_multi, seq_length)

total_size = len(X)

train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]

X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]
```
### Bài 2:  
Xây dựng mô hình RNN bằng PyTorch với cấu hình gợi ý như sau:

- `input_size = 3`
- `hidden_size = 32`
- `output_size = 1`

Yêu cầu trong quá trình huấn luyện:

- Sử dụng hàm mất mát `MSELoss`.
- Sử dụng bộ tối ưu `Adam`.
- Huấn luyện mô hình trong `150 epochs`.
- Lưu lại các giá trị sau:
  - `train loss`
  - `validation loss`
 
```python
class RNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, output_size=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])  
        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

model = RNN(input_size=3, hidden_size=32, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 150
batch_size = 16

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
........
```
--> Kết quả:  
<img width="416" height="384" alt="image" src="https://github.com/user-attachments/assets/961b1925-8607-42ee-aa72-43c8a01f2a74" />  

### Bài 3:  
Sau khi huấn luyện xong, thực hiện các yêu cầu sau:

- Dự đoán trên tập test.
- Tính các chỉ số đánh giá:
  - `MSE`
  - `MAE`
- Vẽ biểu đồ so sánh:
  - giá trị thực
  - giá trị dự đoán
 
```python
model.eval()
predictions = []

with torch.no_grad():
    current_input = X_test[0].unsqueeze(0)
    hidden = model.init_hidden(1)

    for i in range(len(X_test)):
        output, hidden = model(current_input, hidden)
        pred_value = output.item()
        predictions.append(pred_value)
        
        new_step = torch.zeros(1, 1, 3)
        new_step[0, 0, 0] = pred_value           
        new_step[0, 0, 1] = current_input[0, -1, 0] 
        new_step[0, 0, 2] = current_input[0, -1, 1] 

        current_input = torch.cat((current_input[:, 1:, :], new_step), dim=1)

actual_values = y_test[:, 0].numpy() 
predictions_arr = np.array(predictions)

mse = mean_squared_error(actual_values, predictions_arr)
mae = mean_absolute_error(actual_values, predictions_arr)
```

Để đánh giá hiệu suất của mô hình RNN trong việc dự đoán chuỗi thời gian, chúng ta sử dụng hai chỉ số đo lường sai số phổ biến:

#### **a. Sai số bình phương trung bình (Mean Squared Error - MSE)**
MSE đo lường trung bình bình phương của các khoảng sai lệch giữa giá trị thực tế và giá trị dự đoán. Chỉ số này nhấn mạnh các sai số lớn do có lũy thừa bậc 2.

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

#### **b. Sai số tuyệt đối trung bình (Mean Absolute Error - MAE)**
MAE đo lường giá trị trung bình của các khoảng cách tuyệt đối giữa giá trị dự đoán và thực tế. Chỉ số này cho biết trung bình mô hình dự đoán lệch bao nhiêu đơn vị so với thực tế.

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Trong đó:**
* $n$: Tổng số mẫu dữ liệu trong tập kiểm tra (test set).
* $y_i$: Giá trị thực tế tại thời điểm $i$.
* $\hat{y}_i$: Giá trị mà mô hình dự đoán tại thời điểm $i$.

--> Kết quả:  
<img width="1203" height="711" alt="image" src="https://github.com/user-attachments/assets/00ef8d13-5ace-4b2b-8a0e-de032017f2e7" />  

### Bài 4:  
- Thử ít nhất **2 giá trị khác nhau của `seq_length`** (ví dụ: 10, 20, 30) và nhận xét xem độ dài chuỗi ảnh hưởng như thế nào đến kết quả dự đoán.
- Thay đổi số lượng **hidden_size** (ví dụ: 16, 32, 64) để so sánh hiệu quả của mô hình.
- Tăng số epoch huấn luyện và quan sát sự thay đổi của `train loss` và `validation loss`.
- Thử dự đoán **nhiều bước tiếp theo** (ví dụ: 3 bước hoặc 5 bước) thay vì chỉ dự đoán 1 bước.
- Sử dụng **dropout** hoặc tăng số tầng của mô hình RNN để kiểm tra xem mô hình có cải thiện hay không.
- Thay đổi **learning rate** của bộ tối ưu Adam và so sánh kết quả.
- Vẽ thêm biểu đồ thể hiện sai số dự đoán theo từng thời điểm trên tập test.
- Viết phần nhận xét ngắn về những yếu tố làm mô hình dự đoán tốt hơn hoặc kém hơn.

#### Giả sử thay đổi seq_length = 30, hidden_size = 64, lr = 0.1, num_epochs = 200, dự đoán 5 bước, thêm 2 tầng RNN, dropout 0.2  
```python
seq_length = 30
X, y = create_sequences(data_multi, seq_length)
&
class RNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])  
        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size)

model = RNN(input_size=3, hidden_size=64, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 200
batch_size = 16

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
......
```
--> Kết quả:  
<img width="600" height="776" alt="image" src="https://github.com/user-attachments/assets/b7037e50-48e3-44e9-a21b-419a642bc9a1" />  

<img width="753" height="623" alt="image" src="https://github.com/user-attachments/assets/fc36160a-f669-4291-8cd8-b8a1bc127a39" />

#### Giả sử thay đổi seq_length = 20, hidden_size = 32, lr = 0.0001, num_epochs = 150, dự đoán 3 bước, thêm 2 tầng RNN, dropout 0.1  
```python
seq_length = 20
X, y = create_sequences(data_multi, seq_length)
&
class RNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, output_size=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, dropout=0.1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])  
        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size)

model = RNN(input_size=3, hidden_size=32, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 150
batch_size = 16

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
.......
```
--> Kết quả:  
<img width="590" height="641" alt="image" src="https://github.com/user-attachments/assets/874a5458-da64-459e-bcea-e10ded14c42f" />  

<img width="752" height="618" alt="image" src="https://github.com/user-attachments/assets/0edd051c-61a3-444d-8b97-3635d87059e0" />  

#### Nhận xét:  
1. Yếu tố giúp mô hình tốt hơn:
- Tăng seq_length: Mở rộng "cửa sổ quan sát", giúp RNN nắm bắt các phụ thuộc xa hơn (Long-term dependencies) và quy luật chu kỳ của dữ liệu.  
- Tăng hidden_size: Nâng cao khả năng biểu diễn (Representation capacity), giúp mô hình ghi nhớ được nhiều đặc trưng phức tạp hơn.  
- Thêm tầng RNN (num_layers=2): Tạo cấu trúc học sâu, cho phép mô hình trích xuất đặc trưng theo phân cấp (từ thô đến tinh), tối ưu hóa việc học các hàm phi tuyến.  

2. Yếu tố làm mô hình kém đi:  
- Lỗi tích lũy (Recursive Error): Khi dự đoán nhiều bước, sai số của bước trước làm đầu vào cho bước sau, khiến sai số bị phóng đại dần (tích lũy) theo thời gian.  
- Tốc độ học (lr) quá cao: Khiến thuật toán tối ưu bị "nhảy" qua điểm cực tiểu của hàm Loss, gây ra hiện tượng mất ổn định hoặc vụ nổ Gradient.  
- Lạm dụng Dropout (0.2): Với dữ liệu đơn giản, Dropout quá cao có thể gây thiếu khớp (Underfitting) do làm gián đoạn dòng chảy thông tin quan trọng giữa các lớp.  

## Phần 3: Cách sử dụng  
1. Cài pytorch (dùng terminal hoặc cmd):  
pip3 install numpy
pip3 install pandas  

2. Import thư viện  
import numpy as np
import pandas as pd

3. Thay đổi tham số
- Ở mỗi bài tập, ví dụ có thể thay đổi hàm, tham số, ... khác nhau để xem kết quả.

## Phần 4: Tài liệu tham khảo  
1. Tài liệu lý thuyết + thực hành Van Lang University.
2. NumPy Documentation (https://numpy.org/doc/).
3. Pandas Documentation (https://pandas.pydata.org/docs/).  
4. Tài liệu hướng dẫn của W3Schools (https://www.w3schools.com/python/numpy/default.asp). 
5. Recurrent Neural Network Tutorial (RNN).  


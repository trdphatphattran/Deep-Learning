# LAB 5: THUẬT TOÁN CONVOLUTIONAL NEURAL NETWORK (CNN)  
## Thông tin:  
Sinh viên: Trần Đại Phát  
MSSV: 2374802010379  
Môn học: Giới thiệu về học sâu  
GVHD: Nguyễn Thái Anh  
Năm học: 2025 - 2026    
## Phần 1: Lý thuyết về CNN
### 1. CNN là gì?  
CNN (Convolutional Neural Network) là một loại mạng nơ-ron nhân tạo giúp máy tính "nhìn" và hiểu ảnh, tương tự cách con người nhận diện vật thể trong đời thực. Thay vì xem toàn bộ ảnh một lúc như mạng nơ-ron thông thường (fully connected), CNN chia nhỏ ảnh ra, tìm các đặc trưng như đường thẳng, góc, vòng tròn, rồi ghép lại để đoán xem ảnh đó là gì.  
Ví dụ: Khi ta nhìn một con mèo, không cần xem hết cả ảnh ngay lập tức. Ta nhận ra tai mèo (hình tam giác), mắt mèo (hình tròn), ria mèo (đường thẳng), rồi kết luận "Đây là mèo". CNN cũng làm như vậy bằng cách dùng các "kính lúp" nhỏ quét qua ảnh từng phần một.  

### 2. Các thành phần chính của CNN  
#### 2.1 Tầng tích chập (Convolution Layer)  
Đây là bước quan trọng nhất, giống như "đôi mắt" của CNN, giúp tìm các đặc trưng nhỏ trong ảnh như cạnh, góc, hoặc đường cong.  
Ý tưởng cơ bản  
- Chúng ta có 1 ảnh, giả sử kích thước là 6x6 pixel.
- Dùng 1 bộ lọc (filter/kernel), ví dụ 3x3, như một "kính lúp" nhỏ để quét qua ảnh.
- Kết quả là 1 feature map, cho biết chỗ nào trong ảnh có đặc trưng mà bộ lọc tìm được.
##### Công thức tích chập  
![image](https://github.com/user-attachments/assets/cbe1bfc9-70cc-4ff5-8ec8-4613e3dad611)  
Trong đó:  
- I: Ảnh đầu vào.
- K: Bộ lọc (kernel/filter).
- F: Kích thước bộ lọc (ví dụ F = 3 nếu là 3x3).
- S(i, j): Gía trị tại vị trí (j, j) trong feature map.

#### 2.2 Hàm kích hoạt (ReLU)  
Sau khi có feature map từ tầng tích chập, ta dùng hàm ReLU để "lọc" nó, giữ lại các đặc trưng rõ ràng và loại bỏ những phần không quan trọng.  
##### Công thức ReLU  
![image](https://github.com/user-attachments/assets/4e00e067-cb5a-41e4-96e9-c048bf034400)  
Trong đó:  
- Nếu số lớn hơn 0, giữ nguyên.
- Nếu số nhỏ hơn hoặc bằng 0, biến thành 0.

#### 2.3 Tầng Pooling (Pooling Layer)  
Pooling giống như "tóm tắt" feature map, giảm kích thước để tiết kiệm tính toán nhưng vẫn giữ được thông tin quan trọng.  
Loại phố biến: Max Pooling  
- Lấy giá trị lớn nhất trong một vùng nhỏ, thường là 2x2.
##### Công thức Max Pooling  
![image](https://github.com/user-attachments/assets/0cfe5f3b-2d2d-45e4-afa1-e2a851d3f907)  
Giải thích đơn giản: Chia feature map thành các ô 2x2, chọn số lớn nhất trong mỗi ô để tạo feature map nhỏ hơn.  

#### 2.4 Tầng Fully Connected (FC Layer)  
Đây là bước cuối cùng, nơi CNN ghép tất cả đặc trưng lại để đoán xem ảnh là gì.  
##### Công thức của FC Layer  
![image](https://github.com/user-attachments/assets/eb6c1dd1-051b-4c36-911c-029e8f513f63)  
Trong đó:  
x: Vector từ feature map duỗi ra.  
W: Ma trận trọng số.  
b: Bias (độ lệch).  
Giải thích đơn giản: Lấy feature map cuối, "duỗi" thành một hàng số, rồi nhân với trọng số để ra kết quả phân loại.  

### 3. Tổng hợp CNN  
1. Tích chập: Tìm đặc trưng như đường ngang --> Feature map S.
2. ReLU: Lọc bỏ các nét mờ (giá trị âm) --> Feature map Srelu.
3. Pooling: Tóm tắt, giảm kích thước --> Feature map Spooled.
4. Fully Connected: Ghép các đặc trưng, đoán xem là gì.

### 4. Ứng dụng thực tế  
CNN dùng trong nhiều lĩnh vực:  
- Nhận diện khuôn mặt: Facebook dùng CNN để gắn thẻ bạn bè trong ảnh.
- Xe tự lái: Phát hiện biển báo, người đi bộ qua camera.
- Y khoa: Phân tích ảnh X-quang để tìm bệnh.

## Phần 2: Bài tập với CNN  
### Bài 1:  
- **Yêu cầu**: Tăng số lượng epoch từ 5 lên 10 trong phần huấn luyện mô hình.  
- **Hướng dẫn**: Tìm dòng `for epoch in range(5):` và sửa thành `for epoch in range(10):`. Chạy lại code và ghi nhận:  
  - Độ chính xác trên tập test có thay đổi không? Nếu có, tăng hay giảm?  
  - Biểu đồ mất mát (loss) thay đổi thế nào qua 10 epoch? Có xu hướng nào đáng chú ý không (ví dụ: giảm đều, chững lại)?  
-  Giải thích về lý do tại sao số epoch ảnh hưởng đến kết quả.

Cách làm:  
```python
for epoch in range(10):
```
- Độ chính xác trên tập test tăng khi thay đổi số lượng epoch từ 5 lên 10.  
- Biểu đồ mất mát loss giảm mạnh từ epoch 1 đến epoch 2, sau đó nó giảm đều tới epoch 10.  
- Số epoch ảnh hưởng đến kết quả vì số epoch cho biết mô hình sẽ học nhiều hay là học ít. Khi mô hình học ít, nó chưa thể học đầy đủ các đặc trưng của bộ data, dẫn đến kết quả chính xác không cao. Ngược lại, khi epoch tăng, mô hình học được nhiều hơn, nắm được nhiều đặc trưng hơn dẫn đến kết quả chính xác tăng lên đáng kể.

<img width="1169" height="478" alt="image" src="https://github.com/user-attachments/assets/42c59f96-bf04-44e4-9a02-a92250c68fe9" />  

### Bài 2:  
- **Yêu cầu**: Thêm một tầng tích chập thứ ba (`conv3`) vào mô hình `MNIST_CNN`.  
- **Hướng dẫn**:  
  - Trong hàm `__init__`, thêm `self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)` (32 kênh đầu vào từ `conv2`, 64 kênh đầu ra).  
  - Trong hàm `forward`, thêm `x = self.pool(torch.relu(self.conv3(x)))` sau dòng `x = self.pool(torch.relu(self.conv2(x)))`.  
  - Kích thước sau `conv3` và pooling sẽ là 64x1x1 (vì 5x5 -> 3x3 -> 1x1 sau hai lần pooling và tích chập). Sửa tầng `fc1` thành `self.fc1 = nn.Linear(64 * 1 * 1, 10)` và dòng `x.view(-1, 64 * 1 * 1)` tương ứng.  
  - Chạy lại code và ghi nhận độ chính xác mới trên tập test.  
- Viết ngắn gọn về tác dụng của việc thêm tầng tích chập (ví dụ: tìm đặc trưng phức tạp hơn, ảnh hưởng đến độ chính xác).

Cách làm:  
```python
class MNIST_CNN(nn.Module):  
    def __init__(self):  
        super(MNIST_CNN, self).__init__() 
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)  
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.fc1 = nn.Linear(64 * 1 * 1, 10)  

    def forward(self, x): 
        x = self.pool(torch.relu(self.conv1(x))) 
        x = self.pool(torch.relu(self.conv2(x)))  
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 1 * 1)  
        x = self.fc1(x)

        return x
```
- Việc thêm một tầng tích chập giúp mô hình học được các đặc trưng phức tạp hơn của hình ảnh. Các tầng tích chập sâu hơn có khả năng trích xuất những đặc điểm chi tiết hơn, từ đó có thể cải thiện độ chính xác của mô hình trên tập kiểm tra.

<img width="1169" height="478" alt="image" src="https://github.com/user-attachments/assets/b5b5683c-c127-4f98-a6ea-9ff6c6a5e697" />  

### Bài 3:  
- **Yêu cầu**: Thử hai giá trị learning rate khác nhau: 0.001 và 0.1 (thay vì 0.01 ban đầu).  
- **Hướng dẫn**: Tìm dòng `optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)` và thay `lr=0.001` rồi `lr=0.1`. Chạy lại code với từng giá trị và ghi nhận:  
  - Độ chính xác trên tập test với mỗi learning rate.  
  - Biểu đồ mất mát thay đổi ra sao? (Ví dụ: dao động mạnh, giảm chậm, hoặc không hội tụ).  
- Cách learning rate ảnh hưởng đến quá trình học của mô hình như thế nào?

Cách làm:  
- Với lr = 0.001:
```python
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  
```
<img width="1169" height="477" alt="image" src="https://github.com/user-attachments/assets/b94d37d3-9747-4db9-a795-bb7cfd650a23" />  

- Với lr = 0.01:
```python
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  
```
<img width="1171" height="477" alt="image" src="https://github.com/user-attachments/assets/e31f60f0-78d2-4fea-975f-10a8240e0daf" />  

- Biểu đồ mất mát ở lr = 0.001 có sự giảm mạnh từ epoch 1 đến epoch 2, sau đó nó giảm đều từ từ đến epoch 10.  
- Biểu đồ mất mát ở lr = 0.1 có sự dao động nhẹ ở một vài epoch, nhưng nhìn chung biểu đồ có xu hướng giảm.  
- Learning rate quyết định mức độ thay đổi của trọng số sau mỗi lần cập nhật trong quá trình huấn luyện mô hình. Nếu learning rate quá nhỏ, mô hình học rất chậm; nếu quá lớn, quá trình học có thể dao động và không hội tụ. Vì vậy, chọn learning rate phù hợp giúp mô hình hội tụ nhanh và đạt độ chính xác tốt hơn.

### Bài 4  
- **Yêu cầu**: Sửa hàm `visualize_feature_map` để vẽ thêm hai feature map từ tầng `conv2`.  
- **Hướng dẫn**:  
  - Trong hàm `visualize_feature_map`, thêm dòng `conv2_output = torch.relu(self.conv2(self.pool(torch.relu(self.conv1(img)))))` để tính feature map từ `conv2`.  
  - Tăng khung hình từ 3 cột thành 5 cột: `plt.figure(figsize=(20, 4))` và sửa các subplot thành `plt.subplot(1, 5, ...)`.  
  - Thêm hai subplot để vẽ `conv2_output[0, 0]` và `conv2_output[0, 1]` (tương tự như `conv1_output`).  
  - Chạy lại và mô tả sự khác biệt giữa feature map từ `conv1` và `conv2` (ví dụ: chi tiết hơn, trừu tượng hơn).  
- Viết ngắn gọn về sự khác biệt giữa feature map từ các tầng khác nhau.

Cách làm:  
```python
def visualize_feature_map(): 
    model.eval()  
    images, _ = next(iter(test_loader)) 
    img = images[0].unsqueeze(0).to(device)  

    conv1_output = torch.relu(model.conv1(img)) 
    conv2_output = torch.relu(model.conv2(model.pool(torch.relu(model.conv1(img)))))
    plt.figure(figsize=(20, 4))  
    plt.subplot(1, 5, 1)  
    plt.title("Ảnh gốc")  
    plt.imshow(img.cpu().squeeze(), cmap='gray') 
    plt.axis('off')  
    
    plt.subplot(1, 5, 2)  
    plt.title("Feature Map 1")  
    plt.imshow(conv1_output[0, 0].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')  
    
    plt.subplot(1, 5, 3)  
    plt.title("Feature Map 2") 
    plt.imshow(conv1_output[0, 1].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off') 
    
    plt.subplot(1, 5, 4)  
    plt.title("Feature Map 3") 
    plt.imshow(conv2_output[0, 0].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off') 
    
    plt.subplot(1, 5, 5)  
    plt.title("Feature Map 4") 
    plt.imshow(conv2_output[0, 1].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off') 
    plt.show() 

visualize_feature_map()  
```

<img width="1168" height="480" alt="image" src="https://github.com/user-attachments/assets/8940f593-3b53-47ec-bdc9-d1ed069928d8" />  

<img width="1171" height="512" alt="image" src="https://github.com/user-attachments/assets/2acfc71f-e549-49be-a298-1a78ccca788a" />  

- Conv1: hiển thị rõ viền và nét của số 7 gồm đường chéo và đường ngang rõ ràng, độ cong và mép nét đều được giữ lại. Feature map vẫn giữ nhiều chi tiết không gian của ảnh gốc.  
- Conv2: hiển thị các vùng chính của số 7 bao gồm vùng đầu số 7 được nhấn mạnh, các vùng còn lại mờ hơn. Ảnh trở nên thô hơn, tập trung vào hình dạng tổng quát.

## CIFAR-10  
- Khai báo thư viện, load bộ data CIFAR-10 với 10 classes: 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'.
- Thực hiện bước tiền xử lý dữ liệu với transform.Compose():
```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), # cắt ngẫu nhiên một vùng 32x32
    transforms.RandomHorizontalFlip(), # lật ngược ảnh theo chiều ngang
    transforms.RandomRotation(15), # xoay ảnh một góc 15 độ
    transforms.ToTensor(), # chuyển đổi ảnh sang tensor 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # chuẩn hóa dữ liệu
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
```

- Khởi tạo class CIFAR-10 với 6 tầng convolution và 3 tầng fully connected.
```python
class CIFAR10_AdvancedCNN(nn.Module):
    def __init__(self):
        super(CIFAR10_AdvancedCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), # tầng đặc trưng 1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), # tầng đặc trưng 2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), # tầng đặc trưng 3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), # tầng đặc trưng 4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), # tầng đặc trưng 5
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), # tầng đặc trưng 6
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024), # tầng phân loại 1
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512), # tầng phân loại 2
            nn.ReLU(),
            nn.Linear(512, 10) # tầng phân loại 3
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```
- Khởi tạo mô hình CNN
```python
model = CIFAR10_AdvancedCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
```
- Huấn luyện mô hình với 100 epochs:
```python
for epoch in range(100):
    model.train()
    ......
```
- Vẽ biểu đồ sau khi huấn luyện:
<img width="614" height="268" alt="image" src="https://github.com/user-attachments/assets/78d7f213-eb0c-432d-b9a3-8f4703c6f32a" />

- In ra độ chính xác trên tập test:
```python
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\n Độ chính xác trên tập test: {100 * correct / total:.2f}%")
```
<img width="270" height="30" alt="image" src="https://github.com/user-attachments/assets/3b87bfb3-31ea-4558-aed4-ed482a1113c9" />  

- Thực hiện test thử với bộ dữ liệu test:
<img width="1490" height="335" alt="image" src="https://github.com/user-attachments/assets/e1b78166-5a31-401b-9b0a-9f75d9179064" />

## Cat and Dog  
- Khai báo thư viện, định nghĩa cách đọc ảnh với hàm CatDogCustomDataset(), sử dụng các phép tiền xử lý dữ liệu với transform.Compose():
```python
transform_train = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
```
- Khởi tạo class CatDog với 4 tầng 
```python
class CatDog_CNN_Advanced(nn.Module):
    def __init__(self):
        super(CatDog_CNN_Advanced, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
```
- Khởi tạo mô hình CNN:
```python
model = CatDog_CNN_Advanced().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2) 
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
```
- Huấn luyện mô hình với 50 epochs:
```python
for epoch in range(50):
    model.train()
    ........
```
- Vẽ biểu đồ sau huấn luyện:
<img width="614" height="202" alt="image" src="https://github.com/user-attachments/assets/9424b9fb-3d81-40d0-9632-01083b4a5011" />

- In ra độ chính xác trên tập test:
<img width="261" height="27" alt="image" src="https://github.com/user-attachments/assets/a22e13b7-ece6-47a9-b085-fd38452705ee" />

- Thử test với bộ dữ liệu test:
<img width="615" height="155" alt="image" src="https://github.com/user-attachments/assets/8bed81f7-0e86-4b6c-b138-08101ba91846" />

## PlantVillage  
- Khai báo thư viện, thực hiện các bước tiền xử lý ảnh với transform.Compose():
```python
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
```
- Khởi tạo class:
```python
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseCNN, self).__init__()
        def conv_layer(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
        self.features = nn.Sequential(
            conv_layer(3, 32),   
            conv_layer(32, 64),  
            conv_layer(64, 128), 
            conv_layer(128, 256) 
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```
- Khởi tạo mô hình CNN:
```python
model = PlantDiseaseCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
```
- Huấn luyện mô hình với 40 epochs:
```python
for epoch in range(40):
    model.train()
    .......
```
- Vẽ biểu đồ sau huấn luyện:
<img width="615" height="288" alt="image" src="https://github.com/user-attachments/assets/1ad6bf42-bd9d-4117-95ae-07d456f4ec61" />

- In ra độ chính xác trên tập test:
```python
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        out = model(imgs); _, pred = out.max(1); total += lbls.size(0); correct += pred.eq(lbls).sum().item()

print(f"\n Độ chính xác trên tập test: {100. * correct / total:.2f}%")
```
<img width="262" height="29" alt="image" src="https://github.com/user-attachments/assets/620c87d3-5882-4f3d-9b69-0fa50cc0ff70" />  

- Thử test trên bộ dữ liệu test:
<img width="616" height="127" alt="image" src="https://github.com/user-attachments/assets/4e1811c3-26f9-4d93-94b4-549831607937" />  









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
5. PyTorch CNN Tutorial: Build and Train Convolutional Neural Networks in Python.  

















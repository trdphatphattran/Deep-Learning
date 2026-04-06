# XÂY DỰNG WEB CƠ BẢN VỚI CNN  
## Thông tin:  
Sinh viên: Trần Đại Phát  
MSSV: 2374802010379  
Môn học: Giới thiệu về học sâu  
GVHD: Nguyễn Thái Anh  
Năm học: 2025 - 2026    

## Giới thiệu:  
- Xây dựng web với mô hình CNN và 3 bộ data dữ liệu gồm: CIFAR-10, cat and dog, plantvillage giúp cho người dùng có cơ hội trải nghiệm hệ thống dự đoán với mô hình học sâu. Hệ thống cho phép người dùng chọn ảnh bất kỳ như con chó, con mèo, con tàu, lá cây bị bệnh; sau đó hệ thống trả ra kết quả đó là gì cho người dùng có thể nhậm biết.
- Thông qua quá trình huấn luyện 3 bộ dữ liệu với 50 epochs cho mỗi dữ liệu, hệ thống đã có thể nhận diện tương đối chính xác với 3 mô hình.
- Kết quả huấn luyện:
  - CIFAR-10:
  <img width="819" height="362" alt="image" src="https://github.com/user-attachments/assets/b6b2b9b5-1061-4beb-8f69-0d23d5bf834e" />
  - Cat and dog:
  <img width="819" height="368" alt="image" src="https://github.com/user-attachments/assets/03aaf369-3817-4938-83f0-284f93af8125" />
  - Plantvillage:
  <img width="819" height="365" alt="image" src="https://github.com/user-attachments/assets/05a3ee81-d698-49ea-9eff-770a0b0878ff" />

## Cách hoạt động của web  
### 1. Khai báo lớp CNN  
```python
class UniversalCNN(nn.Module):
    def __init__(self, num_classes):
        super(UniversalCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)
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
- Đây là phần định nghĩa cấu trúc mạng CNN. Nếu không có đoạn này, Python sẽ không biết cách lắp ghép các tầng thần kinh để xử lý ảnh.

### 2. Tiền xử lý hình ảnh  
```python
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
```
- Trước khi đưa vào dự đoán, ảnh cần phải trả về đúng kích thước, nếu không sẽ báo lỗi.

### 3. Hàm load model  
```python
def load_ai_model(selection):
    file_map = {
        "Vật thể (CIFAR-10)": "model_cifar.pth",
        "Chó & Mèo": "model_catdog.pth",
        "Bệnh cây trồng": "model_plant.pth"
    }
    
    file_path = file_map[selection]
    
    if not os.path.exists(file_path):
        return None, None
    
    try:
        checkpoint = torch.load(file_path, map_location=device, weights_only=False)
        classes = checkpoint['classes']
        
        model = UniversalCNN(len(classes)).to(device)
        model.load_state_dict(checkpoint['state'])
        model.eval()
        return model, classes
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        return None, None
```
- Đoạn code này có nhiệm vụ tải mô hình đã học tương ứng với lựa chọn của người dùng: nó tìm file trọng số (.pth) trong máy tính, nạp toàn bộ "tri thức" đã học vào khung mạng CNN, đồng thời lấy ra danh sách tên các loại đối tượng để sẵn sàng nhận diện hình ảnh và trả về kết quả.  
### 4. Dự đoán kết quả cuối cùng  
```python
with torch.no_grad():
    output = model(img_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)
            
result = class_names[predicted.item()]
score = confidence.item()
```
- Đây là code trả về kết quả dự đoán cuối cùng và hiển thị ra % đúng của ảnh đó.

## Demo Web  
### 1. Giao diện  
<img width="1321" height="826" alt="image" src="https://github.com/user-attachments/assets/dd0368b1-94ee-418b-96a8-a7e9af2df97f" />  

- Nhìn qua bên trái, người dùng có thể lựa chọn bộ dữ liệu họ muốn dự đoán.

### 2. Load ảnh  
#### CIFAR-10:  
- Thực hiện việc load ảnh thuộc CIFAR-10 ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck').
<img width="1279" height="817" alt="image" src="https://github.com/user-attachments/assets/1d21d564-a3ca-4c5b-be81-e3639e529179" />

#### Cat and dog:  
- Thực hiện việc load ảnh chó hoặc mèo.
<img width="1151" height="809" alt="image" src="https://github.com/user-attachments/assets/4b74dc4a-3432-4577-8a83-9deb814d591c" />

#### Plantvillage:  
- Thực hiện việc load ảnh lá cây bị sâu bệnh lên.
<img width="1149" height="786" alt="image" src="https://github.com/user-attachments/assets/45ae3da1-b13d-4a95-ba5f-ac23bf8cfeee" />

### 3. Hiển thị kết quả  
#### CIFAR-10:  
<img width="1147" height="767" alt="image" src="https://github.com/user-attachments/assets/aa372451-0e02-4987-87ad-0f257340a6d8" />  

#### Cat and dog:  
<img width="1141" height="791" alt="image" src="https://github.com/user-attachments/assets/79c84354-4c2a-4f8f-ae54-2693255440c8" />


#### Plantvillage:  
<img width="1151" height="694" alt="image" src="https://github.com/user-attachments/assets/857595d2-08d6-432e-8075-18115aecfbca" />  











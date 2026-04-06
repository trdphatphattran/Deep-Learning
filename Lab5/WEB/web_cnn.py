import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 3. Hàm Load Model tùy chọn (Có bộ nhớ đệm cache để chạy mượt)
@st.cache_resource
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

# 4. Giao diện Streamlit chính
st.set_page_config(page_title="VLU AI Hub", page_icon="🧪", layout="centered")

st.title("Hệ thống Nhận diện hình ảnh (CNN)")


st.sidebar.header("Cấu hình")
choice = st.sidebar.selectbox("Bạn muốn test bộ dữ liệu nào?", 
                             ["Vật thể (CIFAR-10)", "Chó & Mèo", "Bệnh cây trồng"])

model, class_names = load_ai_model(choice)

if model is None:
    st.warning(f"Model '{choice}' chưa được huấn luyện hoặc thiếu file .pth")
    st.stop() 
else:
    st.sidebar.success(f"Đã kích hoạt bộ não: {choice}")
    st.sidebar.write(f"Nhận diện được: **{len(class_names)}** loại khác nhau.")

# Khu vực Upload ảnh
uploaded_file = st.file_uploader("Tải ảnh lên để AI phân tích...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Ảnh đầu vào", use_container_width=True)
    
    if st.button("Bắt đầu nhận diện"):
        with st.spinner(f'AI đang soi ảnh {choice}...'):
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            result = class_names[predicted.item()]
            score = confidence.item()
            
            st.divider()
            st.balloons()
            st.subheader(f"Kết quả dự đoán: **{result}**")
            st.write(f"Độ tin cậy của AI: **{score:.2%}**")
            
            st.progress(score)


import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# 1. Định nghĩa lại kiến trúc CNN (Phải khớp hoàn toàn với file Train)
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
        x = self.conv1(x); x = self.bn1(x); x = self.relu1(x); x = self.pool1(x)
        x = self.conv2(x); x = self.bn2(x); x = self.relu2(x); x = self.pool2(x)
        x = self.conv3(x); x = self.bn3(x); x = self.relu3(x); x = self.pool3(x)
        x = self.conv4(x); x = self.bn4(x); x = self.relu4(x); x = self.pool4(x)
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 2. Cấu hình Load Model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

@st.cache_resource
def load_cnn_model():
    model = CatDog_CNN_Advanced().to(device)
    checkpoint = torch.load('cat_dog_cnn_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

model = load_cnn_model()
classes = ['Mèo (Cat)', 'Chó (Dog)']

# 3. Tiền xử lý ảnh (Giống hệt transform_test lúc train)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 4. Giao diện Streamlit
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐶")
st.title("Phân loại Chó và Mèo (CNN)")
st.write("Tải lên một tấm hình và mô hình AI sẽ đoán xem đó là chó hay mèo!")

uploaded_file = st.file_uploader("Chọn một tấm ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh đã upload
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Ảnh bạn đã tải lên', use_container_width=True)
    
    if st.button("Bắt đầu nhận diện"):
        with st.spinner('Đang phân tích...'):
            # Tiền xử lý
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            # Dự đoán
            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(prob, 1)
            
            # Hiển thị kết quả
            res_class = classes[predicted.item()]
            conf_score = confidence.item() * 100
            
            st.divider()
            if predicted.item() == 0:
                st.success(f"Dự đoán: **{res_class}**")
            else:
                st.info(f"Dự đoán: **{res_class}**")
            
            st.progress(conf_score / 100)
            st.write(f"Độ tin cậy: **{conf_score:.2f}%**")

st.sidebar.markdown("---")
st.sidebar.write("💻 **Thông số hệ thống:**")
st.sidebar.write(f"Đang chạy trên: `{device}`")
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import streamlit as st

# UNet модель
class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        xe11 = torch.relu(self.e11(x))
        xe12 = torch.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = torch.relu(self.e21(xp1))
        xe22 = torch.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = torch.relu(self.e31(xp2))
        xe32 = torch.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = torch.relu(self.e41(xp3))
        xe42 = torch.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = torch.relu(self.e51(xp4))
        xe52 = torch.relu(self.e52(xe51))

        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = torch.relu(self.d11(xu11))
        xd12 = torch.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = torch.relu(self.d21(xu22))
        xd22 = torch.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = torch.relu(self.d31(xu33))
        xd32 = torch.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = torch.relu(self.d41(xu44))
        xd42 = torch.relu(self.d42(xd41))

        return self.outconv(xd42)

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.write(f"Using device: {device}")
    model_path = '/home/savely/ds_bootcamp/ds-phase-2/Computer-vision-project/model/unet_model.pt'

    try:
        model = torch.load(model_path, map_location=device)
        if isinstance(model, nn.Module):
            model.to(device)
        else:
            raise ValueError("Файл не содержит полную модель, пробуем state_dict...")
    except:
        try:
            model = UNet(1).to(device)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {str(e)}")
            return None

    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def postprocess_mask(mask_tensor, original_size):
    mask_np = torch.sigmoid(mask_tensor).squeeze().cpu().detach().numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[0]  # если размерность [1, H, W]
    mask_np = (mask_np > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_np).resize(original_size).convert('L')
    color_mask = Image.merge("RGB", (mask_img,)*3)
    return mask_img, color_mask

def load_image(uploaded_file=None, url=None):
    try:
        if uploaded_file:
            return Image.open(uploaded_file).convert("RGB")
        elif url:
            response = requests.get(url)
            return Image.open(BytesIO(response.content)).convert("RGB")
    except:
        st.error("Не удалось загрузить изображение")
    return None

def main():
    st.title("Семантическая сегментация спутниковых изображений (UNet)")
    option = st.radio("Источник изображения:", ["Файл", "URL"])

    image = None
    if option == "Файл":
        uploaded_file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            image = load_image(uploaded_file=uploaded_file)
    else:
        url = st.text_input("Введите URL:")
        if url:
            image = load_image(url=url)

    if image:
        st.image(image, caption="Оригинал", use_container_width=True)
        if st.button("Сегментировать"):
            with st.spinner("Модель в работе..."):
                model = load_model()
                if model:
                    device = next(model.parameters()).device
                    input_tensor = preprocess_image(image).to(device)
                    with torch.no_grad():
                        output = model(input_tensor)
                    mask_img, color_mask = postprocess_mask(output, image.size)
                    col1, col2 = st.columns(2)
                    col1.image(image, caption="Оригинал", use_container_width=True)
                    col2.image(color_mask, caption="Сегментация", use_container_width=True)

                    if st.checkbox("Показать наложение маски"):
                        overlay = Image.blend(image.convert("RGBA"),
                                              color_mask.convert("RGBA"),
                                              alpha=0.5)
                        st.image(overlay, caption="Наложение", use_container_width=True)

if __name__ == "__main__":
    main()


import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

page_bg_img = f"""
<style>

.st-emotion-cache-1yiq2ps {{
background-image: url("https://cbeditz.com/public/cbeditz/large/black-gold-youtube-gaming-banner-background-free-images-9ynh0xq2mg.jpg");
background-size: cover;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
# Hàm chuyển đổi RGB sang CMYK
def rgb_to_cmyk(image):
    # Chuyển đổi từ RGB sang CMYK
    img = np.array(image).astype(float) / 255
    K = 1 - np.max(img, axis=2)
    C = (1 - img[..., 0] - K) / (1 - K + 1e-5)
    M = (1 - img[..., 1] - K) / (1 - K + 1e-5)
    Y = (1 - img[..., 2] - K) / (1 - K + 1e-5)
    C = (C * 255).astype(np.uint8)
    M = (M * 255).astype(np.uint8)
    Y = (Y * 255).astype(np.uint8)
    K = (K * 255).astype(np.uint8)
    return cv2.merge((C, M, Y, K))

# Hàm hiển thị hình ảnh
def display_image(image, caption):
    st.image(image, caption=caption, use_container_width=True)

# Hàm chính
def main():
    st.title("Ứng dụng chuyển đổi hệ màu hình ảnh")
    uploaded_file = st.file_uploader("Tải lên hình ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Lựa chọn hệ màu
        color_spaces = ["Grayscale", "HSV", "HSL"]
        option = st.selectbox("Chọn hệ màu", color_spaces)

        if option == "Grayscale":
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            display_image(gray, "Hình ảnh Grayscale")
            result_image = Image.fromarray(gray)
        elif option == "HSV":
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            display_image(hsv_rgb, "Hình ảnh HSV")
            result_image = Image.fromarray(hsv_rgb)
        elif option == "HSL":
            hls = cv2.cvtColor(image_np, cv2.COLOR_RGB2HLS)
            hls_rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
            display_image(hls_rgb, "Hình ảnh HSL")
            result_image = Image.fromarray(hls_rgb)
    

        # Nút tải xuống hình ảnh
        st.markdown("---")
        st.markdown("### 📥 Tải xuống hình ảnh đã chuyển đổi")
        buf = BytesIO()
        result_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Tải xuống",
            data=byte_im,
            file_name=f"converted_{option}.png",
            mime="image/png"
        )
import streamlit as st
import numpy as np
from PIL import Image

def rgb_to_cmyk(img):
    # Chuyển đổi ảnh sang mảng numpy và chuẩn hóa giá trị pixel
    img = np.asarray(img).astype(float) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    # Tính toán kênh K
    k = 1 - np.max([r, g, b], axis=0)

    # Tránh chia cho 0
    denominator = 1 - k
    denominator[denominator == 0] = 1

    # Tính toán các kênh C, M, Y
    c = (1 - r - k) / denominator
    m = (1 - g - k) / denominator
    y = (1 - b - k) / denominator

    # Thay thế các giá trị không xác định bằng 0
    c[np.isnan(c)] = 0
    m[np.isnan(m)] = 0
    y[np.isnan(y)] = 0

    # Chuyển đổi về dạng uint8
    cmyk = (np.dstack((c, m, y, k)) * 255).astype(np.uint8)
    return cmyk

def colorize_channel(channel, color):
    # Tạo ảnh màu từ kênh xám
    zeros = np.zeros_like(channel)
    if color == 'cyan':
        return np.stack([zeros, channel, channel], axis=2)
    elif color == 'magenta':
        return np.stack([channel, zeros, channel], axis=2)
    elif color == 'yellow':
        return np.stack([channel, channel, zeros], axis=2)
    elif color == 'black':
        return np.stack([channel, channel, channel], axis=2)
    else:
        return np.stack([channel, channel, channel], axis=2)
    

if __name__ == "__main__":
    main()
st.logo(
    image="https://fgam.hcmute.edu.vn/Resources/Images/SubDomain/fgam/Logo%20khoa/Logo_FGAM.png",
    size="large", 
    icon_image="https://fgam.hcmute.edu.vn/Resources/Images/SubDomain/fgam/CIP%20FGAM-01.jpg" 
)
import streamlit as st

# Hiển thị logo trong sidebar với chiều rộng 150px
st.sidebar.image("https://fgam.hcmute.edu.vn/Resources/Images/SubDomain/fgam/Logo%20khoa/Logo_FGAM.png", width=150)


def main():
    st.title("Tách Kênh Màu CMYK từ Ảnh RGB")
    uploaded_file = st.file_uploader("Chọn một ảnh", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh Gốc", use_container_width=True)

        cmyk = rgb_to_cmyk(image)
        c, m, y, k = cmyk[..., 0], cmyk[..., 1], cmyk[..., 2], cmyk[..., 3]

        st.write("### Các Kênh Màu CMYK:")
        col1, col2 = st.columns(2)
        with col1:
            st.image(colorize_channel(c, 'cyan'), caption="Kênh Cyan", use_container_width=True)
            st.image(colorize_channel(y, 'yellow'), caption="Kênh Yellow", use_container_width=True)
        with col2:
            st.image(colorize_channel(m, 'magenta'), caption="Kênh Magenta", use_container_width=True)
            st.image(colorize_channel(k, 'black'), caption="Kênh Black (Key)", use_container_width=True)
        
            

if __name__ == "__main__":
    main()


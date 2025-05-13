import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO


st.set_page_config(page_title="Tách màu CMYK", layout="wide")
st.title("🖼️ Tách ảnh thành 4 bản màu Cyan – Magenta – Yellow – Black (CMYK)")

uploaded_file = st.file_uploader("📂 Tải ảnh RGB bất kỳ (PNG/JPG)", type=["jpg", "jpeg", "png"])

# --- Hàm chuyển RGB (0–255) → CMYK (0–1) ---
def rgb_to_cmyk_array(rgb_image):
    rgb_norm = rgb_image / 255.0
    R, G, B = rgb_norm[..., 0], rgb_norm[..., 1], rgb_norm[..., 2]
    K = 1 - np.max(rgb_norm, axis=2)
    C = (1 - R - K) / (1 - K + 1e-8)
    M = (1 - G - K) / (1 - K + 1e-8)
    Y = (1 - B - K) / (1 - K + 1e-8)
    C[np.isnan(C)] = 0
    M[np.isnan(M)] = 0
    Y[np.isnan(Y)] = 0
    return C, M, Y, K

# --- Hàm tô màu cho mỗi kênh CMYK ---
def apply_color_mask_channel(mask, channel="C"):
    """Trả về ảnh chỉ chứa kênh màu CMYK được tách riêng, mô phỏng in thực tế"""
    h, w = mask.shape
    color = {
        "C": (0, 255, 255),    # Cyan
        "M": (255, 0, 255),    # Magenta
        "Y": (255, 255, 0),    # Yellow
        "K": (100, 100, 100)   # Black
    }[channel]
    base = np.ones((h, w, 3), dtype=np.uint8) * 255  # Nền trắng
    overlay = np.zeros_like(base)
    for i in range(3):
        overlay[..., i] = (1 - mask) * 255 + mask * color[i]
    return overlay.astype(np.uint8)


# ---Tải bản tách màu---
def get_download_button(img_array, label="Tải ảnh", filename="split.png"):
    img_pil = Image.fromarray(img_array)
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label=label,
        data=byte_im,
        file_name=filename,
        mime="image/png"
    )



# --- Xử lý khi có ảnh ---
if uploaded_file:
    # Ảnh gốc
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("🖼️ Ảnh gốc")
    
    #  Resize ảnh gốc để hiển thị nhỏ hơn
    display_width = 250  # hoặc 400 nếu bạn muốn nhỏ hơn nữa
    w_percent = display_width / float(image.width)
    new_height = int(float(image.height) * w_percent)
    image_resized = image.resize((display_width, new_height), Image.LANCZOS)

    #  Hiển thị ảnh đã resize
    st.image(image_resized, caption="Ảnh gốc (đã thu nhỏ)", use_container_width=False)

    # Convert → NumPy
    rgb_np = np.array(image)
    C, M, Y, K = rgb_to_cmyk_array(rgb_np)

    # Hiển thị bản tách màu với màu tương ứng
    st.subheader("🖨️ Bốn bản tách màu theo mực in thực tế (CMYK)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 🟦 Cyan")
        cyan_img = apply_color_mask_channel(C, "C")  # xanh cyan
        st.image(cyan_img, use_container_width=True)
        get_download_button(cyan_img, label="⬇️ Tải bản Cyan", filename="cyan_channel.png")


    with col2:
        st.markdown("### 🟣 Magenta")
        magenta_img = apply_color_mask_channel(M, "M")  # tím hồng
        st.image(magenta_img, use_container_width=True)
        get_download_button(magenta_img, "⬇️ Tải bản Magenta", "magenta_channel.png")


    with col3:
        st.markdown("### 🟨 Yellow")
        yellow_img = apply_color_mask_channel(Y, "Y")  # vàng
        st.image(yellow_img, use_container_width=True)
        get_download_button(yellow_img, "⬇️ Tải bản Yellow", "yellow_channel.png")


    with col4:
        st.markdown("### ⚫ Black (Key)")
        black_img = apply_color_mask_channel(K, "K")  # xám đen
        st.image(black_img, use_container_width=True)
        get_download_button(black_img, "⬇️ Tải bản Black", "black_channel.png")

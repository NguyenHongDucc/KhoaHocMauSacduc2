import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.pyplot as plt

#hình background


def analyze_image_colors(image):
    # Chuyển hình ảnh sang chế độ RGB
    image = image.convert("RGB")

    # Lấy kích thước hình ảnh
    width, height = image.size
    total_pixels = width * height

    # Khởi tạo bộ đếm cho các màu R, G, B
    r_count = 0
    g_count = 0
    b_count = 0

    # Duyệt qua từng pixel và đếm số lần xuất hiện của các màu R, G, B
    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))

            # Đếm số pixel có màu đỏ, xanh lá, và xanh dương
            r_count += r > g and r > b  # Đếm pixel có màu đỏ nhiều nhất
            g_count += g > r and g > b  # Đếm pixel có màu xanh lá nhiều nhất
            b_count += b > r and b > g  # Đếm pixel có màu xanh dương nhiều nhất

    # Tính phần trăm mỗi màu
    r_percentage = (r_count / total_pixels) * 100
    g_percentage = (g_count / total_pixels) * 100
    b_percentage = (b_count / total_pixels) * 100

    # Trả về phần trăm
    return r_percentage, g_percentage, b_percentage

# Giao diện Streamlit
st.title("Phân tích tỉ lệ 3 màu R, G, B trong hình ảnh")
st.write("Tải lên hình ảnh của bạn để phân tích phần trăm ba màu đỏ, xanh lá, xanh dương.")

# Tải hình ảnh từ máy tính
uploaded_image = st.file_uploader("Chọn hình ảnh", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Mở hình ảnh đã tải lên
    image = Image.open(uploaded_image)

    # Hiển thị hình ảnh lên giao diện
    st.image(image, caption='Hình ảnh đã tải lên', use_container_width=True)

    # Phân tích và tính toán phần trăm các màu
    r_percentage, g_percentage, b_percentage = analyze_image_colors(image)

    # Hiển thị kết quả phần trăm màu sắc
    st.subheader("Phần trăm màu sắc:")
    st.write(f"Phần trăm màu đỏ (R): {r_percentage:.2f}%")
    st.write(f"Phần trăm màu xanh lá (G): {g_percentage:.2f}%")
    st.write(f"Phần trăm màu xanh dương (B): {b_percentage:.2f}%")

    # Tạo biểu đồ hình tròn để hiển thị phần trăm màu sắc
    colors = ['Red', 'Green', 'Blue']
    percentages = [r_percentage, g_percentage, b_percentage]

    # Vẽ đồ thị hình tròn với nền màu trắng
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(percentages, labels=colors, autopct='%1.1f%%', colors=['red', 'green', 'blue'], startangle=90, wedgeprops={'edgecolor': 'black'})

    # Thêm tiêu đề cho đồ thị với màu trắng
    ax.set_title("Phần trăm màu sắc trong hình ảnh", color='white')

    # Thiết lập nền của đồ thị thành màu trắng
    fig.patch.set_facecolor('white')  # Nền trắng cho đồ thị

    # Vẽ các ô màu tương ứng với các màu sắc
    legend_colors = ['red', 'green', 'blue']
    legend_labels = ['Red', 'Green', 'Blue']

    # Vẽ các ô màu và chữ tương ứng bên ngoài biểu đồ
    for i, (label, color) in enumerate(zip(legend_labels, legend_colors)):
        ax.add_patch(plt.Rectangle((1.2, 1 - 0.1 * i), 0.05, 0.05, color=color))  # Vẽ ô màu
        ax.text(1.3, 1 - 0.1 * i, f"{label}", color='white', fontsize=12, ha='left')  # Vẽ chữ với màu trắng

    
    st.pyplot(fig, transparent=True)
import streamlit as st
from PIL import Image
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def extract_colors(image, num_colors=10):
    image = image.convert('RGB')
    img_array = np.array(image)
    pixels = img_array.reshape(-1, 3)
    counter = Counter(map(tuple, pixels))
    most_common = counter.most_common(num_colors)
    total_pixels = sum(counter.values())
    return [(color, count / total_pixels) for color, count in most_common]

def display_color_palette(colors):
    st.markdown("### 🎨 Bảng màu phổ biến trong ảnh:")
    for color, ratio in colors:
        hex_color = rgb_to_hex(color)
        st.markdown(
            f'<div style="display: flex; align-items: center; margin-bottom: 8px;">'
            f'<div style="width: 40px; height: 40px; background-color: {hex_color}; border: 1px solid #000;"></div>'
            f'<div style="margin-left: 10px;">Mã màu: <b>{hex_color}</b> | Tỉ lệ: <b>{ratio:.2%}</b></div>'
            f'</div>',
            unsafe_allow_html=True
        )

def main():
    st.title("🔍 Phân tích các mã màu dùng trong hình ảnh")
    uploaded_file = st.file_uploader("Tải lên hình ảnh (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Hình ảnh gốc", use_container_width=True)

        num_colors = st.slider("Số lượng màu phổ biến cần phân tích", 3, 20, 8)
        colors = extract_colors(image, num_colors=num_colors)
        display_color_palette(colors)

if __name__ == "__main__":
    main()
def set_background_image():
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("https://cbeditz.com/public/cbeditz/large/black-gold-youtube-gaming-banner-background-free-images-9ynh0xq2mg.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_image()
import streamlit as st

st.logo(
    image="https://fgam.hcmute.edu.vn/Resources/Images/SubDomain/fgam/Logo%20khoa/Logo_FGAM.png",  
    size="large", 
    icon_image="https://fgam.hcmute.edu.vn/Resources/Images/SubDomain/fgam/CIP%20FGAM-01.jpg"  
)
import streamlit as st

# Hiển thị logo trong sidebar với chiều rộng 150px
st.sidebar.image("https://fgam.hcmute.edu.vn/Resources/Images/SubDomain/fgam/Logo%20khoa/Logo_FGAM.png", width=150)






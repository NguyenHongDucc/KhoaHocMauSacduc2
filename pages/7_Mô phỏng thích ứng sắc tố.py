import streamlit as st
import numpy as np
from PIL import Image
from skimage import color, img_as_float
from io import BytesIO
import base64

st.set_page_config(page_title="Thích ứng sắc độ mở rộng", layout="wide")
st.title("🌞 Mô phỏng Thích ứng Sắc độ (Chromatic Adaptation) + CMYK + Ảnh")

M_bradford = np.array([
    [ 0.8951,  0.2664, -0.1614],
    [-0.7502,  1.7135,  0.0367],
    [ 0.0389, -0.0685,  1.0296]
])
M_inv = np.linalg.inv(M_bradford)

white_points = {
    "D65": [95.047, 100.000, 108.883],
    "D50": [96.421, 100.000, 82.519],
    "A":   [109.850, 100.000, 35.585],
    "C":   [98.074, 100.000, 118.232],
    "E":   [100.000, 100.000, 100.000],
    "D75": [94.972, 100.000, 122.638],
}

def lab_to_xyz(L, a, b, white):
    Xn, Yn, Zn = white
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    def f_inv(t):
        delta = 6 / 29
        return t**3 if t > delta else 3 * delta**2 * (t - 4 / 29)
    xr, yr, zr = f_inv(fx), f_inv(fy), f_inv(fz)
    return np.array([xr * Xn, yr * Yn, zr * Zn])

def xyz_to_lab(X, Y, Z, white):
    Xn, Yn, Zn = white
    xr, yr, zr = X/Xn, Y/Yn, Z/Zn
    def f(t):
        delta = 6 / 29
        return t**(1/3) if t > delta**3 else (t / (3 * delta**2)) + 4 / 29
    fx, fy, fz = f(xr), f(yr), f(zr)
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])

def cmyk_to_rgb(c, m, y, k):
    c, m, y, k = [x / 100 for x in (c, m, y, k)]
    r = 1 - min(1, c * (1 - k) + k)
    g = 1 - min(1, m * (1 - k) + k)
    b = 1 - min(1, y * (1 - k) + k)
    return np.array([r, g, b])

def rgb_to_lab(rgb):
    return color.rgb2lab(rgb.reshape(1, 1, 3)).flatten()

def lab_to_rgb(lab):
    rgb = color.lab2rgb(lab.reshape(1, 1, 3)).flatten()
    return np.clip(rgb, 0, 1)

def pil_to_base64(im):
    buf = BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def rgb_to_cmyk(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    K = 1 - np.max(rgb, axis=2)
    C = (1 - r - K) / (1 - K + 1e-8)
    M = (1 - g - K) / (1 - K + 1e-8)
    Y = (1 - b - K) / (1 - K + 1e-8)
    C[np.isnan(C)] = 0
    M[np.isnan(M)] = 0
    Y[np.isnan(Y)] = 0
    return np.stack([C, M, Y, K], axis=2)

def cmyk_to_rgb_image(cmyk):
    C, M, Y, K = cmyk[..., 0], cmyk[..., 1], cmyk[..., 2], cmyk[..., 3]
    R = 1 - np.clip(C * (1 - K) + K, 0, 1)
    G = 1 - np.clip(M * (1 - K) + K, 0, 1)
    B = 1 - np.clip(Y * (1 - K) + K, 0, 1)
    return np.stack([R, G, B], axis=2)



def rgb_to_cmyk(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    K = 1 - np.max(rgb, axis=2)
    C = (1 - r - K) / (1 - K + 1e-8)
    M = (1 - g - K) / (1 - K + 1e-8)
    Y = (1 - b - K) / (1 - K + 1e-8)
    C[np.isnan(C)] = 0
    M[np.isnan(M)] = 0
    Y[np.isnan(Y)] = 0
    return np.stack([C, M, Y, K], axis=2)

def cmyk_to_rgb_image(cmyk):
    C, M, Y, K = cmyk[..., 0], cmyk[..., 1], cmyk[..., 2], cmyk[..., 3]
    R = 1 - np.clip(C * (1 - K) + K, 0, 1)
    G = 1 - np.clip(M * (1 - K) + K, 0, 1)
    B = 1 - np.clip(Y * (1 - K) + K, 0, 1)
    return np.stack([R, G, B], axis=2)

def rgb_to_cmyk_pixel(rgb):
    r, g, b = rgb
    K = 1 - max(r, g, b)
    if K >= 1.0:
        return (0, 0, 0, 1)
    C = (1 - r - K) / (1 - K + 1e-8)
    M = (1 - g - K) / (1 - K + 1e-8)
    Y = (1 - b - K) / (1 - K + 1e-8)
    return (C, M, Y, K)


tab1, tab2 = st.tabs(["🎨 Chuyển đổi màu đơn", "🖼️ Chuyển đổi hình ảnh"])

with tab1:
    st.subheader("🔹 Nhập màu gốc (CMYK)")
    c = st.number_input("C (%)", 0.0, 100.0, 20.0)
    m = st.number_input("M (%)", 0.0, 100.0, 30.0)
    y = st.number_input("Y (%)", 0.0, 100.0, 40.0)
    k = st.number_input("K (%)", 0.0, 100.0, 10.0)

    st.subheader("🔸 Chọn nguồn sáng")
    src = st.selectbox("Nguồn sáng ban đầu", list(white_points.keys()), index=0)
    dst = st.selectbox("Nguồn sáng mục tiêu", list(white_points.keys()), index=1)

    if st.button("🎯 Chuyển đổi màu"):
        wp_src = np.array(white_points[src])
        wp_dst = np.array(white_points[dst])

        rgb = cmyk_to_rgb(c, m, y, k)
        lab = rgb_to_lab(rgb)

        xyz = lab_to_xyz(*lab, wp_src)
        lms_src = M_bradford @ wp_src
        lms_dst = M_bradford @ wp_dst
        adapt_matrix = np.diag(lms_dst / lms_src)
        lms_color = M_bradford @ xyz
        lms_adapted = adapt_matrix @ lms_color
        xyz_adapted = M_inv @ lms_adapted
        lab_adapted = xyz_to_lab(*xyz_adapted, wp_dst)
        rgb_adapted = lab_to_rgb(lab_adapted)

        delta_e = np.linalg.norm(lab - lab_adapted)

        color1 = f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})"
        color2 = f"rgb({int(rgb_adapted[0]*255)}, {int(rgb_adapted[1]*255)}, {int(rgb_adapted[2]*255)})"
        st.markdown(f"""
        <div style='display:flex;gap:40px;justify-content:center;margin-top:20px'>
            <div style='text-align:center'>
                <p style='font-weight:bold;font-size:18px'>🎨 Màu gốc</p>
                <div style='width:300px;height:150px;background:{color1};border:3px solid black'></div>
            </div>
            <div style='text-align:center'>
                <p style='font-weight:bold;font-size:18px'>🌈 Màu sau thích ứng</p>
                <div style='width:300px;height:150px;background:{color2};border:3px solid black'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        
        st.write(f"📏 ∆E = {delta_e:.4f}")
        if 0 <= delta_e <= 1:
            st.info("🔍 Sự sai biệt không cảm nhận được.")
        elif 1 <= delta_e <= 2:
            st.info("🧐 Sự sai biệt rất nhỏ, cảm nhận được bởi người có kinh nghiệm.")
        elif 2 <= delta_e <=3.5:
            st.info("👁️ Sự sai biệt tương đối, có thể cảm nhận bằng người có kinh nghiệm")
        elif 3.5 <= delta_e <= 5:
            st.info("👁️ Sự sai biệt lớn.")
        else:
            st.info("🔴 Rõ ràng có sự khác biệt màu sắc.")

        cmyk_adapted = rgb_to_cmyk_pixel(rgb_adapted)
        cmyk_percent = [round(x * 100, 2) for x in cmyk_adapted]
        st.write(f"🖨️ CMYK sau thích ứng: C={cmyk_percent[0]}%, M={cmyk_percent[1]}%, Y={cmyk_percent[2]}%, K={cmyk_percent[3]}%")


with tab2:
    st.subheader("📂 Tải ảnh RGB (JPG/PNG)")
    uploaded_file = st.file_uploader("Tải ảnh", type=["jpg", "jpeg", "png"])

    src2 = st.selectbox("Nguồn sáng ban đầu (ảnh)", list(white_points.keys()), index=0, key="src2")
    dst2 = st.selectbox("Nguồn sáng mục tiêu (ảnh)", list(white_points.keys()), index=1, key="dst2")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = img_as_float(np.array(image))
        h, w, _ = img_np.shape
        cmyk_img = rgb_to_cmyk(img_np)
        rgb_from_cmyk = cmyk_to_rgb_image(cmyk_img)
        lab_img = color.rgb2lab(rgb_from_cmyk)
        flat_lab = lab_img.reshape(-1, 3)

        wp_src = np.array(white_points[src2])
        wp_dst = np.array(white_points[dst2])

        xyz_flat = np.array([lab_to_xyz(*lab, wp_src) for lab in flat_lab])
        lms_src = M_bradford @ wp_src
        lms_dst = M_bradford @ wp_dst
        adapt_matrix = np.diag(lms_dst / lms_src)
        lms = (M_bradford @ xyz_flat.T).T
        lms_adapted = (adapt_matrix @ lms.T).T
        xyz_adapted = (M_inv @ lms_adapted.T).T
        lab_adapted = np.array([xyz_to_lab(*xyz, wp_dst) for xyz in xyz_adapted])

        delta_e = np.mean(np.linalg.norm(flat_lab - lab_adapted, axis=1))
        st.info(f"📏 ∆E trung bình (CIELAB): {delta_e:.4f}")
        if 0 <= delta_e <= 1:
            st.info("🔍 Sự sai biệt không cảm nhận được.")
        elif 1 <= delta_e <= 2:
            st.info("🧐 Sự sai biệt rất nhỏ, cảm nhận được bởi người có kinh nghiệm.")
        elif 2 <= delta_e <=3.5:
            st.info("👁️ Sự sai biệt tương đối, có thể cảm nhận bằng người có kinh nghiệm")
        elif 3.5 <= delta_e <= 5:
            st.info("👁️ Sự sai biệt lớn.")
        else:
            st.info("🔴 Rõ ràng có sự khác biệt màu sắc.")

        rgb_adapted = color.lab2rgb(lab_adapted.reshape(h, w, 3))
        img_resized = image.resize((300, int(image.height * 300 / image.width)))
        rgb_clipped = np.clip(rgb_adapted, 0, 1)
        rgb_img_pil = Image.fromarray((rgb_clipped * 255).astype(np.uint8))
        adapted_resized = rgb_img_pil.resize((300, int(rgb_img_pil.height * 300 / rgb_img_pil.width)))
        

        b64_img1 = pil_to_base64(img_resized)
        b64_img2 = pil_to_base64(adapted_resized)

        st.markdown(f"""
        <div style='display:flex;gap:40px;justify-content:center;margin-top:20px'>
            <div style='text-align:center'>
                <p style='font-weight:bold;font-size:18px'>🖼️ Ảnh gốc</p>
                <img src='data:image/png;base64,{b64_img1}' width='300px'/>
            </div>
            <div style='text-align:center'>
                <p style='font-weight:bold;font-size:18px'>🌈 Ảnh sau thích ứng</p>
                <img src='data:image/png;base64,{b64_img2}' width='300px'/>
            </div>
        </div>
        """, unsafe_allow_html=True)

        buf = BytesIO()
        rgb_img_pil.save(buf, format="PNG")
        st.download_button("⬇️ Tải ảnh sau thích ứng", data=buf.getvalue(), file_name="adapted_image.png", mime="image/png")
st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("https://img5.thuthuatphanmem.vn/uploads/2021/12/06/anh-nen-full-hd-trang-xam-dep_100112864.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

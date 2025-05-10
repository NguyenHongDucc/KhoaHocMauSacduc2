import streamlit as st
st.set_page_config(
    page_title="Khoa học Màu sắc", layout="wide"
)

st.write("### Chào mừng bạn đến với project Khoa học Màu sắc của tôi!")
st.write("# Mình tên là Nguyễn Hồng Đức, MSSV 22158053")
st.write("# Sinh viên Trường đại học sư phạm kỹ thuật TpHcm")
st.write("# Khoa in và truyền thông")


st.markdown(
    """
    <div style="display: flex; justify-content: flex-end;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/b/b9/Logo_Tr%C6%B0%E1%BB%9Dng_%C4%90%E1%BA%A1i_H%E1%BB%8Dc_S%C6%B0_Ph%E1%BA%A1m_K%E1%BB%B9_Thu%E1%BA%ADt_TP_H%E1%BB%93_Ch%C3%AD_Minh.png" width="130" style="margin-right:10px;">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSQ25mw5Ic55r3Y83GvxAB0h-63315IooQNaA&s" width="130">
    </div>
    """,
    unsafe_allow_html=True
)
import streamlit as st

page_bg_img = f"""
<style>

.st-emotion-cache-1yiq2ps {{
background-image: url("https://i.pinimg.com/736x/79/ab/d7/79abd72250c0004e626c1fa1986c9f35.jpg");
background-size: cover;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

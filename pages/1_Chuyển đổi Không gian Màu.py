import streamlit as st
import numpy as np
import colour

st.set_page_config(layout="wide")

st.write('## Chuyển đổi không gian màu')

chon_khong_gian_mau = st.sidebar.radio("Chuyển đổi không gian màu", ("sRGB to XYZ", "XYZ to sRGB"))

col1, col2 = st.columns(2)

if chon_khong_gian_mau == "sRGB to XYZ":
    with col1:
        st.write("### sRGB to XYZ")
        R = st.number_input('Nhập R:')
        G = st.number_input('Nhập G:')
        B = st.number_input('Nhập B:')
        if st.button('Chuyển sang XYZ'):
            R = R/255.0
            G = G/255.0
            B = B/255.0
            RGB = np.array([R, G, B])
            XYZ = colour.sRGB_to_XYZ(RGB)
            X = XYZ[0]*100
            Y = XYZ[1]*100
            Z = XYZ[2]*100
            s = '### X = %.2f, Y = %.2f, Z = %.2f' % (X, Y, Z)
            st.write(s)
elif chon_khong_gian_mau == "XYZ to sRGB":
    with col1:
        st.write("### XYZ to sRGB")
        X = st.number_input('Nhập X:')
        Y = st.number_input('Nhập Y:')
        Z = st.number_input('Nhập Z:')
        if st.button('Chuyển sang sRGB'):
            X = X/100.0
            Y = Y/100.0
            Z = Z/100.0
            XYZ = np.array([X, Y, Z])
            RGB = colour.XYZ_to_sRGB(XYZ)
            R = RGB[0]*255
            G = RGB[1]*255
            B = RGB[2]*255
            s = '### R = %.0f, G = %.0f, B = %.0f' % (R, G, B)
            st.write(s)
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
